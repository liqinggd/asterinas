// SPDX-License-Identifier: MPL-2.0

use alloc::{boxed::Box, collections::BTreeMap, string::ToString, sync::Arc, vec::Vec};
use core::{fmt::Debug, mem::size_of};

use aster_block::{
    bio::{BioEnqueueError, BioStatus, BioType, SubmittedBio},
    request_queue::{BioRequest, BioRequestSingleQueue},
};
use aster_frame::{
    io_mem::IoMem,
    sync::SpinLock,
    trap::TrapFrame,
    vm::{DmaDirection, DmaReader, DmaStream, DmaWriter, VmAllocOptions, VmIo},
};
use aster_util::{id_allocator::IdAlloc, safe_ptr::SafePtr};
use log::info;
use pod::Pod;

use super::{BlockFeatures, VirtioBlockConfig};
use crate::{
    device::{
        block::{ReqType, RespStatus},
        VirtioDeviceError,
    },
    queue::VirtQueue,
    transport::VirtioTransport,
};

#[derive(Debug)]
pub struct BlockDevice {
    device: DeviceInner,
    /// The software staging queue.
    queue: BioRequestSingleQueue,
}

impl BlockDevice {
    /// Creates a new VirtIO-Block driver and registers it.
    pub(crate) fn init(transport: Box<dyn VirtioTransport>) -> Result<(), VirtioDeviceError> {
        let block_device = {
            let device = DeviceInner::init(transport)?;
            Self {
                device,
                queue: BioRequestSingleQueue::new(),
            }
        };
        aster_block::register_device(super::DEVICE_NAME.to_string(), Arc::new(block_device));
        Ok(())
    }

    /// Dequeues a `BioRequest` from the software staging queue and
    /// processes the request.
    pub fn handle_requests(&self) {
        let request = self.queue.dequeue();
        info!("Handle Request: {:?}", request);
        match request.type_() {
            BioType::Read => self.device.do_read(request),
            BioType::Write => self.device.do_write(request),
            BioType::Flush | BioType::Discard => todo!(),
        }
    }

    /// Negotiate features for the device specified bits 0~23
    pub(crate) fn negotiate_features(features: u64) -> u64 {
        let feature = BlockFeatures::from_bits(features).unwrap();
        let support_features = BlockFeatures::from_bits(features).unwrap();
        (feature & support_features).bits
    }
}

impl aster_block::BlockDevice for BlockDevice {
    fn enqueue(&self, bio: SubmittedBio) -> Result<(), BioEnqueueError> {
        self.queue.enqueue(bio)
    }

    fn handle_irq(&self) {
        info!("Virtio block device handle irq");
        self.device.do_handle_irq();
    }
}

const QUEUE_SIZE: usize = 64;

#[derive(Debug)]
struct DeviceInner {
    config: SafePtr<VirtioBlockConfig, IoMem>,
    queue: SpinLock<VirtQueue<QUEUE_SIZE>>,
    transport: Box<dyn VirtioTransport>,
    block_requests: DmaStream,
    block_responses: DmaStream,
    id_allocator: SpinLock<IdAlloc>,
    submitted_requests: SpinLock<BTreeMap<u16, SubmittedRequest>>,
}

impl DeviceInner {
    /// Creates and inits the device.
    fn init(mut transport: Box<dyn VirtioTransport>) -> Result<Self, VirtioDeviceError> {
        let config = VirtioBlockConfig::new(transport.as_mut());
        let num_queues = transport.num_queues();
        if num_queues != 1 {
            return Err(VirtioDeviceError::QueuesAmountDoNotMatch(num_queues, 1));
        }
        let queue =
            VirtQueue::<QUEUE_SIZE>::new(0, transport.as_mut()).expect("create virtqueue failed");
        let block_requests = {
            let vm_segment = VmAllocOptions::new(1)
                .is_contiguous(true)
                .alloc_contiguous()
                .unwrap();
            DmaStream::map(vm_segment, DmaDirection::Bidirectional, false).unwrap()
        };
        assert!(QUEUE_SIZE * size_of::<BlockReq>() <= block_requests.nbytes());
        let block_responses = {
            let vm_segment = VmAllocOptions::new(1)
                .is_contiguous(true)
                .alloc_contiguous()
                .unwrap();
            DmaStream::map(vm_segment, DmaDirection::Bidirectional, false).unwrap()
        };
        assert!(QUEUE_SIZE * size_of::<BlockResp>() <= block_responses.nbytes());

        let mut device = Self {
            config,
            queue: SpinLock::new(queue),
            transport,
            block_requests,
            block_responses,
            id_allocator: SpinLock::new(IdAlloc::with_capacity(QUEUE_SIZE)),
            submitted_requests: SpinLock::new(BTreeMap::new()),
        };

        device
            .transport
            .register_cfg_callback(Box::new(config_space_change))
            .unwrap();
        device
            .transport
            .register_queue_callback(0, Box::new(handle_block_device), false)
            .unwrap();

        fn handle_block_device(_: &TrapFrame) {
            aster_block::get_device(super::DEVICE_NAME)
                .unwrap()
                .handle_irq();
        }

        fn config_space_change(_: &TrapFrame) {
            info!("Virtio block device config space change");
        }
        device.transport.finish_init();
        Ok(device)
    }

    /// Handles the irq issued from the device
    fn do_handle_irq(&self) {
        loop {
            // Pops the complete request
            let complete_request = {
                let mut queue = self.queue.lock_irq_disabled();
                let Ok((token, _)) = queue.pop_used() else {
                    return;
                };
                self.submitted_requests.lock().remove(&token).unwrap()
            };

            // Handles the response
            let resp_offset = (complete_request.id as usize) * size_of::<BlockResp>();
            let resp_len = size_of::<BlockResp>();
            self.block_responses
                .sync(resp_offset..resp_offset + resp_len)
                .unwrap();
            let resp: BlockResp = self.block_responses.read_val(resp_offset).unwrap();
            self.id_allocator.lock().free(complete_request.id as usize);
            match RespStatus::try_from(resp.status).unwrap() {
                RespStatus::Ok => {}
                _ => panic!("io error in block device"),
            };
            if let BioType::Read = complete_request.bio_request.type_() {
                complete_request.dma_bufs.iter().for_each(|dma_buf| {
                    dma_buf.sync().unwrap();
                });
            }

            // Completes the bio request
            complete_request.bio_request.bios().for_each(|bio| {
                bio.complete(BioStatus::Complete);
            });
        }
    }

    /// Reads data from the device, this function is no-blocking.
    fn do_read(&self, bio_request: BioRequest) {
        let dma_bufs: Vec<_> = bio_request
            .bios()
            .flat_map(|bio| {
                bio.segments().iter().map(|segment| {
                    let dma_stream =
                        DmaStream::map(segment.pages().clone(), DmaDirection::ToDevice, false)
                            .unwrap();
                    DmaStreamBuf::new(dma_stream, segment.offset(), segment.nbytes())
                })
            })
            .collect();

        let id = self.id_allocator.lock().alloc().unwrap();
        let req_reader = {
            let req = BlockReq {
                type_: ReqType::In as _,
                reserved: 0,
                sector: bio_request.sid_range().start.to_raw(),
            };
            let req_offset = id * size_of::<BlockReq>();
            let req_len = size_of::<BlockReq>();
            self.block_requests.write_val(req_offset, &req).unwrap();
            self.block_requests
                .sync(req_offset..req_offset + req_len)
                .unwrap();
            self.block_requests
                .reader()
                .unwrap()
                .skip(req_offset)
                .limit(req_len)
        };

        let resp_offset = id * size_of::<BlockResp>();
        let resp_len = size_of::<BlockResp>();
        let resp_writer = {
            self.block_responses
                .write_val(resp_offset, &BlockResp::default())
                .unwrap();

            self.block_responses
                .writer()
                .unwrap()
                .skip(resp_offset)
                .limit(resp_len)
        };

        let outputs = {
            let mut outputs = Vec::with_capacity(dma_bufs.len() + 1);
            outputs.extend(dma_bufs.iter().map(|dma_buf| dma_buf.writer().unwrap()));
            outputs.push(resp_writer);
            outputs
        };

        let mut queue = self.queue.lock_irq_disabled();
        let token = queue
            .add_dma(&[req_reader], outputs.as_slice())
            .expect("add queue failed");
        if queue.should_notify() {
            queue.notify();
        }

        // Records the submitted request
        let submitted_request = SubmittedRequest::new(id as u16, bio_request, dma_bufs);
        self.submitted_requests
            .lock()
            .insert(token, submitted_request);
    }

    /// Writes data to the device, this function is no-blocking.
    fn do_write(&self, bio_request: BioRequest) {
        let dma_bufs: Vec<_> = bio_request
            .bios()
            .flat_map(|bio| {
                bio.segments().iter().map(|segment| {
                    let dma_stream =
                        DmaStream::map(segment.pages().clone(), DmaDirection::FromDevice, false)
                            .unwrap();
                    DmaStreamBuf::new(dma_stream, segment.offset(), segment.nbytes())
                })
            })
            .collect();

        let id = self.id_allocator.lock().alloc().unwrap();
        let req_reader = {
            let req = BlockReq {
                type_: ReqType::Out as _,
                reserved: 0,
                sector: bio_request.sid_range().start.to_raw(),
            };
            let req_offset = id * size_of::<BlockReq>();
            let req_len = size_of::<BlockReq>();
            self.block_requests.write_val(req_offset, &req).unwrap();
            self.block_requests
                .sync(req_offset..req_offset + req_len)
                .unwrap();
            self.block_requests
                .reader()
                .unwrap()
                .skip(req_offset)
                .limit(req_len)
        };

        let resp_offset = id * size_of::<BlockResp>();
        let resp_len = size_of::<BlockResp>();
        let resp_writer = {
            self.block_responses
                .write_val(resp_offset, &BlockResp::default())
                .unwrap();
            self.block_responses
                .writer()
                .unwrap()
                .skip(resp_offset)
                .limit(resp_len)
        };

        let inputs = {
            let mut inputs = Vec::with_capacity(dma_bufs.len() + 1);
            inputs.push(req_reader);
            inputs.extend(dma_bufs.iter().map(|dma_buf| dma_buf.reader().unwrap()));
            inputs
        };

        let mut queue = self.queue.lock_irq_disabled();
        let token = queue
            .add_dma(inputs.as_slice(), &[resp_writer])
            .expect("add queue failed");
        if queue.should_notify() {
            queue.notify();
        }

        // Records the submitted request
        let submitted_request = SubmittedRequest::new(id as u16, bio_request, dma_bufs);
        self.submitted_requests
            .lock()
            .insert(token, submitted_request);
    }
}

#[derive(Debug)]
struct SubmittedRequest {
    id: u16,
    bio_request: BioRequest,
    dma_bufs: Vec<DmaStreamBuf>,
}

impl SubmittedRequest {
    pub fn new(id: u16, bio_request: BioRequest, dma_bufs: Vec<DmaStreamBuf>) -> Self {
        Self {
            id,
            bio_request,
            dma_bufs,
        }
    }
}

#[derive(Debug)]
struct DmaStreamBuf {
    dma_stream: DmaStream,
    offset: usize,
    len: usize,
}

impl<'a> DmaStreamBuf {
    pub fn new(dma_stream: DmaStream, offset: usize, len: usize) -> Self {
        Self {
            dma_stream,
            offset,
            len,
        }
    }

    pub fn reader(&'a self) -> aster_frame::Result<DmaReader<'a>> {
        Ok(self.dma_stream.reader()?.skip(self.offset).limit(self.len))
    }

    pub fn writer(&'a self) -> aster_frame::Result<DmaWriter<'a>> {
        Ok(self.dma_stream.writer()?.skip(self.offset).limit(self.len))
    }

    pub fn sync(&self) -> aster_frame::Result<()> {
        self.dma_stream.sync(self.offset..self.offset + self.len)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod)]
struct BlockReq {
    pub type_: u32,
    pub reserved: u32,
    pub sector: u64,
}

/// Response of a VirtIOBlock request.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod)]
struct BlockResp {
    pub status: u8,
}

impl Default for BlockResp {
    fn default() -> Self {
        Self {
            status: RespStatus::_NotReady as _,
        }
    }
}
