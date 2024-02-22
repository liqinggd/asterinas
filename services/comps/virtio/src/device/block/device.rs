// SPDX-License-Identifier: MPL-2.0

use alloc::{boxed::Box, string::ToString, sync::Arc, vec::Vec};
use core::{fmt::Debug, hint::spin_loop, mem::size_of};

use aster_block::{
    bio::{BioEnqueueError, BioStatus, BioType, SubmittedBio},
    id::Sid,
    request_queue::{BioRequest, BioRequestSingleQueue},
};
use aster_frame::{
    io_mem::IoMem,
    sync::SpinLock,
    trap::TrapFrame,
    vm::{DmaDirection, DmaReader, DmaStream, DmaWriter, VmAllocOptions, VmIo},
};
use aster_util::safe_ptr::SafePtr;
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
    ///
    /// TODO: Current read and write operations are still synchronous，
    /// it needs to be modified to use the queue-based asynchronous programming pattern.
    pub fn handle_requests(&self) {
        let request = self.queue.dequeue();
        match request.type_() {
            BioType::Read => self.do_read(&request),
            BioType::Write => self.do_write(&request),
            BioType::Flush | BioType::Discard => todo!(),
        }
    }

    fn do_read(&self, request: &BioRequest) {
        let start_sid = request.sid_range().start;
        let dma_stream_bufs: Vec<_> = request
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

        let dma_writers: Vec<_> = dma_stream_bufs
            .iter()
            .map(|dma_stream_buf| dma_stream_buf.writer().unwrap())
            .collect();
        self.device.read(start_sid, dma_writers);

        dma_stream_bufs.iter().for_each(|dma_stream_buf| {
            dma_stream_buf.sync().unwrap();
        });
        drop(dma_stream_bufs);

        request.bios().for_each(|bio| {
            bio.complete(BioStatus::Complete);
        });
    }

    fn do_write(&self, request: &BioRequest) {
        let start_sid = request.sid_range().start;
        let dma_stream_bufs: Vec<_> = request
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

        let dma_readers: Vec<_> = dma_stream_bufs
            .iter()
            .map(|dma_stream_buf| dma_stream_buf.reader().unwrap())
            .collect();
        self.device.write(start_sid, dma_readers);
        drop(dma_stream_bufs);

        request.bios().for_each(|bio| {
            bio.complete(BioStatus::Complete);
        });
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
    }
}

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

#[derive(Debug)]
struct DeviceInner {
    config: SafePtr<VirtioBlockConfig, IoMem>,
    queue: SpinLock<VirtQueue>,
    transport: Box<dyn VirtioTransport>,
    block_requests: DmaStream,
    block_responses: DmaStream,
    id_allocator: SpinLock<Vec<u8>>,
}

impl DeviceInner {
    /// Creates and inits the device.
    pub fn init(mut transport: Box<dyn VirtioTransport>) -> Result<Self, VirtioDeviceError> {
        let config = VirtioBlockConfig::new(transport.as_mut());
        let num_queues = transport.num_queues();
        if num_queues != 1 {
            return Err(VirtioDeviceError::QueuesAmountDoNotMatch(num_queues, 1));
        }
        let queue = VirtQueue::new(0, 64, transport.as_mut()).expect("create virtqueue failed");
        let block_requests = {
            let vm_segment = VmAllocOptions::new(1)
                .is_contiguous(true)
                .alloc_contiguous()
                .unwrap();
            DmaStream::map(vm_segment, DmaDirection::Bidirectional, false).unwrap()
        };
        let block_responses = {
            let vm_segment = VmAllocOptions::new(1)
                .is_contiguous(true)
                .alloc_contiguous()
                .unwrap();
            DmaStream::map(vm_segment, DmaDirection::Bidirectional, false).unwrap()
        };
        let mut device = Self {
            config,
            queue: SpinLock::new(queue),
            transport,
            block_requests,
            block_responses,
            id_allocator: SpinLock::new((0..64).collect()),
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

    /// Reads data from the block device, this function is blocking.
    pub fn read(&self, sector_id: Sid, buf: Vec<DmaWriter>) {
        // FIXME: Handling cases without id.
        let id = self.id_allocator.lock().pop().unwrap() as usize;

        let req_reader = {
            let req = BlockReq {
                type_: ReqType::In as _,
                reserved: 0,
                sector: sector_id.to_raw(),
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
            let mut outputs = buf;
            outputs.push(resp_writer);
            outputs
        };

        let mut queue = self.queue.lock_irq_disabled();
        let token = queue
            .add_dma(&[req_reader], outputs.as_slice())
            .expect("add queue failed");
        queue.notify();
        while !queue.can_pop() {
            spin_loop();
        }
        queue.pop_used_with_token(token).expect("pop used failed");

        self.block_responses
            .sync(resp_offset..resp_offset + resp_len)
            .unwrap();
        let resp: BlockResp = self.block_responses.read_val(resp_offset).unwrap();
        self.id_allocator.lock().push(id as u8);
        match RespStatus::try_from(resp.status).unwrap() {
            RespStatus::Ok => {}
            _ => panic!("io error in block device"),
        };
    }

    /// Writes data to the block device, this function is blocking.
    pub fn write(&self, sector_id: Sid, buf: Vec<DmaReader>) {
        // FIXME: Handling cases without id.
        let id = self.id_allocator.lock().pop().unwrap() as usize;

        let req_reader = {
            let req = BlockReq {
                type_: ReqType::Out as _,
                reserved: 0,
                sector: sector_id.to_raw(),
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
            let mut inputs = buf;
            inputs.insert(0, req_reader);
            inputs
        };

        let mut queue = self.queue.lock_irq_disabled();
        let token = queue
            .add_dma(inputs.as_slice(), &[resp_writer])
            .expect("add queue failed");
        queue.notify();
        while !queue.can_pop() {
            spin_loop();
        }
        queue.pop_used_with_token(token).expect("pop used failed");

        self.block_responses
            .sync(resp_offset..resp_offset + resp_len)
            .unwrap();
        let resp: BlockResp = self.block_responses.read_val(resp_offset).unwrap();
        self.id_allocator.lock().push(id as u8);
        match RespStatus::try_from(resp.status).unwrap() {
            RespStatus::Ok => {}
            _ => panic!("io error in block device:{:?}", resp.status),
        };
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
