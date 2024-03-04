// SPDX-License-Identifier: MPL-2.0

mod dma_coherent;
mod dma_stream;

use alloc::collections::BTreeSet;

pub use dma_coherent::DmaCoherent;
pub use dma_stream::{DmaDirection, DmaStream};
use spin::Once;

use super::Paddr;
use crate::{
    arch::iommu::has_iommu,
    config::PAGE_SIZE,
    sync::SpinLock,
    vm::{VmReader, VmWriter},
};

/// If a device performs DMA to read or write system
/// memory, the addresses used by the device are device addresses.
/// Daddr can distinguish the address space used by cpu side and
/// the address space used by device side.
pub type Daddr = usize;

fn has_tdx() -> bool {
    // FIXME: Support TDX
    false
}

#[derive(PartialEq)]
pub enum DmaType {
    Direct,
    Iommu,
    Tdx,
}

#[derive(Debug)]
pub enum DmaError {
    InvalidArgs,
    AlreadyMapped,
}

pub trait HasDaddr {
    /// Get the base address of the mapping in the
    /// device address space.
    fn daddr(&self) -> Daddr;
}

/// Set of all physical addresses with dma mapping.
static DMA_MAPPING_SET: Once<SpinLock<BTreeSet<Paddr>>> = Once::new();

pub fn dma_type() -> DmaType {
    if has_iommu() {
        DmaType::Iommu
    } else if has_tdx() {
        return DmaType::Tdx;
    } else {
        return DmaType::Direct;
    }
}

pub fn init() {
    DMA_MAPPING_SET.call_once(|| SpinLock::new(BTreeSet::new()));
}

/// Check whether the physical addresses has dma mapping.
/// Fail if they have been mapped, otherwise insert them.
fn check_and_insert_dma_mapping(start_paddr: Paddr, num_pages: usize) -> bool {
    let mut mapping_set = DMA_MAPPING_SET.get().unwrap().lock_irq_disabled();
    for i in 0..num_pages {
        let paddr = start_paddr + (i * PAGE_SIZE);
        if mapping_set.contains(&paddr) {
            return false;
        }
    }
    for i in 0..num_pages {
        let paddr = start_paddr + (i * PAGE_SIZE);
        mapping_set.insert(paddr);
    }
    true
}

/// Remove a physical address from the dma mapping set.
fn remove_dma_mapping(start_paddr: Paddr, num_pages: usize) {
    let mut mapping_set = DMA_MAPPING_SET.get().unwrap().lock_irq_disabled();
    for i in 0..num_pages {
        let paddr = start_paddr + (i * PAGE_SIZE);
        mapping_set.remove(&paddr);
    }
}

pub struct DmaReader<'a> {
    inner: VmReader<'a>,
    start_daddr: Daddr,
}

impl HasDaddr for DmaReader<'_> {
    fn daddr(&self) -> Daddr {
        self.start_daddr
    }
}

impl<'a> DmaReader<'a> {
    /// Constructs a DmReader from a VmReader and a Daddr.
    ///
    /// # Safety
    ///
    /// User must ensure the `start_daddr` is the correct start
    /// device address of the `vm_reader`.
    pub const unsafe fn from_vm(vm_reader: VmReader<'a>, start_daddr: Daddr) -> Self {
        Self {
            inner: vm_reader,
            start_daddr,
        }
    }

    pub const fn remain(&self) -> usize {
        self.inner.remain()
    }

    pub fn skip(self, nbytes: usize) -> Self {
        let reader = self.inner.skip(nbytes);
        Self {
            inner: reader,
            start_daddr: self.start_daddr + nbytes,
        }
    }

    pub const fn limit(self, max_remain: usize) -> Self {
        let reader = self.inner.limit(max_remain);
        Self {
            inner: reader,
            start_daddr: self.start_daddr,
        }
    }

    pub fn read(&mut self, writer: &mut VmWriter<'_>) -> usize {
        let len = self.inner.read(writer);
        self.start_daddr += len;
        len
    }
}

pub struct DmaWriter<'a> {
    inner: VmWriter<'a>,
    start_daddr: Daddr,
}

impl HasDaddr for DmaWriter<'_> {
    fn daddr(&self) -> Daddr {
        self.start_daddr
    }
}

impl<'a> DmaWriter<'a> {
    /// Constructs a DmWriter from a VmWriter and a Daddr.
    ///
    /// # Safety
    ///
    /// User must ensure the `start_daddr` is the correct start
    /// device address of the `vm_writer`.
    pub const unsafe fn from_vm(vm_writer: VmWriter<'a>, start_daddr: Daddr) -> Self {
        Self {
            inner: vm_writer,
            start_daddr,
        }
    }

    pub const fn avail(&self) -> usize {
        self.inner.avail()
    }

    pub fn skip(self, nbytes: usize) -> Self {
        let writer = self.inner.skip(nbytes);
        Self {
            inner: writer,
            start_daddr: self.start_daddr + nbytes,
        }
    }

    pub const fn limit(self, max_remain: usize) -> Self {
        let writer = self.inner.limit(max_remain);
        Self {
            inner: writer,
            start_daddr: self.start_daddr,
        }
    }

    pub fn write(&mut self, reader: &mut VmReader<'_>) -> usize {
        let len = self.inner.write(reader);
        self.start_daddr += len;
        len
    }
}
