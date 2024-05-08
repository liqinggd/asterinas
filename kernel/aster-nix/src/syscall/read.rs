// SPDX-License-Identifier: MPL-2.0

use core::sync::atomic::{AtomicU64, Ordering};

use super::{SyscallReturn, SYS_READ};
use crate::{
    fs::file_table::FileDescripter, log_syscall_entry, prelude::*, util::write_bytes_to_user,
};

pub static TOTAL_NUM: AtomicU64 = AtomicU64::new(0);
pub static READ_BUF_CYCLES: AtomicU64 = AtomicU64::new(0);

core::arch::global_asm!(
    ".globl asm_rdtsc",
    "asm_rdtsc:",
    "rdtsc",
    "shlq $32, %rdx",
    "or %rdx, %rax",
    "ret",
    options(att_syntax)
);

extern "C" {
    pub fn asm_rdtsc() -> u64;
}

pub fn rdtsc() -> u64 {
    unsafe { asm_rdtsc() }
}

pub fn sys_read(fd: FileDescripter, user_buf_addr: Vaddr, buf_len: usize) -> Result<SyscallReturn> {
    log_syscall_entry!(SYS_READ);
    debug!(
        "fd = {}, user_buf_ptr = 0x{:x}, buf_len = 0x{:x}",
        fd, user_buf_addr, buf_len
    );
    let current = current!();
    let file_table = current.file_table().lock();
    let file = file_table.get_file(fd)?;
    let metadata = file.metadata();

    if buf_len == 4096 && metadata.size == 500 * 1024 * 1024 {
        // let start = rdtsc();
        let read_buf =
            unsafe { core::slice::from_raw_parts_mut(user_buf_addr as *mut u8, buf_len) };

        let read_len = file.read(read_buf)?;
        // write_bytes_to_user(user_buf_addr, &read_buf)?;

        // let end = rdtsc();
        // let read_buf_cycles = end - start;
        // READ_BUF_CYCLES.fetch_add(read_buf_cycles, Ordering::Relaxed);

        // TOTAL_NUM.fetch_add(1, Ordering::Relaxed);
        // if TOTAL_NUM.load(Ordering::Relaxed) == 100_0000 {
        //     let read_buf_cycles = READ_BUF_CYCLES.load(Ordering::Relaxed) / 100_0000;
        //     log::error!("read_buf avg cycles: {:?}", read_buf_cycles);
        //     TOTAL_NUM.store(0, Ordering::Relaxed);
        //     READ_BUF_CYCLES.store(0, Ordering::Relaxed);
        // }
        Ok(SyscallReturn::Return(read_len as _))
    } else {
        let mut read_buf = vec![0u8; buf_len];
        let read_len = file.read(&mut read_buf)?;
        write_bytes_to_user(user_buf_addr, &read_buf)?;
        Ok(SyscallReturn::Return(read_len as _))
    }
}
