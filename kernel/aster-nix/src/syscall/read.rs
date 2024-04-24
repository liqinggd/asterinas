// SPDX-License-Identifier: MPL-2.0

use core::sync::atomic::Ordering;

use super::{SyscallReturn, SYS_READ};
use crate::{
    fs::{file_table::FileDescripter, utils::UserIoUnit},
    log_syscall_entry,
    prelude::*,
    util::write_bytes_to_user,
};

lazy_static! {
    static ref BUFFER: Box<[u8]> = unsafe { Box::<[u8]>::new_zeroed_slice(8192).assume_init() };
}

pub fn sys_read(fd: FileDescripter, user_buf_addr: Vaddr, buf_len: usize) -> Result<SyscallReturn> {
    log_syscall_entry!(SYS_READ);
    debug!(
        "fd = {}, user_buf_ptr = 0x{:x}, buf_len = 0x{:x}",
        fd, user_buf_addr, buf_len
    );
    // clflush(BUFFER.as_ptr());
    // let start = rdtsc();
    let current = current!();
    let file_table = current.file_table().lock();
    let file = file_table.get_file(fd)?;

    if buf_len == 8192 {
        let uio = UserIoUnit::new(current.root_vmar(), user_buf_addr, buf_len);
        let read_len = file.read_uio(uio)?;
        // let user_slice = unsafe { core::slice::from_raw_parts_mut(user_buf_addr as *mut u8, buf_len) };
        // unsafe {
        //     core::ptr::copy(BUFFER.as_ptr(), user_buf_addr as *mut u8, buf_len);
        // }
        // // user_slice.copy_from_slice(&BUFFER);
        // let end = rdtsc();
        // let read_buf_cycles = end - start;
        // READ_BUF_CYCLES.fetch_add(read_buf_cycles, Ordering::Relaxed);
        return Ok(SyscallReturn::Return(read_len as _));
    }

    let mut read_buf = vec![0u8; buf_len];
    // let start = rdtsc();
    let read_len = file.read(&mut read_buf)?;
    //      let end = rdtsc();
    //  let read_buf_cycles = end - start;
    //  READ_BUF_CYCLES.fetch_add(read_buf_cycles, Ordering::Relaxed);

    //  let start = rdtsc();
    write_bytes_to_user(user_buf_addr, &read_buf)?;
    //  let end = rdtsc();
    //  let read_buf_cycles = end - start;
    //  READ_BUF_CYCLES.fetch_add(read_buf_cycles, Ordering::Relaxed);

    Ok(SyscallReturn::Return(read_len as _))
}
