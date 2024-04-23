// SPDX-License-Identifier: MPL-2.0

use core::sync::atomic::Ordering;

use super::SyscallReturn;
use crate::{log_syscall_entry, prelude::*, syscall::SYS_GETPID};

pub fn sys_getpid() -> Result<SyscallReturn> {
    let start = rdtsc();
    log_syscall_entry!(SYS_GETPID);
    let pid = current!().pid();
    debug!("[sys_getpid]: pid = {}", pid);
    let end = rdtsc();
    let cycles = end - start;
    GETPID_CYCLES.fetch_add(cycles, Ordering::Relaxed);
    Ok(SyscallReturn::Return(pid as _))
}
