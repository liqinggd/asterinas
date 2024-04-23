// SPDX-License-Identifier: MPL-2.0

use super::{SyscallReturn, SYS_CHROOT};
use crate::{
    fs::{fs_resolver::FsPath, utils::InodeType},
    log_syscall_entry,
    prelude::*,
    syscall::constants::MAX_FILENAME_LEN,
    util::read_cstring_from_user,
};

pub fn sys_chroot(path_ptr: Vaddr) -> Result<SyscallReturn> {
    log_syscall_entry!(SYS_CHROOT);
    let path = read_cstring_from_user(path_ptr, MAX_FILENAME_LEN)?;
    debug!("path = {:?}", path);

    let current = current!();
    let mut fs = current.fs().write();
    let dentrymnt = {
        let path = path.to_string_lossy();
        if path.is_empty() {
            return_errno_with_message!(Errno::ENOENT, "path is empty");
        }
        let fs_path = FsPath::try_from(path.as_ref())?;
        fs.lookup(&fs_path)?
    };
    if dentrymnt.type_() != InodeType::Dir {
        return_errno_with_message!(Errno::ENOTDIR, "must be directory");
    }
    fs.set_root(dentrymnt);
    Ok(SyscallReturn::Return(0))
}
