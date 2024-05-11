// SPDX-License-Identifier: MPL-2.0

use int_to_c_enum::TryFromInt;

#[derive(Clone, Copy, PartialEq, Eq, Debug, TryFromInt)]
#[repr(u8)]
pub enum ThreadStatus {
    Init = 0,
    Running = 1,
    Exited = 2,
    Stopped = 3,
}

impl ThreadStatus {
    pub fn is_running(&self) -> bool {
        *self == ThreadStatus::Running
    }

    pub fn is_exited(&self) -> bool {
        *self == ThreadStatus::Exited
    }

    pub fn is_stopped(&self) -> bool {
        *self == ThreadStatus::Stopped
    }
}
