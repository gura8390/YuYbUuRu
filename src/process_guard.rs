//! A watchdog for a process object already owned through `std::process::Child`.
//! It never looks up a PID or executable name. A non-inheritable duplicate handle
//! keeps the same kernel process object alive even after the original Child drops.
use anyhow::{Context, Result, bail};
use std::{
    io,
    os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle},
    process::Child,
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use windows_sys::Win32::{
    Foundation::{
        DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT,
    },
    System::Threading::{
        CreateEventW, GetCurrentProcess, SetEvent, TerminateProcess, WaitForMultipleObjects,
        WaitForSingleObject,
    },
};

const POLL_MS: u32 = 75;
const CONFIRM_MS: u32 = 5_000;
const CANCEL_EXIT_CODE: u32 = 0xe000_0001;
const TIMEOUT_EXIT_CODE: u32 = 0xe000_0002;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reason {
    Cancelled,
    Expired,
}

#[derive(Default)]
struct State {
    stop: bool,
    reason: Option<Reason>,
    confirmed: bool,
    error: Option<String>,
}

/// Dropping the guard stops supervision and joins the watchdog; it does not kill
/// a live child merely because ownership of the guard ends. If termination has
/// already begun, Drop waits for its bounded confirmation step to finish.
pub struct ProcessGuard {
    state: Arc<Mutex<State>>,
    stop_event: Arc<OwnedHandle>,
    watchdog: Option<JoinHandle<()>>,
}

fn state_lock(state: &Mutex<State>) -> std::sync::MutexGuard<'_, State> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl ProcessGuard {
    /// Watch only this Child's process handle. `Duration::ZERO` disables the
    /// deadline and still observes `closing`. Cancellation wins if both trigger.
    pub fn watch(child: &Child, closing: Arc<AtomicBool>, timeout: Duration) -> Result<Self> {
        let deadline = if timeout.is_zero() {
            None
        } else {
            Some(
                Instant::now()
                    .checked_add(timeout)
                    .context("Process watchdog timeout is too large")?,
            )
        };
        let mut duplicate: HANDLE = ptr::null_mut();
        // SAFETY: Child owns a valid process handle for the duration of this
        // call. The returned duplicate is a new, non-inheritable owned handle.
        let ok = unsafe {
            let current = GetCurrentProcess();
            DuplicateHandle(
                current,
                child.as_raw_handle() as HANDLE,
                current,
                &mut duplicate,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error()).context("Duplicate owned child process handle");
        }
        if duplicate.is_null() {
            bail!("DuplicateHandle returned a null process handle");
        }
        // SAFETY: DuplicateHandle succeeded and ownership transfers exactly once.
        let process = unsafe { OwnedHandle::from_raw_handle(duplicate.cast()) };
        // A private manual-reset event interrupts the kernel wait immediately
        // when an RPC finishes and drops its guard while the worker stays alive.
        let event = unsafe { CreateEventW(ptr::null(), 1, 0, ptr::null()) };
        if event.is_null() {
            return Err(io::Error::last_os_error()).context("Create process-watchdog stop event");
        }
        let stop_event = Arc::new(unsafe { OwnedHandle::from_raw_handle(event.cast()) });
        let state = Arc::new(Mutex::new(State::default()));
        let watched_state = Arc::clone(&state);
        let watched_event = Arc::clone(&stop_event);
        let watchdog = thread::Builder::new()
            .name("owned-process-watchdog".into())
            .spawn(move || supervise(process, closing, deadline, watched_state, watched_event))
            .context("Start owned-process watchdog")?;
        Ok(Self {
            state,
            stop_event,
            watchdog: Some(watchdog),
        })
    }

    pub fn expired(&self) -> bool {
        state_lock(&self.state).reason == Some(Reason::Expired)
    }
    pub fn cancelled(&self) -> bool {
        state_lock(&self.state).reason == Some(Reason::Cancelled)
    }
    pub fn termination_confirmed(&self) -> bool {
        state_lock(&self.state).confirmed
    }
    pub fn termination_error(&self) -> Option<String> {
        state_lock(&self.state).error.clone()
    }
}

fn supervise(
    process: OwnedHandle,
    closing: Arc<AtomicBool>,
    deadline: Option<Instant>,
    state: Arc<Mutex<State>>,
    stop_event: Arc<OwnedHandle>,
) {
    // OwnedHandle is Send. Derive the borrowed raw value only on this thread,
    // and keep its owner alive until every Wait/Terminate call has finished.
    let handle = process.as_raw_handle() as HANDLE;
    let handles = [handle, stop_event.as_raw_handle() as HANDLE];
    loop {
        if state_lock(&state).stop {
            return;
        }
        let wait_ms = match deadline {
            Some(end) => {
                let remaining = end.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    0
                } else {
                    remaining.as_millis().clamp(1, POLL_MS as u128) as u32
                }
            }
            None => POLL_MS,
        };
        // SAFETY: the duplicate is owned and remains valid on this thread.
        let wait = unsafe { WaitForMultipleObjects(2, handles.as_ptr(), 0, wait_ms) };
        let mut status = state_lock(&state);
        if status.stop || wait == WAIT_OBJECT_0 || wait == WAIT_OBJECT_0 + 1 {
            return;
        }
        if wait == WAIT_FAILED {
            status.error = Some(format!(
                "WaitForMultipleObjects while monitoring: {}",
                io::Error::last_os_error()
            ));
            return;
        }
        if wait != WAIT_TIMEOUT {
            status.error = Some(format!("Unexpected process wait status: {wait:#x}"));
            return;
        }
        let reason = if closing.load(Ordering::Acquire) {
            Some(Reason::Cancelled)
        } else if deadline.is_some_and(|end| Instant::now() >= end) {
            Some(Reason::Expired)
        } else {
            None
        };
        let Some(reason) = reason else {
            continue;
        };
        // The mutex linearizes Drop's stop request against beginning termination.
        // Check again for a natural exit so it is never misclassified as timeout.
        if unsafe { WaitForSingleObject(handle, 0) } == WAIT_OBJECT_0 {
            return;
        }
        let exit_code = if reason == Reason::Cancelled {
            CANCEL_EXIT_CODE
        } else {
            TIMEOUT_EXIT_CODE
        };
        if unsafe { TerminateProcess(handle, exit_code) } == 0 {
            let error = io::Error::last_os_error();
            // A process may exit naturally between the last poll and Terminate.
            if unsafe { WaitForSingleObject(handle, 0) } == WAIT_OBJECT_0 {
                return;
            }
            status.reason = Some(reason);
            status.error = Some(format!("Terminate owned child process: {error}"));
            return;
        }
        status.reason = Some(reason);
        drop(status);
        // TerminateProcess is asynchronous. A signaled process handle confirms
        // thread termination; only then can callers rely on its pipe EOF.
        let wait = unsafe { WaitForSingleObject(handle, CONFIRM_MS) };
        let mut status = state_lock(&state);
        if wait == WAIT_OBJECT_0 {
            status.confirmed = true;
        } else if wait == WAIT_FAILED {
            status.error = Some(format!(
                "Confirm terminated child process: {}",
                io::Error::last_os_error()
            ));
        } else {
            status.error = Some(format!(
                "Owned child termination was not confirmed within {CONFIRM_MS}ms (wait={wait:#x})"
            ));
        }
        return;
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        {
            let mut state = state_lock(&self.state);
            state.stop = true;
            // SAFETY: this Arc keeps the event alive until the watchdog joins.
            if unsafe { SetEvent(self.stop_event.as_raw_handle() as HANDLE) } == 0 {
                state.error = Some(format!(
                    "Wake process watchdog: {}",
                    io::Error::last_os_error()
                ));
            }
        }
        if let Some(watchdog) = self.watchdog.take() {
            // The stop event wakes a normal watchdog immediately. One already
            // terminating may spend at most five seconds confirming that action.
            let _ = watchdog.join();
        }
    }
}
