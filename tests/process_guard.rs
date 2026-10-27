#![cfg(windows)]
use std::{
    io::{BufRead, BufReader, Read},
    os::windows::{
        io::{AsRawHandle, FromRawHandle, OwnedHandle},
        process::CommandExt,
    },
    process::{Child, Command, Stdio},
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use windows_sys::Win32::{
    Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE, WAIT_OBJECT_0},
    System::Threading::{
        CREATE_NO_WINDOW, GetCurrentProcess, GetProcessHandleCount, TerminateProcess,
        WaitForSingleObject,
    },
};
use yuybuuru::process_guard::ProcessGuard;

static SERIAL: Mutex<()> = Mutex::new(());

// Each test has an independent cleanup handle to its own child only. Even a
// failed assertion after dropping Child cannot leave the sleeping helper behind.
struct Helper {
    child: Option<Child>,
    cleanup: OwnedHandle,
}
impl Helper {
    fn from_command(mut command: Command) -> Self {
        let mut child = command
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn owned hidden test helper");
        let mut handle: HANDLE = ptr::null_mut();
        let duplicated = unsafe {
            DuplicateHandle(
                GetCurrentProcess(),
                child.as_raw_handle() as HANDLE,
                GetCurrentProcess(),
                &mut handle,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            )
        };
        if duplicated == 0 {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "duplicate cleanup handle: {}",
                std::io::Error::last_os_error()
            );
        }
        Self {
            child: Some(child),
            cleanup: unsafe { OwnedHandle::from_raw_handle(handle.cast()) },
        }
    }
    fn sleeper() -> Self {
        let system_root = std::env::var_os("SystemRoot").expect("Windows SystemRoot");
        let exe = std::path::PathBuf::from(system_root)
            .join("System32/WindowsPowerShell/v1.0/powershell.exe");
        let mut command = Command::new(exe);
        command.args(["-NoProfile", "-NonInteractive", "-Command",
            "[Console]::Out.WriteLine('READY'); [Console]::Out.Flush(); Start-Sleep -Seconds 30; [Console]::Out.WriteLine('DONE')"]);
        Self::from_command(command)
    }
    fn normal() -> Self {
        let mut command =
            Command::new(std::env::var_os("ComSpec").unwrap_or_else(|| "cmd.exe".into()));
        command.args(["/d", "/c", "exit /b 7"]);
        Self::from_command(command)
    }
    fn child(&self) -> &Child {
        self.child.as_ref().unwrap()
    }
    fn child_mut(&mut self) -> &mut Child {
        self.child.as_mut().unwrap()
    }
    fn wait_ready(&mut self) -> BufReader<std::process::ChildStdout> {
        let mut stdout = BufReader::new(self.child_mut().stdout.take().unwrap());
        let mut ready = String::new();
        stdout.read_line(&mut ready).unwrap();
        assert_eq!(ready.trim(), "READY");
        stdout
    }
}
impl Drop for Helper {
    fn drop(&mut self) {
        let handle = self.cleanup.as_raw_handle() as HANDLE;
        unsafe {
            if WaitForSingleObject(handle, 0) != WAIT_OBJECT_0 {
                TerminateProcess(handle, 0xe000_00ff);
                WaitForSingleObject(handle, 5_000);
            }
        }
        if let Some(child) = self.child.as_mut() {
            let _ = child.wait();
        }
    }
}

fn confirmed(guard: &ProcessGuard) {
    let end = Instant::now() + Duration::from_secs(2);
    while !guard.termination_confirmed() && Instant::now() < end {
        assert!(
            guard.termination_error().is_none(),
            "{:?}",
            guard.termination_error()
        );
        thread::sleep(Duration::from_millis(5));
    }
    assert!(guard.termination_confirmed());
    assert!(guard.termination_error().is_none());
}

#[test]
fn natural_exit_is_not_a_cancel_or_timeout() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut helper = Helper::normal();
    let guard = ProcessGuard::watch(
        helper.child(),
        Arc::new(AtomicBool::new(false)),
        Duration::from_secs(3),
    )
    .unwrap();
    assert_eq!(helper.child_mut().wait().unwrap().code(), Some(7));
    assert!(!guard.expired());
    assert!(!guard.cancelled());
    assert!(guard.termination_error().is_none());
}

#[test]
fn cancellation_terminates_owned_child_and_unblocks_stdout_eof() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut helper = Helper::sleeper();
    let closing = Arc::new(AtomicBool::new(false));
    let guard =
        ProcessGuard::watch(helper.child(), closing.clone(), Duration::from_secs(5)).unwrap();
    let mut stdout = helper.wait_ready();
    let began = Instant::now();
    closing.store(true, Ordering::Release);
    let mut rest = String::new();
    stdout.read_to_string(&mut rest).unwrap();
    assert!(began.elapsed() < Duration::from_secs(2));
    assert!(!rest.contains("DONE"));
    assert!(!helper.child_mut().wait().unwrap().success());
    confirmed(&guard);
    assert!(guard.cancelled());
    assert!(!guard.expired());
}

#[test]
fn deadline_terminates_only_the_owned_process() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut helper = Helper::sleeper();
    let mut untouched = Helper::sleeper();
    let untouched_startup = ProcessGuard::watch(
        untouched.child(),
        Arc::new(AtomicBool::new(false)),
        Duration::from_secs(5),
    )
    .unwrap();
    let _untouched_stdout = untouched.wait_ready();
    drop(untouched_startup);
    let closing = Arc::new(AtomicBool::new(false));
    let startup =
        ProcessGuard::watch(helper.child(), closing.clone(), Duration::from_secs(5)).unwrap();
    let mut stdout = helper.wait_ready();
    drop(startup);
    let began = Instant::now();
    let guard = ProcessGuard::watch(helper.child(), closing, Duration::from_millis(180)).unwrap();
    let mut rest = String::new();
    stdout.read_to_string(&mut rest).unwrap();
    assert!(began.elapsed() >= Duration::from_millis(150));
    assert!(began.elapsed() < Duration::from_secs(2));
    helper.child_mut().wait().unwrap();
    confirmed(&guard);
    assert!(guard.expired());
    assert!(!guard.cancelled());
    assert!(
        untouched.child_mut().try_wait().unwrap().is_none(),
        "another same-name owned helper must remain alive"
    );
}

#[test]
fn dropping_guard_stops_monitoring_without_killing_live_child() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut helper = Helper::sleeper();
    let closing = Arc::new(AtomicBool::new(false));
    let guard =
        ProcessGuard::watch(helper.child(), closing.clone(), Duration::from_secs(5)).unwrap();
    let _stdout = helper.wait_ready();
    let began = Instant::now();
    drop(guard);
    assert!(began.elapsed() < Duration::from_millis(500));
    closing.store(true, Ordering::Release);
    thread::sleep(Duration::from_millis(150));
    assert!(helper.child_mut().try_wait().unwrap().is_none());
}

#[test]
fn live_worker_rpc_guard_drop_is_woken_without_polling_latency() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut helper = Helper::sleeper();
    let closing = Arc::new(AtomicBool::new(false));
    let startup =
        ProcessGuard::watch(helper.child(), closing.clone(), Duration::from_secs(5)).unwrap();
    let _stdout = helper.wait_ready();
    drop(startup);
    let mut samples = Vec::new();
    for _ in 0..32 {
        let guard = ProcessGuard::watch(helper.child(), closing.clone(), Duration::ZERO).unwrap();
        // Make the watchdog enter its 75ms kernel wait before ending this RPC.
        thread::sleep(Duration::from_millis(5));
        let began = Instant::now();
        drop(guard);
        samples.push(began.elapsed());
    }
    let total: Duration = samples.iter().sum();
    samples.sort_unstable();
    let median = samples[16];
    eprintln!("32 live-worker guard drops: total={total:?}, median={median:?}");
    assert!(total < Duration::from_secs(1), "total Drop time {total:?}");
    assert!(
        median < Duration::from_millis(10),
        "median Drop time {median:?}"
    );
    assert!(helper.child_mut().try_wait().unwrap().is_none());
}

#[test]
fn duplicate_survives_dropping_original_child_handle() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut helper = Helper::sleeper();
    let closing = Arc::new(AtomicBool::new(false));
    let guard = ProcessGuard::watch(helper.child(), closing.clone(), Duration::ZERO).unwrap();
    let mut stdout = helper.wait_ready();
    drop(helper.child.take());
    closing.store(true, Ordering::Release);
    let mut rest = String::new();
    stdout.read_to_string(&mut rest).unwrap();
    confirmed(&guard);
    assert!(guard.cancelled());
    assert!(!guard.expired());
}

fn handle_count() -> u32 {
    let mut count = 0;
    assert_ne!(
        unsafe { GetProcessHandleCount(GetCurrentProcess(), &mut count) },
        0
    );
    count
}

#[test]
fn repeated_guards_close_their_process_and_thread_handles() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    // Let concurrently scheduled test threads settle before taking the baseline.
    thread::sleep(Duration::from_millis(100));
    for _ in 0..2 {
        let mut h = Helper::normal();
        let g = ProcessGuard::watch(h.child(), Arc::new(AtomicBool::new(false)), Duration::ZERO)
            .unwrap();
        h.child_mut().wait().unwrap();
        drop(g);
        drop(h);
    }
    let before = handle_count();
    for _ in 0..32 {
        let mut h = Helper::normal();
        let g = ProcessGuard::watch(h.child(), Arc::new(AtomicBool::new(false)), Duration::ZERO)
            .unwrap();
        h.child_mut().wait().unwrap();
        drop(g);
        drop(h);
    }
    let after = handle_count();
    assert!(
        after <= before + 2,
        "handle count grew: before={before}, after={after}"
    );
}
