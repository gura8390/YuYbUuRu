//! Explicit test-only foreground ownership. Never inject into another window.
use super::*;
use anyhow::{Context, ensure};
use std::{path::Path, time::Instant};
use windows_sys::Win32::{
    Foundation::*,
    System::{LibraryLoader::GetModuleHandleW, Threading::GetCurrentThreadId},
    UI::WindowsAndMessaging::*,
};

fn wide(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(Some(0)).collect()
}
fn modifiers_up() -> bool {
    [VK_CONTROL, VK_MENU, VK_SHIFT, VK_LWIN, VK_RWIN]
        .iter()
        .all(|key| unsafe { GetAsyncKeyState(*key as i32) as u16 & 0x8000 == 0 })
}

struct OwnEdit {
    hwnd: HWND,
    edit: HWND,
    instance: HINSTANCE,
    class_name: Vec<u16>,
    original: HWND,
    original_thread: u32,
    original_pid: u32,
    closed: bool,
    restored: bool,
    external_focus_change: bool,
}

impl OwnEdit {
    unsafe fn create(original: HWND) -> Result<Self> {
        let mut original_pid = 0;
        let original_thread = GetWindowThreadProcessId(original, &mut original_pid);
        let instance = GetModuleHandleW(ptr::null());
        let class_name = wide(&format!(
            "YuYbUuRuInputSmoke_{}_{}",
            std::process::id(),
            GetCurrentThreadId()
        ));
        let class = WNDCLASSW {
            lpfnWndProc: Some(DefWindowProcW),
            hInstance: instance,
            lpszClassName: class_name.as_ptr(),
            ..Default::default()
        };
        ensure!(
            RegisterClassW(&class) != 0,
            "Unable to register the smoke window"
        );
        let hwnd = CreateWindowExW(
            0,
            class_name.as_ptr(),
            wide("语音输入 · 自有控件输入验收").as_ptr(),
            WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
            200,
            200,
            580,
            160,
            ptr::null_mut(),
            ptr::null_mut(),
            instance,
            ptr::null(),
        );
        if hwnd.is_null() {
            UnregisterClassW(class_name.as_ptr(), instance);
            anyhow::bail!("Unable to create the smoke window");
        }
        let mut own = Self {
            hwnd,
            edit: ptr::null_mut(),
            instance,
            class_name,
            original,
            original_thread,
            original_pid,
            closed: false,
            restored: false,
            external_focus_change: false,
        };
        own.edit = CreateWindowExW(
            WS_EX_CLIENTEDGE,
            wide("EDIT").as_ptr(),
            wide("").as_ptr(),
            WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_AUTOHSCROLL as u32,
            16,
            24,
            530,
            42,
            hwnd,
            ptr::null_mut(),
            instance,
            ptr::null(),
        );
        ensure!(
            !own.edit.is_null(),
            "Unable to create the owned EDIT control"
        );
        Ok(own)
    }

    fn focus_is_owned(&self) -> bool {
        unsafe {
            GetForegroundWindow() == self.hwnd
                && GetFocus() == self.edit
                && GetWindowThreadProcessId(self.hwnd, ptr::null_mut()) == GetCurrentThreadId()
                && GetWindowThreadProcessId(self.edit, ptr::null_mut()) == GetCurrentThreadId()
        }
    }

    unsafe fn activate(&self) -> bool {
        ShowWindow(self.hwnd, SW_SHOWNORMAL);
        SetForegroundWindow(self.hwnd);
        SetFocus(self.edit);
        let deadline = Instant::now() + Duration::from_millis(500);
        while Instant::now() < deadline {
            pump();
            if self.focus_is_owned() {
                return true;
            }
            thread::sleep(Duration::from_millis(5));
        }
        false
    }

    unsafe fn text(&self) -> String {
        let length = GetWindowTextLengthW(self.edit).max(0) as usize;
        let mut units = vec![0u16; length + 1];
        let copied =
            GetWindowTextW(self.edit, units.as_mut_ptr(), units.len() as i32).max(0) as usize;
        String::from_utf16_lossy(&units[..copied])
    }

    unsafe fn cleanup(&mut self) {
        if self.closed {
            return;
        }
        let foreground = GetForegroundWindow();
        if foreground == self.hwnd {
            let mut pid = 0;
            let thread = GetWindowThreadProcessId(self.original, &mut pid);
            if !self.original.is_null()
                && IsWindow(self.original) != 0
                && thread == self.original_thread
                && pid == self.original_pid
            {
                SetForegroundWindow(self.original);
                self.restored = GetForegroundWindow() == self.original;
            }
        } else {
            // A real user focus change wins. Do not steal focus back from it.
            self.external_focus_change = foreground != self.original;
            self.restored = foreground == self.original;
        }
        DestroyWindow(self.hwnd);
        UnregisterClassW(self.class_name.as_ptr(), self.instance);
        if !self.original.is_null() && GetForegroundWindow() == self.original {
            let mut pid = 0;
            let thread = GetWindowThreadProcessId(self.original, &mut pid);
            self.restored = thread == self.original_thread && pid == self.original_pid;
        }
        self.closed = true;
    }
}
impl Drop for OwnEdit {
    fn drop(&mut self) {
        unsafe {
            self.cleanup();
        }
    }
}

unsafe fn pump() {
    let mut message = MSG::default();
    while PeekMessageW(&mut message, ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

#[test]
#[ignore = "Root-coordinated own-EDIT Unicode SendInput smoke; temporarily changes foreground"]
fn native_unicode_edit_smoke() {
    let mut report = serde_json::json!({"status":"prepared", "unicode_exact":false,
        "physical_key_acceptance":false, "clipboard_tested":false,
        "clipboard_skip_reason":"Only plaintext restore is implemented; all clipboard formats must remain untouched",
        "clipboard_opened_or_modified":false, "capture_enabled":false, "models_loaded":false,
        "ffmpeg_started":false, "enter_sent":false, "injected_to_other_app":false,
        "hook_install_uninstall":false});
    let original = unsafe { GetForegroundWindow() };
    let result = (|| -> Result<()> {
        if original.is_null() || unsafe { IsWindow(original) } == 0 {
            report["status"] = "skipped".into();
            report["skip_reason"] = "No original foreground window could be saved; no activation or injection attempted".into();
            return Ok(());
        }
        if !modifiers_up() {
            report["status"] = "skipped".into();
            report["skip_reason"] =
                "A modifier key is held; no focus change or injection attempted".into();
            return Ok(());
        }
        super::super::windows::test_install_uninstall_hooks()?;
        report["hook_install_uninstall"] = true.into();
        // No active Capture context exists while these hooks are installed.
        let mut own = unsafe { OwnEdit::create(original)? };
        report["own_hwnd"] = (own.hwnd as usize).into();
        let input_result = (|| -> Result<()> {
            if !unsafe { own.activate() } {
                report["status"] = "skipped".into();
                report["skip_reason"] =
                    "Foreground/focus ownership could not be obtained; no injection attempted"
                        .into();
                return Ok(());
            }
            let payload = "中文 English 123 😀";
            let inputs = unicode_inputs(payload);
            // This check is immediately adjacent to every actual SendInput call.
            if !own.focus_is_owned() || !modifiers_up() {
                report["status"] = "skipped".into();
                report["skip_reason"] =
                    "Focus changed or a modifier became pressed before injection".into();
                return Ok(());
            }
            send(&inputs).context("Unicode SendInput into the owned EDIT")?;
            report["input_events_sent"] = inputs.len().into();
            let deadline = Instant::now() + Duration::from_millis(750);
            loop {
                unsafe {
                    pump();
                }
                let actual = unsafe { own.text() };
                if actual == payload {
                    report["unicode_exact"] = true.into();
                    report["status"] = "passed".into();
                    break;
                }
                if !own.focus_is_owned() {
                    report["status"] = "skipped".into();
                    report["skip_reason"] = "User changed foreground after the guarded injection; no further input sent".into();
                    break;
                }
                ensure!(
                    Instant::now() < deadline,
                    "Owned EDIT text differs: expected {payload:?}, received {actual:?}"
                );
                thread::sleep(Duration::from_millis(5));
            }
            Ok(())
        })();
        unsafe {
            own.cleanup();
        }
        report["own_window_destroyed"] = (unsafe { IsWindow(own.hwnd) } == 0).into();
        report["original_foreground_restored"] = own.restored.into();
        report["external_focus_change_respected"] = own.external_focus_change.into();
        if report["status"] == "passed" && !own.restored {
            report["status"] = "skipped".into();
            report["skip_reason"] = "Unicode was received exactly, but original foreground restoration could not be confirmed; no further activation attempted".into();
        }
        input_result
    })();
    if let Err(error) = &result {
        report["status"] = "failed".into();
        report["error"] = format!("{error:#}").into();
    }
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("evidence/desktop-input-smoke.json");
    std::fs::write(file, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("{report}");
    result.unwrap();
}
