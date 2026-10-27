#![allow(unsafe_op_in_unsafe_fn)]
//! Win32 output. Tests inspect INPUT values without calling SendInput.
use super::{Shortcut, output_policy};
use anyhow::{Result, bail};
use std::{
    ffi::OsString,
    mem::size_of,
    os::windows::ffi::OsStringExt,
    ptr,
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::Duration,
};
use windows_sys::Win32::{
    Foundation::{CloseHandle, GlobalFree, HWND},
    System::{DataExchange::*, Memory::*, Threading::*},
    UI::{
        Input::KeyboardAndMouse::*,
        WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId},
    },
};

pub(super) const INPUT_TAG: usize = 0x5955_5942;
const CF_UNICODETEXT: u32 = 13;

fn keyboard(vk: u16, scan: u16, flags: u32) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: vk,
                wScan: scan,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: INPUT_TAG,
            },
        },
    }
}

fn unicode_inputs(text: &str) -> Vec<INPUT> {
    text.encode_utf16()
        .flat_map(|unit| {
            [
                keyboard(0, unit, KEYEVENTF_UNICODE),
                keyboard(0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP),
            ]
        })
        .collect()
}

fn shortcut_inputs(key: Shortcut) -> [INPUT; 2] {
    match key {
        Shortcut::CapsLock => [
            keyboard(VK_CAPITAL, 0, 0),
            keyboard(VK_CAPITAL, 0, KEYEVENTF_KEYUP),
        ],
        Shortcut::X2 => [MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP].map(|flags| INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: 0,
                    dy: 0,
                    mouseData: 2,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: INPUT_TAG,
                },
            },
        }),
    }
}

fn send(inputs: &[INPUT]) -> Result<()> {
    if inputs.is_empty() {
        return Ok(());
    }
    let inserted = unsafe {
        SendInput(
            inputs.len() as u32,
            inputs.as_ptr(),
            size_of::<INPUT>() as i32,
        )
    };
    if inserted != inputs.len() as u32 {
        bail!(
            "SendInput only inserted {inserted}/{} events: {}",
            inputs.len(),
            std::io::Error::last_os_error()
        );
    }
    Ok(())
}

pub(super) fn replay(key: Shortcut) -> Result<()> {
    send(&shortcut_inputs(key))
}

pub(super) fn foreground_process() -> String {
    unsafe {
        let foreground = GetForegroundWindow();
        if foreground.is_null() {
            return String::new();
        }
        let mut pid = 0;
        GetWindowThreadProcessId(foreground, &mut pid);
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if process.is_null() {
            return String::new();
        }
        let mut path = vec![0u16; 32_768];
        let mut length = path.len() as u32;
        let ok = QueryFullProcessImageNameW(process, 0, path.as_mut_ptr(), &mut length);
        CloseHandle(process);
        if ok == 0 {
            return String::new();
        }
        let path = OsString::from_wide(&path[..length as usize]);
        std::path::Path::new(&path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

struct Clipboard;
impl Clipboard {
    fn open(owner: HWND) -> Result<Self> {
        for _ in 0..20 {
            if unsafe { OpenClipboard(owner) } != 0 {
                return Ok(Self);
            }
            thread::sleep(Duration::from_millis(5));
        }
        bail!("无法打开剪贴板: {}", std::io::Error::last_os_error())
    }

    fn text(&self) -> Option<Vec<u16>> {
        unsafe {
            let handle = GetClipboardData(CF_UNICODETEXT);
            if handle.is_null() {
                return None;
            }
            let bytes = GlobalSize(handle);
            let data = GlobalLock(handle).cast::<u16>();
            if data.is_null() {
                return None;
            }
            let data_slice = std::slice::from_raw_parts(data, bytes / 2);
            let end = data_slice
                .iter()
                .position(|unit| *unit == 0)
                .unwrap_or(data_slice.len());
            let copy = data_slice[..end].to_vec();
            GlobalUnlock(handle);
            Some(copy)
        }
    }

    fn set_text(&self, text: &[u16]) -> Result<()> {
        unsafe {
            let memory = GlobalAlloc(GMEM_MOVEABLE, (text.len() + 1) * 2);
            if memory.is_null() {
                bail!("GlobalAlloc failed for clipboard text");
            }
            let target = GlobalLock(memory).cast::<u16>();
            if target.is_null() {
                GlobalFree(memory);
                bail!("GlobalLock failed for clipboard text");
            }
            ptr::copy_nonoverlapping(text.as_ptr(), target, text.len());
            *target.add(text.len()) = 0;
            GlobalUnlock(memory);
            if EmptyClipboard() == 0 || SetClipboardData(CF_UNICODETEXT, memory).is_null() {
                GlobalFree(memory);
                bail!("写入剪贴板失败: {}", std::io::Error::last_os_error());
            }
            // Successful SetClipboardData transfers HGLOBAL ownership to Windows.
            Ok(())
        }
    }
}
impl Drop for Clipboard {
    fn drop(&mut self) {
        unsafe {
            CloseClipboard();
        }
    }
}

fn paste(owner: HWND, text: &str, restore: bool) -> Result<()> {
    let prior;
    {
        let clipboard = Clipboard::open(owner)?;
        prior = clipboard.text().unwrap_or_default();
        clipboard.set_text(&text.encode_utf16().collect::<Vec<_>>())?;
    }
    let sequence = unsafe { GetClipboardSequenceNumber() };
    let chord = [
        keyboard(VK_CONTROL, 0, 0),
        keyboard(VK_V, 0, 0),
        keyboard(VK_V, 0, KEYEVENTF_KEYUP),
        keyboard(VK_CONTROL, 0, KEYEVENTF_KEYUP),
    ];
    let insertion = send(&chord);
    if insertion.is_err() {
        let _ = send(&[
            keyboard(VK_V, 0, KEYEVENTF_KEYUP),
            keyboard(VK_CONTROL, 0, KEYEVENTF_KEYUP),
        ]);
    }
    if restore {
        thread::sleep(Duration::from_millis(100));
        let clipboard = Clipboard::open(owner)?;
        // Preserve a user's intervening copy instead of overwriting it.
        if unsafe { GetClipboardSequenceNumber() } == sequence {
            clipboard.set_text(&prior)?;
        }
    }
    insertion
}

pub(super) fn output_text(
    owner: HWND,
    text: &str,
    restore_clipboard: bool,
    alive: &AtomicBool,
) -> Result<()> {
    if text.is_empty() || !alive.load(Ordering::Acquire) {
        return Ok(());
    }
    let policy = output_policy(&foreground_process());
    if policy.paste {
        paste(owner, text, restore_clipboard)?;
    } else {
        for batch in unicode_inputs(text).chunks(512) {
            if !alive.load(Ordering::Acquire) {
                break;
            }
            send(batch)?;
        }
    }
    if let Some(delay) = policy.enter_after {
        thread::sleep(delay);
        if alive.load(Ordering::Acquire) {
            send(&[
                keyboard(VK_RETURN, 0, 0),
                keyboard(VK_RETURN, 0, KEYEVENTF_KEYUP),
            ])?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "input_smoke.rs"]
mod smoke;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unicode_and_surrogate_inputs_have_exact_up_down_pairs_and_recursion_tag() {
        let text = "汉A😀";
        let units: Vec<_> = text.encode_utf16().collect();
        let inputs = unicode_inputs(text);
        assert_eq!(inputs.len(), units.len() * 2);
        for (index, pair) in inputs.chunks_exact(2).enumerate() {
            unsafe {
                assert_eq!(pair[0].r#type, INPUT_KEYBOARD);
                assert_eq!(pair[0].Anonymous.ki.wVk, 0);
                assert_eq!(pair[0].Anonymous.ki.wScan, units[index]);
                assert_eq!(pair[1].Anonymous.ki.wScan, units[index]);
                assert_eq!(pair[0].Anonymous.ki.dwFlags, KEYEVENTF_UNICODE);
                assert_eq!(
                    pair[1].Anonymous.ki.dwFlags,
                    KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
                );
                assert_eq!(pair[0].Anonymous.ki.dwExtraInfo, INPUT_TAG);
            }
        }
    }
    #[test]
    fn short_press_replay_builds_capslock_or_xbutton2_without_sending() {
        let caps = shortcut_inputs(Shortcut::CapsLock);
        let mouse = shortcut_inputs(Shortcut::X2);
        unsafe {
            assert_eq!(caps[0].Anonymous.ki.wVk, VK_CAPITAL);
            assert_eq!(caps[1].Anonymous.ki.dwFlags, KEYEVENTF_KEYUP);
            assert_eq!(mouse[0].Anonymous.mi.mouseData, 2);
            assert_eq!(mouse[0].Anonymous.mi.dwFlags, MOUSEEVENTF_XDOWN);
            assert_eq!(mouse[1].Anonymous.mi.dwFlags, MOUSEEVENTF_XUP);
            assert_eq!(mouse[1].Anonymous.mi.dwExtraInfo, INPUT_TAG);
        }
    }
}
