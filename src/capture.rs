//! PortAudio ownership and the existing float32 48 kHz frame-selection contract.
use crate::desktop::{BudgetEventSender, DesktopEvent};
use anyhow::{Result, bail};
use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[cfg(windows)]
mod portaudio;

pub const INPUT_SAMPLE_RATE: u32 = 48_000;
pub const OUTPUT_SAMPLE_RATE: u32 = 16_000;
pub const INPUT_BLOCK_FRAMES: u32 = 2_400;
pub const HOLD_THRESHOLD: Duration = Duration::from_millis(300);

/// Exact equivalent of the shipped AudioBlock.mono_16k_bytes, without changing
/// its every-third-frame selection, float32 reduction, or NaN payload policy.
pub fn pcm_48k_to_mono_16k(samples: &[f32], channels: usize) -> Result<Vec<f32>> {
    if !(channels == 1 || channels == 2) || samples.len() % channels != 0 {
        bail!("Expected complete one- or two-channel float32 frames");
    }
    let frames = samples.len() / channels;
    let mut output = Vec::with_capacity(frames.div_ceil(3));
    for frame in (0..frames).step_by(3) {
        let left = samples[frame * channels];
        let left_bits = left.to_bits();
        if is_nan_bits(left_bits) {
            output.push(f32::from_bits(left_bits | 0x0040_0000));
        } else if channels == 1 {
            output.push(if left_bits == 0x8000_0000 { 0.0 } else { left });
        } else {
            let right = samples[frame * channels + 1];
            let right_bits = right.to_bits();
            if is_nan_bits(right_bits) {
                output.push(f32::from_bits(right_bits | 0x0040_0000));
            } else {
                // Round the sum to float32 before division, as NumPy does.
                let sum = left + right;
                let sum = if sum == 0.0 { 0.0 } else { sum };
                output.push(sum * 0.5);
            }
        }
    }
    Ok(output)
}

fn is_nan_bits(bits: u32) -> bool {
    bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0
}

enum Session {
    Idle,
    Pending {
        started: Instant,
        raw: Vec<f32>,
        channels: usize,
    },
    Recording,
}

pub(super) struct CaptureGate {
    events: BudgetEventSender,
    session: Session,
    error: Option<String>,
}

impl CaptureGate {
    fn new(events: BudgetEventSender) -> Self {
        Self {
            events,
            session: Session::Idle,
            error: None,
        }
    }

    fn begin(&mut self) {
        if matches!(self.session, Session::Idle) {
            self.session = Session::Pending {
                started: Instant::now(),
                raw: Vec::new(),
                channels: 0,
            };
        }
    }

    fn confirm(&mut self) {
        let prior = std::mem::replace(&mut self.session, Session::Recording);
        match prior {
            Session::Pending { raw, channels, .. } => {
                if let Err(error) = self.events.send(DesktopEvent::Start) {
                    self.fail(error.to_string());
                    return;
                }
                if !raw.is_empty() {
                    self.send_audio(&raw, channels);
                }
            }
            other => self.session = other,
        }
    }

    fn send_audio(&mut self, raw: &[f32], channels: usize) {
        match pcm_48k_to_mono_16k(raw, channels) {
            Ok(samples) if !samples.is_empty() => {
                if let Err(error) = self.events.send(DesktopEvent::Audio(samples)) {
                    self.fail(error.to_string());
                }
            }
            Ok(_) => {}
            Err(error) => self.fail(error.to_string()),
        }
    }

    pub(super) fn accept(&mut self, raw: &[f32], channels: usize) {
        match &mut self.session {
            Session::Idle => return,
            Session::Recording => self.send_audio(raw, channels),
            Session::Pending {
                started,
                raw: pending,
                channels: prior_channels,
            } => {
                if *prior_channels == 0 {
                    *prior_channels = channels;
                }
                if *prior_channels != channels {
                    self.error = Some("Input channel count changed during a recording".into());
                    return;
                }
                pending.extend_from_slice(raw);
                // Bound pre-roll even if a modal Windows dialog delays timers.
                if started.elapsed() >= HOLD_THRESHOLD {
                    self.confirm();
                }
            }
        }
    }

    fn finish(&mut self) {
        self.confirm();
        if matches!(self.session, Session::Recording) {
            self.session = Session::Idle;
            let _ = self.events.send(DesktopEvent::Stop);
        }
    }

    fn cancel(&mut self) {
        self.session = Session::Idle;
    }

    pub(super) fn fail(&mut self, message: String) {
        if matches!(self.session, Session::Idle) {
            return;
        }
        self.session = Session::Idle;
        self.error = Some(message.clone());
        let _ = self.events.send(DesktopEvent::CaptureFailed(message));
    }
}

/// Construction is the only point that opens a microphone. Pure PCM and gate
/// tests never construct an active Capture. The native stream closes before
/// its callback context or dynamically loaded DLL is released.
pub struct Capture {
    #[cfg(windows)]
    _stream: Option<portaudio::Owner>,
    gate: Arc<Mutex<CaptureGate>>,
}

impl Capture {
    pub fn disabled(events: BudgetEventSender) -> Self {
        Self {
            #[cfg(windows)]
            _stream: None,
            gate: Arc::new(Mutex::new(CaptureGate::new(events))),
        }
    }

    pub fn open(dll: &Path, events: BudgetEventSender) -> Result<Self> {
        let capture = Self::disabled(events);
        #[cfg(windows)]
        {
            let stream = portaudio::Owner::open(dll, Arc::clone(&capture.gate))?;
            Ok(Self {
                _stream: Some(stream),
                gate: capture.gate,
            })
        }
        #[cfg(not(windows))]
        {
            let _ = (dll, capture);
            bail!("Native capture requires Windows");
        }
    }

    pub fn begin_press(&self) {
        if let Ok(mut gate) = self.gate.lock() {
            gate.begin();
        }
    }
    pub fn confirm_hold(&self) {
        if let Ok(mut gate) = self.gate.lock() {
            gate.confirm();
        }
    }
    pub fn finish_press(&self) {
        if let Ok(mut gate) = self.gate.lock() {
            gate.finish();
        }
    }
    pub fn cancel_press(&self) {
        if let Ok(mut gate) = self.gate.lock() {
            gate.cancel();
        }
    }
    pub fn take_error(&self) -> Option<String> {
        self.gate.lock().ok()?.error.take()
    }
    pub fn is_ready(&self) -> bool {
        #[cfg(windows)]
        {
            self._stream
                .as_ref()
                .is_some_and(|stream| stream.is_ready())
        }
        #[cfg(not(windows))]
        {
            false
        }
    }
    pub fn reopen(&self) {
        #[cfg(windows)]
        if let Some(stream) = &self._stream {
            stream.reopen();
        }
    }
    pub fn shutdown(&mut self) {
        self.cancel_press();
        #[cfg(windows)]
        {
            self._stream.take();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desktop::event_channel;

    #[test]
    fn frame_selection_and_stereo_reduction_match_existing_contract() {
        assert_eq!(
            pcm_48k_to_mono_16k(&[1., 2., 3., 4., 5., 6., 7.], 1).unwrap(),
            [1., 4., 7.]
        );
        assert_eq!(
            pcm_48k_to_mono_16k(&[1., 3., 7., 9., 9., 11., 4., 8.], 2).unwrap(),
            [2., 6.]
        );
        assert!(pcm_48k_to_mono_16k(&[1.], 2).is_err());
        assert!(pcm_48k_to_mono_16k(&[], 0).is_err());
    }

    #[test]
    fn preserves_nan_payload_precedence_and_zero_rounding() {
        let bits = [0x8000_0000, 0, 0, 0xff80_0123, 0, 0, 0x7fc0_1234];
        let input: Vec<_> = bits.map(f32::from_bits).into();
        let output: Vec<_> = pcm_48k_to_mono_16k(&input, 1)
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(output, [0, 0xffc0_0123, 0x7fc0_1234]);
        let pair = [f32::from_bits(0x7f80_0123), f32::from_bits(0xff80_0456)];
        assert_eq!(
            pcm_48k_to_mono_16k(&pair, 2).unwrap()[0].to_bits(),
            0x7fc0_0123
        );
        assert_eq!(
            pcm_48k_to_mono_16k(&[-0.0, -0.0], 2).unwrap()[0].to_bits(),
            0
        );
        assert_eq!(
            pcm_48k_to_mono_16k(&[f32::from_bits(0x8000_0001), 0.0], 2).unwrap()[0].to_bits(),
            0x8000_0000
        );
        assert!(pcm_48k_to_mono_16k(&[f32::MAX, f32::MAX], 2).unwrap()[0].is_infinite());
    }

    #[test]
    fn threshold_pre_roll_is_kept_in_start_audio_stop_order() {
        let (tx, rx) = event_channel(1024 * 1024);
        let mut gate = CaptureGate::new(tx);
        gate.begin();
        gate.accept(&[1., 2.], 1);
        gate.accept(&[3., 4., 5., 6.], 1);
        assert!(rx.try_recv().is_err());
        gate.confirm();
        gate.accept(&[7., 8., 9.], 1);
        gate.finish();
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::Start));
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::Audio(v) if v == [1., 4.]));
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::Audio(v) if v == [7.]));
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::Stop));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn short_press_and_idle_callbacks_emit_nothing() {
        let (tx, rx) = event_channel(1024 * 1024);
        let mut gate = CaptureGate::new(tx);
        gate.accept(&[1., 2., 3.], 1);
        gate.begin();
        gate.accept(&[1., 2., 3.], 1);
        gate.cancel();
        gate.accept(&[1., 2., 3.], 1);
        gate.finish();
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn budget_failure_emits_once_stops_allocation_and_new_press_recovers() {
        let (tx, rx) = event_channel(4);
        let monitor = tx.clone();
        let mut gate = CaptureGate::new(tx);
        gate.begin();
        gate.confirm();
        gate.accept(&[1.; 9], 1); // Three output samples exceed the four-byte budget.
        for _ in 0..100 {
            gate.accept(&[2.; 2400], 1);
        }
        gate.finish();
        assert!(matches!(gate.session, Session::Idle));
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::Start));
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::CaptureFailed(_)));
        assert!(rx.try_recv().is_err());
        assert_eq!(monitor.queued_audio_bytes(), 0);
        gate.begin();
        gate.confirm();
        gate.accept(&[3.; 3], 1);
        gate.finish();
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::Start));
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::Audio(v) if v == [3.]));
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::Stop));
        assert!(rx.try_recv().is_err());
        assert_eq!(monitor.queued_audio_bytes(), 0);
    }

    #[test]
    fn frozen_numpy_conversion_fixture_is_bit_exact() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../assets/fixtures/pcm-conversion.json")).unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert!(cases.len() >= 135);
        for (index, case) in cases.iter().enumerate() {
            let input: Vec<f32> = case["input_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| f32::from_bits(value.as_u64().unwrap() as u32))
                .collect();
            let expected: Vec<u32> = case["output_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as u32)
                .collect();
            let actual: Vec<_> =
                pcm_48k_to_mono_16k(&input, case["channels"].as_u64().unwrap() as usize)
                    .unwrap()
                    .iter()
                    .map(|sample| sample.to_bits())
                    .collect();
            assert_eq!(actual, expected, "frozen NumPy PCM fixture {index}");
        }
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "Explicitly coordinated hardware-only open/start/reopen/stop smoke; no saved audio"]
    fn native_open_start_reopen_stop_smoke() {
        let dll = std::env::var_os("YUYB_PORTAUDIO_SMOKE_DLL")
            .expect("Explicit PortAudio DLL path is required");
        let (tx, rx) = event_channel(1024 * 1024);
        let started = Instant::now();
        let mut capture = Capture::open(Path::new(&dll), tx).unwrap();
        let wait_ready = |capture: &Capture| {
            let deadline = Instant::now() + Duration::from_secs(8);
            while !capture.is_ready() {
                assert!(
                    Instant::now() < deadline,
                    "Device did not start: {:?}",
                    capture.take_error()
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        wait_ready(&capture);
        let initial_seconds = started.elapsed().as_secs_f64();
        let reopen_started = Instant::now();
        capture.reopen();
        wait_ready(&capture);
        let reopen_seconds = reopen_started.elapsed().as_secs_f64();
        let stop_started = Instant::now();
        capture.shutdown();
        let stop_seconds = stop_started.elapsed().as_secs_f64();
        assert!(!capture.is_ready());
        assert!(
            rx.try_recv().is_err(),
            "Idle smoke must not emit or retain any microphone samples"
        );
        let report = serde_json::json!({"native_open_start": true, "native_close_reopen": true,
            "native_shutdown": true, "initial_seconds": initial_seconds, "reopen_seconds": reopen_seconds,
            "shutdown_seconds": stop_seconds, "desktop_hooks_installed": false,
            "input_injected": false, "audio_recording_started": false, "audio_saved_or_transmitted": false,
            "physical_unplug_replug_tested": false});
        let output =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("evidence/capture-native-smoke.json");
        std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        println!("{report}");
    }
}
