//! Native desktop boundary. Merely importing this module creates no hook,
//! microphone stream, window, or synthesized keyboard/mouse event.
use anyhow::Result;
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Receiver},
    },
    time::{Duration, Instant},
};

#[cfg(windows)]
mod input;
#[cfg(windows)]
mod windows;

#[derive(Debug)]
pub enum DesktopEvent {
    Start,
    Audio(Vec<f32>),
    CaptureFailed(String),
    Stop,
    Transcribe(PathBuf),
    Quit,
}

#[derive(Debug)]
pub enum DesktopOutput {
    /// Already corrected/formatted by the text pipeline. Never strip it again.
    Text(String),
    State(String),
    Error(String),
}

#[derive(Debug)]
pub enum EventSendError {
    AudioBudgetExceeded { limit: usize, attempted: usize },
    Disconnected,
}
impl std::fmt::Display for EventSendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AudioBudgetExceeded { limit, attempted } => write!(
                f,
                "音频排队超过内存限额（{attempted}/{limit} 字节），已取消本次录音，请等待当前任务结束后重试"
            ),
            Self::Disconnected => write!(f, "Desktop event receiver disconnected"),
        }
    }
}
impl std::error::Error for EventSendError {}

struct AudioBudget {
    limit: usize,
    queued: AtomicUsize,
}
struct AudioCredit {
    budget: Arc<AudioBudget>,
    bytes: usize,
}
impl Drop for AudioCredit {
    fn drop(&mut self) {
        self.budget.queued.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
struct EventEnvelope {
    event: DesktopEvent,
    credit: Option<AudioCredit>,
}
impl EventEnvelope {
    fn acknowledge(self) -> DesktopEvent {
        let Self { event, credit } = self;
        drop(credit);
        event
    }
}

#[derive(Clone)]
pub struct BudgetEventSender {
    sender: mpsc::Sender<EventEnvelope>,
    budget: Arc<AudioBudget>,
}
pub struct BudgetEventReceiver {
    receiver: mpsc::Receiver<EventEnvelope>,
}
pub const DEFAULT_AUDIO_BUDGET_BYTES: usize = 1024 * 1024;

pub fn event_channel(audio_budget_bytes: usize) -> (BudgetEventSender, BudgetEventReceiver) {
    let (sender, receiver) = mpsc::channel();
    let budget = Arc::new(AudioBudget {
        limit: audio_budget_bytes,
        queued: AtomicUsize::new(0),
    });
    (
        BudgetEventSender { sender, budget },
        BudgetEventReceiver { receiver },
    )
}

impl BudgetEventSender {
    pub fn send(&self, event: DesktopEvent) -> std::result::Result<(), EventSendError> {
        // Account for the actual owned Vec allocation, not only sample length.
        let bytes = match &event {
            DesktopEvent::Audio(audio) => audio
                .capacity()
                .checked_mul(size_of::<f32>())
                .unwrap_or(usize::MAX),
            _ => 0,
        };
        let credit = if bytes == 0 {
            None
        } else {
            self.budget
                .queued
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                    queued
                        .checked_add(bytes)
                        .filter(|next| *next <= self.budget.limit)
                })
                .map_err(|queued| EventSendError::AudioBudgetExceeded {
                    limit: self.budget.limit,
                    attempted: queued.saturating_add(bytes),
                })?;
            Some(AudioCredit {
                budget: Arc::clone(&self.budget),
                bytes,
            })
        };
        self.sender
            .send(EventEnvelope { event, credit })
            .map_err(|_| EventSendError::Disconnected)
    }
    pub fn queued_audio_bytes(&self) -> usize {
        self.budget.queued.load(Ordering::Acquire)
    }
}
impl BudgetEventReceiver {
    pub fn recv(&self) -> std::result::Result<DesktopEvent, mpsc::RecvError> {
        self.receiver.recv().map(EventEnvelope::acknowledge)
    }
    pub fn try_recv(&self) -> std::result::Result<DesktopEvent, mpsc::TryRecvError> {
        self.receiver.try_recv().map(EventEnvelope::acknowledge)
    }
    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> std::result::Result<DesktopEvent, mpsc::RecvTimeoutError> {
        self.receiver
            .recv_timeout(timeout)
            .map(EventEnvelope::acknowledge)
    }
}

#[derive(Clone, Debug)]
pub struct DesktopOptions {
    pub base_dir: PathBuf,
    pub portaudio_dll: PathBuf,
    pub icon_path: Option<PathBuf>,
    pub enable_hooks: bool,
    pub enable_capture: bool,
    pub restore_clipboard: bool,
}

impl DesktopOptions {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        let base_dir = base_dir.into();
        let candidates = [
            base_dir.join("native/libportaudio64bit.dll"),
            base_dir.join("internal/_sounddevice_data/portaudio-binaries/libportaudio64bit.dll"),
            base_dir.join("assets/libportaudio64bit.dll"),
        ];
        let portaudio_dll = candidates
            .iter()
            .find(|path| path.is_file())
            .cloned()
            .unwrap_or_else(|| candidates[0].clone());
        let icon = base_dir.join("assets/icon.ico");
        Self {
            base_dir,
            portaudio_dll,
            icon_path: icon.is_file().then_some(icon),
            enable_hooks: true,
            enable_capture: true,
            restore_clipboard: true,
        }
    }
}

/// Run the hidden Win32 tray and its message loop. This is an explicit active
/// desktop operation; callers must disable capture/hooks for passive tests.
pub fn run(
    options: DesktopOptions,
    events: BudgetEventSender,
    output: Receiver<DesktopOutput>,
) -> Result<()> {
    #[cfg(windows)]
    {
        windows::run(options, events, output)
    }
    #[cfg(not(windows))]
    {
        let _ = (options, events, output);
        anyhow::bail!("The native desktop client requires Windows")
    }
}

/// Read-only process lookup, also used by the text pipeline's forced-punctuation policy.
pub fn foreground_process_name() -> String {
    #[cfg(windows)]
    {
        input::foreground_process()
    }
    #[cfg(not(windows))]
    {
        String::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shortcut {
    CapsLock,
    X2,
}
impl Shortcut {
    fn index(self) -> usize {
        match self {
            Self::CapsLock => 0,
            Self::X2 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HoldAction {
    Begin,
    Finish,
    Cancel,
    Replay(Shortcut),
}

/// One microphone utterance owns the first held shortcut. Repeated keydown and
/// overlapping shortcuts cannot create duplicate Start/Stop events.
#[derive(Default)]
struct HoldTracker {
    pressed: [Option<Instant>; 2],
    owner: Option<Shortcut>,
}
impl HoldTracker {
    fn press(&mut self, key: Shortcut, now: Instant) -> Vec<HoldAction> {
        if self.pressed[key.index()].is_some() {
            return Vec::new();
        }
        self.pressed[key.index()] = Some(now);
        if self.owner.is_none() {
            self.owner = Some(key);
            vec![HoldAction::Begin]
        } else {
            Vec::new()
        }
    }

    fn release(&mut self, key: Shortcut, now: Instant) -> Vec<HoldAction> {
        let Some(started) = self.pressed[key.index()].take() else {
            return Vec::new();
        };
        let long = now.saturating_duration_since(started) >= crate::capture::HOLD_THRESHOLD;
        let mut actions = Vec::new();
        if self.owner == Some(key) {
            self.owner = None;
            actions.push(if long {
                HoldAction::Finish
            } else {
                HoldAction::Cancel
            });
        }
        if !long {
            actions.push(HoldAction::Replay(key));
        }
        actions
    }

    fn should_confirm(&self, now: Instant) -> bool {
        self.owner
            .and_then(|key| self.pressed[key.index()])
            .is_some_and(|started| {
                now.saturating_duration_since(started) >= crate::capture::HOLD_THRESHOLD
            })
    }
}

#[derive(Debug, PartialEq, Eq)]
struct OutputPolicy {
    paste: bool,
    enter_after: Option<Duration>,
}
fn output_policy(process_name: &str) -> OutputPolicy {
    let name = process_name.to_ascii_lowercase();
    OutputPolicy {
        paste: matches!(name.as_str(), "weixin.exe" | "telegram.exe"),
        enter_after: matches!(name.as_str(), "happ.exe" | "hexin.exe")
            .then_some(Duration::from_millis(500)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn audio_budget_preserves_fifo_and_acknowledges_only_dequeued_audio() {
        let (tx, rx) = event_channel(8);
        tx.send(DesktopEvent::Start).unwrap();
        tx.send(DesktopEvent::Audio(vec![1., 2.])).unwrap();
        assert_eq!(tx.queued_audio_bytes(), 8);
        assert!(matches!(
            tx.send(DesktopEvent::Audio(vec![3.])),
            Err(EventSendError::AudioBudgetExceeded { .. })
        ));
        tx.send(DesktopEvent::CaptureFailed("capacity".into()))
            .unwrap();
        tx.send(DesktopEvent::Stop).unwrap();
        tx.send(DesktopEvent::Quit).unwrap();
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::Start));
        assert_eq!(tx.queued_audio_bytes(), 8);
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::Audio(v) if v == [1., 2.]));
        assert_eq!(tx.queued_audio_bytes(), 0);
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::CaptureFailed(_)));
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::Stop));
        assert!(matches!(rx.recv().unwrap(), DesktopEvent::Quit));
    }
    #[test]
    fn budget_credits_return_on_receiver_drop_and_disconnected_send() {
        let (tx, rx) = event_channel(8);
        tx.send(DesktopEvent::Audio(vec![1., 2.])).unwrap();
        drop(rx);
        assert_eq!(tx.queued_audio_bytes(), 0);
        assert!(matches!(
            tx.send(DesktopEvent::Audio(vec![1., 2.])),
            Err(EventSendError::Disconnected)
        ));
        assert_eq!(tx.queued_audio_bytes(), 0);
        let (tx, _rx) = event_channel(4);
        let mut oversized = Vec::with_capacity(32);
        oversized.push(1.);
        assert!(matches!(
            tx.send(DesktopEvent::Audio(oversized)),
            Err(EventSendError::AudioBudgetExceeded { .. })
        ));
        assert_eq!(tx.queued_audio_bytes(), 0);
    }
    #[test]
    fn concurrent_audio_senders_cannot_oversubscribe_the_budget() {
        let (tx, rx) = event_channel(24);
        let workers: Vec<_> = (0..16)
            .map(|_| {
                let sender = tx.clone();
                std::thread::spawn(move || sender.send(DesktopEvent::Audio(vec![1.; 3])).is_ok())
            })
            .collect();
        let accepted = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .filter(|accepted| *accepted)
            .count();
        assert_eq!(accepted, 2);
        assert_eq!(tx.queued_audio_bytes(), 24);
        for _ in 0..accepted {
            assert!(matches!(rx.recv().unwrap(), DesktopEvent::Audio(_)));
        }
        assert_eq!(tx.queued_audio_bytes(), 0);
    }
    #[test]
    fn short_presses_replay_only_the_suppressed_original_key() {
        for key in [Shortcut::CapsLock, Shortcut::X2] {
            let mut state = HoldTracker::default();
            let now = Instant::now();
            assert_eq!(state.press(key, now), [HoldAction::Begin]);
            assert!(state.press(key, now).is_empty());
            assert_eq!(
                state.release(key, now + Duration::from_millis(299)),
                [HoldAction::Cancel, HoldAction::Replay(key)]
            );
            assert!(state.release(key, now).is_empty());
        }
    }
    #[test]
    fn threshold_boundary_and_overlap_produce_one_utterance() {
        let mut state = HoldTracker::default();
        let now = Instant::now();
        state.press(Shortcut::CapsLock, now);
        assert!(state.press(Shortcut::X2, now).is_empty());
        assert!(!state.should_confirm(now + Duration::from_millis(299)));
        assert!(state.should_confirm(now + Duration::from_millis(300)));
        assert_eq!(
            state.release(Shortcut::CapsLock, now + Duration::from_millis(300)),
            [HoldAction::Finish]
        );
        assert!(
            state
                .release(Shortcut::X2, now + Duration::from_millis(301))
                .is_empty()
        );
        assert!(!state.should_confirm(now + Duration::from_secs(1)));
    }
    #[test]
    fn exact_foreground_names_select_paste_and_enter_without_mutating_text() {
        assert!(output_policy("WeiXin.exe").paste);
        assert!(output_policy("TELEGRAM.EXE").paste);
        assert!(!output_policy("other-Telegram.exe").paste);
        assert_eq!(
            output_policy("hexin.exe").enter_after,
            Some(Duration::from_millis(500))
        );
        assert_eq!(
            output_policy("happ.exe").enter_after,
            Some(Duration::from_millis(500))
        );
        assert_eq!(
            output_policy("notepad.exe"),
            OutputPolicy {
                paste: false,
                enter_after: None
            }
        );
    }
}
