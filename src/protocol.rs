//! Private parent/worker protocol: bounded JSON headers plus exact float32 PCM.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::io::{self, Read, Write};

const MAX_REQUEST_HEADER: usize = 1024 * 1024;
const MAX_REPLY_HEADER: usize = 64 * 1024 * 1024;
pub const MAX_SAMPLES: usize = 80 * 16_000;
const MAGIC: [u8; 4] = *b"YYUR";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum Request {
    Warmup {
        id: u64,
    },
    Segment {
        id: u64,
        task_id: String,
        source: String,
        language: String,
        context: String,
        offset: f64,
        overlap: f64,
        is_final: bool,
        time_start: f64,
        time_submit: f64,
        seed: Option<u32>,
    },
    DropSession {
        id: u64,
        task_id: String,
    },
    Shutdown {
        id: u64,
    },
}
impl Request {
    pub fn id(&self) -> u64 {
        match self {
            Self::Warmup { id }
            | Self::Segment { id, .. }
            | Self::DropSession { id, .. }
            | Self::Shutdown { id } => *id,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Recognition {
    pub task_id: String,
    pub text: String,
    pub text_accu: String,
    pub tokens: Vec<String>,
    pub timestamps: Vec<f64>,
    pub duration: f64,
    pub is_final: bool,
    pub time_start: f64,
    pub time_submit: f64,
    pub time_complete: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reply {
    pub id: u64,
    pub kind: String,
    pub result: Option<Recognition>,
    pub error: Option<String>,
    #[serde(default)]
    pub metrics: serde_json::Value,
}
impl Reply {
    pub fn ok(id: u64, kind: &str) -> Self {
        Self {
            id,
            kind: kind.into(),
            result: None,
            error: None,
            metrics: serde_json::Value::Null,
        }
    }
    pub fn error(id: u64, error: impl ToString) -> Self {
        Self {
            id,
            kind: "error".into(),
            result: None,
            error: Some(error.to_string()),
            metrics: serde_json::Value::Null,
        }
    }
}

pub fn write_packet<T: Serialize>(stream: &mut impl Write, header: &T, pcm: &[f32]) -> Result<()> {
    write_frame(stream, header, pcm, MAX_REQUEST_HEADER, "request")
}

pub fn write_reply(stream: &mut impl Write, reply: &Reply, pcm: &[f32]) -> Result<()> {
    write_frame(stream, reply, pcm, MAX_REPLY_HEADER, "reply")
}

// Serialize within the limit before writing any frame bytes. A rejected header
// leaves the stream intact and cannot allocate an unbounded intermediate Vec.
struct HeaderBuffer {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}
#[derive(Debug)]
pub(crate) struct HeaderLimitExceeded(usize);
impl std::fmt::Display for HeaderLimitExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON header exceeds {}-byte limit", self.0)
    }
}
impl std::error::Error for HeaderLimitExceeded {}

impl Write for HeaderBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let needed = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .filter(|n| *n <= self.limit);
        let Some(needed) = needed else {
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("JSON header exceeds {}-byte limit", self.limit),
            ));
        };
        if needed > self.bytes.capacity() {
            let capacity = self
                .bytes
                .capacity()
                .saturating_mul(2)
                .max(needed)
                .min(self.limit);
            self.bytes.reserve_exact(capacity - self.bytes.len());
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn write_frame<T: Serialize>(
    stream: &mut impl Write,
    header: &T,
    pcm: &[f32],
    max_header: usize,
    direction: &str,
) -> Result<()> {
    ensure!(pcm.len() <= MAX_SAMPLES, "PCM frame exceeds model window");
    let mut data = HeaderBuffer {
        bytes: Vec::new(),
        limit: max_header,
        exceeded: false,
    };
    let serialized = serde_json::to_writer(&mut data, header);
    if data.exceeded {
        return Err(anyhow::Error::new(HeaderLimitExceeded(max_header))
            .context(format!("Worker {direction} header was not sent")));
    }
    serialized.with_context(|| {
        format!("Serialize worker {direction} header (maximum {max_header} bytes)")
    })?;
    let data = data.bytes;
    stream.write_all(&MAGIC)?;
    stream.write_all(&(data.len() as u32).to_le_bytes())?;
    stream.write_all(&(pcm.len() as u32).to_le_bytes())?;
    stream.write_all(&data)?;
    // Windows x64 is little endian. Sending one contiguous allocation avoids per-sample syscalls.
    #[cfg(target_endian = "little")]
    {
        let bytes = unsafe {
            std::slice::from_raw_parts(pcm.as_ptr().cast::<u8>(), std::mem::size_of_val(pcm))
        };
        stream.write_all(bytes)?;
    }
    #[cfg(not(target_endian = "little"))]
    {
        for v in pcm {
            stream.write_all(&v.to_le_bytes())?;
        }
    }
    stream.flush()?;
    Ok(())
}
pub fn read_packet<T: DeserializeOwned>(stream: &mut impl Read) -> Result<Option<(T, Vec<f32>)>> {
    read_frame(stream, MAX_REQUEST_HEADER, "request")
}

pub fn read_reply(stream: &mut impl Read) -> Result<Option<(Reply, Vec<f32>)>> {
    read_frame(stream, MAX_REPLY_HEADER, "reply")
}

fn read_frame<T: DeserializeOwned>(
    stream: &mut impl Read,
    max_header: usize,
    direction: &str,
) -> Result<Option<(T, Vec<f32>)>> {
    let mut prefix = [0u8; 12];
    let first = loop {
        match stream.read(&mut prefix[..1]) {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            other => break other?,
        }
    };
    if first == 0 {
        return Ok(None);
    }
    stream
        .read_exact(&mut prefix[1..])
        .context("Truncated worker packet")?;
    ensure!(prefix[..4] == MAGIC, "Invalid worker packet signature");
    let json_len = u32::from_le_bytes(prefix[4..8].try_into()?) as usize;
    let count = u32::from_le_bytes(prefix[8..12].try_into()?) as usize;
    ensure!(
        json_len <= max_header,
        "Worker {direction} JSON header exceeds {max_header}-byte limit ({json_len} bytes)"
    );
    ensure!(
        count <= MAX_SAMPLES,
        "Worker PCM frame exceeds {MAX_SAMPLES}-sample limit"
    );
    let mut json = vec![0u8; json_len];
    stream.read_exact(&mut json)?;
    let header = serde_json::from_slice(&json)?;
    let mut pcm = vec![0.0f32; count];
    #[cfg(target_endian = "little")]
    {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                pcm.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(pcm.as_slice()),
            )
        };
        stream.read_exact(bytes)?;
    }
    #[cfg(not(target_endian = "little"))]
    {
        for v in &mut pcm {
            let mut b = [0u8; 4];
            stream.read_exact(&mut b)?;
            *v = f32::from_le_bytes(b);
        }
    }
    Ok(Some((header, pcm)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_nan_pcm_and_two_frames() {
        let pcm = [1.0, -0.0, f32::from_bits(0x7fc01234), f32::NEG_INFINITY];
        let mut bytes = Vec::new();
        write_packet(&mut bytes, &Request::Warmup { id: 7 }, &pcm).unwrap();
        write_packet(&mut bytes, &Request::Shutdown { id: 8 }, &[]).unwrap();
        let mut cursor = std::io::Cursor::new(bytes);
        let (h, p) = read_packet::<Request>(&mut cursor).unwrap().unwrap();
        assert_eq!(h.id(), 7);
        assert!(pcm.iter().zip(p).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_eq!(
            read_packet::<Request>(&mut cursor).unwrap().unwrap().0.id(),
            8
        );
        assert!(read_packet::<Request>(&mut cursor).unwrap().is_none());
    }
    #[test]
    fn refuses_truncation_and_oversize() {
        assert!(read_packet::<Request>(&mut &b"Y"[..]).is_err());
        let mut bytes = Vec::from(MAGIC);
        bytes.extend(2u32.to_le_bytes());
        bytes.extend(u32::MAX.to_le_bytes());
        assert!(read_packet::<Request>(&mut bytes.as_slice()).is_err());
    }

    #[test]
    fn large_final_reply_roundtrips_without_relaxing_request_limit() {
        let count = 100_000;
        let mut reply = Reply::ok(19, "result");
        reply.result = Some(Recognition {
            task_id: "long-file".into(),
            text: "长".repeat(count),
            text_accu: "长".repeat(count),
            tokens: vec!["长".into(); count],
            timestamps: (0..count).map(|i| i as f64 * 0.2).collect(),
            duration: count as f64 * 0.2,
            is_final: true,
            ..Default::default()
        });
        let pcm = [1.0, -0.0, f32::from_bits(0x7fc0_1234)];
        let mut bytes = Vec::new();
        write_reply(&mut bytes, &reply, &pcm).unwrap();
        assert!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize > MAX_REQUEST_HEADER);
        assert!(read_packet::<Request>(&mut bytes.as_slice()).is_err());
        let (actual, actual_pcm) = read_reply(&mut bytes.as_slice()).unwrap().unwrap();
        assert_eq!(actual.id, reply.id);
        assert_eq!(actual.kind, reply.kind);
        let expected = reply.result.unwrap();
        let actual = actual.result.unwrap();
        assert_eq!(actual.task_id, expected.task_id);
        assert_eq!(actual.text, expected.text);
        assert_eq!(actual.text_accu, expected.text_accu);
        assert_eq!(actual.tokens, expected.tokens);
        assert_eq!(actual.timestamps, expected.timestamps);
        assert_eq!(actual.duration, expected.duration);
        assert!(actual.is_final);
        assert!(
            pcm.iter()
                .zip(actual_pcm)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );

        let request = Request::Segment {
            id: 20,
            task_id: "long-file".into(),
            source: "file".into(),
            language: "auto".into(),
            context: "x".repeat(MAX_REQUEST_HEADER),
            offset: 0.0,
            overlap: 4.0,
            is_final: true,
            time_start: 0.0,
            time_submit: 0.0,
            seed: None,
        };
        let mut rejected = Vec::new();
        let error = write_packet(&mut rejected, &request, &[]).unwrap_err();
        assert!(error.is::<HeaderLimitExceeded>());
        assert!(rejected.is_empty());
    }

    #[test]
    fn bounded_headers_fail_before_frame_output_and_reply_reads_remain_bounded() {
        let mut output = b"existing".to_vec();
        let error = write_frame(&mut output, &"x".repeat(129), &[], 128, "reply").unwrap_err();
        assert!(error.is::<HeaderLimitExceeded>());
        assert_eq!(output, b"existing");
        let mut prefix = Vec::from(MAGIC);
        prefix.extend(((MAX_REPLY_HEADER + 1) as u32).to_le_bytes());
        prefix.extend(0u32.to_le_bytes());
        let error = read_reply(&mut prefix.as_slice()).unwrap_err();
        assert!(error.to_string().contains("67108864-byte limit"));
    }
}
