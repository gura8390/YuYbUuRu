//! Locate only token_embd.weight in the GGUF. Mapping stays read-only; rows are
//! dequantized on demand, so a second full-precision embedding table is never kept.
use super::llama::{Api, ToFloat};
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use std::{fs::File, path::Path, sync::Arc};

pub(super) struct EmbeddingTable {
    mmap: Mmap,
    _api: Arc<Api>,
    start: usize,
    row_bytes: usize,
    count: usize,
    pub dim: usize,
    qtype: i32,
    convert: Option<ToFloat>,
}
impl EmbeddingTable {
    pub fn open(path: &Path, api: Arc<Api>) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("Open GGUF embedding {}", path.display()))?;
        // Models are immutable inputs, shared with the existing installation.
        let mmap = unsafe { Mmap::map(&file)? };
        let mut c = Cursor {
            bytes: &mmap,
            at: 0,
        };
        ensure!(c.take(4)? == b"GGUF", "Not a GGUF file");
        let version = c.u32()?;
        ensure!(
            version == 2 || version == 3,
            "Unsupported GGUF version {version}"
        );
        let tensors = c.usize64()?;
        let kv = c.usize64()?;
        let mut alignment = 32usize;
        for _ in 0..kv {
            let key = c.string()?;
            let kind = c.u32()?;
            if key == b"general.alignment" {
                ensure!(kind == 4, "Invalid GGUF alignment metadata type");
                alignment = c.u32()? as usize;
            } else {
                c.skip_value(kind, 0)?;
            }
        }
        ensure!(
            alignment.is_power_of_two() && alignment <= 1024 * 1024,
            "Invalid GGUF alignment"
        );
        let mut found = None;
        for _ in 0..tensors {
            let name = c.string()?;
            let n_dims = c.u32()? as usize;
            ensure!((1..=4).contains(&n_dims), "Invalid GGUF tensor rank");
            let mut shape = [1usize; 4];
            for value in shape.iter_mut().take(n_dims) {
                *value = c.usize64()?;
            }
            let kind = c.u32()? as i32;
            let offset = c.usize64()?;
            if name == b"token_embd.weight" {
                ensure!(n_dims == 2, "Embedding must be a matrix");
                found = Some((shape[0], shape[1], kind, offset));
            }
        }
        let header = c.at;
        let (dim, count, qtype, relative) = found.context("GGUF token_embd.weight is missing")?;
        ensure!(
            dim > 0 && dim <= 65536 && count > 0,
            "Invalid GGUF embedding shape"
        );
        // These are the shipped model formats; reject unsupported metadata before
        // calling GGML's enum-indexed C helpers, which can assert on unknown types.
        ensure!(
            [0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15].contains(&qtype),
            "Unsupported GGUF embedding quantization {qtype}"
        );
        let (block, type_bytes, convert) = api.dequantizer(qtype)?;
        ensure!(
            dim % block == 0,
            "Embedding row is not a whole quantization block"
        );
        let row_bytes = (dim / block)
            .checked_mul(type_bytes)
            .context("Embedding row overflow")?;
        let aligned = header
            .checked_add(alignment - 1)
            .context("GGUF header overflow")?
            & !(alignment - 1);
        let start = aligned
            .checked_add(relative)
            .context("GGUF tensor offset overflow")?;
        let end = start
            .checked_add(
                row_bytes
                    .checked_mul(count)
                    .context("GGUF tensor size overflow")?,
            )
            .context("GGUF tensor end overflow")?;
        ensure!(end <= mmap.len(), "GGUF embedding extends past end of file");
        Ok(Self {
            mmap,
            _api: api,
            start,
            row_bytes,
            count,
            dim,
            qtype,
            convert,
        })
    }
    pub fn rows(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        let mut output = vec![
            0.0;
            tokens
                .len()
                .checked_mul(self.dim)
                .context("Embedding allocation overflow")?
        ];
        for (&token, out) in tokens.iter().zip(output.chunks_exact_mut(self.dim)) {
            ensure!(
                token >= 0 && (token as usize) < self.count,
                "Embedding token out of range: {token}"
            );
            let offset = self.start + token as usize * self.row_bytes;
            let source = &self.mmap[offset..offset + self.row_bytes];
            match self.qtype {
                0 => {
                    for (value, raw) in out.iter_mut().zip(source.chunks_exact(4)) {
                        *value = f32::from_le_bytes(raw.try_into().unwrap());
                    }
                }
                1 => {
                    for (value, raw) in out.iter_mut().zip(source.chunks_exact(2)) {
                        *value = half::f16::from_bits(u16::from_le_bytes(raw.try_into().unwrap()))
                            .to_f32();
                    }
                }
                _ => unsafe {
                    self.convert.context("Missing native dequantizer")?(
                        source.as_ptr().cast(),
                        out.as_mut_ptr(),
                        self.dim as i64,
                    );
                },
            }
        }
        Ok(output)
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .context("GGUF metadata offset overflow")?;
        let value = self
            .bytes
            .get(self.at..end)
            .context("Truncated GGUF metadata")?;
        self.at = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn usize64(&mut self) -> Result<usize> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()).try_into()?)
    }
    fn string(&mut self) -> Result<&'a [u8]> {
        let n = self.usize64()?;
        self.take(n)
    }
    fn skip_value(&mut self, kind: u32, depth: usize) -> Result<()> {
        ensure!(depth < 8, "Nested GGUF metadata too deep");
        let fixed = match kind {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4 | 5 | 6 => 4,
            10 | 11 | 12 => 8,
            8 => {
                let _ = self.string()?;
                return Ok(());
            }
            9 => {
                let item = self.u32()?;
                let count = self.usize64()?;
                ensure!(
                    count <= self.bytes.len(),
                    "GGUF array count exceeds file size"
                );
                for _ in 0..count {
                    self.skip_value(item, depth + 1)?;
                }
                return Ok(());
            }
            _ => bail!("Unknown GGUF metadata kind {kind}"),
        };
        let _ = self.take(fixed)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn truncated_metadata_errors_instead_of_indexing() {
        let mut c = Cursor {
            bytes: &[3, 0],
            at: 0,
        };
        assert!(c.u32().is_err());
    }
    #[test]
    fn nested_arrays_skip_exactly() {
        let bytes = [
            0u32.to_le_bytes().as_slice(),
            3u64.to_le_bytes().as_slice(),
            &[1, 2, 3],
        ]
        .concat();
        let mut c = Cursor {
            bytes: &bytes,
            at: 0,
        };
        c.skip_value(9, 0).unwrap();
        assert_eq!(c.at, bytes.len());
    }
}
