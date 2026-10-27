//! NumPy-compatible Whisper/Qwen frontend: f64 Hann/FFT/power, f32 Slaney
//! filters, reflect padding, 160-sample hop, 128 bands, f32 normalized output.
use anyhow::{Result, ensure};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

pub struct MelExtractor {
    fft: Arc<dyn Fft<f64>>,
    window: [f64; 400],
    // Flattened by band, with each band's nonzero bins in ascending order.
    // Omitting positive-zero products preserves the dense f64 accumulation.
    filter_offsets: [usize; 129],
    filters: Vec<(usize, f64)>,
}
impl Default for MelExtractor {
    fn default() -> Self {
        Self::new()
    }
}
impl MelExtractor {
    pub fn new() -> Self {
        let fft = FftPlanner::<f64>::new().plan_fft_forward(400);
        let mut window = [0.0; 400];
        for (i, w) in window.iter_mut().enumerate() {
            *w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 400.0).cos();
        }
        fn hz_to_mel(h: f64) -> f64 {
            if h < 1000.0 {
                h / (200.0 / 3.0)
            } else {
                15.0 + (h / 1000.0).ln() / ((6.4f64).ln() / 27.0)
            }
        }
        fn mel_to_hz(m: f64) -> f64 {
            if m < 15.0 {
                m * (200.0 / 3.0)
            } else {
                1000.0 * (((6.4f64).ln() / 27.0) * (m - 15.0)).exp()
            }
        }
        let max = hz_to_mel(8000.0);
        let mut edges = [0.0; 130];
        for (i, e) in edges.iter_mut().enumerate() {
            *e = mel_to_hz(max * i as f64 / 129.0);
        }
        let mut filters = Vec::new();
        let mut filter_offsets = [0; 129];
        for band in 0..128 {
            filter_offsets[band] = filters.len();
            for bin in 0..201 {
                let frequency = bin as f64 * 40.0;
                let down = (frequency - edges[band]) / (edges[band + 1] - edges[band]);
                let up = (edges[band + 2] - frequency) / (edges[band + 2] - edges[band + 1]);
                let weight =
                    (down.min(up).max(0.0) * (2.0 / (edges[band + 2] - edges[band]))) as f32;
                if weight != 0.0 {
                    // The original first rounds each coefficient to f32 and
                    // promotes it to f64 for the dot product. Keep that order.
                    filters.push((bin, weight as f64));
                }
            }
        }
        filter_offsets[128] = filters.len();
        Self {
            fft,
            window,
            filter_offsets,
            filters,
        }
    }
    /// Returns row-major [128, samples/160], like Python encoder's mel tensor.
    pub fn extract(&self, audio: &[f32]) -> Result<(Vec<f32>, usize)> {
        ensure!(!audio.is_empty(), "Cannot extract mel from empty audio");
        ensure!(
            audio.iter().all(|v| v.is_finite()),
            "Audio contains non-finite samples"
        );
        let frames = audio.len() / 160;
        ensure!(frames > 0, "Audio must contain at least 160 samples");
        // The discarded last frame participates in log_spec.max() in NumPy.
        let fft_frames = 1 + audio.len() / 160;
        let mut logs = vec![0.0f64; 128 * fft_frames];
        let mut buffer = vec![Complex::new(0.0, 0.0); 400];
        let mut scratch = vec![Complex::new(0.0, 0.0); self.fft.get_inplace_scratch_len()];
        let mut powers = [0.0; 201];
        let mut maximum = f64::NEG_INFINITY;
        for frame in 0..fft_frames {
            for (i, value) in buffer.iter_mut().enumerate() {
                let sample = reflect_index(frame as isize * 160 + i as isize - 200, audio.len());
                *value = Complex::new(audio[sample] as f64 * self.window[i], 0.0);
            }
            self.fft.process_with_scratch(&mut buffer, &mut scratch);
            for (power, value) in powers.iter_mut().zip(&buffer[..201]) {
                *power = value.norm_sqr();
            }
            for band in 0..128 {
                let mut energy = 0.0;
                for &(bin, weight) in
                    &self.filters[self.filter_offsets[band]..self.filter_offsets[band + 1]]
                {
                    energy += weight * powers[bin];
                }
                let log = energy.max(1e-10).log10();
                logs[band * fft_frames + frame] = log;
                maximum = maximum.max(log);
            }
        }
        let floor = maximum - 8.0;
        let mut out = vec![0.0; 128 * frames];
        for band in 0..128 {
            for frame in 0..frames {
                out[band * frames + frame] =
                    ((logs[band * fft_frames + frame].max(floor) + 4.0) / 4.0) as f32;
            }
        }
        Ok((out, frames))
    }
}
fn reflect_index(index: isize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }
    let period = (len as isize - 1) * 2;
    let mapped = index.rem_euclid(period);
    if mapped < len as isize {
        mapped as usize
    } else {
        (period - mapped) as usize
    }
}
pub(super) fn output_length(frames: usize) -> usize {
    // Python uses floor division even for -1 in the zero remainder case.
    let left = (frames % 100) as isize;
    let a = (left - 1).div_euclid(2) + 1;
    let b = (a - 1).div_euclid(2) + 1;
    ((b - 1).div_euclid(2) + 1) as usize + (frames / 100) * 13
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reflect_matches_numpy_pattern() {
        let indices = (-4..9).map(|i| reflect_index(i, 4)).collect::<Vec<_>>();
        assert_eq!(indices, vec![2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2]);
    }
    #[test]
    fn silence_matches_python() {
        let (mel, n) = MelExtractor::new().extract(&vec![0.0; 32000]).unwrap();
        assert_eq!(n, 200);
        assert!(mel.iter().all(|v| *v == -1.5));
    }
    #[test]
    fn encoder_frame_boundaries() {
        assert_eq!(output_length(1), 1);
        assert_eq!(output_length(99), 13);
        assert_eq!(output_length(100), 13);
        assert_eq!(output_length(101), 14);
        assert_eq!(output_length(200), 26);
    }
}
