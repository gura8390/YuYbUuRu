//! NumPy RandomState's MT19937 randint(0, 2**31-1), used only for matched tests.
pub struct SamplerSeeds {
    state: [u32; 624],
    index: usize,
}
impl SamplerSeeds {
    pub fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }
    fn next_u32(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let v = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (v >> 1)
                    ^ if v & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut v = self.state[self.index];
        self.index += 1;
        v ^= v >> 11;
        v ^= (v << 7) & 0x9d2c5680;
        v ^= (v << 15) & 0xefc60000;
        v ^= v >> 18;
        v
    }
    pub fn next_seed(&mut self) -> u32 {
        loop {
            let v = self.next_u32() & 0x7fffffff;
            if v < 0x7fffffff {
                return v;
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numpy_randomstate_known_seed() {
        let mut rng = SamplerSeeds::new(0);
        assert_eq!(
            [rng.next_seed(), rng.next_seed(), rng.next_seed()],
            [209652396, 398764591, 924231285]
        );
    }
}
