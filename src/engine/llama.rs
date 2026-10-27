//! ABI copied from the installed ctypes wrapper, not from a potentially newer llama header.
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::{
    ffi::{CString, c_char, c_void},
    path::{Path, PathBuf},
    ptr,
    sync::{Arc, Mutex, OnceLock},
};

type Handle = *mut c_void;
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct ModelParams {
    devices: *mut Handle,
    tensor_buft_overrides: *mut Handle,
    n_gpu_layers: i32,
    split_mode: i32,
    main_gpu: i32,
    tensor_split: *mut f32,
    progress_callback: *mut c_void,
    progress_callback_user_data: Handle,
    kv_overrides: *mut Handle,
    vocab_only: bool,
    use_mmap: bool,
    use_direct_io: bool,
    use_mlock: bool,
    check_tensors: bool,
    use_extra_bufts: bool,
    no_host: bool,
    no_alloc: bool,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct ContextParams {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_threads: i32,
    n_threads_batch: i32,
    rope_scaling_type: i32,
    pooling_type: i32,
    attention_type: i32,
    flash_attn_type: i32,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    defrag_thold: f32,
    cb_eval: Handle,
    cb_eval_user_data: Handle,
    type_k: i32,
    type_v: i32,
    abort_callback: Handle,
    abort_callback_data: Handle,
    embeddings: bool,
    offload_kqv: bool,
    no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,
    samplers: *mut Handle,
    n_samplers: usize,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Batch {
    n_tokens: i32,
    token: *mut i32,
    embd: *mut f32,
    pos: *mut i32,
    n_seq_id: *mut i32,
    seq_id: *mut *mut i32,
    logits: *mut i8,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct SamplerParams {
    no_perf: bool,
}
pub(super) type ToFloat = unsafe extern "C" fn(*const c_void, *mut f32, i64);

macro_rules! api {
    ($($name:ident: $ty:ty),* $(,)?) => {
        pub(super) struct Api {
            pub dir: PathBuf,
            _llama: Library, _base: Library, _ggml: Library, _cpu_runtime: Library,
            pub type_size: unsafe extern "C" fn(i32)->usize,
            pub block_size: unsafe extern "C" fn(i32)->i64,
            pub type_traits: unsafe extern "C" fn(i32)->*const c_void,
            $($name: $ty,)*
        }
        impl Api {
            unsafe fn load(dir: &Path) -> Result<Self> {
                let base = unsafe { load_library(&dir.join("ggml-base.dll"))? };
                let ggml = unsafe { load_library(&dir.join("ggml.dll"))? };
                // CPU plugins use an adjacent OpenMP runtime. Preload it with
                // altered search path so plugin scoring also works without
                // changing the parent application's global PATH/CWD.
                let cpu_runtime = unsafe { load_library(&dir.join("libomp140.x86_64.dll"))? };
                let library = unsafe { load_library(&dir.join("llama.dll"))? };
                let symbols = Self {
                    dir: dir.to_path_buf(),
                    type_size: unsafe { *base.get(b"ggml_type_size\0")? },
                    block_size: unsafe { *base.get(b"ggml_blck_size\0")? },
                    type_traits: unsafe { *base.get(b"ggml_get_type_traits\0")? },
                    $($name: unsafe { *library.get(concat!(stringify!($name), "\0").as_bytes())
                        .with_context(|| concat!("Missing installed llama ABI symbol: ", stringify!($name)))? },)*
                    _llama: library, _base: base, _ggml: ggml, _cpu_runtime: cpu_runtime,
                };
                // Explicit plugin directory avoids changing the process CWD/PATH.
                let load_all = unsafe { symbols._ggml.get::<unsafe extern "C" fn(*const c_char)>(b"ggml_backend_load_all_from_path\0")? };
                let path = path_cstring(dir)?;
                unsafe { load_all(path.as_ptr()); (symbols.llama_backend_init)(); }
                Ok(symbols)
            }
        }
    }
}
api! {
    llama_backend_init: unsafe extern "C" fn(),
    llama_model_default_params: unsafe extern "C" fn()->ModelParams,
    llama_model_load_from_file: unsafe extern "C" fn(*const c_char, ModelParams)->Handle,
    llama_model_free: unsafe extern "C" fn(Handle),
    llama_model_get_vocab: unsafe extern "C" fn(Handle)->Handle,
    llama_model_n_embd: unsafe extern "C" fn(Handle)->i32,
    llama_vocab_n_tokens: unsafe extern "C" fn(Handle)->i32,
    llama_vocab_eos: unsafe extern "C" fn(Handle)->i32,
    llama_tokenize: unsafe extern "C" fn(Handle,*const c_char,i32,*mut i32,i32,bool,bool)->i32,
    llama_token_to_piece: unsafe extern "C" fn(Handle,i32,*mut c_char,i32,i32,bool)->i32,
    llama_context_default_params: unsafe extern "C" fn()->ContextParams,
    llama_init_from_model: unsafe extern "C" fn(Handle,ContextParams)->Handle,
    llama_free: unsafe extern "C" fn(Handle),
    llama_get_memory: unsafe extern "C" fn(Handle)->Handle,
    llama_memory_clear: unsafe extern "C" fn(Handle,bool),
    llama_decode: unsafe extern "C" fn(Handle,Batch)->i32,
    llama_batch_get_one: unsafe extern "C" fn(*mut i32,i32)->Batch,
    llama_get_logits_ith: unsafe extern "C" fn(Handle,i32)->*const f32,
    llama_sampler_chain_default_params: unsafe extern "C" fn()->SamplerParams,
    llama_sampler_chain_init: unsafe extern "C" fn(SamplerParams)->Handle,
    llama_sampler_chain_add: unsafe extern "C" fn(Handle,Handle),
    llama_sampler_init_top_k: unsafe extern "C" fn(i32)->Handle,
    llama_sampler_init_temp: unsafe extern "C" fn(f32)->Handle,
    llama_sampler_init_dist: unsafe extern "C" fn(u32)->Handle,
    llama_sampler_init_greedy: unsafe extern "C" fn()->Handle,
    llama_sampler_sample: unsafe extern "C" fn(Handle,Handle,i32)->i32,
    llama_sampler_free: unsafe extern "C" fn(Handle),
}

unsafe fn load_library(path: &Path) -> Result<Library> {
    #[cfg(windows)]
    {
        Ok(
            unsafe { libloading::os::windows::Library::load_with_flags(path, 0x00000008) }
                .with_context(|| format!("Load native DLL {}", path.display()))?
                .into(),
        )
    }
    #[cfg(not(windows))]
    {
        Ok(unsafe { Library::new(path)? })
    }
}
fn path_cstring(path: &Path) -> Result<CString> {
    let raw = path.to_str().context("Model/DLL path is not Unicode")?;
    let raw = raw.strip_prefix(r"\\?\").unwrap_or(raw);
    Ok(CString::new(raw.replace('\\', "/"))?)
}

static API: OnceLock<Arc<Api>> = OnceLock::new();
static INIT: Mutex<()> = Mutex::new(());
impl Api {
    pub fn get(dir: &Path) -> Result<Arc<Self>> {
        let dir = dir
            .canonicalize()
            .with_context(|| format!("Native DLL directory {}", dir.display()))?;
        let _lock = INIT
            .lock()
            .map_err(|_| anyhow::anyhow!("Native loader poisoned"))?;
        if let Some(api) = API.get() {
            ensure!(
                api.dir == dir,
                "A different llama DLL directory is already active"
            );
            return Ok(api.clone());
        }
        let api = Arc::new(unsafe { Self::load(&dir)? });
        let _ = API.set(api.clone());
        Ok(api)
    }

    pub fn dequantizer(&self, qtype: i32) -> Result<(usize, usize, Option<ToFloat>)> {
        ensure!((0..64).contains(&qtype), "Invalid GGML type {qtype}");
        let block = unsafe { (self.block_size)(qtype) };
        let size = unsafe { (self.type_size)(qtype) };
        ensure!(
            block > 0 && size > 0,
            "Unsupported GGML embedding type {qtype}"
        );
        if qtype == 0 || qtype == 1 {
            return Ok((block as usize, size, None));
        }
        let traits = unsafe { (self.type_traits)(qtype) } as *const u8;
        ensure!(
            !traits.is_null(),
            "No GGML type traits for embedding {qtype}"
        );
        // Both ABI variants occur in llama releases: newer ggml inserted an
        // int64 blck_size_interleave before type_size. Match the DLL's public
        // ggml_type_size result before reading its function pointer.
        let old_size = unsafe { ptr::read_unaligned(traits.add(16).cast::<usize>()) };
        let new_size = unsafe { ptr::read_unaligned(traits.add(24).cast::<usize>()) };
        let offset = if old_size == size {
            32
        } else if new_size == size {
            40
        } else {
            bail!("Unknown installed ggml type-traits ABI");
        };
        let convert: Option<ToFloat> = unsafe { ptr::read_unaligned(traits.add(offset).cast()) };
        ensure!(
            convert.is_some(),
            "No native embedding dequantizer for type {qtype}"
        );
        Ok((block as usize, size, convert))
    }
}

struct ModelInner {
    ptr: Handle,
    api: Arc<Api>,
}
impl Drop for ModelInner {
    fn drop(&mut self) {
        unsafe {
            (self.api.llama_model_free)(self.ptr);
        }
    }
}
pub(super) struct NativeModel {
    inner: Arc<ModelInner>,
    pub api: Arc<Api>,
    pub vocab: Handle,
    pub dim: usize,
    pub eos: i32,
}
// Models are read-only after loading. Every mutable context remains exclusively
// owned by its inference worker; moving an engine does not share its context.
unsafe impl Send for NativeModel {}
impl NativeModel {
    pub fn load(dll_dir: &Path, path: &Path, gpu: bool) -> Result<Self> {
        let stage = std::time::Instant::now();
        let api = Api::get(dll_dir)?;
        super::load_timing("llama.api_init", stage);
        let stage = std::time::Instant::now();
        let mut params = unsafe { (api.llama_model_default_params)() };
        params.n_gpu_layers = -1;
        let mut cpu_devices = [ptr::null_mut()];
        if !gpu {
            params.devices = cpu_devices.as_mut_ptr();
        }
        let file = path_cstring(path)?;
        let raw = unsafe { (api.llama_model_load_from_file)(file.as_ptr(), params) };
        super::load_timing("llama.gguf_load", stage);
        ensure!(!raw.is_null(), "Cannot load GGUF {}", path.display());
        let inner = Arc::new(ModelInner {
            ptr: raw,
            api: api.clone(),
        });
        let vocab = unsafe { (api.llama_model_get_vocab)(raw) };
        ensure!(!vocab.is_null(), "Native GGUF vocabulary is null");
        let dim = unsafe { (api.llama_model_n_embd)(raw) };
        ensure!(dim > 0, "Invalid model embedding dimension {dim}");
        let eos = unsafe { (api.llama_vocab_eos)(vocab) };
        Ok(Self {
            inner,
            api,
            vocab,
            dim: dim as usize,
            eos,
        })
    }
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        let mut out = vec![0; text.len() + 32];
        let mut n = unsafe {
            (self.api.llama_tokenize)(
                self.vocab,
                text.as_ptr().cast(),
                text.len().try_into()?,
                out.as_mut_ptr(),
                out.len().try_into()?,
                false,
                true,
            )
        };
        if n < 0 {
            out.resize(n.unsigned_abs() as usize, 0);
            n = unsafe {
                (self.api.llama_tokenize)(
                    self.vocab,
                    text.as_ptr().cast(),
                    text.len().try_into()?,
                    out.as_mut_ptr(),
                    out.len().try_into()?,
                    false,
                    true,
                )
            };
        }
        ensure!(
            n >= 0 && n as usize <= out.len(),
            "Native tokenization failed"
        );
        out.truncate(n as usize);
        Ok(out)
    }
    pub fn token_id(&self, text: &str) -> Result<i32> {
        let tokens = self.tokenize(text)?;
        ensure!(
            tokens.len() == 1,
            "Expected exactly one special token for {text}"
        );
        Ok(tokens[0])
    }
    pub fn token_bytes(&self, token: i32) -> Result<Vec<u8>> {
        let mut out = vec![0; 256];
        let mut n = unsafe {
            (self.api.llama_token_to_piece)(
                self.vocab,
                token,
                out.as_mut_ptr().cast(),
                out.len().try_into()?,
                0,
                true,
            )
        };
        if n < 0 {
            out.resize(n.unsigned_abs() as usize, 0);
            n = unsafe {
                (self.api.llama_token_to_piece)(
                    self.vocab,
                    token,
                    out.as_mut_ptr().cast(),
                    out.len().try_into()?,
                    0,
                    true,
                )
            };
        }
        ensure!(
            n >= 0 && n as usize <= out.len(),
            "Native token-to-piece failed"
        );
        out.truncate(n as usize);
        Ok(out)
    }
    pub fn sampler(&self, temperature: f32, seed: u32) -> Result<NativeSampler> {
        let api = &self.api;
        let ptr =
            unsafe { (api.llama_sampler_chain_init)((api.llama_sampler_chain_default_params)()) };
        ensure!(!ptr.is_null(), "Native sampler initialization failed");
        let chain = NativeSampler {
            ptr,
            api: api.clone(),
        };
        let add = |filter: Handle| -> Result<()> {
            ensure!(
                !filter.is_null(),
                "Native sampler filter initialization failed"
            );
            unsafe {
                (api.llama_sampler_chain_add)(ptr, filter);
            }
            Ok(())
        };
        unsafe {
            if temperature > 0.0 {
                add((api.llama_sampler_init_top_k)(50))?;
                add((api.llama_sampler_init_temp)(temperature))?;
                add((api.llama_sampler_init_dist)(seed))?;
            } else {
                add((api.llama_sampler_init_greedy)())?;
            }
        }
        Ok(chain)
    }
}

pub(super) struct NativeContext {
    ptr: Handle,
    model: Arc<ModelInner>,
    n_ctx: usize,
    needs_data_clear: bool,
}
unsafe impl Send for NativeContext {}
impl NativeContext {
    pub fn new(
        model: &NativeModel,
        n_ctx: u32,
        n_batch: u32,
        threads: i32,
        batch_threads: i32,
    ) -> Result<Self> {
        let api = &model.api;
        let mut p = unsafe { (api.llama_context_default_params)() };
        p.n_ctx = n_ctx;
        p.n_batch = n_batch;
        p.n_ubatch = 512;
        p.n_seq_max = 1;
        p.embeddings = false;
        p.pooling_type = 0;
        p.flash_attn_type = 1;
        p.offload_kqv = true;
        p.no_perf = true;
        p.n_threads = threads;
        p.n_threads_batch = batch_threads;
        let ptr = unsafe { (api.llama_init_from_model)(model.inner.ptr, p) };
        ensure!(!ptr.is_null(), "Native context initialization failed");
        Ok(Self {
            ptr,
            model: model.inner.clone(),
            n_ctx: n_ctx as usize,
            needs_data_clear: true,
        })
    }
    fn take_data_clear(needs_data_clear: &mut bool) -> bool {
        std::mem::replace(needs_data_clear, false)
    }
    pub fn clear(&mut self) -> Result<()> {
        let memory = unsafe { (self.model.api.llama_get_memory)(self.ptr) };
        ensure!(!memory.is_null(), "Native context memory is null");
        // Initialize every KV slot once. Later clears still reset all metadata,
        // without rewriting the allocated data buffers.
        let clear_data = Self::take_data_clear(&mut self.needs_data_clear);
        unsafe {
            (self.model.api.llama_memory_clear)(memory, clear_data);
        }
        Ok(())
    }
    pub fn prefill(
        &mut self,
        embeddings: &[f32],
        n: usize,
        dim: usize,
        logit_positions: &[usize],
    ) -> Result<()> {
        ensure!(
            n > 0
                && n <= self.n_ctx
                && embeddings.len() == n.checked_mul(dim).context("Embedding length overflow")?,
            "Invalid native prefill shape"
        );
        let mut positions = Vec::with_capacity(n * 4);
        for _ in 0..3 {
            positions.extend((0..n).map(|i| i as i32));
        }
        positions.resize(n * 4, 0);
        let mut n_seq = vec![1; n];
        let mut zero = 0i32;
        let mut seq = vec![&mut zero as *mut i32; n];
        let mut logits = vec![0i8; n];
        logits[n - 1] = 1;
        for &i in logit_positions {
            ensure!(i < n, "Logits index exceeds prompt");
            logits[i] = 1;
        }
        let batch = Batch {
            n_tokens: n.try_into()?,
            token: ptr::null_mut(),
            embd: embeddings.as_ptr().cast_mut(),
            pos: positions.as_mut_ptr(),
            n_seq_id: n_seq.as_mut_ptr(),
            seq_id: seq.as_mut_ptr(),
            logits: logits.as_mut_ptr(),
        };
        let status = unsafe { (self.model.api.llama_decode)(self.ptr, batch) };
        if status != 0 {
            self.needs_data_clear = true;
        }
        ensure!(
            status == 0,
            "Native prefill decode failed with status {status}"
        );
        Ok(())
    }
    pub fn decode_token(&mut self, mut token: i32) -> Result<i32> {
        // token storage outlives decode; returning a batch from a temporary token
        // allocation would leave the native token pointer dangling.
        let batch = unsafe { (self.model.api.llama_batch_get_one)(&mut token, 1) };
        let status = unsafe { (self.model.api.llama_decode)(self.ptr, batch) };
        if status != 0 {
            self.needs_data_clear = true;
        }
        Ok(status)
    }
    pub fn timestamp_argmax(&self, index: usize, limit: usize) -> Result<i32> {
        let api = &self.model.api;
        let vocab = unsafe { (api.llama_model_get_vocab)(self.model.ptr) };
        let vocab_size = unsafe { (api.llama_vocab_n_tokens)(vocab) };
        ensure!(
            limit > 0 && limit <= vocab_size as usize,
            "Timestamp limit exceeds vocabulary"
        );
        let logits = unsafe { (api.llama_get_logits_ith)(self.ptr, index.try_into()?) };
        ensure!(!logits.is_null(), "Native logits at {index} are null");
        let values = unsafe { std::slice::from_raw_parts(logits, limit) };
        let mut best = 0;
        for i in 1..limit {
            if values[i] > values[best] {
                best = i;
            }
        }
        Ok(best as i32)
    }
}
impl Drop for NativeContext {
    fn drop(&mut self) {
        unsafe {
            (self.model.api.llama_free)(self.ptr);
        }
    }
}
pub(super) struct NativeSampler {
    ptr: Handle,
    api: Arc<Api>,
}
impl NativeSampler {
    pub fn sample(&mut self, ctx: &NativeContext) -> Result<i32> {
        let token = unsafe { (self.api.llama_sampler_sample)(self.ptr, ctx.ptr, -1) };
        ensure!(token >= 0, "Native sampler returned invalid token {token}");
        Ok(token)
    }
}
impl Drop for NativeSampler {
    fn drop(&mut self) {
        unsafe {
            (self.api.llama_sampler_free)(self.ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn installed_windows_x64_struct_sizes() {
        assert_eq!(std::mem::size_of::<ModelParams>(), 72);
        assert_eq!(std::mem::size_of::<ContextParams>(), 136);
        assert_eq!(std::mem::size_of::<Batch>(), 56);
        assert_eq!(std::mem::offset_of!(ContextParams, samplers), 120);
        assert_eq!(std::mem::offset_of!(ModelParams, vocab_only), 64);
    }

    #[test]
    fn data_clear_policy_initializes_each_context_once() {
        for _new_context in 0..2 {
            let mut needs_data_clear = true;
            assert!(NativeContext::take_data_clear(&mut needs_data_clear));
            for _later_input in 0..16 {
                assert!(!NativeContext::take_data_clear(&mut needs_data_clear));
            }
        }
    }

    /// Root coordinates this model/GPU microbenchmark. It never encodes audio,
    /// decodes tokens, changes a production setting, or writes under app_dir.
    #[test]
    #[ignore = "Root-coordinated native model clear microbenchmark; reads production models"]
    fn clear_memory_probe() -> Result<()> {
        use serde_json::json;
        use std::time::Instant;

        fn median(values: impl Iterator<Item = f64>) -> f64 {
            let mut values = values.collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        }

        fn probe(
            app_dir: &Path,
            model_relative: &str,
            gpu: bool,
            n_ctx: u32,
            n_batch: u32,
            threads: i32,
            batch_threads: i32,
            cache_mib_from_existing_log: f64,
        ) -> Result<serde_json::Value> {
            let dll_dir = app_dir.join("core/server/engines/llama/bin");
            let model_path = app_dir.join(model_relative);
            let model = NativeModel::load(&dll_dir, &model_path, gpu)?;
            let ctx = NativeContext::new(&model, n_ctx, n_batch, threads, batch_threads)?;
            let api = &model.api;
            // Resolve from the already-loaded installed DLL, without extending
            // the production ABI table or assuming a newer private layout.
            let synchronize = unsafe {
                api._llama
                    .get::<unsafe extern "C" fn(Handle)>(b"llama_synchronize\0")
                    .map(|symbol| *symbol)
            };
            let synchronize_error = synchronize.as_ref().err().map(ToString::to_string);
            let synchronize = synchronize.ok();
            let memory = unsafe { (api.llama_get_memory)(ctx.ptr) };
            ensure!(!memory.is_null(), "Native context memory is null");
            for _ in 0..2 {
                unsafe {
                    (api.llama_memory_clear)(memory, true);
                    if let Some(synchronize) = synchronize {
                        synchronize(ctx.ptr);
                    }
                }
            }
            let mut samples = Vec::with_capacity(30);
            for pair in 0..15 {
                let order = if pair % 2 == 0 {
                    [true, false]
                } else {
                    [false, true]
                };
                for data in order {
                    if let Some(synchronize) = synchronize {
                        unsafe { synchronize(ctx.ptr) };
                    }
                    let started = Instant::now();
                    unsafe { (api.llama_memory_clear)(memory, data) };
                    let return_ms = started.elapsed().as_secs_f64() * 1000.0;
                    if let Some(synchronize) = synchronize {
                        unsafe { synchronize(ctx.ptr) };
                    }
                    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                    samples.push((pair, data, return_ms, elapsed_ms));
                }
            }
            let true_median = median(samples.iter().filter(|s| s.1).map(|s| s.3));
            let false_median = median(samples.iter().filter(|s| !s.1).map(|s| s.3));
            let pair_deltas = (0..15)
                .map(|pair| {
                    let true_ms = samples.iter().find(|s| s.0 == pair && s.1).unwrap().3;
                    let false_ms = samples.iter().find(|s| s.0 == pair && !s.1).unwrap().3;
                    true_ms - false_ms
                })
                .collect::<Vec<_>>();
            Ok(json!({
                "model": model_path,
                "gpu_requested": gpu,
                "n_ctx": n_ctx,
                "n_batch": n_batch,
                "n_ubatch": 512,
                "threads": threads,
                "batch_threads": batch_threads,
                "warmup_true_clears": 2,
                "pairs": 15,
                "synchronize_available": synchronize.is_some(),
                "synchronize_symbol": "llama_synchronize",
                "synchronize_error": synchronize_error,
                "timing_scope": if synchronize.is_some() {
                    "Clear return followed by installed llama_synchronize(context)"
                } else {
                    "Backend buffer clear return only; GPU completion is not guaranteed"
                },
                "cache_mib_from_existing_log": cache_mib_from_existing_log,
                "cache_size_is_memory_saved": false,
                "true_median_ms": true_median,
                "false_median_ms": false_median,
                "paired_saved_ms": pair_deltas,
                "paired_saved_median_ms": median(pair_deltas.iter().copied()),
                "samples": samples.iter().map(|s| json!({
                    "pair": s.0,
                    "clear_data": s.1,
                    "clear_return_ms": s.2,
                    "clear_and_synchronize_ms": synchronize.map(|_| s.3),
                    "elapsed_ms": s.3
                })).collect::<Vec<_>>()
            }))
        }

        let app_dir = std::env::var_os("CAPS_CLEAR_PROBE_APP")
            .map(PathBuf::from)
            .context("Set CAPS_CLEAR_PROBE_APP explicitly before running the native clear probe")?;
        let include_aligner = std::env::var("CAPS_CLEAR_PROBE_ALIGNER").as_deref() == Ok("1");
        let mut cases = Vec::new();
        let outcome = (|| -> Result<()> {
            cases.push(probe(
                &app_dir,
                "models/Qwen3-ASR/Qwen3-ASR-0.6B/qwen3_asr_llm.q4_k.gguf",
                true,
                2048,
                4096,
                1,
                1,
                224.0,
            )?);
            if include_aligner {
                cases.push(probe(
                    &app_dir,
                    "models/Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B/qwen3_aligner_llm.q4_k.gguf",
                    false, 3072, 3072, 16, 32, 336.0,
                )?);
            }
            Ok(())
        })();
        let report = json!({
            "complete": outcome.is_ok(),
            "error": outcome.as_ref().err().map(|error| format!("{error:#}")),
            "app_dir": app_dir,
            "include_aligner": include_aligner,
            "inferences": 0,
            "production_changes": 0,
            "cases": cases,
            "screening_reference": "Existing Rust v0.3 ORT8 r1 first-30 mean latency",
            "screening_reference_mean_ms": 5.444423600449227 / 30.0 * 1000.0,
            "one_percent_screening_ms": 5.444423600449227 / 30.0 * 10.0,
            "limitations": [
                "Freshly loaded contexts and repeated clears only; no recognition, quality or pipeline speed measurement.",
                "Cache capacities come from existing logs and are not measured memory savings.",
                "If synchronize is unavailable, timings cannot establish GPU completion or final performance benefit.",
                "Even synchronized clear deltas are screening evidence; adoption needs exact quality and matched end-to-end verification.",
                "No EmptyWorkingSet, heap trim, native library replacement, model change or production write."
            ]
        });
        let output =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("evidence/clear-memory-probe.json");
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
        eprintln!("clear memory probe: {}", output.display());
        outcome
    }
}
