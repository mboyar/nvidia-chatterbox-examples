# Chatterbox CPU Optimizations (Measured)

All numbers below were measured on an **Intel Core i7-10850H** (Comet Lake, 6 cores / 12 threads, AVX2+FMA, no AVX-512, 30 GB DDR4) with PyTorch 2.11.0 (CUDA wheels, GPU hidden), MKL + oneDNN, Chatterbox-Nano, a 10.3 s reference voice, and `--seed 42`. Test text: two sentences, about 7 s of audio.

## 1. Benchmark summary

| Configuration | Threads | Generation time | Real-time factor (lower = faster) | Playback speed | Peak RSS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 (default)** | 6 (physical) | 17.8 s | **2.52x** | 0.40x RT | 3.65 GB |
| FP32 | 4 | 16.9 s | 2.38x | 0.42x RT | 3.65 GB |
| FP32 | 12 (all logical) | 124.5 s | **17.58x** ❌ | 0.06x RT | 3.65 GB |
| **int8 T3 backbone** (experimental) | 6 | 10.4 s | **1.81x** | 0.55x RT | 3.65 GB |

Other measurements:
- Model load: about 4–8 s (zero-copy safetensors streaming). Voice conditioning: about 1.9 s per reference voice.
- T3 decoding speed (FP32, 6 threads): about 18–22 speech tokens/s. S3Gen (2 meanflow steps): about 1–2 s per sentence.
- FP32 GEMM micro-benchmark (`test_synthesis.sh`): about 193 GFLOPS.

> [!NOTE]
> CPU synthesis on a 6-core laptop is **slower than real-time**, at about 0.4x (FP32) and 0.55x (int8). Use it for batch/offline generation, CI, or machines without a GPU. For interactive use, prefer the GPU skills.

## 2. Thread count: physical cores, never all logical CPUs

Using all 12 logical CPUs made synthesis **about 7x slower** than using 6 physical cores. OpenMP threads pinned to sibling hyperthreads contend for the same FMA units and L1/L2 cache. The per-token T3 matmuls are small (batch 1), so barrier synchronization overhead dominates.

4 and 6 threads perform about the same. T3 decoding is limited by memory bandwidth, so extra cores help only S3Gen. The default (physical cores) is a safe choice. On hosts with more cores, try `--threads` values between half and all physical cores.

## 3. int8 dynamic quantization of T3 (`--quantize int8`)

- HF GPT-2 implements its projections (`c_attn`, `c_proj`, `c_fc`) as `transformers.pytorch_utils.Conv1D`. `quantize_dynamic` only targets `nn.Linear`, so the CLI first converts **48 Conv1D layers** (Nano: 12 blocks × 4) into equivalent `nn.Linear` layers (`W_linear = W_conv1d.T`).
- Only `t3.tfmr` is quantized. The T3 heads stay FP32 because `T3.dtype` reads `speech_head.weight.dtype` to cast the conditioning tensors.
- Result: **about 28% faster** overall (RTF 2.52x → 1.81x). Peak RSS doesn't change, because S3Gen dominates memory use and the FP32 weights are loaded before quantization.
- **Quality caveat:** with the same seed, int8 produced slightly different (shorter) audio for one sentence, because quantization changes the sampling path. Listen to the output before relying on it. CPUs with `avx512_vnni` should see larger int8 gains.

## 4. Forcing CPU with CUDA wheels

`CUDA_VISIBLE_DEVICES=""` is exported before `import torch`, so `torch.cuda.is_available()` returns False. The patched loader then:
- loads checkpoints with `map_location='cpu'`,
- skips the bf16/fp16 conversion of T3 (keeps FP32),
- skips the low-VRAM T3 offload/`empty_cache` path.

The CLI asserts that T3 and S3Gen parameters are on CPU after loading.

## 5. Troubleshooting

| Symptom | Cause / Fix |
| :--- | :--- |
| Extremely slow (≪0.1x RT) | Thread oversubscription. Don't pass `--threads` above the physical core count, and check that no other process sets `OMP_NUM_THREADS` too high. Note that the CLI uses `setdefault`, so an exported `OMP_NUM_THREADS` wins over the default. Pass `--threads` explicitly. |
| `ModuleNotFoundError: chatterbox` | Install the patched package into the venv: `uv pip install --python .venv-cpu/bin/python -e chatterbox`. |
| RAM spike > 6 GB during load | Unpatched Chatterbox (`load_file` duplication). Apply `patch_chatterbox_cpu.py`. |
| BF16/FP16 dtype errors on CPU | Unpatched loader converted T3 to half precision. Apply the patch. |
| `AttributeError: 'function' object has no attribute 'dtype'` | The whole of T3 was quantized (including `speech_head`). Quantize only `t3.tfmr`, as the CLI does. |

