# CPU Architecture Notes for Chatterbox Inference

This reference explains how x86_64 CPU characteristics affect Chatterbox (T3 GPT-2 decoder + S3Gen meanflow vocoder) inference, and why the `cpu-voice-clone` skill picks its defaults.

## 1. Two very different workloads

| Stage | Workload | Bottleneck on CPU |
| :--- | :--- | :--- |
| **T3** (GPT-2 small for Nano, GPT-2 medium for Turbo) | Autoregressive token-by-token decoding with a KV cache, batch = 1 | **Memory bandwidth.** Every generated token streams all T3 weights from RAM. |
| **S3Gen** (flow-matching decoder + HiFT vocoder, 2 meanflow steps) | A few large, parallel convolution/attention passes | **Compute (FLOPS).** Benefits from AVX2/AVX-512 and every physical core. |
| Voice encoder + S3 tokenizer (`prepare_conditionals`) | One-off per reference voice | Negligible (runs once). |

Because T3 is limited by bandwidth, shrinking its weights (int8) helps more than adding threads. S3Gen is limited by compute, so it scales with physical cores.

## 2. Threading model

PyTorch has two thread pools:
- **Intra-op** (`torch.set_num_threads`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`): parallelizes a single matmul/conv. This is the one that matters.
- **Inter-op** (`torch.set_num_interop_threads`): runs independent ops concurrently. Chatterbox's graph is sequential, so the skill sets it to **1** to avoid oversubscription.

**Hyperthreading:** two logical threads on one core share the same FMA units and L1/L2 cache. For dense FP32 GEMM, using all logical CPUs usually gives no gain or makes things slower, because of cache thrashing and OpenMP barrier overhead. The skill defaults to **physical cores**. Override it with `--threads N`.

Thread settings must be applied **before `import torch`**, because MKL and OpenMP read their environment variables when they initialize. `chatterbox_clone_cpu.py` reads `--threads` early and exports the variables before importing torch.

## 3. ISA extensions

| Extension | Effect |
| :--- | :--- |
| **AVX2 + FMA** | Baseline for usable speed. Present on Intel Haswell+ (2013) and AMD Zen. |
| **AVX-512F** | About 1.3–2x FP32 GEMM throughput on server/Ice Lake+ parts. |
| **AVX512_VNNI** | Faster int8 dot products. Makes `--quantize int8` noticeably more effective. |
| **AVX512_BF16 / AMX** | Native BF16 matmul (Cooper Lake, Sapphire Rapids+). **Only** on these CPUs is BF16 worth trying. |

Without AVX512_BF16 or AMX, BF16 matmuls are emulated and are **slower** than FP32. FP16 matmul isn't supported on CPU in eager mode. That's why the skill always runs in **FP32**, with int8 as the optional speed-up.

`test_synthesis.sh` prints the detected extensions and an FP32 GEMM GFLOPS micro-benchmark.

## 4. Memory footprint

There is no VRAM limit. Weights live in host RAM:
- Chatterbox-Nano FP32: T3 (GPT-2 small) + S3Gen (about 1 GB safetensors) + voice encoder.
- Chatterbox-Turbo FP32: T3 (GPT-2 medium) adds about 1.2 GB more.

The zero-copy `safe_open` streaming loader from the patch avoids the 3+ GB dictionary-duplication spike of vanilla `load_file()`. Measured peak RSS values are in [chatterbox_cpu_optimizations.md](./chatterbox_cpu_optimizations.md).

## 5. Telemetry

NVML isn't involved. The CLI reports **peak RSS** through `resource.getrusage(RUSAGE_SELF).ru_maxrss`. For live monitoring, use `htop` or `/usr/bin/time -v`.

