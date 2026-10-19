---
name: cpu-voice-clone
description: >-
  Offline zero-shot voice cloning and speech synthesis on x86_64 CPUs (no GPU required)
  using Chatterbox and Chatterbox-Nano in FP32 with optional int8 quantization.
  Use this skill whenever setting up, optimizing, debugging, or running voice cloning, TTS, or audio generation
  on CPU-only machines, hosts without an NVIDIA GPU, CI runners, or when CPU inference (--device cpu) is requested.
---

# CPU-only Voice Clone Skill (no GPU required)

This skill teaches the agent how to configure, tune, and run fully offline zero-shot voice cloning and text-to-speech synthesis on **x86_64 CPUs** with Resemble AI's **Chatterbox** and **Chatterbox-Nano**. No NVIDIA GPU is needed.

---

## 1. Quick Technical Reference & Rules

> [!IMPORTANT]
> CPU inference runs in **FP32** in host RAM. Speed depends on core count, AVX2/AVX-512 support, and memory bandwidth. Follow these rules:

1. **FP32 only (plus optional int8)**:
   - BF16 is emulated, and slower than FP32, on CPUs without `avx512_bf16`/AMX. FP16 matmul isn't supported on CPU.
   - The patched loader keeps T3 in FP32 whenever the device is not CUDA.
2. **Configure threads before importing torch**:
   - Set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` and call `torch.set_num_threads(N)` and `torch.set_num_interop_threads(1)`.
   - Default to **physical cores** (hyperthreads share the FMA units). `chatterbox_clone_cpu.py` does this automatically. Override with `--threads N`.
3. **Hide the GPU when using CUDA wheels**:
   - The CLI sets `CUDA_VISIBLE_DEVICES=""` so an existing CUDA build of PyTorch can't silently place tensors on a GPU.
4. **Use `torch.inference_mode()`** for the whole load + synthesis pipeline (no autograd bookkeeping).
5. **Prefer Chatterbox-Nano**:
   - The T3 decoder is limited by memory bandwidth on CPU. Nano (GPT-2 small) is much faster than Turbo (GPT-2 medium). Use `--turbo` only if you need its quality.
6. **AVX2 + FMA is the practical minimum**. Run `test_synthesis.sh` to check the ISA and measure FP32 GFLOPS.

For architectural details, see [references/cpu_architecture.md](./references/cpu_architecture.md).

---

## 2. CPU Optimizations

### A. Physical-core thread pinning
Intra-op threads are set to the number of physical cores, and inter-op threads to 1. This avoids OpenMP oversubscription and hyperthread cache contention.

### B. Zero-copy safetensors streaming (inherited from the patch)
`s3gen_meanflow.safetensors` is streamed with `safe_open` directly into pre-allocated parameters. This avoids the 3+ GB RAM spike of vanilla `load_file()`.

### C. Experimental int8 dynamic quantization (`--quantize int8`)
HF GPT-2 uses `transformers.pytorch_utils.Conv1D`, not `nn.Linear`, so `quantize_dynamic` does nothing on it as-is. The CLI first converts every `Conv1D` in `t3.tfmr` into an equivalent `nn.Linear` (transposed weight), then applies `torch.ao.quantization.quantize_dynamic(..., {nn.Linear}, qint8)`. The T3 heads stay in FP32 because `T3.dtype` is derived from `speech_head.weight`.

### D. Self-bootstrapping venv detection
`chatterbox_clone_cpu.py` re-execs into `.venv-cpu/bin/python3` if present, otherwise `.venv/bin/python3`.

Measured effects and troubleshooting: [references/chatterbox_cpu_optimizations.md](./references/chatterbox_cpu_optimizations.md).

---

## 3. Step-by-Step Installation Runbook

### Step 1: Verify CPU & PyTorch runtime
```bash
./.agents/skills/cpu-voice-clone/scripts/test_synthesis.sh
```

### Step 2: Create a CPU-only Python environment
CPU-only wheels are about 200 MB, compared with about 2.5 GB for the CUDA wheels:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing

uv venv .venv-cpu --python 3.12
uv pip install --python .venv-cpu/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-cpu/bin/python pip soundfile librosa huggingface_hub transformers diffusers pyloudnorm numpy psutil
```
> [!TIP]
> An existing CUDA `.venv` works too. The CLI hides the GPU and runs on CPU.

### Step 3: Clone and patch Chatterbox
```bash
git clone https://github.com/resemble-ai/chatterbox.git
python3 .agents/skills/cpu-voice-clone/scripts/patch_chatterbox_cpu.py chatterbox
uv pip install --python .venv-cpu/bin/python -e chatterbox
```
*(The patch at [resources/chatterbox_cpu.patch](./resources/chatterbox_cpu.patch) is identical to the Turing patch. Its loader and offload paths are device-agnostic.)*

---

## 4. Synthesizing Speech with Cloned Voice

Use the CLI runner [scripts/chatterbox_clone_cpu.py](./scripts/chatterbox_clone_cpu.py):

```bash
# Single sentence (Nano, FP32, physical-core threads)
python3 .agents/skills/cpu-voice-clone/scripts/chatterbox_clone_cpu.py \
  --reference path/to/my_voice.wav \
  --text "Hello! This voice clone is running offline on my CPU." \
  --output output.wav

# Full script + playback
python3 .agents/skills/cpu-voice-clone/scripts/chatterbox_clone_cpu.py \
  --reference path/to/my_voice.wav --script input.txt --output full_speech.wav --play

# Experimental int8 T3 backbone, explicit thread count, reproducible seed
python3 .agents/skills/cpu-voice-clone/scripts/chatterbox_clone_cpu.py \
  --reference path/to/my_voice.wav --text "Quantized CPU inference." \
  --quantize int8 --threads 6 --seed 42
```

| Flag | Meaning |
| :--- | :--- |
| `--threads/-j N` | Intra-op threads (default: physical cores) |
| `--quantize {none,int8}` | int8 dynamic quantization of the T3 GPT-2 backbone (experimental) |
| `--turbo` | Full Chatterbox-Turbo instead of Nano (slower on CPU) |
| `--seed N` | Reproducible sampling |
| `--temperature`, `--pause-duration`, `--play` | Same as the GPU skills |

### Reference Audio Best Practices
- 8–12 s (minimum 5.0 s), mono, clean WAV. See [references/reference_audio_guide.md](./references/reference_audio_guide.md).

---

## 5. Measured Performance

Measured on an Intel Core i7-10850H (6C/12T, AVX2+FMA, 30 GB RAM), Chatterbox-Nano, about 7 s of output audio:

| Configuration | Threads | Real-time factor (lower = faster) | Playback speed | Peak RSS |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 (default)** | 6 (physical) | 2.52x | 0.40x RT | 3.65 GB |
| FP32 | 4 | 2.38x | 0.42x RT | 3.65 GB |
| FP32 | 12 (all logical) | **17.58x** ❌ | 0.06x RT | 3.65 GB |
| int8 T3 backbone | 6 | **1.81x** | 0.55x RT | 3.65 GB |

> [!WARNING]
> **Never use all logical CPUs.** Hyperthread oversubscription made synthesis about 7x slower. CPU synthesis on a laptop runs below real-time, so use it for offline or batch generation.

---

## 6. Exporting This Skill to Another Machine / AGY Instance

```bash
# Global install (installs to ~/.gemini/config/skills/cpu-voice-clone)
./.agents/skills/cpu-voice-clone/scripts/install_skill.sh global

# Local workspace install
./.agents/skills/cpu-voice-clone/scripts/install_skill.sh workspace
```

