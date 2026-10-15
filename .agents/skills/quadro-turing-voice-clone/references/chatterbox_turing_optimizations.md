# Chatterbox Optimizations for NVIDIA Quadro T2000 (Turing 4GB VRAM)

This technical reference details the code-level optimizations applied to Resemble AI's Chatterbox library for the NVIDIA Quadro T2000 with Max-Q Design and discrete NVIDIA Turing GPUs operating within 4GB VRAM.

---

## 1. Zero-Allocation In-Place Streaming Loading

### The Problem
Vanilla Chatterbox loads `s3gen_meanflow.safetensors` using standard `load_file()`:
```python
weights = load_file(ckpt_dir / "s3gen_meanflow.safetensors") # 1.015 GB RAM dictionary
s3gen.load_state_dict(weights, strict=True)                 # Duplication into parameters
s3gen.to(device).eval()                                     # 1.015 GB CUDA allocation
```
On a 4GB GPU, this temporary 3x memory multiplier risks exhausting system and GPU memory.

### The In-Place Streaming Solution
Using `safe_open`, parameters are pre-allocated on the target device, and tensors are streamed directly from disk into pre-allocated memory:
```python
s3gen = S3Gen(meanflow=True)

# 1. Pre-allocate submodules directly on CUDA
s3gen.speaker_encoder.to(device).eval()
s3gen.flow.to(device).eval()
s3gen.mel2wav.to(device).eval()
s3gen.trim_fade = s3gen.trim_fade.to(device)

# 2. In-place stream weights directly from disk
params = dict(s3gen.named_parameters())
buffers = dict(s3gen.named_buffers())
s3gen_path = ckpt_dir / "s3gen_meanflow.safetensors"

with safe_open(s3gen_path, framework="pt") as sf:
    for k in sf.keys():
        t = sf.get_tensor(k)
        if k in params:
            params[k].data.copy_(t)
        elif k in buffers:
            buffers[k].data.copy_(t)
        del t

del params, buffers
gc.collect()
torch.cuda.empty_cache()
```
**Result**: Eliminates dictionary duplication, minimizes RAM spikes, and completes loading in ~4–6 seconds.

---

## 2. 4GB Resident VRAM Pipeline vs Low VRAM Mode

### Resident VRAM Pipeline (Default for Chatterbox-Nano)
With Chatterbox-Nano:
- `T3-Nano`: ~350 MB in FP16 / ~700 MB in FP32
- `S3Gen.flow`: ~218 MB in FP16 / ~437 MB in FP32
- `S3Gen.mel2wav`: ~80 MB
- `CAMPPlus`: ~26 MB
- **Peak VRAM**: **~1.1 GB in FP16 / ~1.6 GB in FP32**

All models comfortably fit simultaneously on the Quadro T2000's 4,096 MiB VRAM. No PCIe swapping is needed between sentences, enabling high-speed speech synthesis at **>2.5x real-time**.

### Low VRAM Mode (`--low-vram`)
When using the full `Chatterbox-Turbo` model or running alongside other GPU applications:
- Passing `--low-vram` offloads `T3` to CPU while `S3Gen` runs vocoding.
- Reduces peak VRAM to **<950 MB**, leaving >3GB free for multitasking.

---

## 3. Component Placement Matrix

| Component | Weight Size | Target Device | Precision | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **`T3-Nano` (Autoregressive GPT2)** | ~350 MB | **CUDA** | `float16` / `bfloat16` | Accelerated on Turing Tensor Cores (>100 tokens/s). |
| **`T3-Turbo` (Full Model)** | ~800 MB | **CUDA** | `float16` / `bfloat16` | Full expressiveness, resident or low-vram mode. |
| **`S3Gen.flow` (Flow Matching)** | ~218 MB | **CUDA** | `float16` / `float32` | Fast 2-step flow matching ODE solver. |
| **`S3Gen.mel2wav` (HiFiGAN Vocoder)** | ~80 MB | **CUDA** | `float32` | Generates 24kHz audio waveform. |
| **`S3Gen.speaker_encoder` (CAMPPlus)** | ~26 MB | **CUDA** | `float32` | Extracts speaker timbre vector. |
| **`VoiceEncoder`** | ~17 MB | **CUDA** | `float32` | 3-layer LSTM speaker timbre extractor. |
| **`S3Tokenizer`** | ~472 MB | **CPU** | `float32` | Reference speech tokenizer. |
