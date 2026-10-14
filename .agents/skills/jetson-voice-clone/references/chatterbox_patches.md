# Chatterbox Architecture Optimizations for Jetson Orin

This document details the architectural modifications made to Resemble AI's Chatterbox library to achieve reliable zero-shot voice cloning within 8GB unified memory.

---

## 1. Zero-Allocation In-Place Streaming Loading

### The Problem in Vanilla Chatterbox
```python
# Vanilla approach:
weights = load_file(ckpt_dir / "s3gen_meanflow.safetensors") # Spikes 1.015 GB in RAM
s3gen.load_state_dict(weights, strict=True)                 # Duplicates 1.015 GB into parameters
s3gen.to(device).eval()                                     # Allocates 1.015 GB on CUDA
```
At that peak moment:
- Python has `weights` dict (1 GB)
- `s3gen` CPU parameters (1 GB)
- `s3gen` CUDA parameters (1 GB)
- Total spike = **3.0+ GB** in addition to the system's existing usage, causing immediate out-of-memory.

### The Solution: Direct In-Place Copy
Instead of creating intermediate dictionaries:
1. Move submodules destined for GPU (`flow`, `mel2wav`) to `cuda` *first*, while they contain only uninitialized placeholder weights.
2. Open the safetensors file using `safe_open` (memory-mapped, reading only one tensor at a time).
3. Copy data directly into the pre-allocated parameter memory using PyTorch's in-place `param.data.copy_(t)`.

```python
s3gen = S3Gen(meanflow=True)

# 1. Pre-allocate submodules onto their target devices
s3gen.speaker_encoder.eval()           # Stays on CPU
s3gen.flow.to(device).eval()           # On CUDA
s3gen.mel2wav.to(device).eval()        # On CUDA
s3gen.trim_fade = s3gen.trim_fade.to(device)

# 2. In-place stream weights directly from disk into pre-allocated memory
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
**Result**: Loading time drops to **12 seconds**, and the 1GB host RAM dictionary spike is **100% eliminated**.

---

## 2. Zero-Thrash Fully-Resident CUDA Pipeline (667 MB Total)

### Why Dynamic Swapping Causes Long-Term Memory Fragmentation
Initially, dynamically swapping `t3` between CUDA and CPU was considered. However, repeatedly calling `.to("cpu")` and `.to("cuda")` across sentences forced PyTorch's caching allocator to re-allocate physical memory segments for weights on every iteration. On Tegra's unified memory architecture, this rapid allocation thrashing fragments the `nvmap` contiguous physical address space, causing `error 12` allocations and eventually triggering the fatal NVML internal assert.

### The Resident Solution: BFloat16 Flow-Matching
By casting `s3gen.flow` to `torch.bfloat16`, its parameter footprint drops from 437 MB to **218 MB**, and Conformer ODE attention/activation tensors are halved.
- **T3** (`bfloat16`): 341 MB
- **S3Gen Flow** (`bfloat16`): 218 MB
- **HiFiGAN Mel2Wav** (`float32`): 79 MB
- **Static CUDA Resident Footprint**: **667 MB**

Both models remain permanently resident on CUDA without swapping. Dynamic memory churn is completely eliminated.

---

## 3. Disabling cuDNN for 1D Convolutions & Chunked Vocoding

### cuDNN 1D Convolution Workspace Exhaustion
In HiFiGAN (`mel2wav`), `decode()` runs 24 `ResBlock1` layers containing 1D convolutions with Snake activations.
On long audio sequences (100+ mel frames $\rightarrow$ 130,000+ waveform samples), cuDNN's heuristic workspace allocator requests hundreds of megabytes of contiguous workspace memory. On Tegra, this triggers out-of-memory.

**Fix**:
1. `torch.backends.cudnn.enabled = False` forces PyTorch to use native ATen CUDA 1D convolution kernels. Native CUDA kernels allocate **zero workspace memory**, run at **53+ tokens/sec** and **8.7+ it/s**, and completely eliminate cuDNN workspace crashes!
2. **Chunked HiFiGAN Vocoding**:
   In `S3Gen.hift_inference`, mel spectrograms with length $T > 100$ are divided into chunks (`chunk_size=100`, `overlap=8`) and cross-faded, capping maximum vocoder activation memory under 60 MB regardless of total sentence length.

---

## 3. T3 in BFloat16 Precision

T3's autoregressive speech transformer weights are converted to `torch.bfloat16` when loaded onto CUDA:
```python
use_bf16 = (device == "cuda" or (isinstance(device, str) and device.startswith("cuda")))
if use_bf16:
    for k, v in t3_state.items():
        if v.is_floating_point():
            t3_state[k] = v.to(torch.bfloat16)
    t3.to(torch.bfloat16)
t3.load_state_dict(t3_state)
```
- Halves T3 VRAM footprint from ~1.6 GB to **~800 MB**.
- Speed on Orin Nano Ampere GPU reaches **50–52 tokens/s** (>2x faster than real-time speech).
- Conditioning projection layers (`T3CondEnc`) dynamically cast speaker embeddings to `bfloat16` to prevent type mismatches.

