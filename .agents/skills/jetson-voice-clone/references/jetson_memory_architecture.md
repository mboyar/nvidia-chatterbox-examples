# Jetson Orin Nano Unified Memory Architecture & Optimization Guide

This reference explains the internal memory mechanics, driver quirks, and allocation constraints when running deep learning speech synthesis models (like Chatterbox) on NVIDIA Jetson Orin Nano (8GB Unified LPDDR5 Memory).

---

## 1. Unified Memory vs Discrete GPU

On a desktop workstation with a discrete GPU (e.g., RTX 4090):
- System RAM and VRAM are separate physical memory pools.
- An out-of-memory error on GPU only means GPU VRAM was exceeded, while host RAM remains untouched.

On NVIDIA Jetson Orin Nano (and all Tegra Orin chips):
- **CPU RAM and GPU VRAM share the same physical 8GB LPDDR5 chip**.
- Theoretical memory bandwidth is **68 GB/s**.
- If the operating system or file page cache uses memory, available GPU VRAM shrinks directly.

---

## 2. Linux Page Cache (`buff/cache`) Exhaustion

When large model weights (`safetensors`, `.pt` files, checkpoints) are loaded from disk:
- The Linux kernel automatically caches the file blocks into RAM (`buff/cache`).
- On an 8GB Jetson:
  - OS / system processes: ~2.2 GB
  - Linux page cache: ~3.5 GB
  - **Free unfragmented physical RAM: drops to only ~1.5 GB!**
- Even though `free -m` shows `available: 4.8 GB`, available includes reclaimable clean cache pages. However, low-level Tegra drivers request immediate physical pages from specific memory zones.

### Reclaiming Page Cache
Before running heavy inference models, running the following reclaims 3GB+ of physical RAM:
```bash
sync && sudo sysctl -w vm.drop_caches=3
```

---

## 3. Tegra NvMap Driver & `error 12`

When PyTorch's CUDA caching allocator requests memory via `cudaMalloc`, the NVIDIA userspace library calls the Tegra kernel driver (`/dev/nvmap`).

1. `nvmap` attempts to allocate a physically contiguous memory block using `NvMapMemAllocInternalTagged`.
2. When physical RAM is fragmented by cached pages, a large contiguous block (e.g., >32MB) cannot be allocated.
3. The kernel driver prints to `stderr`:
   ```text
   NvMapMemAllocInternalTagged: 1075072515 error 12
   NvMapMemHandleAlloc: error 0
   ```
4. **Key Understanding**:
   - `error 12` (`ENOMEM`) is the contiguous allocation attempt failing.
   - `error 0` (`SUCCESS`) is the immediate fallback to non-contiguous paged allocation succeeding!
   - This message is **NOT a fatal error**. It is printed by `libnvmap.so` directly to file descriptor 2.
   - In `chatterbox_clone.py`, this driver noise is cleanly filtered from stderr.

---

## 4. The Tegra `NVML_SUCCESS == r` Assert

In PyTorch, when an Out-Of-Memory (OOM) error occurs, PyTorch tries to print a helpful diagnostics message:
```text
RuntimeError: CUDA out of memory. Tried to allocate X bytes...
```
To query GPU memory stats, PyTorch calls `nvmlDeviceGetMemoryInfo()`.

**The Tegra Gotcha**:
- Jetson integrated GPUs (iGPU) do **not** support NVML memory queries.
- `nvmlDeviceGetMemoryInfo` returns `NVML_ERROR_NOT_SUPPORTED`.
- PyTorch has a hard assertion in `CUDACachingAllocator.cpp`:
  ```cpp
  NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:838
  ```
- **Takeaway**: If you see `NVML_SUCCESS == r INTERNAL ASSERT FAILED`, it means **the Jetson ran out of memory**. It is NOT a corrupted PyTorch installation or a missing driver.

---

## 5. Allocation Configuration Rules

### Always Enable `expandable_segments:True`
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```
This tells PyTorch's caching allocator to dynamically map virtual memory addresses to physical pages rather than demanding huge monolithic contiguous segments.

### NEVER Combine with `max_split_size_mb`
In PyTorch 2.5+, `expandable_segments:True` and `max_split_size_mb` are mutually exclusive. Combining them causes PyTorch's allocator initialization to immediately crash.

---

## 6. CPU-GPU Model Placement Strategy

| Model Component | Size | Target Device | Reason |
| :--- | :--- | :--- | :--- |
| **`T3` (Autoregressive Transformer)** | ~800 MB | **CUDA** (`bfloat16`) | Runs at ~50 tokens/s (>2x real-time). Swapped to CPU during vocoding. |
| **`S3Gen.flow` (CFM Flow Matching)** | ~437 MB | **CUDA** (`float32`) | Runs 2-step flow matching in ~1.5s on CUDA. |
| **`S3Gen.mel2wav` (HiFiGAN Vocoder)** | ~80 MB | **CUDA** (`float32`) | High-speed neural vocoder generating 24kHz waveform. |
| **`S3Gen.speaker_encoder` (CAMPPlus)** | ~26 MB | **CPU** (`float32`) | Running on CUDA triggers cuDNN 2D conv workspace allocations that OOM. Runs in ~2.5s on CPU once during conditioning. |
| **`S3Tokenizer`** | ~472 MB | **CPU** (`float32`) | Runs STFT and mel extraction on CPU in ~1s. |
| **`VoiceEncoder`** | ~17 MB | **CPU** (`float32`) | 3-layer LSTM speaker timbre extractor. Runs in ~0.5s on CPU. |

