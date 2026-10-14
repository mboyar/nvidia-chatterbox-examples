# NVIDIA RTX PRO 2000 Blackwell Architecture & Voice Synthesis Guide

This reference explains the internal hardware architecture, memory mechanics, WSL2 virtualization, and performance tuning for running deep learning speech synthesis models (such as Resemble AI's Chatterbox) on the **NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU** (Compute Capability 12.0 / `sm_120`).

---

## 1. Hardware Architecture Overview

| Parameter | NVIDIA RTX PRO 2000 Blackwell Laptop GPU | Jetson Orin Nano 8GB (Tegra Reference) |
| :--- | :--- | :--- |
| **GPU Architecture** | **Blackwell** (`sm_120`) | **Ampere** (`sm_87`) |
| **Compute Capability** | **12.0** | **8.7** |
| **VRAM Type & Capacity** | **8,151 MiB Dedicated GDDR7** | **8 GB Shared LPDDR5 (Unified)** |
| **Host System RAM** | **16 GB DDR5 (Separate Pool)** | **None (Shared with GPU)** |
| **Total System Address Space** | **24 GB Combined** | **8 GB Unified Total** |
| **Memory Bus & Transfer** | High-speed dedicated GDDR7 + PCIe / WSL2 | Unified 68 GB/s LPDDR5 shared bus |
| **Tensor Cores** | 5th Gen (FP4, FP8, BF16, TF32, INT8) | 3rd Gen (BF16, TF32, INT8) |
| **NVML Support** | **Full Support** (`nvidia-smi`, NVML queries) | **Unsupported** (`NVML_ERROR_NOT_SUPPORTED`) |
| **Platform / OS** | x86_64 / Linux on WSL2 (Ubuntu 26.04) | aarch64 / Linux Tegra (JetPack 6) |

---

## 2. Discrete VRAM vs. Unified Memory Mechanics

### The Unified Memory Challenge on Jetson
On Jetson Orin devices, CPU and GPU share the same 8GB physical memory. The Linux kernel page cache (`buff/cache`), desktop manager, and background processes consume 3.5GB+ of that pool. When PyTorch allocates CUDA memory, it competes directly with system memory. If memory runs out:
1. Low-level Tegra driver (`/dev/nvmap`) emits `error 12` (contiguous allocation failure).
2. PyTorch attempts to query GPU memory via NVML and hits `NVML_SUCCESS == r INTERNAL ASSERT FAILED`.
3. To avoid OOM, Jetson requires aggressively offloading the T3 model to CPU between generation stages.

### The Discrete VRAM Advantage on RTX PRO 2000 Blackwell
On the RTX PRO 2000 Blackwell:
1. **8GB Dedicated GDDR7 VRAM**: System OS processes, file page caches, and background applications live in the **16GB host RAM**. The GPU's 8GB VRAM is 100% reserved for CUDA tensors and model weights.
2. **No Memory Swapping Overhead**:
   - T3 (`bfloat16`): ~800 MB
   - S3Gen (`flow` + `mel2wav` in `float32`): ~520 MB
   - CAMPPlus speaker encoder: ~26 MB
   - Intermediate attention buffers & CFM workspaces: ~400–600 MB
   - **Total Peak VRAM Usage**: **~1.85 GB**
3. With 8,151 MiB of dedicated VRAM, **the entire Chatterbox pipeline fits comfortably in GPU memory simultaneously**, leaving over 6GB of headroom.
4. **Resident VRAM Mode**: Models remain resident on the GPU throughout the entire session. This eliminates the PCIe transfer latency and CPU cache synchronization overhead required by stage-swapping architectures.

---

## 3. Blackwell Compute Accelerations

### A. TensorFloat-32 (TF32)
Blackwell includes high-throughput TF32 Tensor Core math for single-precision operations.
Enabling TF32 provides up to **3–5x speedup** on matrix multiplications in S3Gen flow-matching and HiFiGAN vocoding without degrading audio fidelity:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### B. Native BFloat16 Precision
The autoregressive speech model (`T3`) runs natively in `torch.bfloat16` on Blackwell Tensor Cores:
- Halves memory footprint from 1.6GB to ~800MB.
- Preserves full 8-bit dynamic exponent range (preventing underflow during speech token generation).
- Achieves generation speeds in excess of **150–200 tokens/s**.

### C. Scaled Dot-Product Attention (SDPA)
Blackwell fully supports PyTorch SDPA with cuDNN / FlashAttention backends, fusing query-key-value projections and attention softmax into a single high-efficiency GPU kernel.

---

## 4. WSL2 Virtualization & Memory Driver

In Windows Subsystem for Linux 2 (WSL2):
- CUDA communication flows through the virtualized DirectX Graphics driver node (`/dev/dxg`) and `/usr/lib/wsl/lib/libcuda.so.1`.
- **Dynamic Virtual Memory Allocator**:
  Always set:
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  ```
  This enables PyTorch's caching allocator to dynamically map virtual memory segments rather than demanding contiguous blocks, preventing fragmentation in virtualized GPU memory.
- **Audio Output**:
  Audio generated inside WSL2 can be listened to directly through WSLg PulseAudio devices (`paplay`, `aplay`), or played on the Windows host using PowerShell (`powershell.exe -c "(New-Object Media.SoundPlayer 'path.wav').PlaySync()"`).

