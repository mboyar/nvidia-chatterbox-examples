# NVIDIA Quadro T2000 (Turing Architecture) Voice Synthesis Guide

This technical reference explains the internal hardware architecture, memory mechanics, precision optimizations, and performance tuning for running deep learning speech synthesis models (specifically Resemble AI's Chatterbox) on the **NVIDIA Quadro T2000 with Max-Q Design** and other discrete **NVIDIA Turing** GPUs (Compute Capability 7.5 / `sm_75`) within a **4,096 MiB (4 GB)** VRAM budget.

---

## 1. Hardware Architecture Overview

| Parameter | NVIDIA Quadro T2000 Max-Q (This Machine) | NVIDIA RTX PRO 2000 Blackwell | Jetson Orin Nano 8GB |
| :--- | :--- | :--- | :--- |
| **GPU Architecture** | **Turing** (`sm_75`) | **Blackwell** (`sm_120`) | **Ampere** (`sm_87`) |
| **Compute Capability** | **7.5** | **12.0** | **8.7** |
| **VRAM Type & Capacity** | **4,096 MiB Dedicated GDDR6** | **8,151 MiB Dedicated GDDR7** | **8 GB Shared LPDDR5 (Unified)** |
| **Host System RAM** | **30 GB DDR4 (Separate Pool)** | **16 GB DDR5 (Separate Pool)** | **None (Shared with GPU)** |
| **Total System Memory** | **34 GB Combined** | **24 GB Combined** | **8 GB Total Shared** |
| **Tensor Cores** | **2nd Gen (Native FP16, INT8, INT4)** | 5th Gen (FP4, FP8, BF16, TF32) | 3rd Gen (BF16, TF32, INT8) |
| **BF16 Hardware Math** | **Emulated / Software** | Native Hardware Acceleration | Native Hardware Acceleration |
| **TF32 Support** | **Unsupported** | Native Hardware Acceleration | Native Hardware Acceleration |
| **NVML Telemetry** | **Full Support** (`nvidia-smi`) | **Full Support** (`nvidia-smi`) | **Unsupported** (`NVML assert`) |
| **Platform / OS** | x86_64 / Linux (Ubuntu 26.04) | x86_64 / Linux (Ubuntu under WSL2) | aarch64 / Linux Tegra (JetPack 6) |

---

## 2. Discrete 4GB VRAM Mechanics

### The 4GB VRAM Envelope
With 4,096 MiB of dedicated VRAM, the Quadro T2000 has half the VRAM of 8GB GPUs like the Blackwell RTX PRO 2000 or Jetson Orin. However, unlike Jetson's unified memory where the OS and page cache steal memory from the GPU, the Quadro T2000 has **dedicated GDDR VRAM** completely isolated from host RAM (30 GB).

To run Chatterbox reliably without Out-Of-Memory (OOM) errors:
1. **Zero-Allocation In-Place Streaming Loading**:
   Vanilla Chatterbox loads checkpoints into a Python dictionary, creating a 1.015 GB RAM spike before duplicating into model parameters. With `safe_open`, parameters are populated directly in-place.
2. **Chatterbox-Nano Recommended**:
   - `T3-Nano` (autoregressive GPT2 backbone) consumes ~350 MB in FP16 / ~700 MB in FP32.
   - `S3Gen.flow` consumes ~218 MB in FP16 / ~437 MB in FP32.
   - `S3Gen.mel2wav` (HiFiGAN vocoder) consumes ~80 MB.
   - `CAMPPlus` speaker encoder consumes ~26 MB.
   - Peak VRAM during speech generation is **~1.1 GB in FP16 / ~1.6 GB in FP32**, leaving over **2.4 GB of VRAM headroom**.
3. **Chatterbox-Turbo in Low-VRAM Mode**:
   For the full 800MB T3-Turbo model, passing `--low-vram` offloads T3 to host RAM during vocoding, keeping peak VRAM below **950 MB**.

---

## 3. Turing Compute & Precision Tuning

### A. FP16 (Half Precision) Tensor Core Math
Turing Tensor Cores provide **fast native FP16** matrix multiplication:
- Generates speech tokens at **>90–120 tokens/s** with Chatterbox-Nano.
- Cuts model weight footprint and intermediate KV-cache memory in half.

### B. BF16 vs. FP16 on Turing (`sm_75`)
Hardware BF16 Tensor Cores were introduced in Ampere (`sm_80`) and Blackwell (`sm_120`). On Turing (`sm_75`):
- PyTorch emulates BF16 operations on CUDA cores rather than using Tensor Cores.
- For optimal throughput and numerical stability on Turing, **`torch.float16`** or **`torch.float32`** is used for critical paths.

### C. Expandable Segments Allocator
To prevent VRAM fragmentation on 4GB GPUs:
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```
PyTorch's caching allocator uses virtual memory mapping to expand segments incrementally, eliminating out-of-memory errors caused by memory fragmentation.
