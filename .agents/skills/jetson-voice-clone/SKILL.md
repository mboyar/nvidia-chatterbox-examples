---
name: jetson-voice-clone
description: >-
  Offline zero-shot voice cloning and speech synthesis on NVIDIA Jetson Orin Nano (and Orin series)
  using Chatterbox and Chatterbox-Nano within 8GB unified memory. Use this skill whenever setting up,
  optimizing, debugging, or running voice cloning, TTS, or audio generation on NVIDIA Jetson or Tegra devices.
---

# Jetson Orin Voice Clone Skill

This skill teaches the agent how to configure, patch, optimize, and run 100% offline zero-shot voice cloning on an **NVIDIA Jetson Orin Nano** (and other Orin devices with 8GB unified memory) using Resemble AI's **Chatterbox** / **Chatterbox-Nano**.

---

## 1. Quick Technical Reference & Rules

> [!IMPORTANT]
> The Jetson Orin Nano has 8GB shared LPDDR5 Unified Memory (CPU and GPU share the same pool). You **MUST** strictly adhere to the following rules:

1. **Allocator Configuration**:
   - Always set `export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`.
   - **NEVER** set `max_split_size_mb` alongside `expandable_segments:True` in PyTorch 2.5+. They are mutually exclusive and will cause an immediate crash.
2. **The Signature Jetson "OOM" Error**:
   - `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:838`
   - This occurs when PyTorch runs out of memory and tries to query NVML for GPU memory stats. Jetson iGPU does not support NVML memory queries, throwing this assert. **It is an Out-Of-Memory indicator, not a library bug.**
3. **Ignore NvMap Driver Fallback Notices**:
   - `NvMapMemAllocInternalTagged: ... error 12` followed by `NvMapMemHandleAlloc: error 0` is the kernel driver falling back from contiguous to paged allocation. It is not fatal (`error 0` = success).
4. **Reclaim Linux Page Cache**:
   - When loading large checkpoints, the Linux page cache (`buff/cache`) can tie up 3.5GB+ of RAM. Drop clean page cache via:
     ```bash
     sync && sudo sysctl -w vm.drop_caches=3
     ```
5. **Submodule Placement Rules**:
   - **CPU**: Keep `speaker_encoder` (CAMPPlus), `s3gen.tokenizer`, and `VoiceEncoder` on CPU. Moving them to CUDA triggers cuDNN 2D convolution workspace allocations that immediately exhaust memory.
   - **CUDA**: Run `flow` and `mel2wav` on CUDA (`float32`).
   - **BFloat16**: Run autoregressive speech model `t3` in `torch.bfloat16` on CUDA (halves VRAM from ~1.6GB to ~800MB and generates speech at >50 tokens/s).

For deep technical details on memory mechanics, read [references/jetson_memory_architecture.md](./references/jetson_memory_architecture.md).

---

## 2. Core Architectural Breakthroughs

Standard Chatterbox crashes on 8GB Jetson devices because both model loading and inference exceed physical RAM. This skill applies two foundational breakthroughs:

### A. Zero-Allocation In-Place Streaming Loading
Instead of using `load_file()` which duplicates 1.015 GB into a Python dictionary:
1. Submodules destined for GPU (`flow`, `mel2wav`) are moved to CUDA *first* while uninitialized.
2. Weights are memory-mapped using `safetensors.torch.safe_open`.
3. Tensors are copied directly into the pre-allocated parameter memory in-place via `param.data.copy_(t)`.
- Eliminates 1GB+ host RAM spikes. Loading completes in **~12 seconds**.

### B. Stage-Aware Sequential Memory Swapping
Generation is strictly sequential:
- **Stage 1 (Text $\rightarrow$ Speech Tokens)**: T3 runs on CUDA at ~50 tokens/s.
- **Stage 2 (Tokens $\rightarrow$ Audio Waveform)**: T3 moves to CPU (takes only **140 ms** across 68 GB/s unified memory) and clears CUDA cache. This frees **800 MB** of VRAM, giving S3Gen ample headroom to run flow-matching and HiFiGAN without OOM.
- If additional sentences remain, T3 returns to CUDA in 150 ms.
- Peak VRAM never exceeds 900 MB. Synthesis runs at **~1.7x real-time**.

Read the full architectural breakdown in [references/chatterbox_patches.md](./references/chatterbox_patches.md).

---

## 3. Step-by-Step Installation Runbook

When setting up on a new Jetson machine or workspace:

### Step 1: Create Virtual Environment with Tegra PyTorch
Ensure JetPack 6.x is installed (Ubuntu 22.04 aarch64, CUDA 12.6):
```bash
python3 -m venv .venv
source .venv/bin/activate
echo 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' >> .venv/bin/activate
```
Install the official NVIDIA JetPack PyTorch 2.5 L4T wheel or verify torch has CUDA enabled:
```bash
python3 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

### Step 2: Clone and Patch Chatterbox
```bash
git clone https://github.com/resemble-ai/chatterbox.git
# Run the automated patcher script:
python3 scripts/patch_chatterbox.py chatterbox
# Install Chatterbox in editable mode:
pip install -e chatterbox
```
*(The patch file is located at [resources/chatterbox_jetson.patch](./resources/chatterbox_jetson.patch))*

### Step 3: Install Additional Dependencies
```bash
pip install soundfile librosa huggingface_hub transformers diffusers pyloudnorm
```

---

## 4. Synthesizing Speech with Cloned Voice

Use the included battle-tested CLI tool [scripts/chatterbox_clone.py](./scripts/chatterbox_clone.py):

### Synthesize a Single Sentence:
```bash
./scripts/chatterbox_clone.py \
  --reference path/to/my_voice.wav \
  --text "Hello! This voice clone is running offline on my Jetson Orin Nano." \
  --output output.wav
```

### Synthesize a Full Text Script:
```bash
./scripts/chatterbox_clone.py \
  --reference path/to/my_voice.wav \
  --script speech_script.txt \
  --output full_speech.wav
```

### Reference Audio Best Practices
- **Duration**: 8 to 12 seconds (minimum 5.0 seconds).
- **Channels**: Mono (1 channel).
- **Format**: Clean WAV.
- See [references/reference_audio_guide.md](./references/reference_audio_guide.md) for recording commands and reading passages.

---

## 5. Exporting This Skill to Another Machine / AGY Instance

To use this skill on any other machine running `agy`:
1. Copy this skill directory or extract the tarball:
   ```bash
   tar -xzf jetson-voice-clone-skill.tar.gz
   ```
2. Run the installer:
   ```bash
   ./scripts/install_skill.sh global
   ```
   *(Installs to `~/.gemini/config/skills/jetson-voice-clone`)*
3. Any Antigravity agent on the new machine will immediately recognize the skill and know how to operate Chatterbox on Jetson!

