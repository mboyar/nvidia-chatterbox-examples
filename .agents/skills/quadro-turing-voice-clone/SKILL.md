---
name: quadro-turing-voice-clone
description: >-
  Offline zero-shot voice cloning and speech synthesis on NVIDIA Quadro T2000
  (and discrete NVIDIA Turing series GPUs) using Chatterbox and Chatterbox-Nano within 4GB GDDR6 VRAM.
  Use this skill whenever setting up, optimizing, debugging, or running voice cloning, TTS, or audio generation
  on NVIDIA Turing (Compute Capability 7.5, sm_75) or memory-constrained discrete GPUs under Linux.
---

# NVIDIA Quadro T2000 (Turing 4GB VRAM) Voice Clone Skill

This skill teaches the agent how to configure, patch, optimize, and run 100% offline zero-shot voice cloning and text-to-speech synthesis on an **NVIDIA Quadro T2000 with Max-Q Design** (and other discrete NVIDIA Turing GPUs) using Resemble AI's **Chatterbox** and **Chatterbox-Nano**.

---

## 1. Quick Technical Reference & Rules

> [!IMPORTANT]
> The Quadro T2000 GPU has **4,096 MiB dedicated GDDR6 VRAM** backed by separate host DDR4 RAM, running under Linux with **Compute Capability 7.5** (`sm_75`). Follow these rules:

1. **Dedicated 4GB VRAM Envelope**:
   - Unlike Jetson Orin's unified memory where the operating system and page cache compete for GPU memory, the Quadro T2000 has 4GB of **dedicated GDDR VRAM**.
   - With Chatterbox-Nano, static weights + intermediate activations take **~1.1 GB in FP16 / ~1.6 GB in FP32**, leaving over **2.4 GB of VRAM headroom**.
   - Keep models resident in VRAM for high-throughput sentence streaming without PCIe bus swapping.
2. **FP16 Half-Precision Tensor Core Math**:
   - Turing's 2nd Generation Tensor Cores provide native, hardware-accelerated **FP16** matrix math, doubling compute throughput compared to FP32.
   - Generates speech tokens at **>90–120 tokens/s** with Chatterbox-Nano.
3. **TF32 and BF16 Considerations**:
   - Hardware BF16 Tensor Cores and TF32 were introduced in Ampere (`sm_80`+) and Blackwell (`sm_120`). On Turing (`sm_75`), TF32 is unsupported, and BF16 is emulated in software.
   - The Turing patch dynamically selects `torch.bfloat16` when hardware-accelerated (`torch.cuda.is_bf16_supported()`) and otherwise uses native FP16 (`torch.float16`) to maximize 2nd-gen Tensor Core performance, with FP32 softmax in T3 to prevent FP16 underflow.
4. **PyTorch Caching Allocator Configuration**:
   - Always export expandable segments:
     ```bash
     export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
     ```
   - In PyTorch 2.5+, never combine `expandable_segments:True` with `max_split_size_mb`.
5. **Full NVML Telemetry**:
   - Quadro T2000 fully supports NVML queries. `nvidia-smi`, `torch.cuda.memory_allocated()`, and VRAM monitoring work reliably without errors.
6. **Universal sm_75 Binary Kernel Support**:
   - Turing `sm_75` is natively supported across all modern PyTorch releases (CUDA 12.1 through CUDA 12.8+).

For deep architectural details, see [references/turing_gpu_architecture.md](./references/turing_gpu_architecture.md).

---

## 2. Core Architectural Breakthroughs

### A. Zero-Allocation In-Place Streaming Loading
Vanilla Chatterbox loads `s3gen_meanflow.safetensors` via `load_file()`, duplicating 1.015 GB into a Python dictionary and causing a 3+ GB RAM spike.
- Submodules (`flow`, `mel2wav`, `speaker_encoder`) are moved to CUDA *first* while uninitialized.
- Weights are mapped directly from disk using `safetensors.torch.safe_open`.
- Tensors are copied directly into pre-allocated GPU parameters via `param.data.copy_(t)`.
- **Result**: Eliminates dictionary duplication and completes loading in **~4–6 seconds**.

### B. High-Throughput Resident VRAM Mode
With Chatterbox-Nano:
- Both `T3` and `S3Gen` remain resident on CUDA concurrently.
- Synthesizes long text scripts at **>2.5x faster than real-time**.

### C. Low VRAM Multitasking Mode (`--low-vram`)
If running heavy local LLMs or image generation pipelines on the same 4GB GPU, passing `--low-vram` automatically offloads T3 to host RAM during vocoder inference, keeping peak VRAM **under 950 MB**.

### D. Self-Bootstrapping Venv Auto-Detection
`chatterbox_clone_turing.py` automatically resolves and re-execs with the project's virtual environment `.venv/bin/python3`, preventing `ModuleNotFoundError` when run from the command line.

Read full optimization details in [references/chatterbox_turing_optimizations.md](./references/chatterbox_turing_optimizations.md).

---

## 3. Step-by-Step Installation Runbook

When configuring this machine or any Turing workstation:

### Step 1: Verify Hardware & CUDA Runtime
Run the diagnostic smoke-test:
```bash
./.agents/skills/quadro-turing-voice-clone/scripts/test_synthesis.sh
```

### Step 2: Create Python Virtual Environment with CUDA PyTorch
Using `uv`:
```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create Python 3.12 virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate
echo 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' >> .venv/bin/activate

# Install PyTorch with CUDA support
uv pip install --python .venv/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install pip and core ML/audio dependencies
uv pip install --python .venv/bin/python pip soundfile librosa huggingface_hub transformers diffusers pyloudnorm numpy
```

### Step 3: Clone and Patch Chatterbox for Quadro Turing
```bash
git clone https://github.com/resemble-ai/chatterbox.git
python3 .agents/skills/quadro-turing-voice-clone/scripts/patch_chatterbox_turing.py chatterbox
uv pip install --python .venv/bin/python -e chatterbox
```
*(The patch file is located at [resources/chatterbox_turing.patch](./resources/chatterbox_turing.patch))*

---

## 4. Synthesizing Speech with Cloned Voice

Use the CLI runner [scripts/chatterbox_clone_turing.py](./scripts/chatterbox_clone_turing.py):

### Synthesize a Single Sentence:
```bash
python3 .agents/skills/quadro-turing-voice-clone/scripts/chatterbox_clone_turing.py \
  --reference path/to/my_voice.wav \
  --text "Hello! This voice clone is running offline on my NVIDIA Quadro T2000 GPU." \
  --output output.wav
```

### Synthesize a Full Script:
```bash
python3 .agents/skills/quadro-turing-voice-clone/scripts/chatterbox_clone_turing.py \
  --reference path/to/my_voice.wav \
  --script input.txt \
  --output full_speech.wav
```

### Synthesize & Playback Audio:
```bash
python3 .agents/skills/quadro-turing-voice-clone/scripts/chatterbox_clone_turing.py \
  --reference path/to/my_voice.wav \
  --text "Playing cloned audio directly on the workstation." \
  --play
```

### Reference Audio Best Practices
- **Duration**: 8 to 12 seconds (minimum 5.0 seconds).
- **Channels**: Mono (1 channel).
- **Format**: Clean WAV.
- See [references/reference_audio_guide.md](./references/reference_audio_guide.md) for recording guidelines.

---

## 5. Exporting This Skill to Another Machine / AGY Instance

To install this skill globally across all Antigravity agent sessions:
```bash
# Global install (installs to ~/.gemini/config/skills/quadro-turing-voice-clone)
./.agents/skills/quadro-turing-voice-clone/scripts/install_skill.sh global

# Local workspace install
./.agents/skills/quadro-turing-voice-clone/scripts/install_skill.sh workspace
```
