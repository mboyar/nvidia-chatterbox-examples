# NVIDIA Chatterbox Voice Cloning & Speech Synthesis

Zero-shot, 100% offline voice cloning and text-to-speech synthesis using [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) across NVIDIA GPU architectures:
- **NVIDIA RTX PRO 2000 Blackwell Laptop GPU** (and discrete RTX GPUs, Compute Capability 12.0)
- **NVIDIA Jetson Orin Nano** (and Orin series unified memory, Compute Capability 8.7)

---

## Supported Architectures & Skills

| Feature | NVIDIA RTX PRO 2000 Blackwell (`rtx-blackwell-voice-clone`) | NVIDIA Jetson Orin Nano (`jetson-voice-clone`) |
| :--- | :--- | :--- |
| **GPU Architecture** | **Blackwell** (`sm_120`, CC 12.0) | **Ampere Tegra** (`sm_87`, CC 8.7) |
| **VRAM** | **8,151 MiB Dedicated GDDR7** | **8 GB Shared LPDDR5 (Unified)** |
| **Host System RAM** | **16 GB DDR5 (Separate)** | **None (Shared with GPU)** |
| **Platform** | x86_64 / Linux (Ubuntu 26.04 under WSL2) | aarch64 / Linux Tegra (JetPack 6.x) |
| **Pipeline Strategy** | **Resident VRAM** (All models on CUDA) | **Sequential Stage-Swapping** (T3 $\leftrightarrow$ CPU) |
| **Peak VRAM** | ~1.85 GB (fits easily within 8GB VRAM) | <900 MB (prevents Tegra iGPU OOM) |
| **Hardware Math** | 5th Gen Tensor Cores, **TF32** & **BF16** | 3rd Gen Tensor Cores, **BF16** |
| **Inference Speed** | **>3.5x real-time** (>150 tokens/s) | **~1.7x real-time** (>50 tokens/s) |
| **NVML Telemetry** | Full Support (`nvidia-smi`, memory tracking) | Unsupported (Tegra fallback assert) |

---

## 🚀 Quick Start: NVIDIA RTX PRO 2000 Blackwell

### 1. Diagnostic Verification
```bash
./.agents/skills/rtx-blackwell-voice-clone/scripts/test_synthesis.sh
```

### 2. Setup Environment
```bash
# Create venv and install PyTorch with CUDA
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install soundfile librosa huggingface_hub transformers diffusers pyloudnorm

# Clone and patch Chatterbox for RTX Blackwell
git clone https://github.com/resemble-ai/chatterbox.git
python3 .agents/skills/rtx-blackwell-voice-clone/scripts/patch_chatterbox_rtx.py chatterbox
pip install -e chatterbox
```

### 3. Synthesize Speech
```bash
# Single sentence synthesis
python3 .agents/skills/rtx-blackwell-voice-clone/scripts/chatterbox_clone_rtx.py \
  --reference path/to/sample.wav \
  --text "Hello! This voice clone is running offline on my NVIDIA RTX PRO 2000 Blackwell GPU." \
  --output output.wav

# Synthesize full script and play audio in WSL2
python3 .agents/skills/rtx-blackwell-voice-clone/scripts/chatterbox_clone_rtx.py \
  --reference path/to/sample.wav \
  --script speech_script.txt \
  --output full_speech.wav \
  --play
```

---

## 🤖 Quick Start: NVIDIA Jetson Orin Nano

### 1. Diagnostic Verification
```bash
./.agents/skills/jetson-voice-clone/scripts/test_synthesis.sh
```

### 2. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
echo 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' >> .venv/bin/activate

# Clone and patch Chatterbox for Jetson Unified Memory
git clone https://github.com/resemble-ai/chatterbox.git
python3 .agents/skills/jetson-voice-clone/scripts/patch_chatterbox.py chatterbox
pip install -e chatterbox
pip install soundfile librosa huggingface_hub transformers diffusers pyloudnorm
```

### 3. Synthesize Speech
```bash
python3 .agents/skills/jetson-voice-clone/scripts/chatterbox_clone.py \
  --reference path/to/sample.wav \
  --text "Hello! This voice clone is running offline on my Jetson Orin Nano." \
  --output output.wav
```

---

## Repository Structure

```
nvidia-chatterbox-examples/
├── .agents/skills/
│   ├── rtx-blackwell-voice-clone/      # Optimized for RTX PRO 2000 Blackwell & Discrete RTX GPUs
│   │   ├── plugin.json                 # Antigravity skill manifest
│   │   ├── SKILL.md                    # Agent skill documentation & runbook
│   │   ├── scripts/
│   │   │   ├── chatterbox_clone_rtx.py # Resident VRAM CLI synthesizer
│   │   │   ├── patch_chatterbox_rtx.py # Blackwell optimization patcher
│   │   │   ├── test_synthesis.sh       # Hardware & WSL2 diagnostic test
│   │   │   └── install_skill.sh        # Antigravity (AGY) skill installer
│   │   ├── resources/
│   │   │   └── chatterbox_rtx_blackwell.patch
│   │   └── references/
│   │       ├── blackwell_gpu_architecture.md
│   │       ├── chatterbox_blackwell_optimizations.md
│   │       └── reference_audio_guide.md
│   │
│   └── jetson-voice-clone/             # Optimized for Jetson Orin Nano (8GB Unified Memory)
│       ├── plugin.json
│       ├── SKILL.md
│       ├── scripts/
│       │   ├── chatterbox_clone.py     # Stage-swapping CLI synthesizer
│       │   ├── patch_chatterbox.py     # Jetson unified memory patcher
│       │   ├── test_synthesis.sh       # Tegra diagnostic test
│       │   └── install_skill.sh
│       ├── resources/
│       │   └── chatterbox_jetson.patch
│       └── references/
│           ├── jetson_memory_architecture.md
│           ├── chatterbox_patches.md
│           └── reference_audio_guide.md
└── README.md
```

---

## Antigravity (AGY) Skill Installation

Install either skill for automatic discovery across your Antigravity agent workflows:

```bash
# Install RTX Blackwell skill
./.agents/skills/rtx-blackwell-voice-clone/scripts/install_skill.sh workspace

# Install Jetson Orin skill
./.agents/skills/jetson-voice-clone/scripts/install_skill.sh workspace
```

## License

MIT License. See individual submodules and upstream repositories for third-party licenses.
