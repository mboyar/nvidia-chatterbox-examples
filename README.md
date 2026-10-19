# NVIDIA Chatterbox Voice Cloning & Speech Synthesis

Zero-shot, 100% offline voice cloning and text-to-speech synthesis using [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) across CPU and NVIDIA GPU architectures:
- **NVIDIA Quadro T2000 with Max-Q Design** (and discrete Turing GPUs within 4GB VRAM, Compute Capability 7.5)
- **NVIDIA RTX PRO 2000 Blackwell Laptop GPU** (and discrete RTX GPUs, Compute Capability 12.0)
- **NVIDIA Jetson Orin Nano** (and Orin series unified memory, Compute Capability 8.7)
- **x86_64 CPU (No GPU Required)** (Intel/AMD with AVX2/FMA, optimized for physical cores in FP32 with optional int8)

---

## Supported Architectures & Skills

| Feature | NVIDIA Quadro T2000 Turing (`quadro-turing-voice-clone`) | NVIDIA RTX PRO 2000 Blackwell (`rtx-blackwell-voice-clone`) | NVIDIA Jetson Orin Nano (`jetson-voice-clone`) | x86_64 CPU (`cpu-voice-clone`) |
| :--- | :--- | :--- | :--- | :--- |
| **GPU / Host Architecture** | **Turing** (`sm_75`, CC 7.5) | **Blackwell** (`sm_120`, CC 12.0) | **Ampere Tegra** (`sm_87`, CC 8.7) | **x86_64 CPU** (AVX2 / FMA / AVX-512) |
| **VRAM** | **4,096 MiB Dedicated GDDR6** | **8,151 MiB Dedicated GDDR7** | **8 GB Shared LPDDR5 (Unified)** | **None (No GPU required)** |
| **Host System RAM** | **30 GB DDR4 (Separate)** | **16 GB DDR5 (Separate)** | **None (Shared with GPU)** | **$\ge$ 8 GB recommended** (~3.65 GB peak RSS) |
| **Platform** | x86_64 / Linux (Ubuntu 26.04) | x86_64 / Linux (Ubuntu 26.04 under WSL2) | aarch64 / Linux Tegra (JetPack 6.x) | x86_64 / Linux / WSL2 / macOS |
| **Pipeline Strategy** | **Resident VRAM** (Nano) / **Low-VRAM** (Turbo) | **Resident VRAM** (All models on CUDA) | **Sequential Stage-Swapping** (T3 $\leftrightarrow$ CPU) | **Host RAM Resident** (Nano FP32 or int8) |
| **Peak VRAM / RAM** | ~1.1 GB (Nano) / <950 MB (Turbo Low-VRAM) | ~1.85 GB (fits easily within 8GB VRAM) | <900 MB (prevents Tegra iGPU OOM) | **~3.65 GB RAM (Peak RSS)** |
| **Hardware Math** | 2nd Gen Tensor Cores, **Native FP16** | 5th Gen Tensor Cores, **TF32** & **BF16** | 3rd Gen Tensor Cores, **BF16** | Vector SIMD (**FP32** MKL/oneDNN + opt. **int8**) |
| **Inference Speed** | **>2.5x real-time** (>90–120 tokens/s) | **>3.5x real-time** (>150 tokens/s) | **~1.7x real-time** (>50 tokens/s) | **~0.40x–0.55x real-time** (~18–25 tokens/s) |
| **NVML Telemetry** | Full Support (`nvidia-smi`, memory tracking) | Full Support (`nvidia-smi`, memory tracking) | Unsupported (Tegra fallback assert) | N/A (Peak RSS reported via `getrusage`) |


---

## 💻 Quick Start: NVIDIA Quadro T2000 & Turing GPUs

### 1. Diagnostic Verification
```bash
./.agents/skills/quadro-turing-voice-clone/scripts/test_synthesis.sh
```

### 2. Setup Environment
```bash
# Create Python 3.12 venv with uv and install PyTorch with CUDA support
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install --python .venv/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python pip soundfile librosa huggingface_hub transformers diffusers pyloudnorm numpy

# Clone and patch Chatterbox for Quadro Turing 4GB VRAM
git clone https://github.com/resemble-ai/chatterbox.git
python3 .agents/skills/quadro-turing-voice-clone/scripts/patch_chatterbox_turing.py chatterbox
uv pip install --python .venv/bin/python -e chatterbox
```

### 3. Synthesize Speech
```bash
# Single sentence synthesis (Chatterbox-Nano default, ~1.1GB peak VRAM)
python3 .agents/skills/quadro-turing-voice-clone/scripts/chatterbox_clone_turing.py \
  --reference path/to/sample.wav \
  --text "Hello! This voice clone is running offline on my NVIDIA Quadro T2000 GPU." \
  --output output.wav

# Synthesize full script and play audio
python3 .agents/skills/quadro-turing-voice-clone/scripts/chatterbox_clone_turing.py \
  --reference path/to/sample.wav \
  --script input.txt \
  --output full_speech.wav \
  --play
```

---

## 🚀 Quick Start: NVIDIA RTX PRO 2000 Blackwell

### 1. Diagnostic Verification
```bash
./.agents/skills/rtx-blackwell-voice-clone/scripts/test_synthesis.sh
```

### 2. Setup Environment
```bash
# Create venv and install PyTorch with CUDA 12.8+ (native sm_120 Blackwell support)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install --python .venv/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python pip soundfile librosa huggingface_hub transformers diffusers pyloudnorm numpy

# Clone and patch Chatterbox for RTX Blackwell
git clone https://github.com/resemble-ai/chatterbox.git
python3 .agents/skills/rtx-blackwell-voice-clone/scripts/patch_chatterbox_rtx.py chatterbox
uv pip install --python .venv/bin/python -e chatterbox
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

## 🖥️ Quick Start: CPU-only (x86_64, No GPU Required)

### 1. Diagnostic Verification
```bash
./.agents/skills/cpu-voice-clone/scripts/test_synthesis.sh
```

### 2. Setup Environment
```bash
# Create dedicated CPU virtualenv and install lightweight CPU-only PyTorch (~200MB wheels)
uv venv .venv-cpu --python 3.12
source .venv-cpu/bin/activate
uv pip install --python .venv-cpu/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-cpu/bin/python pip soundfile librosa huggingface_hub transformers diffusers pyloudnorm numpy psutil

# Clone and patch Chatterbox for CPU
git clone https://github.com/resemble-ai/chatterbox.git
python3 .agents/skills/cpu-voice-clone/scripts/patch_chatterbox_cpu.py chatterbox
uv pip install --python .venv-cpu/bin/python -e chatterbox
```
*(Note: You can also run against existing CUDA virtual environments; the CLI will automatically hide GPUs and enforce CPU execution).*

### 3. Synthesize Speech
```bash
# Single sentence synthesis (Nano default, FP32, automatic physical-core thread pinning)
python3 .agents/skills/cpu-voice-clone/scripts/chatterbox_clone_cpu.py \
  --reference path/to/sample.wav \
  --text "Hello! This voice clone is running entirely offline on an x86_64 CPU." \
  --output output.wav

# Synthesize full script and play audio
python3 .agents/skills/cpu-voice-clone/scripts/chatterbox_clone_cpu.py \
  --reference path/to/sample.wav \
  --script input.txt \
  --output full_speech.wav \
  --play

# Experimental int8 dynamic quantization of T3 backbone (25-30% faster)
python3 .agents/skills/cpu-voice-clone/scripts/chatterbox_clone_cpu.py \
  --reference path/to/sample.wav \
  --text "Quantized inference on CPU." \
  --quantize int8 \
  --threads 6
```

---

## Repository Structure

```
nvidia-chatterbox-examples/
├── .agents/skills/
│   ├── cpu-voice-clone/                # Optimized for x86_64 CPU-only systems (No GPU required)
│   │   ├── plugin.json                 # Antigravity skill manifest
│   │   ├── SKILL.md                    # Agent skill documentation & runbook
│   │   ├── scripts/
│   │   │   ├── chatterbox_clone_cpu.py # CPU-tuned CLI synthesizer (physical cores, int8 option)
│   │   │   ├── patch_chatterbox_cpu.py # Device-agnostic memory-streaming patcher
│   │   │   ├── test_synthesis.sh       # Hardware & CPU diagnostic test
│   │   │   └── install_skill.sh        # Antigravity (AGY) skill installer
│   │   ├── resources/
│   │   │   └── chatterbox_cpu.patch
│   │   └── references/
│   │       ├── cpu_architecture.md
│   │       ├── chatterbox_cpu_optimizations.md
│   │       └── reference_audio_guide.md
│   │
│   ├── quadro-turing-voice-clone/      # Optimized for Quadro T2000 & Turing GPUs (4GB VRAM)
│   │   ├── plugin.json                 # Antigravity skill manifest
│   │   ├── SKILL.md                    # Agent skill documentation & runbook
│   │   ├── scripts/
│   │   │   ├── chatterbox_clone_turing.py # 4GB VRAM CLI synthesizer
│   │   │   ├── patch_chatterbox_turing.py # Turing optimization patcher
│   │   │   ├── test_synthesis.sh       # Hardware & CUDA diagnostic test
│   │   │   └── install_skill.sh        # Antigravity (AGY) skill installer
│   │   ├── resources/
│   │   │   └── chatterbox_turing.patch
│   │   └── references/
│   │       ├── turing_gpu_architecture.md
│   │       ├── chatterbox_turing_optimizations.md
│   │       └── reference_audio_guide.md
│   │
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

Install the appropriate skill for automatic discovery across your Antigravity agent workflows:

```bash
# Install CPU-only skill
./.agents/skills/cpu-voice-clone/scripts/install_skill.sh workspace

# Install Quadro Turing skill
./.agents/skills/quadro-turing-voice-clone/scripts/install_skill.sh workspace

# Install RTX Blackwell skill
./.agents/skills/rtx-blackwell-voice-clone/scripts/install_skill.sh workspace

# Install Jetson Orin skill
./.agents/skills/jetson-voice-clone/scripts/install_skill.sh workspace
```

## License

MIT License. See individual submodules and upstream repositories for third-party licenses.
