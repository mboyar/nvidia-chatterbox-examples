# NVIDIA Jetson Chatterbox Voice Cloning

Zero-shot, 100% offline voice cloning and text-to-speech synthesis on **NVIDIA Jetson Orin Nano** (and Orin series) within 8 GB unified memory using [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox).

## Key Highlights

- **8 GB Unified Memory Optimized**: Peak VRAM stays under 900 MB, preventing Tegra iGPU OOM crashes.
- **Zero-Allocation Streaming**: Loads weights directly into pre-allocated memory via `safetensors`, eliminating ~1 GB host RAM spikes.
- **Stage-Aware Memory Swapping**: Swaps T3 model between CPU and CUDA across 68 GB/s unified memory between synthesis stages.
- **Fast Offline Inference**: Generates audio at ~1.7x real-time (>50 tokens/s for T3 in `bfloat16`).

## Prerequisites

- **Hardware**: NVIDIA Jetson Orin Nano / Orin NX / AGX Orin (JetPack 6.x, CUDA 12.6, Ubuntu 22.04 aarch64).
- **Environment**: PyTorch with CUDA enabled.
- **Allocator Config**:
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  ```
  *(Note: Never set `max_split_size_mb` alongside `expandable_segments:True` on PyTorch 2.5+).*

## Installation & Setup

1. **Clone and patch Chatterbox**:
   ```bash
   git clone https://github.com/resemble-ai/chatterbox.git
   python3 .agents/skills/jetson-voice-clone/scripts/patch_chatterbox.py chatterbox
   pip install -e chatterbox
   ```

2. **Install audio dependencies**:
   ```bash
   pip install soundfile librosa huggingface_hub transformers diffusers pyloudnorm
   ```

## Quick Start

Synthesize speech using the optimized CLI runner:

```bash
# Single sentence synthesis
python3 .agents/skills/jetson-voice-clone/scripts/chatterbox_clone.py \
  --reference path/to/sample.wav \
  --text "Hello! This voice clone is running offline on my Jetson Orin Nano." \
  --output output.wav

# Synthesize full script
python3 .agents/skills/jetson-voice-clone/scripts/chatterbox_clone.py \
  --reference path/to/sample.wav \
  --script speech_script.txt \
  --output full_speech.wav
```

> **Audio Reference Tip**: Use 8–12 seconds of clean 24kHz mono WAV audio with minimal background noise.

## Repository Structure

```
nvidia-chatterbox-examples/
└── .agents/skills/jetson-voice-clone/
    ├── plugin.json                 # Antigravity skill manifest
    ├── SKILL.md                    # Agent skill documentation & runbook
    ├── scripts/
    │   ├── chatterbox_clone.py     # End-to-end voice cloning CLI
    │   ├── patch_chatterbox.py     # Jetson memory optimization patcher
    │   ├── test_synthesis.sh       # Hardware & environment smoke test
    │   └── install_skill.sh        # Installer for Antigravity (AGY)
    ├── resources/
    │   └── chatterbox_jetson.patch # Patch for safe_open & sequential swapping
    └── references/                 # Architectural deep-dives & audio guides
```

## Antigravity (AGY) Skill Installation

Install as a reusable skill across your Antigravity agent workflows:

```bash
# Global install (available to all agent sessions)
./.agents/skills/jetson-voice-clone/scripts/install_skill.sh global

# Local workspace install
./.agents/skills/jetson-voice-clone/scripts/install_skill.sh workspace
```

## License

MIT License. See individual submodules and upstream repositories for third-party licenses.

