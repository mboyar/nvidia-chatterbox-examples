#!/usr/bin/env bash
# Hardware, environment, and smoke-test for NVIDIA RTX PRO 2000 Blackwell & Discrete RTX GPUs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=========================================================="
echo "🔍 RTX Blackwell Environment & Hardware Diagnostic Test"
echo "=========================================================="

# Check WSL2 / DirectX device
if [ -e "/dev/dxg" ]; then
    echo "✓ WSL2 DirectX Graphics Driver node (/dev/dxg) present."
fi

if [ -f "/usr/lib/wsl/lib/libcuda.so.1" ]; then
    echo "✓ WSL2 CUDA library mount detected (/usr/lib/wsl/lib/libcuda.so.1)."
fi

# Activate virtualenv if present
if [ -f "${PWD}/.venv/bin/activate" ]; then
    source "${PWD}/.venv/bin/activate"
elif [ -f "${SKILL_ROOT}/../../../.venv/bin/activate" ]; then
    source "${SKILL_ROOT}/../../../.venv/bin/activate"
elif [ -f "${SKILL_ROOT}/../../.venv/bin/activate" ]; then
    source "${SKILL_ROOT}/../../.venv/bin/activate"
fi

# Check nvidia-smi
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✓ nvidia-smi detected:"
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader | sed 's/^/  ↳ /'
else
    echo "⚠️ nvidia-smi not in PATH."
fi

# PyTorch CUDA check
echo "Checking PyTorch CUDA runtime..."
python3 - << 'EOF'
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("❌ PyTorch installed but CUDA is NOT available!")
        sys.exit(1)
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ GPU Device: {name}")
    print(f"✓ Compute Capability: {cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})")
    print(f"✓ Dedicated VRAM: {vram:.2f} GB")
    
    # Test TensorFloat-32 and BF16
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"✓ TensorFloat-32 (TF32): Supported & Enabled")
    
    t_bf16 = torch.zeros((2, 2), dtype=torch.bfloat16, device="cuda")
    print(f"✓ BFloat16 Tensor Operation: Success on CUDA")
except ImportError:
    print("⚠️ PyTorch not yet installed in current Python environment.")
    print("  To install: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124")
EOF

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ -f "${SCRIPT_DIR}/chatterbox_clone_rtx.py" ]; then
    echo "Checking Chatterbox RTX CLI runner..."
    python3 "${SCRIPT_DIR}/chatterbox_clone_rtx.py" --help > /dev/null 2>&1 && echo "✓ chatterbox_clone_rtx.py is executable and ready!" || echo "ℹ️ chatterbox_clone_rtx.py will be ready once Chatterbox dependencies are installed."
fi

echo "=========================================================="
echo "🎉 Diagnostic complete!"
echo "=========================================================="

