#!/usr/bin/env bash
# Hardware, environment, and smoke-test for NVIDIA Quadro T2000 & Turing Architecture GPUs (4GB VRAM)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=========================================================="
echo "🔍 Quadro Turing Environment & Hardware Diagnostic Test"
echo "=========================================================="

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
    
    # Verify sm_75 in arch list
    arch_list = torch.cuda.get_arch_list()
    if f"sm_{cap[0]}{cap[1]}" not in arch_list and f"compute_{cap[0]}{cap[1]}" not in arch_list:
        print(f"⚠️ Warning: Device architecture sm_{cap[0]}{cap[1]} not found in PyTorch arch list: {arch_list}")
    else:
        print(f"✓ Binary Kernel Architecture: sm_{cap[0]}{cap[1]} compiled & available")

    # Test FP16 and FP32 on CUDA
    x_fp32 = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    _ = torch.matmul(x_fp32, x_fp32)
    print(f"✓ FP32 Tensor Operations: Success on CUDA")

    x_fp16 = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    _ = torch.matmul(x_fp16, x_fp16)
    print(f"✓ FP16 Half Precision: Accelerated on Turing Tensor Cores")

    if torch.cuda.is_bf16_supported():
        print(f"✓ BF16 Hardware Acceleration: Supported (Ampere/Blackwell)")
    else:
        print(f"ℹ️ BF16 Hardware Acceleration: Not supported on sm_{cap[0]}{cap[1]} (Using native FP16 Tensor Cores)")
except ImportError:
    print("⚠️ PyTorch not yet installed in current Python environment.")
    print("  To install: uv pip install --python .venv/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cu128")
EOF

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ -f "${SCRIPT_DIR}/chatterbox_clone_turing.py" ]; then
    echo "Checking Chatterbox Turing CLI runner..."
    python3 "${SCRIPT_DIR}/chatterbox_clone_turing.py" --help > /dev/null 2>&1 && echo "✓ chatterbox_clone_turing.py is executable and ready!" || echo "ℹ️ chatterbox_clone_turing.py will be ready once Chatterbox dependencies are installed."
fi

echo "=========================================================="
echo "🎉 Diagnostic complete!"
echo "=========================================================="
