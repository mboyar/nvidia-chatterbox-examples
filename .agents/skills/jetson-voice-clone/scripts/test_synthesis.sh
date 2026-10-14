#!/usr/bin/env bash
# Quick smoke-test for Jetson Chatterbox Voice Clone

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${PWD}/.venv/bin/activate" ]; then
    source "${PWD}/.venv/bin/activate"
elif [ -f "${SKILL_ROOT}/../../../.venv/bin/activate" ]; then
    source "${SKILL_ROOT}/../../../.venv/bin/activate"
elif [ -f "${SKILL_ROOT}/../../.venv/bin/activate" ]; then
    source "${SKILL_ROOT}/../../.venv/bin/activate"
fi

echo "Checking CUDA device on Jetson..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('✓ CUDA device:', torch.cuda.get_device_name(0))"

echo "Checking PyTorch allocator configuration..."
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ -f "${SCRIPT_DIR}/chatterbox_clone.py" ]; then
    echo "Running Chatterbox CLI smoke test..."
    python3 "${SCRIPT_DIR}/chatterbox_clone.py" --help > /dev/null
    echo "✓ CLI is ready and executable!"
fi
