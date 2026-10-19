#!/usr/bin/env bash
# Hardware, environment, and smoke-test for CPU-only Chatterbox voice cloning (no GPU required)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=========================================================="
echo "🔍 CPU Environment & Hardware Diagnostic Test"
echo "=========================================================="

# Activate virtualenv if present (prefer CPU-only venv)
for candidate in \
    "${PWD}/.venv-cpu" "${SKILL_ROOT}/../../../.venv-cpu" \
    "${PWD}/.venv" "${SKILL_ROOT}/../../../.venv"; do
    if [ -f "${candidate}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${candidate}/bin/activate"
        echo "✓ Using virtualenv: $(cd "${candidate}" && pwd)"
        break
    fi
done

# CPU topology & ISA
if command -v lscpu >/dev/null 2>&1; then
    echo "✓ CPU topology:"
    lscpu | grep -E "^(Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU\(s\)):" | sed 's/^/  ↳ /'
fi
FLAGS="$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null || true)"
ISA=""
for f in avx2 fma avx512f avx512_vnni avx512_bf16 amx_tile amx_bf16; do
    if echo "${FLAGS}" | grep -qw "${f}"; then ISA="${ISA} ${f}"; fi
done
echo "✓ ISA extensions:${ISA:- baseline (no AVX2!)}"
if ! echo "${FLAGS}" | grep -qw avx2; then
    echo "⚠️ AVX2 not detected: CPU inference will be very slow."
fi

# Memory
if command -v free >/dev/null 2>&1; then
    echo "✓ System memory:"
    free -h | sed 's/^/  ↳ /'
fi

# Force CPU for the PyTorch checks
export CUDA_VISIBLE_DEVICES=""

echo "Checking PyTorch CPU runtime..."
python3 - << 'EOF'
import sys, time
try:
    import torch
except ImportError:
    print("⚠️ PyTorch not yet installed in current Python environment.")
    print("  To install: uv pip install --python .venv-cpu/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(0)

build = f"CUDA {torch.version.cuda} build (GPU hidden for CPU test)" if torch.version.cuda else "CPU-only build"
print(f"✓ PyTorch version: {torch.__version__} ({build})")
print(f"✓ MKL available: {torch.backends.mkl.is_available()}")
print(f"✓ oneDNN (mkldnn) available: {torch.backends.mkldnn.is_available()}")
print(f"✓ Default intra-op threads: {torch.get_num_threads()}")

x = torch.randn(1024, 1024, dtype=torch.float32)
for _ in range(3):
    _ = x @ x  # warmup
iters = 20
t0 = time.perf_counter()
for _ in range(iters):
    _ = x @ x
dt = time.perf_counter() - t0
gflops = (2 * 1024**3 * iters) / dt / 1e9
print(f"✓ FP32 matmul 1024x1024: {gflops:.1f} GFLOPS")
EOF

if [ -f "${SCRIPT_DIR}/chatterbox_clone_cpu.py" ]; then
    echo "Checking Chatterbox CPU CLI runner..."
    python3 "${SCRIPT_DIR}/chatterbox_clone_cpu.py" --help > /dev/null 2>&1 \
        && echo "✓ chatterbox_clone_cpu.py is executable and ready!" \
        || echo "ℹ️ chatterbox_clone_cpu.py will be ready once Chatterbox dependencies are installed."
fi

echo "=========================================================="
echo "🎉 Diagnostic complete!"
echo "=========================================================="

