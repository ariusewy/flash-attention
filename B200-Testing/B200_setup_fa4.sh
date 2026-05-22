#!/usr/bin/env bash
# Author: ywangmu from HKUST
#
# B200 FA4 environment setup script (container / bare-metal edition).
# Assumes a container with Python 3.10+ and pip already available.
# No conda involved — installs directly into the system Python.
#
# Prerequisites:
#   - CUDA 12.8+ installed (check /usr/local/cuda/bin/nvcc)
#   - python3 + pip in PATH
#   - Internet access (for pip installs)
#
# Usage:
#   bash B200_setup_fa4.sh            # full setup
#   bash B200_setup_fa4.sh --verify   # only verify, don't install

set -euo pipefail

VERIFY_ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify)     VERIFY_ONLY="1"; shift ;;
    *)            echo "Unknown arg: $1"; shift ;;
  esac
done

echo "============================================="
echo "B200 FA4 Environment Setup (container mode)"
echo "============================================="
echo "  python   : $(python3 --version 2>&1 || echo '?')"
echo "  pip      : $(pip3 --version 2>&1 | head -1 || echo '?')"
echo "  verify   : $([ -n "$VERIFY_ONLY" ] && echo 'yes' || echo 'no (will install)')"
echo "============================================="

# ---------------------------------------------------------------------------
# 1. Check CUDA
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Checking CUDA..."
if [[ -f /usr/local/cuda/bin/nvcc ]]; then
    /usr/local/cuda/bin/nvcc --version | head -5
else
    echo "[error] nvcc not found at /usr/local/cuda/bin/nvcc"
    echo "        Install CUDA Toolkit 12.8+ first"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Check ncu
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Checking Nsight Compute..."
if command -v ncu &>/dev/null; then
    ncu --version | head -3
else
    echo "[warn] ncu not on PATH; it may be at /usr/local/cuda/bin/ncu"
    if [[ -f /usr/local/cuda/bin/ncu ]]; then
        echo "       Found: /usr/local/cuda/bin/ncu"
        export PATH="/usr/local/cuda/bin:$PATH"
    else
        echo "[error] ncu not found"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 3. Install PyTorch + flash-attn-4 + dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Checking packages..."

# PyTorch
PYTORCH_INSTALLED=$(python3 -c "
import torch
print(torch.__version__)
" 2>/dev/null || echo "")

if [[ -n "$PYTORCH_INSTALLED" ]]; then
    echo "  PyTorch $PYTORCH_INSTALLED already installed."
else
    if [[ -n "$VERIFY_ONLY" ]]; then
        echo "[error] PyTorch not installed. Run without --verify to install."
        exit 1
    fi
    echo "  Installing PyTorch..."
    pip3 install torch --index-url https://download.pytorch.org/whl/cu130
fi

# flash-attn-4
FA4_INSTALLED=$(python3 -c "
from flash_attn_interface import flash_attn_func
print('ok')
" 2>/dev/null || echo "")

if [[ "$FA4_INSTALLED" == "ok" ]]; then
    echo "  flash-attn-4 already installed."
else
    if [[ -n "$VERIFY_ONLY" ]]; then
        echo "[warn] flash-attn-4 not installed. Run without --verify to install."
    else
        echo "  Installing flash-attn-4..."
        pip3 install flash-attn-4
    fi
fi

# Common dependencies
for pkg in einops numpy; do
    python3 -c "import ${pkg}" 2>/dev/null && \
        echo "  ${pkg}: $(python3 -c "import ${pkg}; print(${pkg}.__version__)" 2>/dev/null)" || {
        if [[ -z "$VERIFY_ONLY" ]]; then
            echo "  Installing ${pkg}..."
            pip3 install "$pkg"
        else
            echo "  [warn] ${pkg}: not installed"
        fi
    }
done

# ---------------------------------------------------------------------------
# 4. Final verification
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Final verification..."
python3 -c "
import sys
print('=== Package Versions ===')
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} (SM {props.major}.{props.minor}, {props.multi_processor_count} SMs)')

try:
    from flash_attn_interface import flash_attn_func
    print('  FA4: available (flash_attn_interface)')
except ImportError:
    try:
        from flash_attn import flash_attn_func
        print('  FA4: available (flash_attn)')
    except ImportError:
        print('  FA4: NOT AVAILABLE')
        sys.exit(1)

try:
    import einops
    print(f'  einops: {einops.__version__}')
except ImportError:
    print('  einops: not installed')

import numpy as np
print(f'  numpy: {np.__version__}')

# Quick functional test
if torch.cuda.is_available():
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16, device='cuda')
    from flash_attn_interface import flash_attn_func
    out = flash_attn_func(q, k, v)
    torch.cuda.synchronize()
    print(f'  FA4 smoke test: PASS (output={out.shape})')
" 2>&1

echo ""
echo "============================================="
echo "Setup Complete"
echo "============================================="
echo ""
echo "To profile:   bash ncu_profile_fa4.sh --seqlen 128 --headdim 64"
echo "To benchmark: python3 bench_fa4_simfa.py --mode perf --cases small"
