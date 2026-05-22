#!/usr/bin/env bash
# Author: ywangmu from HKUST
#
# B200 FA4 environment setup script.
# Run this once on the B200 server to set up the conda environment.
#
# Prerequisites:
#   - CUDA 13.0+ installed (check /usr/local/cuda/bin/nvcc)
#   - conda or miniconda installed
#   - Internet access (for pip installs)
#
# Usage:
#   bash B200_setup_fa4.sh            # full setup
#   bash B200_setup_fa4.sh --verify   # only verify, don't install
#   bash B200_setup_fa4.sh --env-name my_env  # custom env name

set -euo pipefail

ENV_NAME="${ENV_NAME:-b200_fa3}"
VERIFY_ONLY=""
PYTHON_VER="${PYTHON_VER:-3.12}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify)     VERIFY_ONLY="1"; shift ;;
    --env-name)   ENV_NAME="$2"; shift 2 ;;
    --python)     PYTHON_VER="$2"; shift 2 ;;
    *)            echo "Unknown arg: $1"; shift ;;
  esac
done

echo "============================================="
echo "B200 FA4 Environment Setup"
echo "============================================="
echo "  env name : $ENV_NAME"
echo "  python   : $PYTHON_VER"
echo "  verify   : $([ -n "$VERIFY_ONLY" ] && echo 'yes' || echo 'no (will install)')"
echo "============================================="

# ---------------------------------------------------------------------------
# 1. Check CUDA
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Checking CUDA..."
if [[ -f /usr/local/cuda/bin/nvcc ]]; then
    /usr/local/cuda/bin/nvcc --version | head -5
else
    echo "[error] nvcc not found at /usr/local/cuda/bin/nvcc"
    echo "        Install CUDA Toolkit 13.0+ first"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Check ncu
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Checking Nsight Compute..."
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
# 3. Create/verify conda environment
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Conda environment '$ENV_NAME'..."
if conda env list 2>/dev/null | grep -q "$ENV_NAME"; then
    echo "  Environment '$ENV_NAME' already exists."
else
    if [[ -n "$VERIFY_ONLY" ]]; then
        echo "[error] Environment '$ENV_NAME' not found. Run without --verify to create it."
        exit 1
    fi
    echo "  Creating environment..."
    conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
fi

# Helper to run in conda env
run_in_env() {
    conda run -n "$ENV_NAME" --no-banner "$@"
}

# ---------------------------------------------------------------------------
# 4. Install PyTorch
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Checking PyTorch..."
PYTORCH_INSTALLED=$(run_in_env python3 -c "
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
    echo "  Installing PyTorch (CUDA 13.0)..."
    run_in_env pip install torch --index-url https://download.pytorch.org/whl/cu130
fi

# ---------------------------------------------------------------------------
# 5. Install flash-attn-4 and dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Checking FA4 and dependencies..."
FA4_INSTALLED=$(run_in_env python3 -c "
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
        run_in_env pip install flash-attn-4

        # Install common dependencies
        echo "  Installing dependencies..."
        run_in_env pip install einops numpy
    fi
fi

# ---------------------------------------------------------------------------
# 6. Final verification
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Final verification..."
run_in_env python3 -c "
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
echo "To activate:  conda activate $ENV_NAME"
echo "To profile:   bash ncu_profile_fa4.sh --seqlen 128 --headdim 64"
echo "To benchmark:  python bench_fa4_simfa.py --mode perf --cases small"
