#!/usr/bin/env bash
# Author: ywangmu from HKUST
#
# B200 environment probe - detect GPU, driver, CUDA, ncu, Python, disk, etc.
# Run this first on the B200 server to verify the environment is ready.
#
# Usage:
#   bash B200_probe.sh          # print to stdout
#   bash B200_probe.sh > env_report.txt

set -euo pipefail

GPU_ID="${GPU_ID:-0}"

echo "============================================="
echo "B200 Environment Probe"
echo "============================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo ""

# ---------------------------------------------------------------------------
# 1. OS & CPU
# ---------------------------------------------------------------------------
echo "--- OS ---"
uname -a
echo ""

echo "--- CPU ---"
lscpu | grep -E "^(Model name|Socket|Core|Thread|CPU\(s\):)" | head -6
echo ""

# ---------------------------------------------------------------------------
# 2. GPU
# ---------------------------------------------------------------------------
echo "--- GPU ---"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version,persistence_mode, \
           clocks.max.sm,memory.total,memory.free,pcie.link.gen.max \
           --format=csv 2>/dev/null || nvidia-smi
echo ""

echo "--- GPU Inventory ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | while read line; do
    echo "  $line"
done
echo ""

# ---------------------------------------------------------------------------
# 3. CUDA Toolkit
# ---------------------------------------------------------------------------
echo "--- CUDA Toolkit ---"
if command -v nvcc &>/dev/null; then
    nvcc --version 2>&1 | head -5
else
    echo "nvcc not found on PATH"
    if [[ -f /usr/local/cuda/bin/nvcc ]]; then
        echo "  (found at /usr/local/cuda/bin/nvcc)"
        /usr/local/cuda/bin/nvcc --version 2>&1 | head -5
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Nsight Compute
# ---------------------------------------------------------------------------
echo "--- Nsight Compute ---"
if command -v ncu &>/dev/null; then
    ncu --version 2>&1 | head -3
    echo ""

    # Check if profiling is accessible
    echo "Profiling permission check:"
    if grep -q "RestrictProfilingToAdminUsers.*=.*1" /proc/driver/nvidia/params 2>/dev/null; then
        echo "  WARNING: RestrictProfilingToAdminUsers=1 (need sudo or modprobe override)"
        echo "  Fix: sudo sh -c 'echo options nvidia NVreg_RestrictProfilingToAdminUsers=0 > /etc/modprobe.d/nvidia-profiling.conf && modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia && modprobe nvidia'"
    else
        echo "  Profiling appears accessible (no restriction detected)"
    fi
else
    echo "ncu not found on PATH"
fi
echo ""

# ---------------------------------------------------------------------------
# 5. Python packages
# ---------------------------------------------------------------------------
echo "--- Python ---"
echo "python: $(python3 --version 2>/dev/null || echo 'not found')"
echo "pip:    $(pip3 --version 2>/dev/null | head -1 || echo 'not found')"
echo ""
echo "--- Key packages ---"
pip3 list 2>/dev/null | grep -iE "torch|flash-attn|cutlass|quack|numpy|einops" || \
  echo "  (no matching packages found)"
echo ""

# ---------------------------------------------------------------------------
# 6. Disk
# ---------------------------------------------------------------------------
echo "--- Disk ---"
df -h /home 2>/dev/null | tail -1
df -h /mnt 2>/dev/null | tail -1 || true
# Check for nvme
for p in /mnt/nvme*; do
    if [[ -d "$p" ]]; then
        echo "NVMe mount: $p"
        df -h "$p" 2>/dev/null | tail -1
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 7. Quick CUDA smoke test
# ---------------------------------------------------------------------------
echo "--- CUDA Smoke Test ---"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}, SM {props.major}.{props.minor}, '
              f'{props.multi_processor_count} SMs, '
              f'{props.total_memory / 1e9:.1f} GB')

    # Quick FA4 check — try multiple import paths
    fa4_source = None
    for _mod_path in [
        "flash_attn_interface",       # FA4 official pip package
        "flash_attn.cute.interface",  # FA4 editable / CuTeDSL path
        "flash_attn",                 # FA3 / FA2 fallback (MHA only)
    ]:
        try:
            _mod = __import__(_mod_path, fromlist=["flash_attn_func"])
            flash_attn_func = _mod.flash_attn_func
            fa4_source = _mod_path
            break
        except (ImportError, AttributeError):
            continue

    if fa4_source is not None:
        print(f'FA4: flash_attn_func available  source={fa4_source}')

        # Minimal test
        q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16, device='cuda')
        result = flash_attn_func(q, k, v)
        torch.cuda.synchronize()
        out = result[0] if isinstance(result, (tuple, list)) else result
        print(f'FA4 smoke test: PASS (output shape={out.shape})')
    else:
        print('FA4: NOT AVAILABLE (install flash-attn-4)')
    except Exception as e:
        print(f'FA4 smoke test: FAILED ({e})')
else:
    print('CUDA not available!')
" 2>&1 || echo "(python3 check failed)"
echo ""

# ---------------------------------------------------------------------------
# 8. Clock grid
# ---------------------------------------------------------------------------
echo "--- Supported SM Clocks (GPU $GPU_ID, first 10) ---"
nvidia-smi -i "$GPU_ID" --query-supported-clocks=graphics --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | sort -un -r | head -10 || echo "(query failed)"
echo ""

echo "============================================="
echo "Probe Complete"
echo "============================================="
