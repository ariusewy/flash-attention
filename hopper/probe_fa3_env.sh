#!/usr/bin/env bash
# Author: ywangmu from HKUST
#
# Read-only environment survey for FA3 latency evaluation on an H800 server.
# Checks all dependencies required by the FA3 pipeline:
#   1) compile:  Python, PyTorch, CUDA Toolkit, setuptools, ninja, packaging, wheel
#   2) run:      nsys (Nsight Systems), nvtx Python bindings
#   3) lock clk: nvidia-smi + sudo for clock locking
#
# No kernel is launched, no sudo is required (unless PROBE_WITH_SUDO=1),
# no file outside the current directory is touched.
#
# Usage (on H800 host):
#   bash probe_fa3_env.sh
#   bash probe_fa3_env.sh /tmp/fa3_env.txt       # custom output path
#   GPU_ID=0 bash probe_fa3_env.sh               # scope GPU queries to a specific card
#   GPU_ID=0 PROBE_WITH_SUDO=1 bash probe_fa3_env.sh  # also test sudo + clock lock

set -u
LC_ALL=C
export LC_ALL

OUT="${1:-fa3_env_$(hostname -s 2>/dev/null || echo host)_$(date +%Y%m%d-%H%M%S).txt}"
: > "${OUT}"

# ---------------------------------------------------------------------------
# Helpers (same style as probe_h800_env.sh)
# ---------------------------------------------------------------------------
section() {
    printf '\n================================================================\n'  | tee -a "${OUT}"
    printf '  %s\n' "$*"                                                            | tee -a "${OUT}"
    printf '================================================================\n'    | tee -a "${OUT}"
}

run() {
    local label="$1"; shift
    printf '\n--- %s ---\n$ %s\n' "${label}" "$*" | tee -a "${OUT}"
    if "$@" >>"${OUT}" 2>&1; then
        :
    else
        local rc=$?
        printf '[probe] %s -> exit %d (kept going)\n' "${label}" "${rc}" | tee -a "${OUT}"
    fi
}

has() { command -v "$1" >/dev/null 2>&1; }

PASS=()
FAIL=()
WARN=()

check_pass() { PASS+=("$1"); }
check_fail() { FAIL+=("$1"); }
check_warn() { WARN+=("$1"); }

# ---------------------------------------------------------------------------
# Target GPU selection
# ---------------------------------------------------------------------------
GPU_ID="${GPU_ID:-}"
if [[ -n "${GPU_ID}" && ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU_ID=${GPU_ID} is not a non-negative integer." >&2
    exit 2
fi
SMI_I="${GPU_ID:-0}"

# ---------------------------------------------------------------------------
# Report header
# ---------------------------------------------------------------------------
section "FA3 environment probe (read-only, no kernel launches)"
{
    echo "generated_at   : $(date -Is 2>/dev/null || date)"
    echo "hostname       : $(hostname -f 2>/dev/null || hostname)"
    echo "kernel         : $(uname -a)"
    echo "script         : ${BASH_SOURCE[0]}"
    echo "output_path    : ${OUT}"
    echo "GPU_ID (target): ${GPU_ID:-<unset, using GPU 0 for queries>}"
    echo "probe_with_sudo: ${PROBE_WITH_SUDO:-0}"
} | tee -a "${OUT}"

# ===================================================================
# [1/9] OS & system resources
# ===================================================================
section "[1/9] OS & system resources"
run "cat /etc/os-release" cat /etc/os-release
run "glibc version"  bash -c 'ldd --version | head -1'
run "nproc"          bash -c 'printf "nproc = "; nproc'
run "free -h"        free -h
run "disk space"     bash -c 'df -h . | head -5'

# ===================================================================
# [2/9] GPU inventory & idle check
# ===================================================================
section "[2/9] GPU inventory"
if ! has nvidia-smi; then
    echo "FATAL: nvidia-smi not found in PATH." | tee -a "${OUT}"
    check_fail "nvidia-smi"
else
    check_pass "nvidia-smi"
    run "nvidia-smi" nvidia-smi
    run "per-GPU CSV" nvidia-smi \
        --query-gpu=index,name,compute_cap,driver_version,memory.total,memory.free \
        --format=csv

    run "GPU ${SMI_I} idle check" bash -c "
        util=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i ${SMI_I} 2>/dev/null)
        mem=\$(nvidia-smi --query-gpu=memory.used    --format=csv,noheader,nounits -i ${SMI_I} 2>/dev/null)
        name=\$(nvidia-smi --query-gpu=name           --format=csv,noheader         -i ${SMI_I} 2>/dev/null)
        cc=\$(nvidia-smi --query-gpu=compute_cap      --format=csv,noheader         -i ${SMI_I} 2>/dev/null)
        echo \"GPU ${SMI_I}: name=\${name}  cc=\${cc}  util=\${util}%  mem_used=\${mem}MB\"
    "

    run "supported SM clocks on GPU ${SMI_I} (top 20)" bash -c "
        nvidia-smi -i ${SMI_I} --query-supported-clocks=graphics --format=csv,noheader 2>/dev/null | \
        awk '{print \$1}' | sort -u -nr | head -20
    "

    run "is 1830 MHz supported?" bash -c "
        nvidia-smi -i ${SMI_I} --query-supported-clocks=graphics --format=csv,noheader 2>/dev/null | \
        awk '{print \$1}' | grep -qw 1830 && echo 'YES' || echo 'NO (pick from the list above)'
    "
fi

# ===================================================================
# [3/9] CUDA Toolkit
# ===================================================================
section "[3/9] CUDA Toolkit"
run "which nvcc"          bash -c 'command -v nvcc || echo "(nvcc not in PATH)"'
run "nvcc --version"      bash -c 'command -v nvcc >/dev/null && nvcc --version || echo "(skipped)"'

NVCC_VER=""
if has nvcc; then
    NVCC_VER=$(nvcc --version | grep -oP 'release \K[\d.]+' | head -1)
    run "nvcc version parsed" bash -c "echo '${NVCC_VER}'"
    if python3 -c "
from packaging.version import Version
v = Version('${NVCC_VER}')
if v < Version('12.3'):
    print('FAIL: CUDA {} < 12.3 (FA3 requires >= 12.3)'.format(v))
    raise SystemExit(1)
else:
    print('OK: CUDA {} >= 12.3'.format(v))
" >>"${OUT}" 2>&1; then
        check_pass "CUDA >= 12.3"
    else
        check_fail "CUDA >= 12.3 (got ${NVCC_VER})"
    fi
else
    check_fail "nvcc (not in PATH)"
fi

run "CUDA_HOME env"     bash -c 'echo "CUDA_HOME=${CUDA_HOME:-<unset>}"'
run "ldconfig libcuda"  bash -c 'ldconfig -p 2>/dev/null | grep libcuda | head -3 || echo "(no ldconfig or no libcuda)"'
run "cuda_runtime.h"    bash -c '
    for p in /usr/local/cuda/include/cuda_runtime.h /usr/include/cuda_runtime.h; do
        [ -f "$p" ] && { echo "found: $p"; grep -m1 "#define CUDART_VERSION" "$p"; break; }
    done || echo "(not found)"
'

# ===================================================================
# [4/9] Python & PyTorch
# ===================================================================
section "[4/9] Python & PyTorch"
run "which python3"       bash -c 'command -v python3 || echo "(python3 not in PATH)"'
run "python3 --version"   bash -c 'command -v python3 >/dev/null && python3 --version || echo "(skipped)"'
run "which pip / pip3"    bash -c 'command -v pip3 >/dev/null && echo "pip3=$(which pip3)" || { command -v pip >/dev/null && echo "pip=$(which pip)" || echo "(no pip)"; }'

if has python3; then
    PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    run "Python version parsed" bash -c "echo '${PYTHON_VER}'"

    # PyTorch
    run "import torch" bash -c 'python3 -c "import torch; print(\"torch\", torch.__version__)" || echo "(torch not installed)"'

    TORCH_OK=0
    TORCH_VER=""
    if python3 -c "import torch" >>"${OUT}" 2>&1; then
        TORCH_OK=1
        TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        run "torch.cuda.is_available()" bash -c "python3 -c 'import torch; print(torch.cuda.is_available())'"
        run "torch.version.cuda"        bash -c "python3 -c 'import torch; print(torch.version.cuda)'"
        run "torch CUDA arch list"      bash -c "python3 -c 'import torch; print(torch.cuda.get_arch_list())'"
        check_pass "PyTorch ${TORCH_VER}"
    else
        check_fail "PyTorch (not installed)"
    fi

    # Key Python packages
    for pkg in setuptools wheel packaging ninja einops nvtx; do
        run "pip: ${pkg}" bash -c "
            python3 -c 'import ${pkg}; print(\"${pkg}\", ${pkg}.__version__)' 2>/dev/null \
            || echo \"(${pkg} not installed or no __version__)\"
        "
        if python3 -c "import ${pkg}" >>"${OUT}" 2>&1; then
            check_pass "pip: ${pkg}"
        else
            case "${pkg}" in
                einops|nvtx) check_warn "pip: ${pkg} (needed at runtime only)";;
                *)           check_fail "pip: ${pkg}";;
            esac
        fi
    done
else
    check_fail "python3 (not in PATH)"
fi

# ===================================================================
# [5/9] Host compiler
# ===================================================================
section "[5/9] Host compiler"
run "gcc --version"    bash -c 'command -v gcc  >/dev/null && gcc  --version | head -1 || echo "(no gcc)"'
run "g++ --version"    bash -c 'command -v g++  >/dev/null && g++  --version | head -1 || echo "(no g++)"'
run "make --version"   bash -c 'command -v make >/dev/null && make --version | head -1 || echo "(no make)"'

if has g++; then
    check_pass "g++"
else
    check_fail "g++ (required by PyTorch CUDAExtension)"
fi

# ===================================================================
# [6/9] Nsight Systems (nsys)
# ===================================================================
section "[6/9] Nsight Systems (nsys)"
NSYS_BIN="$(command -v nsys 2>/dev/null || true)"
if [ -z "${NSYS_BIN}" ]; then
    for p in /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys; do
        [ -x "$p" ] && { NSYS_BIN="$p"; break; }
    done
fi

if [ -n "${NSYS_BIN}" ]; then
    run "nsys --version" "${NSYS_BIN}" --version
    check_pass "nsys"
else
    echo "nsys NOT FOUND" | tee -a "${OUT}"
    check_fail "nsys (Nsight Systems)"
fi

# ===================================================================
# [7/9] Ninja build (accelerates CUDA compilation)
# ===================================================================
section "[7/9] Ninja build system"
run "which ninja"    bash -c 'command -v ninja || echo "(ninja not in PATH)"'
run "ninja --version" bash -c 'command -v ninja >/dev/null && ninja --version || echo "(skipped)"'
if has ninja; then
    check_pass "ninja (CLI)"
else
    check_warn "ninja (CLI not in PATH, but pip:ninja may still work for setup.py)"
fi

# ===================================================================
# [8/9] Privileges & clock-lock test
# ===================================================================
section "[8/9] Privileges & clock-lock capability"
run "am I root?" bash -c 'id; [ "$(id -u)" = 0 ] && echo "STATE: running as root" || echo "NOT root"'

run "sudo capability" bash -c '
    if ! command -v sudo >/dev/null; then
        echo "STATE: sudo NOT installed"
    elif sudo -n true 2>/dev/null; then
        echo "STATE: passwordless sudo"
    else
        echo "STATE: sudo available, requires password"
    fi
'

if [ "${PROBE_WITH_SUDO:-0}" = "1" ]; then
    section "[8b/9] sudo + clock-lock live test (PROBE_WITH_SUDO=1)"
    run "sudo -v (may prompt once)" sudo -v
    run "test lock clock to 1830 MHz on GPU ${SMI_I}" bash -c "
        sudo nvidia-smi -i ${SMI_I} --lock-gpu-clocks=1830,1830 2>&1 && echo 'LOCK OK' || echo 'LOCK FAILED'
        sudo nvidia-smi -i ${SMI_I} -rgc 2>/dev/null
        echo '(clock restored)'
    "
else
    echo ""                                                           | tee -a "${OUT}"
    echo "[hint] To test clock-locking end-to-end (one sudo prompt):"| tee -a "${OUT}"
    echo "  GPU_ID=${GPU_ID:-0} PROBE_WITH_SUDO=1 bash ${BASH_SOURCE[0]}" | tee -a "${OUT}"
fi

# ===================================================================
# [9/9] Disk & workspace
# ===================================================================
section "[9/9] Disk & workspace"
run "pwd"       bash -c 'echo "cwd = $(pwd)"'
run "df -h ."   df -h .
run "write test" bash -c '
    t="$(mktemp -p . 2>/dev/null)" && rm -f "$t" && echo "write ok" || echo "NOT writable"
'

# Estimate build space
if has python3 && [ "${TORCH_OK}" = "1" ]; then
    run "estimated build space check" bash -c '
        available=$(df --output=avail -BG . | tail -1 | tr -d " G")
        echo "Available: ${available} GB"
        if [ "${available}" -lt 10 ]; then
            echo "WARN: < 10 GB free, FA3 build may need ~5-10 GB"
        else
            echo "OK: >= 10 GB free for build"
        fi
    '
fi

# ===================================================================
# Summary
# ===================================================================
section "Dependency summary"
{
    echo ""
    echo "--- PASSED (${#PASS[@]}) ---"
    for p in "${PASS[@]:-}"; do [ -n "$p" ] && echo "  [OK]   $p"; done

    echo ""
    echo "--- WARNINGS (${#WARN[@]}) ---"
    for w in "${WARN[@]:-}"; do [ -n "$w" ] && echo "  [WARN] $w"; done

    echo ""
    echo "--- MISSING / FAILED (${#FAIL[@]}) ---"
    for f in "${FAIL[@]:-}"; do [ -n "$f" ] && echo "  [FAIL] $f"; done

    echo ""
    echo "=============================================="
    if [ "${#FAIL[@]}" -eq 0 ]; then
        echo " VERDICT: ALL critical dependencies met."
        echo " You can proceed with FA3 compilation and evaluation."
    else
        echo " VERDICT: ${#FAIL[@]} critical dependency(ies) MISSING."
        echo " Install them before running the FA3 pipeline."
    fi
    echo "=============================================="
    echo ""
    echo "Next steps (after all dependencies are met):"
    echo "  1) cd hopper/"
    echo "  2) FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=20 pip install . --no-build-isolation -v > log_dtl 2> log_err"
    echo "  3) sudo nvidia-smi -i ${SMI_I} -lgc 1830,1830"
    echo "  4) source run_all_nsys.sh"
    echo "  5) python3 extract_device_kernel.py"
    echo ""
    echo "Report written to: ${OUT}"
} | tee -a "${OUT}"

echo ""
echo "[done] wrote ${OUT}"
echo "scp this file back or paste it, and I will tell you what to install."
