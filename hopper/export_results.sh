#!/usr/bin/env bash
# Author: ywangmu from HKUST
#
# Export FA3 profiling results into a single plain-text file for
# copy-paste from the server terminal (no file download needed).
#
# Prerequisites: run_all_nsys.sh and extract_device_kernel.py must have
# already been run in this directory.
#
# Usage:
#   bash export_results.sh                  # -> COPYME_FA3_H800.txt
#   bash export_results.sh /tmp/out.txt     # custom output path
#   LOCK_MHZ=1830 bash export_results.sh    # specify locked clock

set -euo pipefail

LOCK_MHZ="${LOCK_MHZ:-1830}"
OUT="${1:-COPYME_FA3_H800.txt}"
NSYS_DIR="nsys_reports"
KERNEL_CSV="device_kernel_summary.csv"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
{
    echo "FA3_H800 COPY/PASTE EXPORT"
    echo "generated_at    = $(date -Is 2>/dev/null || date)"
    echo "generated_from  = $(pwd)"
    echo "platform        = $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || echo unknown)"
    echo "lock_mhz        = ${LOCK_MHZ}"
    echo ""

    # GPU info
    echo "gpu_info:"
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader 2>/dev/null | while IFS= read -r line; do
        echo "  ${line}"
    done
    echo ""

    # Current clock (verify lock is active)
    CUR_CLK=$(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits -i 0 2>/dev/null || echo "N/A")
    echo "current_sm_clock_MHz = ${CUR_CLK}"
    echo ""

    # nsys version
    echo "nsys_version = $(nsys --version 2>/dev/null | head -1 || echo unknown)"
    echo ""

    # PyTorch version
    echo "pytorch_version = $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
    echo ""

    echo "This file is intentionally plain text so it can be opened on the server"
    echo "and copied directly without downloading files."
    echo ""
} > "${OUT}"

# ---------------------------------------------------------------------------
# [1] device_kernel_summary.csv — the main latency result
# ---------------------------------------------------------------------------
{
    echo "========================"
    echo "BEGIN device_kernel_summary.csv"
    echo "========================"
    if [ -f "${KERNEL_CSV}" ]; then
        cat "${KERNEL_CSV}"
        ROWS=$(($(wc -l < "${KERNEL_CSV}") - 1))
        echo ""
        echo "rows = ${ROWS}  (9 expected: 3 heads x 3 seq_lens)"
    else
        echo "(file not found: ${KERNEL_CSV})"
        echo "Run: python3 extract_device_kernel.py"
    fi
    echo ""
} >> "${OUT}"

# ---------------------------------------------------------------------------
# [2] Per-configuration nsys kernel stats CSVs (raw, for reference)
# ---------------------------------------------------------------------------
{
    echo "========================================"
    echo "BEGIN per-config kernel_stats (raw nsys)"
    echo "========================================"
    if [ -d "${NSYS_DIR}" ]; then
        FOUND=0
        for csv_file in $(ls "${NSYS_DIR}/"*_kernel_stats.csv* 2>/dev/null | sort); do
            FOUND=1
            echo ""
            echo "--- $(basename "${csv_file}") ---"
            cat "${csv_file}"
        done
        if [ "${FOUND}" = "0" ]; then
            echo "(no *_kernel_stats.csv files found in ${NSYS_DIR}/)"
        fi
    else
        echo "(directory not found: ${NSYS_DIR}/)"
    fi
    echo ""
} >> "${OUT}"

# ---------------------------------------------------------------------------
# [3] nsys stdout/stderr logs (fmha.py print output per config)
# ---------------------------------------------------------------------------
{
    echo "============================"
    echo "BEGIN per-config fmha output"
    echo "============================"
    if [ -d "${NSYS_DIR}" ]; then
        FOUND=0
        # nsys stores stdout/stderr in files next to the .nsys-rep
        for log_file in $(ls "${NSYS_DIR}"/fmha_profile_h*_s*.nsys-rep 2>/dev/null | sort); do
            # The .nsys-rep is binary, but nsys also writes a .sqlite next to it
            # We want the textual output — check for companion stdout files
            base="${log_file%.nsys-rep}"
            # nsys may create {base}.stdout or {base}_stdout.txt depending on version
            for companion in "${base}.stdout" "${base}_stdout.txt"; do
                if [ -f "${companion}" ]; then
                    FOUND=1
                    echo ""
                    echo "--- $(basename "${companion}") ---"
                    cat "${companion}"
                fi
            done
        done
        if [ "${FOUND}" = "0" ]; then
            echo "(no stdout companion files found in ${NSYS_DIR}/)"
            echo "(fmha.py output was captured by nsys and is embedded in the .nsys-rep)"
            echo "(use 'nsys stats' to extract, or check the terminal output from run_all_nsys.sh)"
        fi
    else
        echo "(directory not found: ${NSYS_DIR}/)"
    fi
    echo ""
} >> "${OUT}"

# ---------------------------------------------------------------------------
# [4] File listing of nsys_reports/ (audit trail)
# ---------------------------------------------------------------------------
{
    echo "=========================="
    echo "BEGIN nsys_reports listing"
    echo "=========================="
    if [ -d "${NSYS_DIR}" ]; then
        ls -lh "${NSYS_DIR}/" 2>/dev/null || echo "(empty or unreadable)"
    else
        echo "(directory not found: ${NSYS_DIR}/)"
    fi
    echo ""
} >> "${OUT}"

# ---------------------------------------------------------------------------
# [5] Disk usage summary
# ---------------------------------------------------------------------------
{
    echo "==============="
    echo "BEGIN disk info"
    echo "==============="
    echo "workspace: $(pwd)"
    df -h . | head -5
    echo ""
    echo "nsys_reports size: $(du -sh ${NSYS_DIR} 2>/dev/null || echo N/A)"
    echo "kernel summary   : $(ls -lh ${KERNEL_CSV} 2>/dev/null || echo N/A)"
} >> "${OUT}"

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
{
    echo ""
    echo "================"
    echo "END OF EXPORT"
    echo "================"
    echo ""
    echo "To use this data on your local machine:"
    echo "  1) cat ${OUT}  (on the server)"
    echo "  2) Copy-paste the entire output into a local file"
    echo "  3) Each section is delimited by BEGIN/END markers"
} >> "${OUT}"

echo ""
echo "=============================================="
echo " Export complete."
echo " Output: ${OUT}"
echo " Size:   $(du -sh "${OUT}" | cut -f1)"
echo ""
echo " To view: cat ${OUT}"
echo " To copy: select all text from the terminal"
echo "=============================================="
