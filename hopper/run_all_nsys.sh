#!/bin/bash
# =============================================================================
# Flash Attention 3 - INDEPENDENT nsys Profiling Script
# ONLY performs nsys profiling (no trace generation)
# 9 combinations (3 heads × 3 seq lens) with group_size=8
# Generates 9 separate .nsys-rep reports
# =============================================================================

set -e  # Exit immediately if any command fails

# ----------------------------- Configuration -----------------------------
HEADS=(128 64 32)
SEQ_LENS=(512 1024 2048)
K_HEADS=8

NSYS_REPORT_DIR="nsys_reports"
NSYS_TAR_FILE="fa3_all_nsys_reports.tar.gz"

# ----------------------------- Sanity Checks -----------------------------
if [ ! -f "fmha.py" ]; then
    echo "Error: fmha.py not found in current directory."
    exit 1
fi

command -v nsys >/dev/null 2>&1 || {
    echo "Error: nsys command not found. Please install Nsight Systems or load the module."
    exit 1
}

# ----------------------------- Cleanup previous runs -----------------------------
echo "Cleaning up old nsys reports..."
rm -f "${NSYS_TAR_FILE}"
rm -rf "${NSYS_REPORT_DIR}"
mkdir -p "${NSYS_REPORT_DIR}"

echo "=================================================================="
echo "Starting INDEPENDENT nsys profiling of Flash Attention 3"
echo "Total combinations: ${#HEADS[@]} heads × ${#SEQ_LENS[@]} seq lens = 9"
echo "nsys reports will be saved to: ${NSYS_REPORT_DIR}/"
echo "NOTE: This script does NOT generate any traces or call gen_all_sim_trace.py"
echo "=================================================================="

# ----------------------------- Main Double Loop -----------------------------
for h in "${HEADS[@]}"; do
    for s in "${SEQ_LENS[@]}"; do
        GROUP_SIZE=$((h / K_HEADS))
        NSYS_OUTPUT_BASE="${NSYS_REPORT_DIR}/fmha_profile_h${h}_s${s}"

        echo ""
        echo ">>> Profiling: Heads=${h} | SeqLen=${s} | GroupSize=${GROUP_SIZE} <<<"
        echo "    Output report: ${NSYS_OUTPUT_BASE}.nsys-rep"

        # Run fmha.py under nsys (pure profiling, no trace generation)
        nsys profile \
            --trace=cuda,nvtx,osrt \
            --output="${NSYS_OUTPUT_BASE}" \
            --force-overwrite true \
            python3 fmha.py \
                --num-heads "${h}" \
                --seq-len "${s}" \
                --group-size "${GROUP_SIZE}" \
                --no-warmup
        # 在 nsys profile 命令之后，紧接着添加：
        echo ">>> Extracting kernel stats for Heads=${h} | SeqLen=${s} <<<"
        nsys stats --report cuda_gpu_kern_sum \
                --format csv \
                --force-overwrite true \
                --output "${NSYS_OUTPUT_BASE}_kernel_stats.csv" \
                "${NSYS_OUTPUT_BASE}.nsys-rep"

        echo "Kernel stats saved to: ${NSYS_OUTPUT_BASE}_kernel_stats.csv"

        echo "    Completed: ${NSYS_OUTPUT_BASE}.nsys-rep"
    done
done

# ----------------------------- Final Packaging -----------------------------
echo ""
echo "=================================================================="
echo "All 9 nsys reports generated successfully."
echo "Creating compressed archive..."
echo "=================================================================="

tar -czvf "${NSYS_TAR_FILE}" "${NSYS_REPORT_DIR}"

echo ""
echo "=================================================================="
echo "Task completed successfully!"
echo "nsys report package: ${NSYS_TAR_FILE}"
echo "Individual reports are in: ${NSYS_REPORT_DIR}/"
echo ""
echo "To view a report:"
echo "   nsys-ui ${NSYS_REPORT_DIR}/fmha_profile_h128_s2048.nsys-rep"
echo "=================================================================="