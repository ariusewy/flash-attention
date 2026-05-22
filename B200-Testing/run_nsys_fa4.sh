#!/usr/bin/env bash
# Author: ywangmu from HKUST
#
# FA4 nsys profiling — same approach as FA3 on H800.
# Runs bench_fa4_simfa.py --mode run_once under nsys for each shape,
# then extracts kernel latency via nsys stats.
#
# Usage:
#   bash run_nsys_fa4.sh
#   CUDA_VISIBLE_DEVICES=5 bash run_nsys_fa4.sh
#   LOCK_MHZ=1830 bash run_nsys_fa4.sh
#
# Output:
#   nsys_reports/
#     fa4_profile_h{heads}_s{seqlen}.nsys-rep
#     fa4_profile_h{heads}_s{seqlen}_kernel_stats.csv
#   device_kernel_summary.csv   (aggregated latency table)

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
LOCK_MHZ="${LOCK_MHZ:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$HERE/bench_fa4_simfa.py"
NSYS_DIR="nsys_reports"
NSYS_TAR="fa4_all_nsys_reports.tar.gz"

# Shape definitions: heads_q heads_kv seqlen headdim
SHAPES=(
  "8   8  512  128"
  "32  8  1024 128"
  "32  8  2048 128"
  "64  8  4096 128"
  "128 8  8192 128"
)

# Lock clock if requested
if [[ -n "$LOCK_MHZ" ]]; then
  echo "Locking SM clock to ${LOCK_MHZ} MHz on GPU ${GPU_ID}..."
  nvidia-smi -i "$GPU_ID" -pm 1 >/dev/null 2>&1 || true
  nvidia-smi -i "$GPU_ID" --lock-gpu-clocks="$LOCK_MHZ,$LOCK_MHZ" >/dev/null 2>&1 || {
    echo "[warn] clock lock failed"
    LOCK_MHZ=""
  }
fi

cleanup() {
  if [[ -n "$LOCK_MHZ" ]]; then
    echo "Restoring clocks on GPU ${GPU_ID}..."
    nvidia-smi -i "$GPU_ID" -rgc >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

# Check prerequisites
if [ ! -f "$BENCH" ]; then
  echo "Error: bench_fa4_simfa.py not found at $BENCH"
  exit 1
fi
command -v nsys >/dev/null 2>&1 || {
  echo "Error: nsys not found in PATH"
  exit 1
}

# Clean previous runs
rm -f "$NSYS_TAR"
rm -rf "$NSYS_DIR"
mkdir -p "$NSYS_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "=================================================================="
echo "FA4 nsys Profiling ($(date))"
echo "Shapes: ${#SHAPES[@]}"
echo "GPU: $GPU_ID  Lock: ${LOCK_MHZ:-none}"
echo "=================================================================="

for shape in "${SHAPES[@]}"; do
  read hq hkv s d <<< "$shape"
  BASE="${NSYS_DIR}/fa4_profile_h${hq}_s${s}"

  echo ""
  echo ">>> Profiling: H_Q=${hq} H_KV=${hkv} S=${s} D=${d} <<<"

  # Warmup + single invocation under nsys
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --output="${BASE}" \
    --force-overwrite true \
    python3 "$BENCH" \
      --mode run_once \
      --heads "$hq" \
      --heads-kv "$hkv" \
      --seqlen "$s" \
      --headdim "$d" \
      --batch 1 \
      --no-backward

  # Extract kernel stats
  echo "    Extracting kernel stats..."
  nsys stats --report cuda_gpu_kern_sum \
    --format csv \
    --force-overwrite true \
    --output "${BASE}_kernel_stats" \
    "${BASE}.nsys-rep" 2>/dev/null || true

  echo "    Done: ${BASE}.nsys-rep"
done

# ---------------------------------------------------------------------------
# Aggregate kernel latency (same format as FA3 extract_device_kernel.py)
# ---------------------------------------------------------------------------
echo ""
echo "Aggregating kernel latency -> device_kernel_summary.csv"

python3 -c "
import csv, glob, re, os

input_dir = '${NSYS_DIR}'
output_file = 'device_kernel_summary.csv'
header = ['Heads_Q', 'Heads_KV', 'SeqLen', 'Total_Time_us', 'Instances', 'Avg_us', 'Med_us', 'Min_us', 'Max_us', 'StdDev_us']
data = [header]

for csv_file in sorted(glob.glob(f'{input_dir}/*_kernel_stats.csv*')):
    basename = os.path.basename(csv_file)
    match = re.search(r'fa4_profile_h(\d+)_s(\d+)', basename)
    if not match:
        continue
    hq = int(match.group(1))
    s = int(match.group(2))
    # derive hkv from our shape definitions
    hkv = 8  # all shapes use hkv=8

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) > 8 and 'flash' in row[-1].lower() or 'cutlass' in row[-1].lower() or 'attention' in row[-1].lower():
                try:
                    total_ns = float(row[1])
                    instances = int(row[2])
                    avg_ns = float(row[3])
                    med_ns = float(row[4])
                    min_ns = float(row[5])
                    max_ns = float(row[6])
                    std_ns = float(row[7])
                    data.append([
                        hq, hkv, s,
                        f'{total_ns/1000:.3f}',
                        instances,
                        f'{avg_ns/1000:.3f}',
                        f'{med_ns/1000:.3f}',
                        f'{min_ns/1000:.3f}',
                        f'{max_ns/1000:.3f}',
                        f'{std_ns/1000:.3f}',
                    ])
                except (ValueError, IndexError):
                    pass

with open(output_file, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerows(data)
print(f'Written {len(data)-1} rows to {output_file}')
"

# Package
tar -czf "$NSYS_TAR" "$NSYS_DIR"

echo ""
echo "=================================================================="
echo "Done!"
echo "Kernel summary: device_kernel_summary.csv"
echo "Raw reports:    ${NSYS_DIR}/"
echo "Archive:        ${NSYS_TAR}"
echo "=================================================================="
