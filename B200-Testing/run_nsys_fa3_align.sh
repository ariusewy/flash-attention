#!/usr/bin/env bash
# Author: ywangmu from HKUST
#
# FA4 nsys profiling for the 9 FA3-aligned shapes.
# These shapes match exactly what fa3 profiling on H800 covered:
#   H_Q in {32, 64, 128} x S in {512, 1024, 2048},  H_KV=8, headdim=128, batch=1
#
# Usage:
#   CUDA_VISIBLE_DEVICES=5 bash run_nsys_fa3_align.sh
#   LOCK_MHZ=1830 CUDA_VISIBLE_DEVICES=5 bash run_nsys_fa3_align.sh

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
LOCK_MHZ="${LOCK_MHZ:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$HERE/bench_fa4_simfa.py"
NSYS_DIR="nsys_reports_fa3align"
NSYS_TAR="fa4_fa3align_nsys_reports.tar.gz"

# FA3-aligned shapes: heads_q heads_kv seqlen headdim
SHAPES=(
  "32  8   512 128"
  "64  8   512 128"
  "128 8   512 128"
  "32  8  1024 128"
  "64  8  1024 128"
  "128 8  1024 128"
  "32  8  2048 128"
  "64  8  2048 128"
  "128 8  2048 128"
)

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

if [ ! -f "$BENCH" ]; then
  echo "Error: bench_fa4_simfa.py not found at $BENCH"
  exit 1
fi
command -v nsys >/dev/null 2>&1 || { echo "Error: nsys not in PATH"; exit 1; }

rm -f "$NSYS_TAR"
rm -rf "$NSYS_DIR"
mkdir -p "$NSYS_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "=================================================================="
echo "FA4 nsys Profiling (FA3-aligned, 9 shapes) at $(date)"
echo "GPU: $GPU_ID  Lock: ${LOCK_MHZ:-none}"
echo "=================================================================="

for shape in "${SHAPES[@]}"; do
  read hq hkv s d <<< "$shape"
  BASE="${NSYS_DIR}/fa4_profile_h${hq}_s${s}"

  echo ""
  echo ">>> Profiling: H_Q=${hq} H_KV=${hkv} S=${s} D=${d} <<<"

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

  echo "    Extracting kernel stats..."
  nsys stats --report cuda_gpu_kern_sum \
    --format csv \
    --force-overwrite true \
    --output "${BASE}_kernel_stats" \
    "${BASE}.nsys-rep" 2>/dev/null || true

  echo "    Done: ${BASE}.nsys-rep"
done

echo ""
echo "Aggregating kernel latency -> device_kernel_summary_fa3align.csv"

python3 -c "
import csv, glob, re, os

input_dir = '${NSYS_DIR}'
output_file = 'device_kernel_summary_fa3align.csv'
header = ['Heads_Q','Heads_KV','SeqLen','Total_Time_us','Instances','Avg_us','Med_us','Min_us','Max_us','StdDev_us','Kernel_Name']
data = [header]

for csv_file in sorted(glob.glob(f'{input_dir}/*_kernel_stats*.csv*')):
    basename = os.path.basename(csv_file)
    match = re.search(r'fa4_profile_h(\d+)_s(\d+)', basename)
    if not match: continue
    hq, s, hkv = int(match.group(1)), int(match.group(2)), 8

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            if len(row) < 9: continue
            name = row[-1].lower()
            if 'flash' in name or 'cutlass' in name or 'attention' in name:
                try:
                    total_ns = float(row[1])
                    inst = int(row[2])
                    avg_ns, med_ns = float(row[3]), float(row[4])
                    min_ns, max_ns, std_ns = float(row[5]), float(row[6]), float(row[7])
                    data.append([
                        hq, hkv, s,
                        f'{total_ns/1000:.3f}', inst,
                        f'{avg_ns/1000:.3f}', f'{med_ns/1000:.3f}',
                        f'{min_ns/1000:.3f}', f'{max_ns/1000:.3f}',
                        f'{std_ns/1000:.3f}', row[-1][:80],
                    ])
                except (ValueError, IndexError):
                    pass

with open(output_file, 'w', newline='') as f:
    csv.writer(f).writerows(data)
print(f'Written {len(data)-1} rows to {output_file}')
"

tar -czf "$NSYS_TAR" "$NSYS_DIR"

echo ""
echo "=================================================================="
echo "Done!"
echo "Kernel summary: device_kernel_summary_fa3align.csv"
echo "Raw reports:    ${NSYS_DIR}/"
echo "Archive:        ${NSYS_TAR}"
echo "=================================================================="
