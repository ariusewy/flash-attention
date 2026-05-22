#!/usr/bin/env bash
# Author: ywangmu from HKUST
#
# NCU profiling automation for FA4 on B200 (Blackwell, sm_100).
#
# Profiles a single FA4 fwd kernel invocation using bench_fa4_simfa.py
# with Nsight Compute, collects key sections, and exports results.
#
# Prerequisites:
#   - conda activate b200_fa3
#   - flash-attn-4 (beta 8+) installed
#   - ncu available on PATH
#   - sudo / root for GPU counter access (or RestrictProfiling disabled)
#
# Usage:
#   # Minimal smoke test (seqlen=128, headdim=64)
#   bash ncu_profile_fa4.sh
#
#   # Custom shape (GQA)
#   OUTDIR=/path/to/out sudo -E bash ncu_profile_fa4.sh \
#       --seqlen 2048 --heads 32 --heads-kv 8 --headdim 128 --no-backward --full
#
#   # Full metric set (slower, more passes)
#   bash ncu_profile_fa4.sh --full
#
#   # Specific GPU
#   GPU_ID=2 bash ncu_profile_fa4.sh
#
#   # Lock clock for reproducibility
#   LOCK_MHZ=1600 bash ncu_profile_fa4.sh
#
# Output:
#   results/ncu-fa4-<seqlen>-<headdim>-<timestamp>/
#     env.txt              environment fingerprint
#     profile.ncu-rep      raw NCU report (SQLite)
#     profile.csv          exported CSV (details page)
#     summary.json         parsed key metrics (via parse_ncu_report.py)
#     app_stdout.txt       benchmark stdout
#     ncu_run.log          NCU stderr

set -euo pipefail

# Preserve PATH under sudo
export PATH="/usr/local/cuda/bin:${PATH:-}"

GPU_ID="${GPU_ID:-0}"
LOCK_MHZ="${LOCK_MHZ:-}"
PROFILE_TIMEOUT="${PROFILE_TIMEOUT:-600}"
CONDA_ENV="${CONDA_ENV:-b200_fa3}"
FULL_METRICS="${FULL_METRICS:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Parse our own flags (everything before -- is ours, after -- goes to bench)
SEQLEN=128
HEADDIM=64
HEADS=2
HEADS_KV=""
BATCH=1
NO_BWD=""
EXTRA_BENCH_ARGS=()
FULL_MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seqlen)   SEQLEN="$2";   shift 2 ;;
    --headdim)  HEADDIM="$2";  shift 2 ;;
    --heads)    HEADS="$2";    shift 2 ;;
    --heads-kv) HEADS_KV="$2"; shift 2 ;;
    --batch)    BATCH="$2";    shift 2 ;;
    --no-backward) NO_BWD="--no-backward"; shift ;;
    --full)     FULL_MODE="1"; shift ;;
    *)          EXTRA_BENCH_ARGS+=("$1"); shift ;;
  esac
done

[[ -z "$HEADS_KV" ]] && HEADS_KV="$HEADS"

TS="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${OUTDIR:-}" ]]; then
  OUTDIR="$HERE/results/ncu-fa4-${SEQLEN}-${HEADDIM}-${TS}"
fi
mkdir -p "$OUTDIR"

echo "============================================="
echo "NCU FA4 Profile"
echo "============================================="
echo "  GPU_ID    : $GPU_ID"
echo "  shape     : batch=$BATCH seqlen=$SEQLEN heads_q=$HEADS heads_kv=$HEADS_KV headdim=$HEADDIM"
echo "  backward  : $([ -n "$NO_BWD" ] && echo 'disabled' || echo 'enabled')"
echo "  full      : $([ -n "$FULL_MODE" ] && echo 'yes' || echo 'no')"
echo "  timeout   : ${PROFILE_TIMEOUT}s"
echo "  output    : $OUTDIR"
echo "============================================="

# ---------------------------------------------------------------------------
# 1. Environment fingerprint
# ---------------------------------------------------------------------------
{
  echo "=== nvidia-smi ==="
  nvidia-smi --query-gpu=name,driver_version,persistence_mode,compute_mode,clocks.sm,memory.total \
             --format=csv -i "$GPU_ID" 2>/dev/null || nvidia-smi -i "$GPU_ID"
  echo
  echo "=== nvcc ==="
  nvcc --version 2>&1 | head -5 || echo "(nvcc not found)"
  echo
  echo "=== ncu ==="
  ncu --version 2>&1 | head -3 || echo "(ncu not found)"
  echo
  echo "=== conda env ==="
  conda list -n "$CONDA_ENV" 2>/dev/null | grep -iE "torch|flash-attn|cutlass|quack" || echo "(conda env '$CONDA_ENV' not found)"
  echo
  echo "=== /proc/driver/nvidia/params ==="
  grep -i "RestrictProfiling" /proc/driver/nvidia/params 2>/dev/null \
    || echo "(no RestrictProfiling line)"
  echo
  echo "=== GPU topology ==="
  nvidia-smi topo -m 2>/dev/null | head -20 || true
} > "$OUTDIR/env.txt"

echo "[ncu] Environment logged to env.txt"

# ---------------------------------------------------------------------------
# 2. Lock SM clock (optional, for reproducibility)
# ---------------------------------------------------------------------------
if [[ -n "$LOCK_MHZ" ]]; then
  echo "[ncu] Locking SM clock to ${LOCK_MHZ} MHz on GPU ${GPU_ID}..."
  nvidia-smi -i "$GPU_ID" -pm 1 >/dev/null 2>&1 || true
  nvidia-smi -i "$GPU_ID" --lock-gpu-clocks="$LOCK_MHZ" >/dev/null 2>&1 || {
    echo "[warn] Failed to lock clock (may need sudo). Continuing without lock."
    LOCK_MHZ=""
  }
fi

cleanup() {
  if [[ -n "$LOCK_MHZ" ]]; then
    echo "[ncu] Restoring default clocks on GPU ${GPU_ID}..."
    nvidia-smi -i "$GPU_ID" -rgc >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# 3. Build the NCU command
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Python command
PYTHON_CMD="conda run -n $CONDA_ENV --no-banner python $HERE/bench_fa4_simfa.py"
PYTHON_CMD="$PYTHON_CMD --mode run_once --seqlen $SEQLEN --headdim $HEADDIM --heads $HEADS --heads-kv $HEADS_KV --batch $BATCH $NO_BWD"

# NCU sections
if [[ -n "$FULL_MODE" ]]; then
  SECTIONS=(
    --section MemoryWorkloadAnalysis
    --section ComputeWorkloadAnalysis
    --section SpeedOfLight
    --section LaunchStats
    --section SchedulerStats
    --section InstructionStats
    --section Occupancy
    --section MemoryFootprint
    --section SourceCounters
  )
  METRICS_FLAG="--metrics-all"
else
  SECTIONS=(
    --section MemoryWorkloadAnalysis
    --section ComputeWorkloadAnalysis
    --section LaunchStats
    --section Occupancy
  )
  # Targeted metrics for Sim-FA calibration
  METRICS_FLAG=(
    --metrics
    "dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
lts__t_sectors.sum,\
lts__t_sector_hit_rate.pct,\
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,\
l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld.sum,\
l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum,\
l1tex__m_l1tex2xbar_write_sectors_mem_global_op_tma_st.sum,\
l1tex__m_xbar2l1tex_read_bytes_pipe_tma.sum,\
l1tex__m_l1tex2xbar_write_bytes_pipe_tma.sum,\
sm__sass_inst_executed_op_tma_ld.sum,\
sm__sass_inst_executed_op_tma_st.sum,\
l1tex__tmain_requests.sum,\
sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tma_cycles_active.sum.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu.sum,\
sm__inst_executed.sum,\
smsp__inst_executed_pipe_tensor.sum,\
smsp__average_warp_latency_issue_stalled_long_scoreboard.pct,\
smsp__average_warp_latency_issue_stalled_barrier.pct,\
smsp__average_warp_latency_issue_stalled_mio_throttle.pct,\
smsp__average_warp_latency_issue_stalled_not_selected.pct,\
gpu__time_duration.sum,\
sm__warps_active.avg.pct_of_peak"
  )
fi

# ---------------------------------------------------------------------------
# 4. Run NCU
# ---------------------------------------------------------------------------
echo "[ncu] Profiling FA4 kernel..."
echo "[ncu] Command: ncu ${SECTIONS[*]} ${METRICS_FLAG[*]} -o $OUTDIR/profile -- $PYTHON_CMD"

set +e
timeout "$PROFILE_TIMEOUT" ncu \
  --target-processes all \
  --clock-control=none \
  "${SECTIONS[@]}" \
  "${METRICS_FLAG[@]}" \
  -o "$OUTDIR/profile" \
  bash -c "$PYTHON_CMD" \
  > "$OUTDIR/app_stdout.txt" \
  2> "$OUTDIR/ncu_run.log"
NCU_RC=$?
set -e

echo "[ncu] NCU exit code: $NCU_RC (124 = timeout)"

# ---------------------------------------------------------------------------
# 5. Export CSV from raw report
# ---------------------------------------------------------------------------
if [[ -f "$OUTDIR/profile.ncu-rep" ]]; then
  echo "[ncu] Exporting CSV from profile.ncu-rep..."
  ncu --import "$OUTDIR/profile.ncu-rep" --csv --page details \
      > "$OUTDIR/profile.csv" 2> "$OUTDIR/ncu_import.log" || true
  echo "[ncu] CSV rows: $(wc -l < "$OUTDIR/profile.csv")"
else
  echo "[ncu] WARNING: no profile.ncu-rep produced. Check ncu_run.log for errors."
fi

# ---------------------------------------------------------------------------
# 6. Parse summary (if parse script exists)
# ---------------------------------------------------------------------------
if [[ -f "$HERE/parse_ncu_report.py" ]] && [[ -f "$OUTDIR/profile.ncu-rep" ]]; then
  echo "[ncu] Parsing report to summary.json..."
  conda run -n "$CONDA_ENV" --no-banner python "$HERE/parse_ncu_report.py" \
    "$OUTDIR/profile.ncu-rep" -o "$OUTDIR/summary.json" 2>/dev/null || {
    echo "[warn] parse_ncu_report.py failed; skipping summary.json"
  }
fi

# ---------------------------------------------------------------------------
# 7. Print summary
# ---------------------------------------------------------------------------
{
  echo "============================================="
  echo "FA4 NCU Profile Summary"
  echo "============================================="
  echo "Shape: batch=$BATCH seqlen=$SEQLEN heads=$HEADS headdim=$HEADDIM"
  echo "NCU exit code: $NCU_RC"
  echo
  echo "=== App stdout ==="
  head -20 "$OUTDIR/app_stdout.txt" 2>/dev/null || echo "(empty)"
  echo
  echo "=== NCU errors ==="
  grep -cE "ERR_NVGPUCTRPERM|ERROR|Failed" "$OUTDIR/ncu_run.log" 2>/dev/null || echo "0"
  echo
  if [[ -f "$OUTDIR/summary.json" ]]; then
    echo "=== Key Metrics ==="
    python3 -c "
import json, sys
with open('$OUTDIR/summary.json') as f:
    d = json.load(f)
kernels = d.get('kernels', [d])
for k in kernels[:3]:
    name = k.get('kernel_name', k.get('name', 'unknown'))
    print(f'Kernel: {name[:80]}')
    print(f'  Grid: {k.get(\"grid\", \"?\")}  Block: {k.get(\"block\", \"?\")}')
    print(f'  Duration: {k.get(\"duration_us\", \"?\")} us')
    print(f'  DRAM read: {k.get(\"dram_bytes_read\", \"?\")} bytes')
    print(f'  DRAM write: {k.get(\"dram_bytes_write\", \"?\")} bytes')
    print(f'  TMA ld: {k.get(\"tma_ld_bytes\", \"?\")} bytes  '
          f'st: {k.get(\"tma_st_bytes\", \"?\")} bytes')
    for key in ['stall_long_scoreboard_pct', 'stall_barrier_pct', 'stall_not_selected_pct']:
        if key in k:
            print(f'  {key}: {k[key]}')
" 2>/dev/null || echo "(summary parse failed)"
  fi
  echo
  echo "Output directory: $OUTDIR"
} | tee "$OUTDIR/summary.txt"

echo ""
echo "[ncu] Done. All output -> $OUTDIR/"
