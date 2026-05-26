#!/usr/bin/env bash
# Author: ywangmu from HKUST
#
# NCU profiling automation for FA4 on B200 (Blackwell, sm_100).
#
# Profiles a single FA4 fwd kernel invocation using bench_fa4_simfa.py
# with Nsight Compute, collects key sections, and exports results.
#
# Prerequisites:
#   - python3 + pip in PATH
#   - flash-attn-4 installed (pip install flash-attn-4)
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
#   # Clock locking is NOT handled here — lock manually if you want it:
#   #   sudo nvidia-smi -i $GPU_ID -lgc 1830,1830
#   #   sudo nvidia-smi -i $GPU_ID -rgc
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
PROFILE_TIMEOUT="${PROFILE_TIMEOUT:-600}"
CONDA_ENV="${CONDA_ENV:-}"
FULL_METRICS="${FULL_METRICS:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Detect Python with torch — sudo strips conda PATH.
# Override with: PYTHON_BIN=$(which python3) sudo -E bash ncu_profile_fa4.sh ...
_find_python() {
  for c in \
    "${PYTHON_BIN:-}" \
    "$(command -v python3 2>/dev/null || true)" \
    /mnt/nvme3n1/conda/envs/b200_fa3/bin/python3 \
    /mnt/nvme2n1/conda/envs/b200_fa3/bin/python3 \
    /opt/conda/envs/b200_fa3/bin/python3 \
    /opt/conda/bin/python3 \
    /usr/local/bin/python3; do
    [[ -z "$c" ]] && continue
    if "$c" -c "import torch" 2>/dev/null; then echo "$c"; return 0; fi
  done
  return 1
}
PYTHON_BIN="$(_find_python)" || {
  echo "[error] No python3 with torch found. Set PYTHON_BIN explicitly:"
  echo "  PYTHON_BIN=\$(which python3) sudo -E bash $0 ..."
  exit 1
}
echo "[ncu] Python: $PYTHON_BIN  torch=$("$PYTHON_BIN" -c 'import torch; print(torch.__version__)' 2>/dev/null)"

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
  echo "=== pip packages ==="
  pip3 list 2>/dev/null | grep -iE "torch|flash-attn|cutlass|quack" || echo "(pip list failed)"
  echo
  echo "=== /proc/driver/nvidia/params ==="
  grep -i "RestrictProfiling" /proc/driver/nvidia/params 2>/dev/null \
    || echo "(no RestrictProfiling line)"
  echo
  echo "=== GPU topology ==="
  nvidia-smi topo -m 2>/dev/null | head -20 || true
} > "$OUTDIR/env.txt"

echo "[ncu] Environment logged to env.txt"
echo "[ncu] Note: clock locking is the user's responsibility (not handled here)"

# ---------------------------------------------------------------------------
# 2. Build the NCU command
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Python command
PYTHON_CMD="$PYTHON_BIN $HERE/bench_fa4_simfa.py"
PYTHON_CMD="$PYTHON_CMD --mode run_once --seqlen $SEQLEN --headdim $HEADDIM --heads $HEADS --heads-kv $HEADS_KV --batch $BATCH $NO_BWD"

# NCU sections + metrics
# Targeted metrics for Sim-FA calibration (TMA/DRAM/L2/TC/stalls + latency)
TARGETED_METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,\
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
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_elapsed,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
l1tex__t_sectors_pipe_lsu.sum,\
sm__inst_executed.sum,\
smsp__inst_executed_pipe_tensor.sum,\
smsp__sass_thread_inst_executed_op_fp32_pred_on,\
smsp__sass_thread_inst_executed_op_fadd_pred_on,\
smsp__sass_thread_inst_executed_op_fmul_pred_on,\
smsp__sass_thread_inst_executed_op_ffma_pred_on,\
smsp__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on,\
smsp__sass_thread_inst_executed_op_conversion_pred_on,\
smsp__sass_thread_inst_executed_op_misc_pred_on,\
smsp__sass_thread_inst_executed_op_inter_thread_communication_pred_on,\
smsp__sass_inst_executed_op_tmem,\
smsp__sass_inst_executed_op_tmem_ldt,\
smsp__sass_inst_executed_op_tmem_stt,\
smsp__sass_inst_executed_op_utccp,\
smsp__sass_inst_executed_op_utcmma,\
smsp__average_warp_latency_issue_stalled_long_scoreboard.pct,\
smsp__average_warp_latency_issue_stalled_short_scoreboard.pct,\
smsp__average_warp_latency_issue_stalled_barrier.pct,\
smsp__average_warp_latency_issue_stalled_membar.pct,\
smsp__average_warp_latency_issue_stalled_math_pipe_throttle.pct,\
smsp__average_warp_latency_issue_stalled_wait.pct,\
smsp__average_warp_latency_issue_stalled_mio_throttle.pct,\
smsp__average_warp_latency_issue_stalled_not_selected.pct,\
smsp__warps_eligible.avg.per_cycle_active,\
gpu__time_duration.sum,\
sm__warps_active.avg.pct_of_peak,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__waves_per_multiprocessor,\
launch__registers_per_thread,\
launch__shared_mem_per_block_dynamic"

if [[ -n "$FULL_MODE" ]]; then
  SECTIONS=(
    --section SpeedOfLight
    --section MemoryWorkloadAnalysis
    --section ComputeWorkloadAnalysis
    --section LaunchStats
    --section Occupancy
  )
else
  SECTIONS=(
    --section MemoryWorkloadAnalysis
    --section ComputeWorkloadAnalysis
    --section LaunchStats
    --section Occupancy
  )
fi
METRICS_FLAG=(--metrics "$TARGETED_METRICS")

# bench_fa4_simfa.py --mode run_once does: warmup (5 calls) + 1 timed call.
# Each flash attention call launches 1 kernel.
# Use --launch-skip 5 --launch-count 1 to profile only the 6th (timed) call.
LAUNCH_FILTER=(--launch-skip 5 --launch-count 1)

# ---------------------------------------------------------------------------
# 3. Run NCU
# ---------------------------------------------------------------------------
echo "[ncu] Profiling FA4 kernel..."
echo "[ncu] Command: ncu ${LAUNCH_FILTER[*]} ${SECTIONS[*]} ${METRICS_FLAG[*]} -o $OUTDIR/profile -- $PYTHON_CMD"

set +e
timeout "$PROFILE_TIMEOUT" ncu \
  --target-processes all \
  --clock-control=none \
  "${LAUNCH_FILTER[@]}" \
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
# 4. Export CSV from raw report (do this in the SAME env that ran ncu)
# ---------------------------------------------------------------------------
_discover_ncu() {
  for c in \
    "${NCU_BIN:-}" \
    "$(command -v ncu 2>/dev/null || true)" \
    /usr/local/cuda/bin/ncu; do
    [[ -z "$c" ]] && continue
    [[ -x "$c" ]] && echo "$c" && return 0
  done
  return 1
}
_discover_sections() {
  local ncu="$1"
  if [[ -n "${NSIGHT_COMPUTE_SECTIONS_PATH:-}" && -d "$NSIGHT_COMPUTE_SECTIONS_PATH" ]]; then
    echo "$NSIGHT_COMPUTE_SECTIONS_PATH"; return 0
  fi
  local root
  root="$(dirname "$(dirname "$(readlink -f "$ncu" 2>/dev/null || echo "$ncu")")")"
  if [[ -d "$root/sections" ]]; then echo "$root/sections"; return 0; fi
  for d in /usr/local/cuda/nsight-compute/sections /opt/nvidia/nsight-compute/sections; do
    [[ -d "$d" ]] && echo "$d" && return 0
  done
  return 1
}

if [[ -f "$OUTDIR/profile.ncu-rep" ]]; then
  NCU_BIN="$(_discover_ncu)" || NCU_BIN="ncu"
  SECTIONS="$(_discover_sections "$NCU_BIN" 2>/dev/null || true)"
  SF=()
  [[ -n "$SECTIONS" ]] && SF=(--section-folder "$SECTIONS")

  echo "[ncu] Exporting profile_raw.csv (preferred for offline parse)..."
  ncu --import "$OUTDIR/profile.ncu-rep" "${SF[@]}" --page raw --csv \
    > "$OUTDIR/profile_raw.csv" 2> "$OUTDIR/ncu_import_raw.log" || true
  echo "[ncu] profile_raw.csv rows: $(wc -l < "$OUTDIR/profile_raw.csv" 2>/dev/null || echo 0)"

  echo "[ncu] Exporting profile.csv (details page, optional)..."
  ncu --import "$OUTDIR/profile.ncu-rep" "${SF[@]}" --page details --csv \
    > "$OUTDIR/profile.csv" 2> "$OUTDIR/ncu_import.log" || true
  echo "[ncu] profile.csv rows: $(wc -l < "$OUTDIR/profile.csv" 2>/dev/null || echo 0)"
else
  echo "[ncu] WARNING: no profile.ncu-rep produced. Check ncu_run.log for errors."
fi

# ---------------------------------------------------------------------------
# 5. Parse summary (if parse script exists) — NO sudo
# ---------------------------------------------------------------------------
if [[ -f "$HERE/parse_ncu_report.py" ]] && [[ -f "$OUTDIR/profile.ncu-rep" ]]; then
  echo "[ncu] Parsing report to summary.json (via $PYTHON_BIN, no sudo)..."
  "$PYTHON_BIN" "$HERE/parse_ncu_report.py" \
    "$OUTDIR/profile.ncu-rep" -o "$OUTDIR/summary.json" \
    --csv "$OUTDIR/profile_calib.csv" 2>/dev/null || {
    echo "[warn] parse_ncu_report.py failed; re-run later:"
    echo "       bash $HERE/reparse_ncu_fa3align.sh"
  }
fi

# ---------------------------------------------------------------------------
# 6. Print summary
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
    "$PYTHON_BIN" -c "
import json, sys
with open('$OUTDIR/summary.json') as f:
    d = json.load(f)
if isinstance(d, list):
    kernels = []
    for item in d:
        kernels.extend(item.get('kernels', [item]) if isinstance(item, dict) else [])
else:
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
