#!/usr/bin/env bash
# Author: ywangmu from HKUST
#
# One-click B200 profiling runner for FA4 (container / bare-metal).
# Only includes: env probe, correctness, perf, NCU profiling, collect results.
# Environment setup (pip install) is NOT included — run B200_setup_fa4.sh separately.
#
# Modes:
#   --smoke   Quick validation (~5 min, no sudo needed).
#             Runs: env probe + correctness + perf/small + 1 NCU shape (skipped
#             if ncu not available or RestrictProfiling is set).
#
#   --full    Complete data collection (~60-90 min, sudo required for NCU).
#             Runs: env probe + correctness + perf/all + NCU for 5 shapes
#             (MHA smoke + Llama3-8B/70B/405B-like GQA configs).
#
# Usage:
#   bash run_b200_all.sh --smoke
#   sudo bash run_b200_all.sh --full --gpu 5
#   sudo bash run_b200_all.sh --full --outdir /mnt/nvme3n1/b200_results --gpu 5
#
# Clock locking is NOT handled here — lock the SM clock manually before running:
#   sudo nvidia-smi -i 5 -lgc 1830,1830
#   sudo nvidia-smi -i 5 -rgc        # restore afterwards
#
# Output:
#   <OUTDIR>/
#     env_report.txt
#     correctness.log
#     perf_small.json
#     perf_ncu_sweep.json        (full only)
#     perf_llama3_all.json       (full only)
#     ncu_<shape>/
#       summary.json
#       profile_calib.csv
#       profile.ncu-rep
#     ALL_RESULTS.csv            (aggregated by collect_results.py)
#     run_summary.txt

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SMOKE=""
FULL=""
GPU_ID="${GPU_ID:-0}"
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)     SMOKE=1; shift ;;
    --full)      FULL=1;  shift ;;
    --outdir)    OUTDIR="$2"; shift 2 ;;
    --gpu)       GPU_ID="$2"; shift 2 ;;
    *) echo "[error] Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$SMOKE" && -z "$FULL" ]]; then
  echo "Usage: bash $0 --smoke | --full [--outdir DIR] [--gpu N]"
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d-%H%M%S)"
[[ -z "$OUTDIR" ]] && OUTDIR="$HERE/results/b200_run_${TS}"
mkdir -p "$OUTDIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

PYTHON="python3"
BENCH="$HERE/bench_fa4_simfa.py"
NCU_SCRIPT="$HERE/ncu_profile_fa4.sh"
PARSE="$HERE/parse_ncu_report.py"
COLLECT="$HERE/collect_results.py"

# Log everything to run_summary.txt as well as stdout
exec > >(tee -a "$OUTDIR/run_summary.txt") 2>&1

echo "============================================================"
echo "B200 FA4 Profiling Run"
echo "============================================================"
echo "Mode     : $([ -n "$FULL" ] && echo 'FULL' || echo 'SMOKE')"
echo "GPU      : $GPU_ID"
echo "Output   : $OUTDIR"
echo "Date     : $(date)"
echo "Note     : clock locking is the user's responsibility (not handled here)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Helper: run a step, log pass/fail
# ---------------------------------------------------------------------------
STEP_LOG=()
step_ok()   { STEP_LOG+=("  [OK]   $1"); }
step_fail() { STEP_LOG+=("  [FAIL] $1  ($2)"); }
step_skip() { STEP_LOG+=("  [SKIP] $1"); }

check_ncu_perm() {
  if ! command -v ncu &>/dev/null; then return 1; fi
  grep -qi "RestrictProfiling = 0" /proc/driver/nvidia/params 2>/dev/null && return 0
  if ncu --target-processes all --metrics gpu__time_duration.sum \
         -o /dev/null python3 -c "import torch; torch.cuda.synchronize()" \
         >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

# ---------------------------------------------------------------------------
# Step 1: Environment probe
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Environment probe..."
bash "$HERE/B200_probe.sh" > "$OUTDIR/env_report.txt" 2>&1 && \
  step_ok "env_probe" || step_fail "env_probe" "B200_probe.sh failed"
echo "      Saved to env_report.txt"

# ---------------------------------------------------------------------------
# Step 2: FA4 correctness check
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] FA4 correctness check..."
$PYTHON "$BENCH" --mode correctness --cases minimal \
  > "$OUTDIR/correctness.log" 2>&1
CORR_RC=$?
if [[ $CORR_RC -eq 0 ]]; then
  step_ok "correctness (minimal)"
  echo "      PASS"
else
  step_fail "correctness" "exit $CORR_RC — check $OUTDIR/correctness.log"
  echo "      FAIL — check correctness.log"
fi

if [[ -n "$FULL" ]]; then
  $PYTHON "$BENCH" --mode correctness --cases llama3_8b \
    >> "$OUTDIR/correctness.log" 2>&1 && \
    step_ok "correctness (llama3_8b GQA)" || \
    step_fail "correctness llama3_8b" "check correctness.log"
fi

# ---------------------------------------------------------------------------
# Step 3: Performance benchmarks
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Performance benchmarks..."

$PYTHON "$BENCH" --mode perf --cases small \
  --no-backward -o "$OUTDIR/perf_small.json" \
  >> "$OUTDIR/perf.log" 2>&1 && \
  step_ok "perf/small" || step_fail "perf/small" "check perf.log"

if [[ -n "$FULL" ]]; then
  $PYTHON "$BENCH" --mode perf --cases ncu_sweep \
    --no-backward -o "$OUTDIR/perf_ncu_sweep.json" \
    >> "$OUTDIR/perf.log" 2>&1 && \
    step_ok "perf/ncu_sweep" || step_fail "perf/ncu_sweep" "check perf.log"

  $PYTHON "$BENCH" --mode perf --cases llama3_all \
    --no-backward -o "$OUTDIR/perf_llama3_all.json" \
    >> "$OUTDIR/perf.log" 2>&1 && \
    step_ok "perf/llama3_all" || step_fail "perf/llama3_all" "check perf.log"
fi

# ---------------------------------------------------------------------------
# Step 4: NCU profiling
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] NCU profiling..."

if check_ncu_perm; then
  NCU_AVAIL=1
  echo "      NCU counter access: OK"
else
  NCU_AVAIL=""
  echo "      NCU counter access: RESTRICTED (need sudo or RestrictProfiling=0)"
  echo "      Skipping NCU profiling. Re-run with sudo to collect hardware counters."
fi

run_ncu_shape() {
  local seqlen=$1 heads_q=$2 heads_kv=$3 headdim=$4
  local label="s${seqlen}_hq${heads_q}_hkv${heads_kv}_d${headdim}"
  local shape_dir="$OUTDIR/ncu_${label}"
  mkdir -p "$shape_dir"

  echo "      Profiling: seqlen=$seqlen heads_q=$heads_q heads_kv=$heads_kv headdim=$headdim"

  OUTDIR="$shape_dir" \
  GPU_ID="$GPU_ID" \
  FULL_METRICS="$([ -n "$FULL" ] && echo 1)" \
  bash "$NCU_SCRIPT" \
    --seqlen "$seqlen" \
    --headdim "$headdim" \
    --heads "$heads_q" \
    --heads-kv "$heads_kv" \
    --batch 1 \
    --no-backward \
    $( [[ -n "$FULL" ]] && echo "--full" ) \
    >> "$OUTDIR/../ncu_run.log" 2>&1 && \
    step_ok "ncu/${label}" || step_fail "ncu/${label}" "check ncu_run.log"

  local rep="$shape_dir/profile.ncu-rep"
  if [[ -f "$rep" ]] && [[ -f "$PARSE" ]]; then
    $PYTHON "$PARSE" "$rep" \
      -o "$shape_dir/summary.json" \
      --csv "$shape_dir/profile_calib.csv" \
      >> "$OUTDIR/../ncu_parse.log" 2>&1 || true
  fi
}

if [[ -n "$NCU_AVAIL" ]]; then
  if [[ -n "$SMOKE" ]]; then
    run_ncu_shape 512 8 8 128
  else
    run_ncu_shape  512  8   8 128
    run_ncu_shape 1024 32  8 128
    run_ncu_shape 2048 32  8 128
    run_ncu_shape 4096 64  8 128
    run_ncu_shape 8192 128 8 128
  fi
else
  step_skip "ncu_profiling (no counter access)"
fi

# ---------------------------------------------------------------------------
# Step 5: Aggregate results
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Aggregating results..."
if [[ -f "$COLLECT" ]]; then
  $PYTHON "$COLLECT" "$OUTDIR" \
    -o "$OUTDIR/ALL_RESULTS.csv" >> "$OUTDIR/collect.log" 2>&1 && \
    step_ok "collect_results" || step_fail "collect_results" "check collect.log"
  echo "      ALL_RESULTS.csv written"
else
  step_skip "collect_results (collect_results.py not found)"
fi

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "STEP SUMMARY"
echo "============================================================"
for s in "${STEP_LOG[@]}"; do echo "$s"; done
echo ""
echo "Output directory: $OUTDIR/"
ls -lh "$OUTDIR/" 2>/dev/null | head -20
