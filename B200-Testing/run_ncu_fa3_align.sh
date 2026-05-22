#!/usr/bin/env bash
# Author: ywangmu from HKUST
#
# FA4 NCU profiling for the 9 FA3-aligned shapes.
# Same shape set as run_nsys_fa3_align.sh:
#   H_Q in {32, 64, 128} x S in {512, 1024, 2048},  H_KV=8, headdim=128, batch=1
#
# Each shape invokes ncu_profile_fa4.sh, which captures the curated
# DRAM/L2/TMA/Tensor-Core/stall metrics (plus Memory/Compute/Launch/Occupancy
# sections). Pass FULL_METRICS=1 to additionally include SpeedOfLight.
#
# NCU needs profiling counter access — run with `sudo -E` unless
# /proc/driver/nvidia/params has `RestrictProfiling = 0`.
#
# Clock locking is NOT handled here — lock the SM clock manually before running:
#   sudo nvidia-smi -i 5 -lgc 1830,1830
#   CUDA_VISIBLE_DEVICES=5 sudo -E bash run_ncu_fa3_align.sh
#   sudo nvidia-smi -i 5 -rgc
#
# Output:
#   ncu_reports_fa3align/
#     ncu_s<seq>_hq<HQ>_hkv8_d128/
#       profile.ncu-rep        raw NCU report
#       profile.csv            CSV export of the details page
#       summary.json           parsed calibration metrics
#       env.txt, app_stdout.txt, ncu_run.log
#   ncu_fa3align_results.csv   aggregated table over all 9 shapes

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCU_DRIVER="$HERE/ncu_profile_fa4.sh"
# collect_results.py lives in B200-Testing/ in the local checkout but in the
# flat layout used by the fork it sits next to this script. Try both.
COLLECT=""
for _cand in \
  "$HERE/collect_results.py" \
  "$HERE/B200-Testing/collect_results.py"; do
  if [ -f "$_cand" ]; then COLLECT="$_cand"; break; fi
done
OUT_ROOT="${OUT_ROOT:-$HERE/ncu_reports_fa3align}"
SUMMARY_CSV="${SUMMARY_CSV:-$HERE/ncu_fa3align_results.csv}"
NCU_TAR="${NCU_TAR:-fa4_fa3align_ncu_reports.tar.gz}"

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

if [ ! -x "$NCU_DRIVER" ] && [ ! -f "$NCU_DRIVER" ]; then
  echo "Error: ncu_profile_fa4.sh not found at $NCU_DRIVER"
  exit 1
fi
command -v ncu >/dev/null 2>&1 || { echo "Error: ncu not in PATH"; exit 1; }

rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"

# Forward optional flags to ncu_profile_fa4.sh (passed by the caller via env).
EXTRA_FLAGS=()
[[ -n "${FULL_METRICS:-}" ]] && EXTRA_FLAGS+=("--full")

echo "=================================================================="
echo "FA4 NCU Profiling (FA3-aligned, ${#SHAPES[@]} shapes) at $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "FULL_METRICS=${FULL_METRICS:-0}"
echo "Output root : $OUT_ROOT"
echo "Note: clock locking is the user's responsibility (not handled here)"
echo "=================================================================="

# ---------------------------------------------------------------------------
# Per-shape NCU runs (each shape is one full ncu invocation with replays)
# ---------------------------------------------------------------------------
FAILED=()
for shape in "${SHAPES[@]}"; do
  read hq hkv s d <<< "$shape"
  LABEL="s${s}_hq${hq}_hkv${hkv}_d${d}"
  SHAPE_OUT="${OUT_ROOT}/ncu_${LABEL}"
  mkdir -p "$SHAPE_OUT"

  echo ""
  echo ">>> [${LABEL}] H_Q=${hq} H_KV=${hkv} S=${s} D=${d}"
  echo "    OUTDIR=${SHAPE_OUT}"

  set +e
  OUTDIR="$SHAPE_OUT" bash "$NCU_DRIVER" \
    --seqlen "$s" \
    --headdim "$d" \
    --heads "$hq" \
    --heads-kv "$hkv" \
    --batch 1 \
    --no-backward \
    "${EXTRA_FLAGS[@]}" \
    > "${SHAPE_OUT}/driver.log" 2>&1
  RC=$?
  set -e

  if [[ $RC -ne 0 ]]; then
    echo "    [FAIL] ncu_profile_fa4.sh returned $RC — check ${SHAPE_OUT}/driver.log"
    FAILED+=("$LABEL")
  else
    echo "    [OK]   profile.ncu-rep + summary.json written"
  fi
done

# ---------------------------------------------------------------------------
# Aggregate every summary.json into a single CSV
# ---------------------------------------------------------------------------
echo ""
echo "Aggregating per-shape summary.json -> ${SUMMARY_CSV}"

if [[ -n "$COLLECT" && -f "$COLLECT" ]]; then
  python3 "$COLLECT" "$OUT_ROOT" -o "$SUMMARY_CSV" || \
    echo "[warn] collect_results.py exited non-zero; check manually"
else
  echo "[warn] collect_results.py not found — falling back to inline collector"
  python3 - <<PY
import csv, json
from pathlib import Path

out_root = Path("${OUT_ROOT}")
out_csv  = Path("${SUMMARY_CSV}")
cols = [
    "label","seqlen","heads_q","heads_kv","headdim",
    "duration_us","dram_bytes_read","dram_bytes_write","dram_throughput_pct",
    "lts_sectors","lts_hit_rate_pct",
    "tma_pipe_util_pct","tma_ld_bytes","tma_st_bytes",
    "tensor_pipe_util_pct","warps_active_pct","ipc_active",
    "stall_long_scoreboard_pct","stall_barrier_pct",
    "kernel_name",
]
rows = []
for sjson in sorted(out_root.glob("ncu_*/summary.json")):
    label = sjson.parent.name[4:]
    parts = {p[0]: p[1:] for p in label.split("_")} if False else {}
    tokens = label.split("_")
    parsed = {}
    for t in tokens:
        if t.startswith("s") and t[1:].isdigit():   parsed["seqlen"]   = int(t[1:])
        elif t.startswith("hq"):                    parsed["heads_q"]  = int(t[2:])
        elif t.startswith("hkv"):                   parsed["heads_kv"] = int(t[3:])
        elif t.startswith("d") and t[1:].isdigit(): parsed["headdim"]  = int(t[1:])
    data = json.loads(sjson.read_text())
    kernels = data if isinstance(data, list) else [data]
    for k in kernels:
        if isinstance(k, dict) and "kernels" in k:
            for kk in k["kernels"]:
                row = {"label": label, **parsed}
                row["kernel_name"] = kk.get("Kernel Name", kk.get("kernel_name",""))
                for c in cols:
                    row.setdefault(c, kk.get(c, ""))
                rows.append(row)
        elif isinstance(k, dict):
            row = {"label": label, **parsed}
            row["kernel_name"] = k.get("Kernel Name", k.get("kernel_name",""))
            for c in cols:
                row.setdefault(c, k.get(c, ""))
            rows.append(row)
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print(f"Written {len(rows)} rows -> {out_csv}")
PY
fi

# ---------------------------------------------------------------------------
# Pack raw reports
# ---------------------------------------------------------------------------
rm -f "$NCU_TAR"
tar -czf "$NCU_TAR" -C "$(dirname "$OUT_ROOT")" "$(basename "$OUT_ROOT")" 2>/dev/null || true

echo ""
echo "=================================================================="
echo "Done!"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed shapes (${#FAILED[@]}): ${FAILED[*]}"
fi
echo "Per-shape reports: ${OUT_ROOT}/"
echo "Aggregated CSV   : ${SUMMARY_CSV}"
echo "Archive          : ${NCU_TAR}"
echo "=================================================================="
