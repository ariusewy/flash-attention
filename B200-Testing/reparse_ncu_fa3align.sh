#!/usr/bin/env bash
# Author: ywangmu from HKUST
#
# Re-parse existing FA3-aligned NCU reports WITHOUT sudo.
#
# `sudo python3 parse_ncu_report.py` often breaks because sudo uses a
# different PATH/ncu and NSIGHT_COMPUTE_SECTIONS_PATH is unset. The manual
# `ncu --import ... --page raw --csv` that worked for you was run as the
# normal user — this script does the same.
#
# Usage (from B200-Testing/ or scripts/):
#   bash reparse_ncu_fa3align.sh
#   ROOT=ncu_reports_fa3align bash reparse_ncu_fa3align.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$HERE/ncu_reports_fa3align}"
PARSE="$HERE/parse_ncu_report.py"
COLLECT=""
for _c in "$HERE/collect_results.py" "$HERE/B200-Testing/collect_results.py"; do
  [[ -f "$_c" ]] && COLLECT="$_c" && break
done
SUMMARY="${SUMMARY:-$HERE/ncu_fa3align_results.csv}"

command -v ncu >/dev/null 2>&1 || {
  echo "[error] ncu not in PATH — activate your conda env first (b200_fa3)"
  exit 1
}
[[ -d "$ROOT" ]] || { echo "[error] not found: $ROOT"; exit 1; }
[[ -f "$PARSE" ]] || { echo "[error] not found: $PARSE"; exit 1; }

if [[ "$(id -u)" -eq 0 ]]; then
  echo "[error] Do NOT run this script with sudo."
  echo "        Activate conda and run as your normal user."
  exit 1
fi

echo "=================================================================="
echo "Re-export + re-parse NCU reports (no sudo)"
echo "ROOT : $ROOT"
echo "ncu  : $(command -v ncu)"
echo "=================================================================="

# Step 1: export raw CSV next to each .ncu-rep (same as your manual test)
for rep in "$ROOT"/ncu_*/profile.ncu-rep; do
  [[ -f "$rep" ]] || continue
  dir="$(dirname "$rep")"
  label="$(basename "$dir")"
  echo ""
  echo ">>> export raw CSV: $label"
  if ncu --import "$rep" --page raw --csv > "$dir/profile_raw.csv" 2>"$dir/ncu_reexport.log"; then
    rows="$(wc -l < "$dir/profile_raw.csv")"
    echo "    profile_raw.csv: ${rows} lines"
  else
    echo "    [warn] ncu export failed — see $dir/ncu_reexport.log"
    head -5 "$dir/ncu_reexport.log" 2>/dev/null || true
  fi
done

# Step 2: parse (reads profile_raw.csv first, no ncu --import needed)
echo ""
echo ">>> parse summaries"
for rep in "$ROOT"/ncu_*/profile.ncu-rep; do
  [[ -f "$rep" ]] || continue
  dir="$(dirname "$rep")"
  python3 "$PARSE" "$rep" \
    -o "$dir/summary.json" \
    --csv "$dir/profile_calib.csv"
done

# Step 3: aggregate
echo ""
echo ">>> aggregate -> $SUMMARY"
if [[ -n "$COLLECT" ]]; then
  python3 "$COLLECT" "$ROOT" -o "$SUMMARY"
else
  echo "[warn] collect_results.py not found — skip aggregate"
fi

echo ""
echo "Done. Inspect: column -t -s, $SUMMARY | head -15"
