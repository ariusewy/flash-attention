#!/usr/bin/env python3
# Author: ywangmu from HKUST
#
# Aggregate all NCU summary JSON files produced by run_b200_all.sh into a
# single ALL_RESULTS.csv for downstream plotting.
#
# Usage:
#   python collect_results.py <outdir> -o ALL_RESULTS.csv
#
# The script scans <outdir>/ncu_*/summary.json and also picks up any
# perf_*.json benchmark results, merging them into one flat CSV.

import argparse
import csv
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Columns of interest from NCU summary.json
# These match the friendly names emitted by parse_ncu_report.py (CALIB_METRICS)
# ---------------------------------------------------------------------------
NCU_COLS = [
    "duration_us",
    "lts_sectors",
    "lts_hit_rate_pct",
    "lts_miss_sectors",
    "dram_bytes_read",
    "dram_bytes_write",
    "dram_throughput_pct",
    "tma_pipe_util_pct",
    "tma_ld_bytes",
    "tma_ld_sectors",
    "tma_st_bytes",
    "tma_st_sectors",
    "tma_pipe_read_bytes",
    "tma_pipe_write_bytes",
    "tma_ld_inst",
    "tma_st_inst",
    "tma_requests",
    "tensor_pipe_util_pct",
    "fma_pipe_util_pct",
    "warps_active_pct",
    "occupancy_pct",
    "stall_barrier_pct",
    "stall_long_scoreboard_pct",
    "ipc_active",
    "inst_executed",
    "sm_count",
    "waves_per_sm",
    "block_size",
    "grid_size",
    "thread_count",
    "regs_per_thread",
    "smem_dynamic_bytes",
]

SHAPE_COLS = ["seqlen", "heads_q", "heads_kv", "headdim", "batch", "gqa_group"]
META_COLS  = ["kernel_name", "kernel_id", "source"]

ALL_COLS = META_COLS + SHAPE_COLS + NCU_COLS


def _parse_shape_from_dir(dir_name: str) -> dict:
    """Extract shape params from directory name 's2048_hq32_hkv8_d128'."""
    shape = {}
    tokens = dir_name.split("_")
    for t in tokens:
        if t.startswith("s") and t[1:].isdigit():
            shape["seqlen"] = int(t[1:])
        elif t.startswith("hq"):
            shape["heads_q"] = int(t[2:])
        elif t.startswith("hkv"):
            shape["heads_kv"] = int(t[3:])
        elif t.startswith("d") and t[1:].isdigit():
            shape["headdim"] = int(t[1:])
    hq  = shape.get("heads_q", 0)
    hkv = shape.get("heads_kv", hq)
    shape["gqa_group"] = hq // hkv if hkv > 0 else 1
    shape.setdefault("batch", 1)
    return shape


def load_ncu_summaries(outdir: Path) -> list[dict]:
    rows = []
    for ncu_dir in sorted(outdir.glob("ncu_*")):
        summary = ncu_dir / "summary.json"
        if not summary.exists():
            continue
        shape_info = _parse_shape_from_dir(ncu_dir.name[4:])  # strip 'ncu_' prefix

        with open(summary) as f:
            data = json.load(f)

        # summary.json is [{source, kernels:[...]}] from parse_ncu_report.py
        kernels = []
        if isinstance(data, dict):
            kernels = data.get("kernels", [data])
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "kernels" in item:
                    kernels.extend(item["kernels"])
                elif isinstance(item, dict):
                    kernels.append(item)

        for kid, kernel in enumerate(kernels):
            row = {
                "source":      "ncu",
                "kernel_name": kernel.get("Kernel Name", kernel.get("kernel_name", "")),
                "kernel_id":   kernel.get("ID", kid),
            }
            row.update(shape_info)
            for col in NCU_COLS:
                row[col] = kernel.get(col, "")
            rows.append(row)

    return rows


def load_perf_results(outdir: Path) -> list[dict]:
    rows = []
    for perf_file in sorted(outdir.glob("perf_*.json")):
        with open(perf_file) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for entry in data:
            hq  = entry.get("heads_q",  entry.get("heads", 0))
            hkv = entry.get("heads_kv", hq)
            row = {
                "source":      "perf",
                "kernel_name": "bench_fa4",
                "kernel_id":   entry.get("case", ""),
                "batch":       entry.get("batch", 1),
                "seqlen":      entry.get("seqlen", ""),
                "heads_q":     hq,
                "heads_kv":    hkv,
                "headdim":     entry.get("headdim", ""),
                "gqa_group":   hq // hkv if hkv > 0 else 1,
                # Map perf-specific fields into NCU_COLS where sensible
                "duration_us": round(entry.get("fwd_ms", 0) * 1000, 3),
            }
            # Zero-fill remaining NCU columns
            for col in NCU_COLS:
                row.setdefault(col, "")
            # Add perf-specific extras as extra columns (appended)
            row["fwd_ms"]       = entry.get("fwd_ms", "")
            row["bwd_ms"]       = entry.get("bwd_ms", "")
            row["fwd_tflops"]   = entry.get("fwd_tflops", "")
            row["total_tflops"] = entry.get("total_tflops", "")
            rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path):
    if not rows:
        print("[warn] No data rows found.", file=sys.stderr)
        return

    # Build column list: fixed order + any extra keys not already in ALL_COLS
    extra_cols = []
    for r in rows:
        for k in r:
            if k not in ALL_COLS and k not in extra_cols:
                extra_cols.append(k)
    fieldnames = ALL_COLS + extra_cols

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Written {len(rows)} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate B200 NCU + perf results into ALL_RESULTS.csv")
    parser.add_argument("outdir", help="Top-level results directory")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path (default: <outdir>/ALL_RESULTS.csv)")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    if not outdir.exists():
        print(f"[error] Directory not found: {outdir}", file=sys.stderr)
        sys.exit(1)

    out_csv = Path(args.output) if args.output else outdir / "ALL_RESULTS.csv"

    ncu_rows  = load_ncu_summaries(outdir)
    perf_rows = load_perf_results(outdir)

    all_rows = ncu_rows + perf_rows
    print(f"NCU rows: {len(ncu_rows)}")
    print(f"Perf rows: {len(perf_rows)}")
    print(f"Total: {len(all_rows)}")

    write_csv(all_rows, out_csv)


if __name__ == "__main__":
    main()
