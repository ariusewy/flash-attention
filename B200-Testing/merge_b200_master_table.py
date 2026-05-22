#!/usr/bin/env python3
# Author: ywangmu from HKUST
#
# Merge B200 FA4 profiling results into one master table:
#   - nsys latency (ground truth)
#   - NCU duration + memory/TMA/TC metrics
#
# Supports two input suites:
#   fa3_align   : 9 shapes  H_Q in {32,64,128} x S in {512,1024,2048}, H_KV=8
#   ncu_sweep   : 5 shapes  (512,h8), (1024,h32), (2048,h32), (4096,h64), (8192,h128)
#
# Usage:
#   python3 merge_b200_master_table.py \
#       --ncu-fa3align ncu_fa3align_results.csv \
#       --nsys-fa3align device_kernel_summary_fa3align.csv \
#       --nsys-sweep  device_kernel_summary.csv \
#       --ncu-sweep   ALL_RESULTS.csv \
#       -o B200_MASTER_TABLE.csv

import argparse
import csv
import sys
from pathlib import Path

MASTER_COLS = [
    "suite", "label",
    "batch", "seqlen", "heads_q", "heads_kv", "headdim", "gqa_group",
    "nsys_avg_us", "nsys_med_us", "nsys_min_us", "nsys_max_us", "nsys_instances",
    "ncu_duration_us",
    "ncu_dram_read_gb", "ncu_dram_write_gb", "ncu_dram_bw_pct",
    "ncu_l2_hit_pct", "ncu_l2_sectors_m",
    "ncu_tma_pipe_pct", "ncu_tma_ld_gb", "ncu_tma_st_gb",
    "ncu_tensor_pipe_pct", "ncu_warps_active_pct", "ncu_occupancy_pct",
    "ncu_stall_barrier_pct", "ncu_stall_memdep_pct",
    "ncu_ipc", "ncu_grid", "ncu_block", "ncu_smem_kb",
    "tflops_nsys", "mfu_pct",
]

# B200 SXM BF16 TC peak (approx, for MFU reference only)
B200_PEAK_TFLOPS = 2250.0


def _f(v, default=""):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v, default=""):
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _label(seqlen, heads_q, heads_kv=8, headdim=128):
    gqa = heads_q // heads_kv if heads_kv else 1
    return f"s{seqlen}_hq{heads_q}_hkv{heads_kv}_d{headdim}_g{gqa}"


def _tflops(seqlen, heads_q, heads_kv, headdim, latency_us):
    if not latency_us or latency_us <= 0:
        return "", ""
    # FA fwd FLOPs ≈ 4 * B * H_q * S^2 * D  (same as bench_fa4_simfa)
    flops = 4.0 * 1 * heads_q * (seqlen ** 2) * headdim
    tflops = flops / (latency_us * 1e-6) / 1e12
    mfu = 100.0 * tflops / B200_PEAK_TFLOPS
    return round(tflops, 2), round(mfu, 2)


def _nsys_avg_to_us(row, key="Avg_us"):
    """Old sweep CSV stores nanoseconds in *_us columns; new FA3 align may be µs."""
    v = _f(row.get(key))
    if v == "":
        return ""
    # Heuristic: values > 1000 are almost certainly nanoseconds
    if v > 1000:
        return round(v / 1000.0, 3)
    return round(v, 3)


def load_nsys_csv(path: Path, suite: str) -> dict:
    """Return dict keyed by (seqlen, heads_q, heads_kv) -> row."""
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = _i(row.get("SeqLen") or row.get("seqlen"))
            hq = _i(row.get("Heads_Q") or row.get("heads_q"))
            hkv = _i(row.get("Heads_KV") or row.get("heads_kv"), 8)
            if s == "" or hq == "":
                continue
            key = (s, hq, hkv)
            out[key] = {
                "suite": suite,
                "seqlen": s, "heads_q": hq, "heads_kv": hkv,
                "headdim": _i(row.get("headdim"), 128),
                "nsys_avg_us": _nsys_avg_to_us(row, "Avg_us"),
                "nsys_med_us": _nsys_avg_to_us(row, "Med_us"),
                "nsys_min_us": _nsys_avg_to_us(row, "Min_us"),
                "nsys_max_us": _nsys_avg_to_us(row, "Max_us"),
                "nsys_instances": _i(row.get("Instances")),
            }
    return out


def load_ncu_csv(path: Path, suite: str) -> dict:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = _i(row.get("seqlen"))
            hq = _i(row.get("heads_q"))
            hkv = _i(row.get("heads_kv"), 8)
            if s == "" or hq == "":
                continue
            key = (s, hq, hkv)
            dur = _f(row.get("duration_us"))
            stall_b = _f(row.get("stall_barrier_pct"))
            stall_m = _f(row.get("stall_long_scoreboard_pct"))
            # Raw counter misparsed as pct when value >> 100
            if isinstance(stall_b, float) and stall_b > 100:
                stall_b = ""
            if isinstance(stall_m, float) and stall_m > 100:
                stall_m = ""
            out[key] = {
                "ncu_duration_us": dur,
                "ncu_dram_read_gb": round(_f(row.get("dram_bytes_read")) / 1e9, 4)
                    if _f(row.get("dram_bytes_read")) != "" else "",
                "ncu_dram_write_gb": round(_f(row.get("dram_bytes_write")) / 1e9, 4)
                    if _f(row.get("dram_bytes_write")) != "" else "",
                "ncu_dram_bw_pct": _f(row.get("dram_throughput_pct")),
                "ncu_l2_hit_pct": _f(row.get("lts_hit_rate_pct")),
                "ncu_l2_sectors_m": round(_f(row.get("lts_sectors")) / 1e6, 3)
                    if _f(row.get("lts_sectors")) != "" else "",
                "ncu_tma_pipe_pct": _f(row.get("tma_pipe_util_pct")),
                "ncu_tma_ld_gb": round(_f(row.get("tma_ld_bytes")) / 1e9, 4)
                    if _f(row.get("tma_ld_bytes")) != "" else "",
                "ncu_tma_st_gb": round(_f(row.get("tma_st_bytes")) / 1e9, 4)
                    if _f(row.get("tma_st_bytes")) != "" else "",
                "ncu_tensor_pipe_pct": _f(row.get("tensor_pipe_util_pct")),
                "ncu_warps_active_pct": _f(row.get("warps_active_pct")),
                "ncu_occupancy_pct": _f(row.get("occupancy_pct")),
                "ncu_stall_barrier_pct": stall_b,
                "ncu_stall_memdep_pct": stall_m,
                "ncu_ipc": _f(row.get("ipc_active")),
                "ncu_grid": row.get("grid_size", ""),
                "ncu_block": row.get("block_size", ""),
                "ncu_smem_kb": round(_f(row.get("smem_dynamic_bytes")) / 1024, 1)
                    if _f(row.get("smem_dynamic_bytes")) != "" else "",
            }
    return out


def merge_suite(nsys: dict, ncu: dict, suite: str) -> list:
    keys = sorted(set(nsys) | set(ncu))
    rows = []
    for key in keys:
        s, hq, hkv = key
        n = nsys.get(key, {})
        c = ncu.get(key, {})
        headdim = n.get("headdim") or 128
        gqa = hq // hkv if hkv else 1
        nsys_avg = n.get("nsys_avg_us", "")
        tflops, mfu = _tflops(s, hq, hkv, headdim, nsys_avg)
        row = {
            "suite": suite,
            "label": _label(s, hq, hkv, headdim),
            "batch": 1, "seqlen": s, "heads_q": hq, "heads_kv": hkv,
            "headdim": headdim, "gqa_group": gqa,
            "nsys_avg_us": nsys_avg,
            "nsys_med_us": n.get("nsys_med_us", ""),
            "nsys_min_us": n.get("nsys_min_us", ""),
            "nsys_max_us": n.get("nsys_max_us", ""),
            "nsys_instances": n.get("nsys_instances", ""),
            "tflops_nsys": tflops, "mfu_pct": mfu,
        }
        row.update(c)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Merge B200 nsys + NCU into master table")
    ap.add_argument("--ncu-fa3align", default="ncu_fa3align_results.csv")
    ap.add_argument("--nsys-fa3align", default="device_kernel_summary_fa3align.csv")
    ap.add_argument("--nsys-sweep", default="device_kernel_summary.csv")
    ap.add_argument("--ncu-sweep", default="ALL_RESULTS.csv")
    ap.add_argument("-o", "--output", default="B200_MASTER_TABLE.csv")
    args = ap.parse_args()

    all_rows = []

    p = Path(args.nsys_fa3align)
    q = Path(args.ncu_fa3align)
    if p.is_file() and q.is_file():
        all_rows += merge_suite(
            load_nsys_csv(p, "fa3_align"),
            load_ncu_csv(q, "fa3_align"),
            "fa3_align",
        )
        print(f"fa3_align: {len(all_rows)} rows", file=sys.stderr)
    else:
        print(f"[warn] skip fa3_align — missing {p} or {q}", file=sys.stderr)

    n_before = len(all_rows)
    p = Path(args.nsys_sweep)
    q = Path(args.ncu_sweep)
    if p.is_file() and q.is_file():
        all_rows += merge_suite(
            load_nsys_csv(p, "ncu_sweep"),
            load_ncu_csv(q, "ncu_sweep"),
            "ncu_sweep",
        )
        print(f"ncu_sweep: {len(all_rows) - n_before} rows", file=sys.stderr)
    else:
        print(f"[warn] skip ncu_sweep — missing {p} or {q}", file=sys.stderr)

    if not all_rows:
        print("[error] no rows merged", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"Written {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
