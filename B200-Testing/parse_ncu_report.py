#!/usr/bin/env python3
# Author: ywangmu from HKUST
#
# Parse NCU .ncu-rep files via `ncu --import --page raw --csv`.
# No external dependencies (stdlib only: subprocess, csv, json, os, sys, argparse).
#
# The raw page CSV has 3 leading rows:
#   row 0 — column headers
#   row 1 — units
#   row 2+ — one row per kernel launch
#
# Outputs:
#   --output / -o  <file.json>   full per-kernel JSON (all metrics)
#   --csv          <file.csv>    flat CSV of calibration-relevant metrics
#   --pretty                     human-readable summary to stdout
#
# Usage:
#   python parse_ncu_report.py profile.ncu-rep -o summary.json
#   python parse_ncu_report.py profile.ncu-rep --csv results.csv
#   python parse_ncu_report.py *.ncu-rep -o all.json --csv all.csv --pretty

import argparse
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

NCU_BIN = "ncu"

# Calibration-relevant metrics extracted into top-level fields.
# Maps NCU metric name -> friendly field name in the output dict.
CALIB_METRICS = {
    # Timing
    "gpu__time_duration.sum":                                          "duration_us",
    # L2 / LTS
    "lts__t_sectors.sum":                                              "lts_sectors",
    "lts__t_sector_hit_rate.pct":                                      "lts_hit_rate_pct",
    "lts__t_sectors_lookup_miss.sum":                                  "lts_miss_sectors",
    "lts__t_sectors_lookup_miss_data_promoted.sum":                    "lts_miss_sectors_promoted",
    # DRAM (only present when profiled with explicit --metrics)
    "dram__bytes_read.sum":                                            "dram_bytes_read",
    "dram__bytes_write.sum":                                           "dram_bytes_write",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed":              "dram_throughput_pct",
    # TMA pipe + traffic (GMEM <-> SMEM via TMA)
    "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active":     "tma_pipe_util_pct",
    "sm__pipe_tma_cycles_active.sum.pct_of_peak_sustained_elapsed":    "tma_pipe_elapsed_pct",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum":         "tma_ld_bytes",
    "l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld.sum":       "tma_ld_sectors",
    "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum":        "tma_st_bytes",
    "l1tex__m_l1tex2xbar_write_sectors_mem_global_op_tma_st.sum":      "tma_st_sectors",
    "l1tex__m_xbar2l1tex_read_bytes_pipe_tma.sum":                     "tma_pipe_read_bytes",
    "l1tex__m_l1tex2xbar_write_bytes_pipe_tma.sum":                    "tma_pipe_write_bytes",
    "sm__sass_inst_executed_op_tma_ld.sum":                            "tma_ld_inst",
    "sm__sass_inst_executed_op_tma_st.sum":                            "tma_st_inst",
    "l1tex__tmain_requests.sum":                                       "tma_requests",
    # Tensor / MMA pipe
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active":  "tensor_pipe_util_pct",
    "sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_elapsed": "tensor_pipe_elapsed_pct",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active":     "fma_pipe_util_pct",
    # Warp / occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active":               "warps_active_pct",
    "sm__maximum_warps_per_active_cycle_pct":                          "occupancy_pct",
    # Stall breakdown
    "smsp__average_warp_latency_issue_stalled_barrier.pct":            "stall_barrier_pct",
    "smsp__average_warp_latency_issue_stalled_long_scoreboard.pct":    "stall_long_scoreboard_pct",
    # Instructions
    "inst_executed":                                                   "inst_executed",
    "sm__inst_executed.sum.per_cycle_active":                          "ipc_active",
    # Launch config
    "launch__block_size":                                              "block_size",
    "launch__grid_size":                                               "grid_size",
    "launch__thread_count":                                            "thread_count",
    "launch__registers_per_thread":                                    "regs_per_thread",
    "launch__shared_mem_per_block_dynamic":                            "smem_dynamic_bytes",
    "launch__waves_per_multiprocessor":                                "waves_per_sm",
    "launch__sm_count":                                                "sm_count",
}

# Column names that come directly from the fixed NCU report columns (not raw metrics)
FIXED_COLS = {
    "ID":           "id",
    "Kernel Name":  "kernel_name",
    "Block Size":   "block_size_str",
    "Grid Size":    "grid_size_str",
    "Device":       "device",
    "CC":           "cc",
    "Context":      "context_id",
    "Stream":       "stream_id",
}


def _strip_num(s: str):
    """Remove thousands separators and trailing spaces, try to cast to number."""
    s = s.strip().replace(",", "")
    if s == "" or s == "-":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def export_csv_text(ncu_path: str, ncu_bin: str = NCU_BIN) -> str:
    """Run `ncu --import <file> --page raw --csv --units base` and return stdout text.

    `--units base` forces fixed base units (ns / byte / cycle) instead of
    adaptive (us/ms, KB/MB/GB), so downstream parsing can trust the values.
    """
    cmd = [ncu_bin, "--import", ncu_path, "--page", "raw", "--csv",
           "--units", "base"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"`{ncu_bin}` not found. Ensure ncu is on PATH."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ncu timed out on {ncu_path}")

    if result.returncode != 0:
        raise RuntimeError(
            f"ncu exited {result.returncode}:\n{result.stderr[:500]}"
        )
    return result.stdout


def parse_ncu_csv(text: str, source: str) -> dict:
    """
    Parse the raw-page CSV text returned by ncu.

    Returns a dict:
      {
        "source":  "<filename>",
        "kernels": [ {kernel fields + metrics...}, ... ]
      }
    """
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    if len(rows) < 3:
        return {"source": source, "kernels": [], "note": "no kernel data in report"}

    headers = rows[0]   # column names
    # rows[1] is units — skip
    data_rows = rows[2:]

    # Build lookup: header string -> column index
    col_idx = {h: i for i, h in enumerate(headers)}

    kernels = []
    for row in data_rows:
        if not row or row[0].strip() == "":
            continue

        entry = {"_source": source}

        # Fixed identity columns
        for header, field in FIXED_COLS.items():
            if header in col_idx:
                val = row[col_idx[header]] if col_idx[header] < len(row) else ""
                entry[field] = val.strip()

        # Cast id to int
        if "id" in entry:
            try:
                entry["id"] = int(entry["id"])
            except (ValueError, TypeError):
                pass

        # Calibration metrics
        # With --units base, NCU emits ns/byte. Convert to friendly units
        # to keep field-name semantics (duration_us, dram_bytes_*, tma_*_bytes).
        for metric, field in CALIB_METRICS.items():
            if metric in col_idx:
                raw = row[col_idx[metric]] if col_idx[metric] < len(row) else ""
                val = _strip_num(raw)
                if isinstance(val, (int, float)):
                    if field.endswith("_us"):
                        # gpu__time_duration.sum is in ns under --units base
                        val = val / 1000.0
                    # Note: dram_bytes_* / tma_*_bytes remain in raw bytes
                    # (field name is already in bytes, no conversion needed).
                entry[field] = val

        # Keep the full raw metric dict so callers can extract anything else
        raw_metrics = {}
        for h, i in col_idx.items():
            if h in FIXED_COLS:
                continue
            val = row[i] if i < len(row) else ""
            raw_metrics[h] = _strip_num(val)
        entry["_raw"] = raw_metrics

        kernels.append(entry)

    return {"source": source, "kernels": kernels}


def parse_ncu_rep(ncu_path: str, ncu_bin: str = NCU_BIN) -> dict:
    """Top-level entry: export + parse a single .ncu-rep file."""
    text = export_csv_text(ncu_path, ncu_bin=ncu_bin)
    return parse_ncu_csv(text, os.path.basename(ncu_path))


# ── CSV output ────────────────────────────────────────────────────────────────

# Ordered columns written to the flat output CSV (one row per kernel).
CSV_COLS = [
    "source",
    "id",
    "kernel_name",
    "block_size_str",
    "grid_size_str",
    "device",
    "cc",
    "duration_us",
    "lts_sectors",
    "lts_hit_rate_pct",
    "lts_miss_sectors",
    "lts_miss_sectors_promoted",
    "dram_bytes_read",
    "dram_bytes_write",
    "dram_throughput_pct",
    "tma_pipe_util_pct",
    "tma_pipe_elapsed_pct",
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
    "tensor_pipe_elapsed_pct",
    "fma_pipe_util_pct",
    "warps_active_pct",
    "occupancy_pct",
    "stall_barrier_pct",
    "stall_long_scoreboard_pct",
    "inst_executed",
    "ipc_active",
    "block_size",
    "grid_size",
    "thread_count",
    "regs_per_thread",
    "smem_dynamic_bytes",
    "waves_per_sm",
    "sm_count",
]


def flatten_to_csv_rows(results: list) -> list:
    """Flatten list of parsed reports into flat rows for CSV."""
    rows = []
    for r in results:
        for k in r.get("kernels", []):
            row = {col: "" for col in CSV_COLS}
            row["source"] = r["source"]
            for col in CSV_COLS:
                if col in k and k[col] is not None:
                    row[col] = k[col]
            rows.append(row)
    return rows


# ── Pretty print ──────────────────────────────────────────────────────────────

def _fmt(val, unit=""):
    if val is None or val == "":
        return "—"
    if isinstance(val, float):
        return f"{val:.3f}{unit}"
    return f"{val}{unit}"


def pretty_print(results: list):
    for r in results:
        print(f"\n{'='*70}")
        print(f"Report: {r['source']}")
        kernels = r.get("kernels", [])
        if not kernels:
            print(f"  (no kernels — {r.get('note', '')})")
            continue
        for k in kernels:
            name = k.get("kernel_name", "unknown")
            short = name[:80] + ("…" if len(name) > 80 else "")
            print(f"\n  [{k.get('id', '?')}] {short}")
            print(f"       Block/Grid : {k.get('block_size_str','?')} / {k.get('grid_size_str','?')}")
            print(f"       Duration   : {_fmt(k.get('duration_us'), ' us')}")
            print(f"       LTS hit    : {_fmt(k.get('lts_hit_rate_pct'), '%')}  "
                  f"(miss sectors: {_fmt(k.get('lts_miss_sectors'))})")
            print(f"       TMA pipe   : {_fmt(k.get('tma_pipe_util_pct'), '%')} active  "
                  f"| {_fmt(k.get('tma_pipe_elapsed_pct'), '%')} elapsed")
            if k.get("tma_ld_bytes") is not None or k.get("tma_st_bytes") is not None:
                print(f"       TMA traffic: ld={_fmt(k.get('tma_ld_bytes'))} B "
                      f"({_fmt(k.get('tma_ld_sectors'))} sectors)  "
                      f"st={_fmt(k.get('tma_st_bytes'))} B "
                      f"({_fmt(k.get('tma_st_sectors'))} sectors)")
                print(f"       TMA inst   : ld={_fmt(k.get('tma_ld_inst'))}  "
                      f"st={_fmt(k.get('tma_st_inst'))}  "
                      f"requests={_fmt(k.get('tma_requests'))}")
            print(f"       Tensor pipe: {_fmt(k.get('tensor_pipe_util_pct'), '%')} active  "
                  f"| {_fmt(k.get('tensor_pipe_elapsed_pct'), '%')} elapsed")
            print(f"       Warps act  : {_fmt(k.get('warps_active_pct'), '%')}  "
                  f"Occupancy: {_fmt(k.get('occupancy_pct'), '%')}")
            print(f"       Stalls     : barrier={_fmt(k.get('stall_barrier_pct'), '%')}  "
                  f"memdep={_fmt(k.get('stall_long_scoreboard_pct'), '%')}")
            if k.get("dram_bytes_read") is not None:
                print(f"       DRAM read  : {_fmt(k.get('dram_bytes_read'))} bytes  "
                      f"write: {_fmt(k.get('dram_bytes_write'))} bytes  "
                      f"throughput: {_fmt(k.get('dram_throughput_pct'), '%')}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parse NCU .ncu-rep files via `ncu --import --page raw --csv`. "
            "No external Python packages required."
        )
    )
    parser.add_argument("reports", nargs="+", help=".ncu-rep file(s) to parse")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON file (full metrics, all kernels)")
    parser.add_argument("--csv", default=None,
                        help="Output flat CSV (calibration metrics, one row/kernel)")
    parser.add_argument("--pretty", action="store_true",
                        help="Print human-readable summary to stdout")
    parser.add_argument("--ncu-bin", default=NCU_BIN,
                        help=f"Path to ncu binary (default: {NCU_BIN})")
    args = parser.parse_args()

    ncu_bin = args.ncu_bin

    results = []
    for path in args.reports:
        if not os.path.exists(path):
            print(f"[warn] {path} not found, skipping", file=sys.stderr)
            continue
        print(f"[parse] {path} ...", file=sys.stderr)
        try:
            r = parse_ncu_rep(path, ncu_bin=ncu_bin)
            n_k = len(r.get("kernels", []))
            print(f"[parse]   -> {n_k} kernel(s)", file=sys.stderr)
            results.append(r)
        except Exception as e:
            print(f"[error] {path}: {e}", file=sys.stderr)
            results.append({"source": os.path.basename(path), "error": str(e), "kernels": []})

    # JSON — strip _raw to keep file small; caller can pass --keep-raw if needed
    if args.output:
        out = []
        for r in results:
            entry = {k: v for k, v in r.items() if k != "kernels"}
            entry["kernels"] = [{k: v for k, v in kk.items() if k != "_raw"}
                                for kk in r.get("kernels", [])]
            out.append(entry)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[parse] JSON -> {args.output}", file=sys.stderr)

    # CSV
    if args.csv:
        rows = flatten_to_csv_rows(results)
        if rows:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"[parse] CSV  -> {args.csv}  ({len(rows)} rows)", file=sys.stderr)
        else:
            print("[warn] no kernel rows to write to CSV", file=sys.stderr)

    # Pretty
    if args.pretty or (not args.output and not args.csv):
        pretty_print(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
