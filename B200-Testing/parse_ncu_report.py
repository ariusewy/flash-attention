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
    # Softmax-adjacent scalar/math instruction mix
    "smsp__inst_executed_op_mufu.sum":                                 "mufu_inst",
    "smsp__inst_executed_op_fadd_pred_on.sum":                         "fadd_inst",
    "smsp__inst_executed_op_fmul_pred_on.sum":                         "fmul_inst",
    "smsp__inst_executed_op_ffma_pred_on.sum":                         "ffma_inst",
    "smsp__inst_executed_op_conversion_pred_on.sum":                   "conversion_inst",
    # Warp / occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active":               "warps_active_pct",
    "smsp__warps_eligible.avg.per_cycle_active":                       "warps_eligible_per_cycle",
    "sm__maximum_warps_per_active_cycle_pct":                          "occupancy_pct",
    # Stall breakdown
    "smsp__average_warp_latency_issue_stalled_barrier.pct":            "stall_barrier_pct",
    "smsp__average_warp_latency_issue_stalled_membar.pct":             "stall_membar_pct",
    "smsp__average_warp_latency_issue_stalled_long_scoreboard.pct":    "stall_long_scoreboard_pct",
    "smsp__average_warp_latency_issue_stalled_short_scoreboard.pct":   "stall_short_scoreboard_pct",
    "smsp__average_warp_latency_issue_stalled_math_pipe_throttle.pct": "stall_math_pipe_throttle_pct",
    "smsp__average_warp_latency_issue_stalled_wait.pct":               "stall_wait_pct",
    "smsp__average_warp_latency_issue_stalled_mio_throttle.pct":       "stall_mio_throttle_pct",
    "smsp__average_warp_latency_issue_stalled_not_selected.pct":       "stall_not_selected_pct",
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


def _looks_like_csv(text: str) -> bool:
    """Cheap sanity check: does this look like NCU's --csv output?

    Reject NCU diagnostic text (==WARNING== / ==ERROR== / libprotobuf) that
    sometimes lands on stdout when `ncu --import` fails under sudo or when
    section files are missing.
    """
    if not text or not text.strip():
        return False
    head = text.lstrip().splitlines()[0].strip()
    if head.startswith("==WARNING==") or head.startswith("==ERROR=="):
        return False
    if head.startswith("[libprotobuf"):
        return False
    if head.startswith('"ID"') or head.startswith('"Process ID"'):
        return True
    if "Metric Name" in head and "Metric Value" in head:
        return True
    return False


def _discover_ncu_bin(explicit: str = "") -> str:
    """Resolve the same `ncu` the operator uses in their conda shell."""
    from shutil import which

    candidates = [
        explicit,
        os.environ.get("NCU_BIN", ""),
        which("ncu") or "",
        "/usr/local/cuda/bin/ncu",
        "/opt/nvidia/nsight-compute/ncu",
    ]
    for c in candidates:
        if not c:
            continue
        p = c if os.path.isabs(c) else (which(c) or "")
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return os.path.realpath(p)
    return explicit or "ncu"


def _find_cached_csv(ncu_path: str):
    """Return (csv_text, tag) if a pre-exported CSV sits next to the .ncu-rep."""
    rep = Path(ncu_path).resolve()
    d = rep.parent
    for name, tag in (
        ("profile_raw.csv", "cached_raw"),
        ("profile.csv", "cached_details"),
    ):
        p = d / name
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _looks_like_csv(text):
            return text, tag
    return None, None


def _discover_sections_path(ncu_bin: str) -> str:
    """Best-effort lookup of <ncu install>/sections so --import can
    re-evaluate rule results without warnings."""
    env = os.environ.get("NSIGHT_COMPUTE_SECTIONS_PATH", "").strip()
    if env and os.path.isdir(env):
        return env
    # Resolve `ncu` to its install root (typically .../bin/ncu).
    try:
        from shutil import which
        path = which(ncu_bin) or ""
    except Exception:
        path = ""
    if path:
        root = os.path.dirname(os.path.dirname(os.path.realpath(path)))
        candidate = os.path.join(root, "sections")
        if os.path.isdir(candidate):
            return candidate
    # Common install locations to try as a last resort.
    for cand in [
        "/usr/local/cuda/nsight-compute/sections",
        "/opt/nvidia/nsight-compute/sections",
    ]:
        if os.path.isdir(cand):
            return cand
    return ""


def _run_ncu_import(cmd: list, timeout: int = 120):
    """Invoke `ncu --import ...` and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout)
    except FileNotFoundError:
        raise RuntimeError(f"`{cmd[0]}` not found. Ensure ncu is on PATH.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ncu timed out: {' '.join(cmd)}")
    return result.returncode, result.stdout, result.stderr


def export_csv_text(ncu_path: str, ncu_bin: str = NCU_BIN) -> tuple:
    """Return (csv_text, page_used) for a .ncu-rep file.

    Strategy:
      1. Try `--page raw --csv --units base` (one row per kernel, base units).
      2. If that fails (newer NCU drops the `raw` page or `--units base`
         on some reports), fall back to `--page details --csv` (one row per
         metric per kernel; parser auto-detects layout).

    Raises RuntimeError with both attempts' stderr if every variant fails.
    """
    # NCU 2025.3 quirks:
    #   - `--units` is rejected on --import (prints "unrecognised option"
    #     to stderr but still exits 0 with empty stdout).
    #   - The in-report Section/Rule blob is re-evaluated on import; if
    #     NSIGHT_COMPUTE_SECTIONS_PATH is unset and the rule files aren't
    #     auto-discovered, ncu exits 1 *after* writing the full CSV body
    #     to stdout (protobuf "missing required fields" error in stderr).
    #
    # So:
    #   1. Try `--section-folder <auto-found path>` first to silence rule
    #      re-eval and get a clean exit 0.
    #   2. If that fails, fall back to the same command without the section
    #      flag and accept any output that looks like CSV regardless of
    #      returncode — the data we want is the CSV body, not the rule blob.
    sections_dir = _discover_sections_path(ncu_bin)
    section_flag = (["--section-folder", sections_dir] if sections_dir
                    else [])

    attempts = []
    if section_flag:
        attempts += [
            ("raw_withSections",
                [ncu_bin, "--import", ncu_path] + section_flag
                + ["--page", "raw", "--csv"]),
            ("details_withSections",
                [ncu_bin, "--import", ncu_path] + section_flag
                + ["--page", "details", "--csv"]),
        ]
    attempts += [
        ("raw",
            [ncu_bin, "--import", ncu_path, "--page", "raw", "--csv"]),
        ("details",
            [ncu_bin, "--import", ncu_path, "--page", "details", "--csv"]),
        ("raw_baseUnits",
            [ncu_bin, "--import", ncu_path, "--page", "raw", "--csv",
             "--units", "base"]),
        ("details_baseUnits",
            [ncu_bin, "--import", ncu_path, "--page", "details", "--csv",
             "--units", "base"]),
    ]

    errors = []
    for tag, cmd in attempts:
        rc, out, err = _run_ncu_import(cmd)
        # Accept whenever ncu emitted something that looks like CSV, even if
        # rc != 0. The export step writes the CSV before the post-processing
        # that may fail; the body is the source of truth.
        if _looks_like_csv(out):
            if rc != 0:
                # Keep the warning visible so the operator can fix the
                # underlying issue (typically NSIGHT_COMPUTE_SECTIONS_PATH).
                print(f"[parse] [{tag}] ncu rc={rc}, but stdout has CSV body "
                      f"— accepting. stderr head: "
                      f"{err.strip().splitlines()[0] if err.strip() else '(empty)'}",
                      file=sys.stderr)
            return out, tag
        errors.append(f"[{tag}] rc={rc} stderr={err.strip()[:300] or '(empty)'} "
                      f"stdout_head={out.strip()[:120] or '(empty)'}")

    raise RuntimeError("ncu --import produced no CSV for any variant:\n"
                       + "\n".join(errors))


def _normalise_to_base(val, unit: str):
    """Convert NCU adaptive units to ns / byte / cycle (== `--units base`).

    NCU 2025.x does not accept `--units` on `--import`, so the raw-page CSV
    always uses adaptive units (e.g. us, Mbyte, Kbyte, Ghz). Normalising here
    means downstream logic can treat every metric as if it had been exported
    with `--units base`.

    Unknown units (e.g. %, sector, inst, warp, register/thread, byte/block)
    pass through unchanged.
    """
    if not isinstance(val, (int, float)):
        return val
    u = unit.strip().lower()
    # time -> ns
    if u in ("us", "usecond"):                return val * 1e3
    if u in ("ms", "msecond"):                return val * 1e6
    if u in ("s",  "second"):                 return val * 1e9
    if u in ("ns", "nsecond"):                return val
    # bytes -> byte
    if u in ("byte",):                        return val
    if u in ("kbyte", "kib"):                 return val * 1024.0
    if u in ("mbyte", "mib"):                 return val * 1024.0 ** 2
    if u in ("gbyte", "gib"):                 return val * 1024.0 ** 3
    if u in ("tbyte", "tib"):                 return val * 1024.0 ** 4
    # throughput per-second -> bytes/s
    if u in ("kbyte/s", "kib/s"):             return val * 1024.0
    if u in ("mbyte/s", "mib/s"):             return val * 1024.0 ** 2
    if u in ("gbyte/s", "gib/s"):             return val * 1024.0 ** 3
    if u in ("tbyte/s", "tib/s"):             return val * 1024.0 ** 4
    # frequency -> Hz
    if u in ("khz",):                         return val * 1e3
    if u in ("mhz",):                         return val * 1e6
    if u in ("ghz",):                         return val * 1e9
    return val


def _finalise_kernel(entry: dict, metric_map: dict) -> dict:
    """Promote CALIB_METRICS into top-level fields and store raw map.

    `metric_map` must already be normalised to ns / byte / cycle.
    """
    for metric, field in CALIB_METRICS.items():
        if metric in metric_map:
            val = metric_map[metric]
            if isinstance(val, (int, float)) and field.endswith("_us"):
                # gpu__time_duration.sum lives in ns (after normalisation).
                val = val / 1000.0
            entry[field] = val
    entry["_raw"] = metric_map
    return entry


def _parse_raw_page(rows: list, source: str) -> dict:
    """NCU 'raw' page CSV: header + units row + one row per kernel."""
    if len(rows) < 3:
        return {"source": source, "kernels": [],
                "note": "raw page: <3 rows"}

    headers   = rows[0]
    units_row = rows[1]
    data_rows = rows[2:]
    col_idx   = {h: i for i, h in enumerate(headers)}

    kernels = []
    for row in data_rows:
        if not row or row[0].strip() == "":
            continue

        entry = {"_source": source}
        for header, field in FIXED_COLS.items():
            if header in col_idx and col_idx[header] < len(row):
                entry[field] = row[col_idx[header]].strip()
        if "id" in entry:
            try:
                entry["id"] = int(entry["id"])
            except (ValueError, TypeError):
                pass

        metric_map = {}
        for h, i in col_idx.items():
            if h in FIXED_COLS:
                continue
            raw_val = row[i] if i < len(row) else ""
            val = _strip_num(raw_val)
            unit = units_row[i] if i < len(units_row) else ""
            val = _normalise_to_base(val, unit)
            metric_map[h] = val

        kernels.append(_finalise_kernel(entry, metric_map))

    return {"source": source, "kernels": kernels}


def _parse_details_page(rows: list, source: str) -> dict:
    """NCU 'details' page CSV: long format with one row per (kernel, metric).

    Columns typically include:
      ID, Process ID, Process Name, Host Name, Kernel Name,
      Block Size, Grid Size, Device, CC, Section Name,
      Metric Name, Metric Unit, Metric Value
    """
    if len(rows) < 2:
        return {"source": source, "kernels": [],
                "note": "details page: empty"}

    headers = rows[0]
    col_idx = {h: i for i, h in enumerate(headers)}

    name_col  = col_idx.get("Metric Name")
    value_col = col_idx.get("Metric Value")
    unit_col  = col_idx.get("Metric Unit")
    if name_col is None or value_col is None:
        return {"source": source, "kernels": [],
                "note": "details page: missing Metric Name/Value columns"}

    # Group rows by (kernel ID, Kernel Name) so multi-kernel reports stay split.
    grouped = {}  # key -> dict(identity + metric_map)
    for row in rows[1:]:
        if not row:
            continue
        identity = {}
        for header, field in FIXED_COLS.items():
            if header in col_idx and col_idx[header] < len(row):
                identity[field] = row[col_idx[header]].strip()

        key = (identity.get("id", ""), identity.get("kernel_name", ""))
        bucket = grouped.setdefault(key, {"identity": identity,
                                          "metrics": {}})

        metric_name = row[name_col].strip() if name_col < len(row) else ""
        raw_val     = row[value_col].strip() if value_col < len(row) else ""
        if not metric_name:
            continue
        val  = _strip_num(raw_val)
        unit = row[unit_col].strip() if (unit_col is not None
                                          and unit_col < len(row)) else ""
        val = _normalise_to_base(val, unit)
        bucket["metrics"][metric_name] = val

    kernels = []
    for (kid, _), bucket in grouped.items():
        entry = {"_source": source}
        entry.update(bucket["identity"])
        if "id" in entry:
            try:
                entry["id"] = int(entry["id"])
            except (ValueError, TypeError):
                pass
        kernels.append(_finalise_kernel(entry, bucket["metrics"]))

    # Stable sort by ID when available
    kernels.sort(key=lambda k: (k.get("id") if isinstance(k.get("id"), int)
                                else 1 << 30))
    return {"source": source, "kernels": kernels}


def parse_ncu_csv(text: str, source: str) -> dict:
    """Parse NCU --csv text, auto-detecting raw vs details page layout."""
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return {"source": source, "kernels": [], "note": "empty csv"}

    headers = rows[0]
    if "Metric Name" in headers and "Metric Value" in headers:
        return _parse_details_page(rows, source)
    return _parse_raw_page(rows, source)


def parse_ncu_rep(ncu_path: str, ncu_bin: str = NCU_BIN) -> dict:
    """Top-level entry: parse a .ncu-rep (cached CSV first, else ncu --import)."""
    cached, tag = _find_cached_csv(ncu_path)
    if cached is not None:
        result = parse_ncu_csv(cached, os.path.basename(ncu_path))
        result["page_used"] = tag
        return result

    text, page_used = export_csv_text(ncu_path, ncu_bin=ncu_bin)
    result = parse_ncu_csv(text, os.path.basename(ncu_path))
    result["page_used"] = page_used
    return result


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

    ncu_bin = _discover_ncu_bin(args.ncu_bin)

    if os.geteuid() == 0:
        print(
            "[warn] Running as root/sudo: `ncu --import` often fails because "
            "PATH and NSIGHT_COMPUTE_SECTIONS_PATH differ from your conda shell.\n"
            "       Prefer: python3 parse_ncu_report.py ...   (no sudo)\n"
            "       Or export CSV first:\n"
            "         ncu --import profile.ncu-rep --page raw --csv > profile_raw.csv",
            file=sys.stderr,
        )

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
