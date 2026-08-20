"""FA4 cutez.trace per-tile runner for one attention shape.
Author: ywangmu from HKUST

Usage:
    python run_trace.py --seqlen S --heads-q HQ --heads-kv HKV --headdim D [--batch B] [--causal]
    python run_trace.py --shape ncu_sweep          # run all 5 ncu_sweep shapes
    python run_trace.py --shape fa3_align          # run all 9 fa3_align shapes
    python run_trace.py --shape all                # run all 14 shapes (default)

Output:
    results/traces/<label>/fa4_trace.json   (one Chrome trace file per shape)
    results/trace_summary.csv               (summary table, appended per run)

Each shape runs in a fresh subprocess so that TRACE_FA4_PATH / USE_TRACE_FA4
env vars (read at flash_attn.cute.interface import time) are set correctly
and the kernel is re-compiled with the right shape.

The trace captures per-tile pipeline stages of every warp role:
    load / mma(QK+PV internal) / softmax0 / softmax1 / correction / epilogue
Coarse *_tile scopes are filtered by hidden_scopes in interface.py; only
fine-grained dependency-revealing scopes (wait_*, *_compute, load_tma_*,
gemm_*, corr_wait_stats, ...) are emitted.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results" / "traces"

# ---------------------------------------------------------------------------
# Shape presets (mirrors bench_fa4_simfa.py CASES for trace collection)
# Tuple: (label, batch, seqlen, heads_q, heads_kv, headdim)
# ---------------------------------------------------------------------------
SHAPES = {
    "ncu_sweep": [
        ("s512_h8",    1,  512,   8, 8, 128),
        ("s1024_h32",  1, 1024,  32, 8, 128),
        ("s2048_h32",  1, 2048,  32, 8, 128),
        ("s4096_h64",  1, 4096,  64, 8, 128),
        ("s8192_h128", 1, 8192, 128, 8, 128),
    ],
    "fa3_align": [
        ("s512_h32",   1,  512, 32, 8, 128),
        ("s1024_h32",  1, 1024, 32, 8, 128),
        ("s2048_h32",  1, 2048, 32, 8, 128),
        ("s512_h64",   1,  512, 64, 8, 128),
        ("s1024_h64",  1, 1024, 64, 8, 128),
        ("s2048_h64",  1, 2048, 64, 8, 128),
        ("s512_h128",  1,  512,128, 8, 128),
        ("s1024_h128", 1, 1024,128, 8, 128),
        ("s2048_h128", 1, 2048,128, 8, 128),
    ],
}
SHAPES["all"] = SHAPES["ncu_sweep"] + SHAPES["fa3_align"]

# ---------------------------------------------------------------------------
# Single-shape worker (run in subprocess with clean env)
# ---------------------------------------------------------------------------
WORKER_CODE = '''
import json, os, sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("USE_TRACE_FA4", "1")
# TRACE_FA4_PATH and TRACE_FA4_LABEL are set by parent before exec
out_dir = Path(os.environ["TRACE_FA4_OUTDIR"])
trace_path = Path(os.environ["TRACE_FA4_PATH"])
out_dir.mkdir(parents=True, exist_ok=True)

import torch
from flash_attn.cute.interface import flash_attn_func, _CUTEZ_TRACE_SESSION

b  = int(os.environ["SHAPE_B"])
s  = int(os.environ["SHAPE_S"])
hq = int(os.environ["SHAPE_HQ"])
hkv= int(os.environ["SHAPE_HKV"])
hd = int(os.environ["SHAPE_HD"])
causal = os.environ.get("SHAPE_CAUSAL", "0") == "1"

print(f"  compiling+running FA4 forward: B={b} S={s} H_Q={hq} H_KV={hkv} D={hd} causal={causal}")
q = torch.randn(b, s, hq, hd, dtype=torch.bfloat16, device="cuda")
k = torch.randn(b, s, hkv, hd, dtype=torch.bfloat16, device="cuda")
v = torch.randn(b, s, hkv, hd, dtype=torch.bfloat16, device="cuda")
out, lse = flash_attn_func(q, k, v, causal=causal)
torch.cuda.synchronize()
print(f"  forward OK, out shape={tuple(out.shape)}")

# per-segment non-zero word count
SEG = {0:"softmax0",1:"softmax1",2:"correction",3:"mma",4:"epilogue",5:"load"}
sess = _CUTEZ_TRACE_SESSION
buf = sess.buffer_tensor.cpu().tolist()
sw, spb = sess.segment_words, sess.segments_per_block
nz_corr = 0
for blk in range(sess.total_blocks):
    for seg in range(spb):
        chunk = buf[(blk*spb+seg)*sw:(blk*spb+seg+1)*sw]
        nz = sum(1 for w in chunk if w != 0)
        if SEG.get(seg) == "correction":
            nz_corr += nz
print(f"  buffer segment_words={sw}, total_blocks={sess.total_blocks}, correction nz={nz_corr}")

with open(trace_path) as f:
    data = json.load(f)
events = data.get("traceEvents", data) if isinstance(data, dict) else data
scope_counts = Counter(e.get("name","") for e in events if e.get("ph")=="X")
total = sum(scope_counts.values())
top = ", ".join(f"{n}={c}" for n,c in scope_counts.most_common(6))
print(f"  total X events: {total}")
print(f"  top scopes: {top}")
print(f"  trace -> {trace_path}")

# write per-shape summary json (parent reads it)
summary = {
    "total_events": total,
    "correction_words": nz_corr,
    "verdict": "PASS" if nz_corr > 4 else "INCONCLUSIVE",
    "scope_counts": dict(scope_counts),
}
with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
'''


def run_one_subprocess(label, b, s, hq, hkv, hd, causal, skip_existing=True) -> dict:
    """Launch a subprocess for one shape (clean env, fresh kernel compile)."""
    out_dir = RESULTS_DIR / label
    trace_path = out_dir / "fa4_trace.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: skip shapes whose trace + summary already exist.
    if skip_existing and trace_path.exists() and (out_dir / "summary.json").exists():
        with open(out_dir / "summary.json") as f:
            summ = json.load(f)
        print(f"\n[{label}] SKIP (trace already exists, {summ['total_events']} events)")
        return {
            "label": label, "batch": b, "seqlen": s,
            "heads_q": hq, "heads_kv": hkv, "headdim": hd,
            "gqa_group": hq // hkv, "causal": int(causal),
            "total_events": summ["total_events"],
            "correction_words": summ["correction_words"],
            "verdict": summ["verdict"],
            "trace_file": f"results/traces/{label}/fa4_trace.json",
        }

    env = os.environ.copy()
    env["USE_TRACE_FA4"] = "1"
    env["TRACE_FA4_PATH"] = str(trace_path)
    env["TRACE_FA4_OUTDIR"] = str(out_dir)
    env["SHAPE_B"] = str(b)
    env["SHAPE_S"] = str(s)
    env["SHAPE_HQ"] = str(hq)
    env["SHAPE_HKV"] = str(hkv)
    env["SHAPE_HD"] = str(hd)
    env["SHAPE_CAUSAL"] = "1" if causal else "0"

    print(f"\n{'='*70}")
    print(f"  TRACE: {label}  B={b} S={s} H_Q={hq} H_KV={hkv} D={hd} causal={causal}")
    print(f"{'='*70}")
    print(f"  Q-tiles approx = {s // 128}")

    # pass worker code via stdin
    proc = subprocess.run(
        [sys.executable, "-c", WORKER_CODE],
        env=env, text=True,
    )
    rc = proc.returncode

    # load summary written by worker
    summary_path = out_dir / "summary.json"
    if rc == 0 and summary_path.exists():
        with open(summary_path) as f:
            summ = json.load(f)
    else:
        summ = {"total_events": 0, "correction_words": 0,
                "verdict": f"FAIL(rc={rc})", "scope_counts": {}}
        print(f"  [ERROR] subprocess exited rc={rc}")

    return {
        "label": label, "batch": b, "seqlen": s,
        "heads_q": hq, "heads_kv": hkv, "headdim": hd,
        "gqa_group": hq // hkv, "causal": int(causal),
        "total_events": summ["total_events"],
        "correction_words": summ["correction_words"],
        "verdict": summ["verdict"],
        "trace_file": f"results/traces/{label}/fa4_trace.json",
    }


def append_summary(rows):
    csv_path = HERE / "results" / "trace_summary.csv"
    fields = ["label", "batch", "seqlen", "heads_q", "heads_kv", "headdim",
              "gqa_group", "causal", "total_events", "correction_words",
              "verdict", "trace_file"]
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nsummary appended -> {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", choices=list(SHAPES.keys()),
                    default="all", help="shape preset (default: all=14 shapes)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--heads-q", type=int, default=8)
    ap.add_argument("--heads-kv", type=int, default=8)
    ap.add_argument("--headdim", type=int, default=128)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--label", help="override label for single shape")
    args = ap.parse_args()

    if args.shape:
        targets = SHAPES[args.shape]
        print(f"=== Running {len(targets)} shapes from preset '{args.shape}' ===")
        rows = []
        for i, (label, b, s, hq, hkv, hd) in enumerate(targets, 1):
            print(f"\n[{i}/{len(targets)}] >>> {label}")
            row = run_one_subprocess(label, b, s, hq, hkv, hd, args.causal)
            rows.append(row)
        append_summary(rows)
        print(f"\n=== DONE: {len(rows)} traces under {RESULTS_DIR} ===")
    else:
        label = args.label or f"s{args.seqlen}_h{args.heads_q}_d{args.headdim}"
        row = run_one_subprocess(label, args.batch, args.seqlen, args.heads_q,
                                 args.heads_kv, args.headdim, args.causal)
        append_summary([row])
        print(f"\n=== DONE: trace -> {RESULTS_DIR/label} ===")


if __name__ == "__main__":
    main()
