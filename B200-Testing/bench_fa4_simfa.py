#!/usr/bin/env python3
# Author: ywangmu from HKUST
#
# FA4 benchmark for B200 profiling. Supports MHA and GQA.
# Three modes:
#   correctness  - compare FA4 output vs PyTorch SDPA reference
#   perf         - measure fwd (+bwd) TFLOPS over multiple iterations
#   run_once     - single fwd call, suitable for NCU profiling
#
# Case tuples: (batch, seqlen, heads_q, heads_kv, headdim)
#   MHA  -> heads_kv == heads_q
#   GQA  -> heads_kv <  heads_q  (e.g. Llama3 8B: heads_q=32, heads_kv=8)
#
# Usage:
#   python bench_fa4_simfa.py --mode correctness --cases small
#   python bench_fa4_simfa.py --mode perf --cases llama3_8b -o perf.json
#   python bench_fa4_simfa.py --mode perf --cases llama3_all -o perf_all.json
#   python bench_fa4_simfa.py --mode run_once --seqlen 2048 --heads 32 --heads-kv 8 --headdim 128
#
# For NCU profiling:
#   ncu --set full -o profile python bench_fa4_simfa.py --mode run_once \
#       --seqlen 2048 --heads 32 --heads-kv 8 --headdim 128 --no-backward

import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F

flash_attn_func = None
_flash_attn_source = "none"

# Try multiple import paths for flash_attn_func.
# Priority: FA4 pip package > FA4 editable > FA3 fallback
for _mod_path in [
    "flash_attn_interface",       # FA4 official pip package on B200
    "flash_attn.cute.interface",  # FA4 editable / cuTile path
    "flash_attn",                 # FA3 / FA2 fallback (MHA only)
]:
    try:
        _mod = __import__(_mod_path, fromlist=["flash_attn_func"])
        flash_attn_func = _mod.flash_attn_func
        _flash_attn_source = _mod_path
        break
    except (ImportError, AttributeError):
        continue

if flash_attn_func is None:
    print("[warn] flash_attn_func not found; correctness/perf modes will fail",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# Case presets
# Tuple format: (batch, seqlen, heads_q, heads_kv, headdim)
# ---------------------------------------------------------------------------
CASES = {
    # ---- smoke / debug ----
    "minimal": [
        (1, 128, 2, 2, 64),
    ],
    # ---- MHA baselines ----
    "small": [
        (1, 128, 2, 2, 64),
        (1, 128, 2, 2, 128),
        (2, 256, 4, 4, 64),
        (2, 256, 4, 4, 128),
    ],
    "medium": [
        (1,  512, 4, 4, 128),
        (2,  512, 8, 8, 128),
        (1, 1024, 8, 8, 128),
        (2, 1024, 8, 8, 128),
    ],
    "large": [
        (1, 2048,  8,  8, 128),
        (2, 2048,  8,  8, 128),
        (1, 4096,  8,  8, 128),
        (1, 8192,  8,  8, 128),
    ],
    # ---- Llama 3 GQA configs (from paper Table 6) ----
    # Llama3-8B:   H_Q=32, H_KV=8,  G=4,  D=128
    "llama3_8b": [
        (1,  1024, 32, 8, 128),
        (1,  2048, 32, 8, 128),
        (1,  4096, 32, 8, 128),
    ],
    # Llama3-70B:  H_Q=64, H_KV=8,  G=8,  D=128
    "llama3_70b": [
        (1, 2048, 64, 8, 128),
        (1, 4096, 64, 8, 128),
        (1, 8192, 64, 8, 128),
    ],
    # Llama3-405B: H_Q=128, H_KV=8, G=16, D=128
    "llama3_405b": [
        (1, 4096,  128, 8, 128),
        (1, 8192,  128, 8, 128),
        (1, 16384, 128, 8, 128),
    ],
    # ---- B200 NCU profiling shapes (5 representative shapes) ----
    # seqlen x headdim, MHA for simplicity in first pass
    "ncu_sweep": [
        (1,  512,  8, 8, 128),   # smoke baseline
        (1, 1024, 32, 8, 128),   # Llama3-8B  small seq
        (1, 2048, 32, 8, 128),   # Llama3-8B
        (1, 4096, 64, 8, 128),   # Llama3-70B-like
        (1, 8192, 128, 8, 128),  # Llama3-405B-like
    ],
}

# Combine all Llama3 configs into one preset
CASES["llama3_all"] = (
    CASES["llama3_8b"] + CASES["llama3_70b"] + CASES["llama3_405b"]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_inputs(batch, seqlen, heads_q, heads_kv, headdim,
                dtype=torch.bfloat16, device="cuda"):
    """Create Q [B,S,H_Q,D] and K,V [B,S,H_KV,D]."""
    q = torch.randn(batch, seqlen, heads_q,  headdim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen, heads_kv, headdim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen, heads_kv, headdim, dtype=dtype, device=device)
    return q, k, v


def pytorch_sdpa_ref(q, k, v, heads_q, heads_kv):
    """PyTorch SDPA reference. Handles GQA by expanding KV heads."""
    # q: [B,S,H_Q,D] -> [B,H_Q,S,D]
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)  # [B,H_KV,S,D]
    vt = v.transpose(1, 2)

    if heads_kv < heads_q:
        # Expand KV from [B,H_KV,S,D] to [B,H_Q,S,D]
        g = heads_q // heads_kv
        kt = kt.repeat_interleave(g, dim=1)
        vt = vt.repeat_interleave(g, dim=1)

    out = F.scaled_dot_product_attention(qt, kt, vt)
    return out.transpose(1, 2)  # [B,S,H_Q,D]


def run_fa4(q, k, v, heads_kv):
    """Call flash_attn_func. Passes num_heads_kv for GQA."""
    if flash_attn_func is None:
        raise RuntimeError("flash_attn_func not available")
    heads_q = q.shape[2]
    if heads_kv == heads_q:
        return flash_attn_func(q, k, v)
    else:
        # FA4 / FA3 API: pass num_heads_kv (or heads_kv) for GQA
        try:
            return flash_attn_func(q, k, v, num_heads_kv=heads_kv)
        except TypeError:
            # Older FA API uses keyword head_dim or similar; try positional
            return flash_attn_func(q, k, v)


def attn_flops(batch, seqlen, heads_q, headdim, fwd_ms, include_bwd=False):
    """Attention TFLOPS. FWD = 4 * B * H_Q * S^2 * D."""
    flops = 4 * batch * heads_q * seqlen * seqlen * headdim
    if include_bwd:
        flops *= 2.5
    return flops / (fwd_ms * 1e-3) / 1e12


def _case_label(b, s, hq, hkv, d):
    gqa = f"GQA(g={hq//hkv})" if hkv < hq else "MHA"
    return f"B={b} S={s} H_Q={hq} H_KV={hkv} D={d} [{gqa}]"


# ---------------------------------------------------------------------------
# Mode: correctness
# ---------------------------------------------------------------------------
def mode_correctness(args):
    print("=" * 60)
    print("FA4 Correctness Check (vs PyTorch SDPA)")
    print("=" * 60)

    cases = CASES[args.cases]
    all_pass = True

    for i, (b, s, hq, hkv, d) in enumerate(cases):
        print(f"\n--- Case {i}: {_case_label(b, s, hq, hkv, d)} ---")
        q, k, v = make_inputs(b, s, hq, hkv, d)

        try:
            out_fa4 = run_fa4(q, k, v, hkv)
        except Exception as e:
            print(f"  FA4 FAILED: {e}")
            all_pass = False
            continue

        out_ref = pytorch_sdpa_ref(q, k, v, hq, hkv)

        max_diff  = (out_fa4.float() - out_ref.float()).abs().max().item()
        mean_diff = (out_fa4.float() - out_ref.float()).abs().mean().item()
        cos_sim = F.cosine_similarity(
            out_fa4.float().flatten().unsqueeze(0),
            out_ref.float().flatten().unsqueeze(0),
        ).item()

        passed = cos_sim > 0.99
        status = "PASS" if passed else "FAIL"
        print(f"  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  "
              f"cos_sim={cos_sim:.6f}  [{status}]")
        if not passed:
            all_pass = False

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


# ---------------------------------------------------------------------------
# Mode: perf
# ---------------------------------------------------------------------------
def mode_perf(args):
    print("=" * 60)
    print("FA4 Performance Benchmark")
    print("=" * 60)

    cases = CASES[args.cases]
    results = []

    for i, (b, s, hq, hkv, d) in enumerate(cases):
        print(f"\n--- Case {i}: {_case_label(b, s, hq, hkv, d)} ---")
        q, k, v = make_inputs(b, s, hq, hkv, d)

        # Warmup
        for _ in range(args.warmup):
            _ = run_fa4(q, k, v, hkv)
        torch.cuda.synchronize()

        # Timed runs
        fwd_times, bwd_times = [], []
        for _ in range(args.iters):
            qg = q.clone().requires_grad_(True)
            kg = k.clone().requires_grad_(True)
            vg = v.clone().requires_grad_(True)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = run_fa4(qg, kg, vg, hkv)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fwd_ms = (t1 - t0) * 1000

            bwd_ms = 0.0
            if not args.no_backward:
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                out.sum().backward()
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                bwd_ms = (t3 - t2) * 1000

            fwd_times.append(fwd_ms)
            bwd_times.append(bwd_ms)

        avg_fwd = sum(fwd_times) / len(fwd_times)
        avg_bwd = sum(bwd_times) / len(bwd_times)
        fwd_tf  = attn_flops(b, s, hq, d, avg_fwd)
        tot_tf  = attn_flops(b, s, hq, d, avg_fwd + avg_bwd,
                             include_bwd=(avg_bwd > 0))

        result = {
            "case": i,
            "batch": b, "seqlen": s,
            "heads_q": hq, "heads_kv": hkv, "headdim": d,
            "gqa_group": hq // hkv,
            "fwd_ms":      round(avg_fwd, 4),
            "bwd_ms":      round(avg_bwd, 4),
            "fwd_tflops":  round(fwd_tf, 2),
            "total_tflops": round(tot_tf, 2),
        }
        results.append(result)
        print(f"  fwd={avg_fwd:.3f}ms  bwd={avg_bwd:.3f}ms  "
              f"fwd_TFLOPS={fwd_tf:.2f}  total_TFLOPS={tot_tf:.2f}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        g = r["gqa_group"]
        label = f"GQA(g={g})" if g > 1 else "MHA"
        print(f"  S={r['seqlen']} H_Q={r['heads_q']} H_KV={r['heads_kv']} "
              f"D={r['headdim']} [{label}]: "
              f"fwd={r['fwd_ms']:.3f}ms ({r['fwd_tflops']:.2f}T) "
              f"bwd={r['bwd_ms']:.3f}ms tot={r['total_tflops']:.2f}T")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


# ---------------------------------------------------------------------------
# Mode: run_once (for NCU profiling — single kernel invocation)
# ---------------------------------------------------------------------------
def mode_run_once(args):
    b   = args.batch
    s   = args.seqlen
    hq  = args.heads
    hkv = args.heads_kv
    d   = args.headdim

    print(f"run_once: {_case_label(b, s, hq, hkv, d)}")

    q, k, v = make_inputs(b, s, hq, hkv, d)
    out = run_fa4(q, k, v, hkv)
    print(f"  output: {out.shape}  dtype={out.dtype}")

    if not args.no_backward:
        out.sum().backward()
        print("  backward done")

    torch.cuda.synchronize()
    print("  cuda synchronized")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FA4 Benchmark for B200 / NCU Profiling")
    parser.add_argument("--mode", choices=["correctness", "perf", "run_once"],
                        required=True)
    parser.add_argument("--cases", default="minimal",
                        choices=list(CASES.keys()),
                        help="Case preset (see CASES dict for options)")
    # Per-shape args (used by run_once and as override for single-shape perf)
    parser.add_argument("--batch",    type=int, default=1)
    parser.add_argument("--seqlen",   type=int, default=512)
    parser.add_argument("--heads",    type=int, default=8,
                        help="Number of Q heads")
    parser.add_argument("--heads-kv", type=int, default=None,
                        dest="heads_kv",
                        help="Number of KV heads (default: same as --heads, i.e. MHA)")
    parser.add_argument("--headdim",  type=int, default=128)
    # Misc
    parser.add_argument("--no-backward", action="store_true",
                        help="Skip backward pass")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters",  type=int, default=10)
    parser.add_argument("--output", "-o", default=None,
                        help="Save perf results as JSON")

    args = parser.parse_args()

    # Default heads_kv to heads (MHA)
    if args.heads_kv is None:
        args.heads_kv = args.heads

    # Print environment
    print(f"PyTorch:  {torch.__version__}")
    print(f"CUDA:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:      {torch.cuda.get_device_name(0)}")
        print(f"SM count: {props.multi_processor_count}")
        print(f"Memory:   {props.total_memory / 1e9:.1f} GB")
        cc = f"{props.major}.{props.minor}"
        print(f"SM arch:  {cc}  {'(Blackwell sm_10x)' if props.major == 10 else ''}"
              f"{'(Blackwell sm_12x)' if props.major == 12 else ''}"
              f"{'(Hopper sm_90)' if props.major == 9 else ''}")
    print(f"flash_attn: {flash_attn_func is not None}  source={_flash_attn_source}")
    print()

    if args.mode == "correctness":
        return mode_correctness(args)
    elif args.mode == "perf":
        return mode_perf(args)
    elif args.mode == "run_once":
        return mode_run_once(args)


if __name__ == "__main__":
    sys.exit(main())
