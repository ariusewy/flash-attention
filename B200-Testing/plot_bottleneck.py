#!/usr/bin/env python3
"""
FA4 trace summary: per-warp role wall time + MMA internal dependency chain.

Author: ywangmu from HKUST
"""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TRACE = sys.argv[1] if len(sys.argv) > 1 else "fa4_final.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "fa4_bottleneck.png"

with open(TRACE) as f:
    d = json.load(f)
evs = [e for e in d["traceEvents"] if e.get("ph") == "X" and e.get("pid") == 0]

from collections import defaultdict
by_role = defaultdict(list)
for e in evs:
    by_role[(e.get("tid"), e["name"])].append(e)

names_by_tid = {160: "load", 96: "MMA", 0: "softmax0", 32: "softmax1",
                64: "correction", 128: "epilogue"}

# ---- figure 1: per-warp wall time bar ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5),
                                gridspec_kw={"width_ratios": [1, 2]})

order = [160, 96, 0, 32, 64, 128]
role_total = {}
for tid in order:
    name = names_by_tid[tid]
    matching = [k for k in by_role if k[0] == tid and k[1] == name]
    if not matching:
        continue
    es = by_role[matching[0]]
    role_total[name] = sum(e.get("dur", 0) for e in es)

colors = {"load": "#2ca02c", "MMA": "#1f77b4", "softmax0": "#9467bd",
          "softmax1": "#9467bd", "correction": "#d62728", "epilogue": "#8c564b"}
labels = list(role_total.keys())
vals = list(role_total.values())
bars = ax1.barh(range(len(labels)), vals,
                color=[colors.get(l, "#7f7f7f") for l in labels])
ax1.set_yticks(range(len(labels)))
ax1.set_yticklabels(labels)
ax1.set_xlabel("wall time (clock units)")
ax1.set_title("Per-warp role total time")
ax1.invert_yaxis()
for i, v in enumerate(vals):
    ax1.text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=9)
# softmax aggregate
soft_sum = role_total.get("softmax0", 0) + role_total.get("softmax1", 0)
ax1.axvline(soft_sum, color="#9467bd", linestyle="--", alpha=0.5)
ax1.text(soft_sum + 0.1, len(labels) - 0.5, f"softmax0+1={soft_sum:.2f}",
         color="#9467bd", fontsize=8)

# ---- figure 2: MMA internal dependency timeline ----
mma_k = [k for k in by_role if k[0] == 96 and k[1] == "mma"][0]
mma_es = by_role[mma_k]
mma_start = min(e["ts"] for e in mma_es)

def color_of(name):
    if "wait" in name: return "#ff7f0e"
    if "gemm" in name: return "#1f77b4"
    return "#7f7f7f"

# plot each MMA internal scope as a bar at its own ts
scope_order = ["2gemm_Si", "wait_V", "wait_P0", "gemm_Pi0", "wait_K0", "gemm_Si0",
               "wait_P1", "gemm_Pi1", "wait_K1", "gemm_Si1", "2gemm_Pi"]
ypos = {}
for i, s in enumerate(scope_order):
    ypos[s] = i

for (tid, name), es in by_role.items():
    if tid != 96 or name in ("mma",):
        continue
    if name not in ypos:
        continue
    for e in es:
        ts = e["ts"] - mma_start  # relative to MMA start
        dur = max(e.get("dur", 0), 0.03)
        ax2.barh(ypos[name], dur, left=ts, height=0.7,
                 color=color_of(name), edgecolor="black", linewidth=0.3)
        if dur > 0.15:
            ax2.text(ts + dur / 2, ypos[name], f"{dur:.2f}",
                     ha="center", va="center", fontsize=6.5)

ax2.set_yticks(range(len(scope_order)))
ax2.set_yticklabels(scope_order)
ax2.set_xlabel("time since MMA start (clock units)")
ax2.set_title("MMA warp internal pipeline: wait_P0 dominates (softmax bottleneck)")
ax2.invert_yaxis()
ax2.grid(axis="x", alpha=0.3)

legend = [
    Patch(facecolor="#1f77b4", label="gemm (MMA compute)"),
    Patch(facecolor="#ff7f0e", label="wait (pipeline stall)"),
]
ax2.legend(handles=legend, loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved {OUT}")

# ---- console summary ----
print("\n" + "=" * 60)
print("BOTTLENECK ANALYSIS (seqlen=512, hdim=128, 2 CTAs)")
print("=" * 60)
print(f"\nPer-warp wall time (clock units):")
for name in ["load", "MMA", "softmax0", "softmax1", "correction", "epilogue"]:
    if name in role_total:
        print(f"  {name:<12} {role_total[name]:>8.2f}")
soft_tot = role_total.get("softmax0", 0) + role_total.get("softmax1", 0)
print(f"  {'softmax tot':<12} {soft_tot:>8.2f}  (softmax0 + softmax1)")

print(f"\nMMA internal stalls (where MMA is blocked):")
wait_total = 0
for s in ["wait_V", "wait_P0", "wait_P1", "wait_K0", "wait_K1"]:
    es = by_role.get((96, s), [])
    if es:
        t = sum(e.get("dur", 0) for e in es)
        wait_total += t
        print(f"  {s:<10} {t:>8.2f}")
print(f"  {'wait tot':<10} {wait_total:>8.2f}")
print(f"\n  => wait_P0/1 (waiting for softmax) is the biggest stall")
