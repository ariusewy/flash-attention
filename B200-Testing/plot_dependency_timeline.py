#!/usr/bin/env python3
"""
Plot an FA4 cutez.trace dependency timeline for one traced CTA/block.

This is intentionally closer to a trace-viewer zoom than to a summary chart:
it keeps real event ts/dur positions, separates the FA4 warp roles into lanes,
and draws inferred dependency arrows between producer stages and wait/consumer
stages in the same visible time window.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch


ROLE_BY_TID = {
    160: "load",
    0: "softmax0",
    32: "softmax1",
    96: "MMA",
    64: "correction",
    128: "epilogue",
}

LANE_Y = {
    "load": 5,
    "softmax0": 4,
    "softmax1": 3,
    "MMA": 2,
    "correction": 1,
    "epilogue": 0,
}

STAGE_STYLE = {
    "load_tma_Q": ("#8dd3c7", -0.18),
    "load_tma_K": ("#1b9e77", 0.00),
    "load_tma_V": ("#66a61e", 0.18),
    "wait_S": ("#d8b365", -0.24),
    "wait_Si": ("#c7a76c", -0.16),
    "softmax_compute": ("#8e44ad", 0.03),
    "store_P": ("#c77cff", 0.22),
    "2gemm_Si": ("#1f78b4", 0.30),
    "wait_V": ("#d95f02", -0.32),
    "wait_P0": ("#e31a1c", -0.22),
    "gemm_Pi0": ("#fb9a99", -0.10),
    "wait_K0": ("#fdbf6f", 0.02),
    "gemm_Si0": ("#377eb8", 0.14),
    "wait_P1": ("#b2182b", -0.22),
    "gemm_Pi1": ("#ef8a62", -0.10),
    "wait_K1": ("#fdbf6f", 0.02),
    "gemm_Si1": ("#4daf4a", 0.14),
    "2gemm_Pi": ("#984ea3", 0.30),
    "corr_wait_stats": ("#e7298a", 0.00),
    "corr_wait_O": ("#a65628", 0.20),
    "epi_wait_O": ("#ff7f00", 0.00),
}

ARROW_STYLE = {
    "P -> PV": "#7b3294",
    "K/V -> QK": "#008837",
    "P/stats -> correction": "#c51b7d",
    "O -> epilogue": "#a6611a",
}


def load_events(trace_path: Path, pid: int) -> list[dict]:
    with trace_path.open() as f:
        data = json.load(f)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    xs = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("pid") == pid
        and e.get("tid") in ROLE_BY_TID
        and "ts" in e
    ]
    for e in xs:
        e["end"] = e["ts"] + e.get("dur", 0.0)
        e["role"] = ROLE_BY_TID[e["tid"]]
    return sorted(xs, key=lambda e: (e["ts"], e.get("tid", 0), e["name"]))


def visible(events: list[dict], start: float, end: float) -> list[dict]:
    return [e for e in events if e["end"] >= start and e["ts"] <= end]


def y_of(e: dict) -> float:
    color, off = STAGE_STYLE.get(e["name"], ("#999999", 0.0))
    return LANE_Y[e["role"]] + off


def clipped_bar(e: dict, start: float, end: float) -> tuple[float, float]:
    x0 = max(e["ts"], start)
    x1 = min(e["end"], end)
    return x0 - start, max(x1 - x0, 0.02)


def latest_before(events: list[dict], names: set[str], tid: int | None, t: float, max_gap: float) -> dict | None:
    candidates = [
        e
        for e in events
        if e["name"] in names
        and (tid is None or e["tid"] == tid)
        and e["ts"] <= t
        and t - e["end"] <= max_gap
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e["end"])


def next_after(events: list[dict], names: set[str], tid: int | None, t: float, max_gap: float) -> dict | None:
    candidates = [
        e
        for e in events
        if e["name"] in names
        and (tid is None or e["tid"] == tid)
        and e["ts"] >= t
        and e["ts"] - t <= max_gap
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda e: e["ts"])


def add_arrow(ax, src: dict, dst: dict, start: float, label: str, rad: float = 0.12) -> None:
    color = ARROW_STYLE[label]
    x0 = src["end"] - start
    x1 = dst["ts"] - start
    y0 = y_of(src)
    y1 = y_of(dst)
    if x1 < 0 or x0 < 0:
        return
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="->",
        mutation_scale=8,
        linewidth=1.0,
        color=color,
        alpha=0.75,
        connectionstyle=f"arc3,rad={rad}",
        zorder=5,
    )
    ax.add_patch(arrow)


def infer_arrows(events: list[dict], win_events: list[dict], start: float, end: float, max_arrows: int) -> list[tuple[dict, dict, str]]:
    arrows: list[tuple[dict, dict, str]] = []

    # Softmax P handoff: store_P from softmax0/1 releases wait_P0/1, then gemm_Pi.
    for wait_name, gemm_name, soft_tid in [
        ("wait_P0", "gemm_Pi0", 0),
        ("wait_P1", "gemm_Pi1", 32),
    ]:
        waits = [e for e in win_events if e["tid"] == 96 and e["name"] == wait_name]
        for w in waits:
            producer = latest_before(events, {"store_P"}, soft_tid, w["end"], max_gap=1.2)
            consumer = next_after(events, {gemm_name}, 96, w["end"], max_gap=0.5)
            if producer and consumer and start <= producer["end"] <= end and start <= consumer["ts"] <= end:
                arrows.append((producer, consumer, "P -> PV"))
            if len(arrows) >= max_arrows:
                return arrows

    # Load handoff: K/V TMA issue scopes leading into MMA-side waits/compute.
    for wait_name, producer_name in [
        ("wait_V", "load_tma_V"),
        ("wait_K0", "load_tma_K"),
        ("wait_K1", "load_tma_K"),
    ]:
        waits = [e for e in win_events if e["tid"] == 96 and e["name"] == wait_name]
        for w in waits:
            producer = latest_before(events, {producer_name}, 160, w["end"], max_gap=3.0)
            consumer = next_after(events, {"gemm_Si0", "gemm_Si1"}, 96, w["end"], max_gap=0.6)
            if producer and consumer and start <= producer["ts"] <= end and start <= consumer["ts"] <= end:
                arrows.append((producer, consumer, "K/V -> QK"))
            if len(arrows) >= max_arrows:
                return arrows

    # Softmax stats/P are consumed by correction.
    for c in [e for e in win_events if e["tid"] == 64 and e["name"] == "corr_wait_stats"]:
        producer = latest_before(events, {"store_P", "softmax_compute"}, None, c["end"], max_gap=0.8)
        if producer and start <= producer["end"] <= end:
            arrows.append((producer, c, "P/stats -> correction"))
        if len(arrows) >= max_arrows:
            return arrows

    # Correction/MMA output handoff into epilogue wait.
    for epi in [e for e in win_events if e["tid"] == 128 and e["name"] == "epi_wait_O"]:
        producer = latest_before(events, {"corr_wait_O", "2gemm_Pi"}, None, epi["end"], max_gap=1.0)
        if producer and start <= producer["end"] <= end:
            arrows.append((producer, epi, "O -> epilogue"))
        if len(arrows) >= max_arrows:
            return arrows

    return arrows


def plot(trace_path: Path, out_path: Path, pid: int, start: float, duration: float, max_arrows: int) -> None:
    end = start + duration
    events = load_events(trace_path, pid)
    win_events = visible(events, start, end)

    fig, ax = plt.subplots(figsize=(18, 7.2))
    seen_label: set[str] = set()
    label_budget = defaultdict(int)

    for e in win_events:
        color, _ = STAGE_STYLE.get(e["name"], ("#999999", 0.0))
        x, w = clipped_bar(e, start, end)
        y = y_of(e)
        ax.broken_barh(
            [(x, w)],
            (y - 0.07, 0.14),
            facecolors=color,
            edgecolors="black",
            linewidth=0.25,
            alpha=0.88,
            zorder=3,
        )
        if w > duration * 0.018 and label_budget[e["name"]] < 3:
            ax.text(
                x + w / 2,
                y + 0.095,
                e["name"],
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=0,
                clip_on=True,
            )
            label_budget[e["name"]] += 1

    for src, dst, label in infer_arrows(events, win_events, start, end, max_arrows=max_arrows):
        add_arrow(ax, src, dst, start, label)

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.55, 5.55)
    ax.set_yticks([LANE_Y[r] for r in ["load", "softmax0", "softmax1", "MMA", "correction", "epilogue"]])
    ax.set_yticklabels(["load", "softmax0", "softmax1", "MMA", "correction", "epilogue"])
    ax.set_xlabel(f"time within selected window (trace units), absolute start = {start:.3f}")
    ax.set_title(
        f"FA4 cutez.trace dependency timeline: {trace_path.name}, pid/block {pid}, "
        f"window [{start:.3f}, {end:.3f}]"
    )
    ax.grid(axis="x", alpha=0.25)

    stage_legend = [
        Patch(facecolor="#1b9e77", label="load_tma_K/V/Q"),
        Patch(facecolor="#8e44ad", label="softmax_compute"),
        Patch(facecolor="#c77cff", label="store_P"),
        Patch(facecolor="#e31a1c", label="wait_P*"),
        Patch(facecolor="#fb9a99", label="gemm_Pi*"),
        Patch(facecolor="#fdbf6f", label="wait_K*/V"),
        Patch(facecolor="#377eb8", label="gemm_Si*"),
        Patch(facecolor="#e7298a", label="corr_wait_stats/O"),
        Patch(facecolor="#ff7f00", label="epi_wait_O"),
    ]
    arrow_legend = [
        Patch(facecolor=ARROW_STYLE["P -> PV"], label="arrow: store_P -> gemm_Pi"),
        Patch(facecolor=ARROW_STYLE["K/V -> QK"], label="arrow: load K/V -> gemm_Si"),
        Patch(facecolor=ARROW_STYLE["P/stats -> correction"], label="arrow: P/stats -> correction"),
        Patch(facecolor=ARROW_STYLE["O -> epilogue"], label="arrow: O -> epilogue"),
    ]
    ax.legend(
        handles=stage_legend + arrow_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"visible events: {len(win_events)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--pid", type=int, default=0, help="Trace pid/block to plot")
    ap.add_argument("--start", type=float, default=0.0, help="Absolute trace-window start")
    ap.add_argument("--duration", type=float, default=16.0, help="Window duration")
    ap.add_argument("--max-arrows", type=int, default=40)
    args = ap.parse_args()
    plot(args.trace, args.out, args.pid, args.start, args.duration, args.max_arrows)


if __name__ == "__main__":
    main()
