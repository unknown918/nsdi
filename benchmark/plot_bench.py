#!/usr/bin/env python3
"""
SGLang Benchmark Plot - Unified Style
- No group distinction
- No grid lines
- Zero gaps between bars in a cluster
- Fixed axis disconnect
- Dynamic y-limit with top padding
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 0,
    "ytick.major.size": 4,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
})

BASE_DIR = Path(__file__).parent

CONFIGS = {
    "qwen3/sglang_v0.4.2/dp": {"label": "SGLang", "color": "#8F3B3E", "hatch": ""},
    "qwen3/megascale/a1e7-21": {"label": r"Megascale$^{*}$", "color": "#2B5B84", "hatch": ""},
    "qwen3/alloscale/a1e7-21": {"label": "AlloScale A1E7", "color": "#E6842A", "hatch": "///"},
    "qwen3/alloscale/a2e6": {"label": "AlloScale A2E6", "color": "#3C763D", "hatch": "\\\\\\\\"},
}

METRICS = [
    ("output_throughput", "Throughput (tok/s)", "Output Throughput"),
    ("mean_tpot_ms", "TPOT (ms)", "Mean TPOT"),
    ("p99_tpot_ms", "TPOT (ms)", "p99 TPOT"),
]


def load_data(cfg_dir: Path) -> dict:
    data = {}
    for fpath in sorted(cfg_dir.glob("bench_batch_size_*.jsonl")):
        m = re.search(r"bench_batch_size_(\d+)\.jsonl$", fpath.name)
        if not m:
            continue
        bs = int(m.group(1))
        runs = defaultdict(list)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    for key, _, _ in METRICS:
                        if key in rec:
                            runs[key].append(rec[key])
        data[bs] = dict(runs)
    return data


def main():
    all_data = {
        name: load_data(BASE_DIR / name)
        for name in CONFIGS
        if (BASE_DIR / name).is_dir()
    }

    if not all_data:
        raise RuntimeError(f"No valid config directories found under: {BASE_DIR}")

    all_bs = sorted({bs for cd in all_data.values() for bs in cd})
    if not all_bs:
        raise RuntimeError("No batch size data found.")

    cfg_names = list(CONFIGS.keys())

    bar_w = 0.38
    cluster_gap = 0.8

    n_bars = len(cfg_names)
    cluster_w = n_bars * bar_w
    x_centers = np.arange(len(all_bs)) * (cluster_w + cluster_gap)

    offsets = [(i - (n_bars - 1) / 2) * bar_w for i in range(n_bars)]
    cfg_offsets = dict(zip(cfg_names, offsets))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (metric_key, y_label, title) in zip(axes, METRICS):
        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_ylabel(y_label)
        ax.grid(False)

        ymax = 0.0
        oom_points = []

        for name in cfg_names:
            if name not in all_data:
                continue

            style = CONFIGS[name]
            offset = cfg_offsets[name]

            for bi, bs in enumerate(all_bs):
                xc = x_centers[bi] + offset
                vals = all_data[name].get(bs, {}).get(metric_key, [])

                if vals:
                    mean = float(np.mean(vals))
                    ymax = max(ymax, mean)

                    ax.bar(
                        xc, mean, bar_w,
                        color=style["color"],
                        hatch=style["hatch"],
                        edgecolor="#222222",
                        linewidth=0.8,
                        zorder=3
                    )
                else:
                    oom_points.append(xc)

        if "tpot" in metric_key:
            ymax = max(ymax, 200.0)
            ax.axhline(y=200, color="#CC0000", linestyle="--", linewidth=1.2, zorder=4)

        if ymax <= 0:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, ymax * 1.15)

        ymin, ymax_local = ax.get_ylim()
        y_text = ymin + (ymax_local - ymin) * 0.05

        for xc in oom_points:
            ax.text(
                xc, y_text * 0.2, "OOM",
                color="#CC0000",
                fontsize=7,
                ha="center",
                va="bottom",
                fontweight="bold",
                rotation=90,
                zorder=4
            )

        ax.spines["bottom"].set_position(("data", 0))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        ax.set_xticks(x_centers)
        ax.set_xticklabels([f"BS={b}" for b in all_bs])
        ax.set_xlim(x_centers[0] - cluster_w / 2 - 0.5,
                    x_centers[-1] + cluster_w / 2 + 0.5)

    legend_handles = [
        mpatches.Patch(
            facecolor=v["color"],
            hatch=v["hatch"],
            edgecolor="#222222",
            label=v["label"]
        )
        for v in CONFIGS.values()
    ]
    legend_handles.append(Line2D([0], [0], color="#CC0000", linestyle="--", label="SLO (200ms)"))

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.05),
        columnspacing=1.5
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out_pdf = BASE_DIR / "bench-e21.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved to {out_pdf}")


if __name__ == "__main__":
    main()
