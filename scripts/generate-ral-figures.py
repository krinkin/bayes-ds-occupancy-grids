#!/usr/bin/env python3
"""Generate RA-L-specific figures.

Produces:
- fig_violin_multirobot_bdry.pdf: compact violin for multi-robot boundary sharpness
- fig_realdata_combined.pdf: combined Intel Lab + Freiburg bar chart

Usage::

    python scripts/generate-ral-figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_FIG_DIR = Path("results/publication/figures")
_DPI = 300
_FONT_SIZE = 9

plt.rcParams.update({
    "font.family": "serif",
    "font.size": _FONT_SIZE,
    "axes.titlesize": _FONT_SIZE + 1,
    "axes.labelsize": _FONT_SIZE,
    "xtick.labelsize": _FONT_SIZE - 1,
    "ytick.labelsize": _FONT_SIZE - 1,
    "legend.fontsize": _FONT_SIZE - 1,
    "figure.dpi": _DPI,
    "savefig.dpi": _DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

_ARM_COLORS = {
    "bayesian_count": "#2166ac",
    "dstbm": "#b2182b",
}
_ARM_LABELS = {
    "bayesian_count": "Bayesian",
    "dstbm": "Dempster",
}


def _load_h003_runs(results_dir: Path, arm_names: list[str]) -> list[dict]:
    runs = []
    idx = 0
    while True:
        run_dir = results_dir / f"run_{idx}"
        if not run_dir.exists():
            break
        run_data = {}
        for arm in arm_names:
            p = run_dir / arm / "metrics.json"
            if p.exists():
                with p.open() as fh:
                    run_data[arm] = json.load(fh)
        if run_data:
            runs.append(run_data)
        idx += 1
    return runs


def _extract_values(runs: list[dict], arm: str, metric: str) -> list[float]:
    return [r[arm][metric] for r in runs if arm in r and metric in r[arm]]


# ---------------------------------------------------------------------------
# Fig 1: Compact multi-robot boundary sharpness violin
# ---------------------------------------------------------------------------

def figure_violin_multirobot_bdry(
    dyn_runs: list[dict],
) -> None:
    metric = "boundary_sharpness"
    arms = ["bayesian_count", "dstbm"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    data = []
    colors = []
    labels = []
    for arm in arms:
        vals = _extract_values(dyn_runs, arm, metric)
        data.append(vals)
        colors.append(_ARM_COLORS[arm])
        labels.append(_ARM_LABELS[arm])

    parts = ax.violinplot(
        data, positions=[0, 1], showmeans=True, showmedians=False,
        widths=0.7,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)
        pc.set_alpha(0.7)
    for key in ["cmeans", "cmins", "cmaxes", "cbars"]:
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(0.8)

    for i, vals in enumerate(data):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
        ax.scatter(
            [i + j for j in jitter], vals,
            c=colors[i], s=20, alpha=0.6, zorder=5, edgecolors="black",
            linewidths=0.3,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Boundary Sharpness")
    ax.set_title("Multi-robot (dynamic baseline, n=15)")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Annotate k/N
    bayes_mean = np.mean(data[0])
    ds_mean = np.mean(data[1])
    ax.annotate(
        f"k/N = 15/15\n$\\Delta$ = {bayes_mean - ds_mean:+.3f}",
        xy=(0.5, max(bayes_mean, ds_mean) + 0.005),
        ha="center", va="bottom", fontsize=_FONT_SIZE - 2,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", lw=0.5),
    )

    fig.tight_layout()
    fig.savefig(_FIG_DIR / "fig_violin_multirobot_bdry.png")
    fig.savefig(_FIG_DIR / "fig_violin_multirobot_bdry.pdf")
    plt.close(fig)
    print("  fig_violin_multirobot_bdry saved")


# ---------------------------------------------------------------------------
# Fig 3: Combined real-data (Intel Lab + Freiburg)
# ---------------------------------------------------------------------------

def figure_realdata_combined() -> None:
    intel_path = Path("results/intel-lab/results.json")
    freiburg_path = Path("results/freiburg-079/results.json")

    if not intel_path.exists() or not freiburg_path.exists():
        print("  fig_realdata_combined: SKIPPED (missing data)")
        return

    with intel_path.open() as fh:
        intel = json.load(fh)
    with freiburg_path.open() as fh:
        freiburg = json.load(fh)

    setups = ["single_robot", "split_2_robots", "split_4_robots"]
    setup_labels = ["1-source", "2-source", "4-source"]
    metrics = ["cell_accuracy", "boundary_sharpness", "brier_score"]
    metric_labels = ["Cell Accuracy", "Bdry Sharpness", "Brier Score"]

    datasets = [
        ("Intel Lab", intel),
        ("Freiburg 079", freiburg),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7, 4))

    for row, (ds_name, ds_data) in enumerate(datasets):
        for col, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]
            bc_vals = [ds_data[s]["bayesian_count"][metric] for s in setups]
            ds_vals = [ds_data[s]["dstbm"][metric] for s in setups]

            x = np.arange(len(setups))
            width = 0.32

            ax.bar(
                x - width / 2, bc_vals, width,
                color=_ARM_COLORS["bayesian_count"],
                label="Bayesian" if row == 0 and col == 0 else None,
                alpha=0.85, edgecolor="black", linewidth=0.3,
            )
            ax.bar(
                x + width / 2, ds_vals, width,
                color=_ARM_COLORS["dstbm"],
                label="Dempster" if row == 0 and col == 0 else None,
                alpha=0.85, edgecolor="black", linewidth=0.3,
            )

            ax.set_xticks(x)
            ax.set_xticklabels(setup_labels, fontsize=_FONT_SIZE - 2)
            if row == 0:
                ax.set_title(mlabel)
            if col == 0:
                ax.set_ylabel(ds_name, fontweight="bold")
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)

            # Tight y-axis around data
            all_vals = bc_vals + ds_vals
            ymin = min(all_vals)
            ymax = max(all_vals)
            margin = (ymax - ymin) * 0.15 if ymax > ymin else 0.01
            ax.set_ylim(ymin - margin, ymax + margin)

    fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=_FONT_SIZE - 1)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(_FIG_DIR / "fig_realdata_combined.png")
    fig.savefig(_FIG_DIR / "fig_realdata_combined.pdf")
    plt.close(fig)
    print("  fig_realdata_combined saved")


# ---------------------------------------------------------------------------

def main() -> None:
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    arm_names = ["bayesian", "bayesian_count", "dstbm"]

    print("Loading data...", flush=True)
    dyn_runs = _load_h003_runs(
        Path("results/hybrid-pgo-ds/pub-dynamic-baseline"), arm_names
    )
    print(f"  H-003 dynamic: {len(dyn_runs)} runs", flush=True)

    print("Generating RA-L figures...", flush=True)
    if dyn_runs:
        figure_violin_multirobot_bdry(dyn_runs)
    figure_realdata_combined()
    print(f"All RA-L figures saved to {_FIG_DIR}/", flush=True)


if __name__ == "__main__":
    main()
