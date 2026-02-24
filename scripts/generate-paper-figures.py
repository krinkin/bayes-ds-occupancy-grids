#!/usr/bin/env python3
"""Generate publication-quality figures for the paper.

Produces:
- Figure 1: H-002 violin plots (single-agent, 3 arms x 3 metrics)
- Figure 2: H-003 violin plots (multi-robot, 2 conditions x 3 metrics)
- Figure 3: Forest plot of effect sizes (Cohen's d) across experiments
- Figure 4: Mechanism test - boundary sharpness vs number of robots
- Figure 5: Intel Lab comparison (real data vs simulation direction)

Usage::

    python scripts/generate-paper-figures.py
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

from src.publication.stats import cohens_d

# ---------------------------------------------------------------------------
# Output config
# ---------------------------------------------------------------------------

_FIG_DIR = Path("results/publication/figures")
_DPI = 300
_FONT_SIZE = 10

plt.rcParams.update({
    "font.size": _FONT_SIZE,
    "axes.titlesize": _FONT_SIZE + 1,
    "axes.labelsize": _FONT_SIZE,
    "xtick.labelsize": _FONT_SIZE - 1,
    "ytick.labelsize": _FONT_SIZE - 1,
    "legend.fontsize": _FONT_SIZE - 1,
    "figure.dpi": _DPI,
    "savefig.dpi": _DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

_ARM_COLORS = {
    "bayesian_count": "#55A868",
    "dstbm": "#C44E52",
}
_ARM_LABELS = {
    "bayesian_count": "Bayesian",
    "dstbm": "DS/TBM",
}

# Metric display names
_METRIC_LABELS = {
    "cell_accuracy": "Cell Accuracy",
    "boundary_sharpness": "Boundary Sharpness",
    "brier_score": "Brier Score",
    "map_entropy": "Map Entropy",
}


# ---------------------------------------------------------------------------
# Data loading (reused from analyze-publication.py)
# ---------------------------------------------------------------------------

def _load_h002_runs(results_dir: Path) -> list[dict]:
    runs = []
    idx = 0
    while True:
        path = results_dir / f"metrics_run{idx}.json"
        if not path.exists():
            break
        with path.open() as fh:
            runs.append(json.load(fh))
        idx += 1
    return runs


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
# Figure 1: H-002 violin plots
# ---------------------------------------------------------------------------

def figure_h002_violins(runs: list[dict]) -> None:
    metrics = ["cell_accuracy", "boundary_sharpness", "brier_score"]
    arms = ["bayesian_count", "dstbm"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    for ax, metric in zip(axes, metrics):
        data = []
        colors = []
        labels = []
        for arm in arms:
            vals = _extract_values(runs, arm, metric)
            data.append(vals)
            colors.append(_ARM_COLORS[arm])
            labels.append(_ARM_LABELS[arm])

        parts = ax.violinplot(data, positions=range(len(arms)), showmeans=True, showmedians=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        for key in ["cmeans", "cmins", "cmaxes", "cbars"]:
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(0.8)

        # Add individual points
        for i, vals in enumerate(data):
            ax.scatter([i] * len(vals), vals, c=colors[i], s=15, alpha=0.5, zorder=5)

        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(_METRIC_LABELS.get(metric, metric))
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("H-002: Single-Agent Comparison (n=15)", fontsize=_FONT_SIZE + 2, y=1.02)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "fig1_h002_violins.png")
    fig.savefig(_FIG_DIR / "fig1_h002_violins.pdf")
    plt.close(fig)
    print("  Figure 1: H-002 violins saved")


# ---------------------------------------------------------------------------
# Figure 2: H-003 violin plots
# ---------------------------------------------------------------------------

def figure_h003_violins(
    dyn_runs: list[dict],
    noisy_runs: list[dict],
) -> None:
    metrics = ["cell_accuracy", "boundary_sharpness", "map_entropy"]
    arms = ["bayesian_count", "dstbm"]
    conditions = [
        ("Dynamic Baseline", dyn_runs),
        ("Noisy Sensor", noisy_runs),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for row, (cond_name, runs) in enumerate(conditions):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            data = []
            colors = []
            labels = []
            for arm in arms:
                vals = _extract_values(runs, arm, metric)
                data.append(vals)
                colors.append(_ARM_COLORS[arm])
                labels.append(_ARM_LABELS[arm])

            if data and all(len(d) > 0 for d in data):
                parts = ax.violinplot(data, positions=range(len(arms)), showmeans=True, showmedians=False)
                for i, pc in enumerate(parts["bodies"]):
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.7)
                for key in ["cmeans", "cmins", "cmaxes", "cbars"]:
                    if key in parts:
                        parts[key].set_color("black")
                        parts[key].set_linewidth(0.8)

                for i, vals in enumerate(data):
                    ax.scatter([i] * len(vals), vals, c=colors[i], s=15, alpha=0.5, zorder=5)

            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels(labels, rotation=15, ha="right")
            if row == 0:
                ax.set_title(_METRIC_LABELS.get(metric, metric))
            ax.grid(axis="y", alpha=0.3)

            if col == 0:
                ax.set_ylabel(cond_name)

    fig.suptitle("H-003: Multi-Robot Comparison (n=15)", fontsize=_FONT_SIZE + 2, y=1.02)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "fig2_h003_violins.png")
    fig.savefig(_FIG_DIR / "fig2_h003_violins.pdf")
    plt.close(fig)
    print("  Figure 2: H-003 violins saved")


# ---------------------------------------------------------------------------
# Figure 3: Forest plot of effect sizes
# ---------------------------------------------------------------------------

def figure_forest_plot(
    h002_runs: list[dict],
    h003_dyn_runs: list[dict],
    h003_noisy_runs: list[dict],
) -> None:
    experiments = [
        ("H-002 single", h002_runs, ["cell_accuracy", "boundary_sharpness", "brier_score"]),
        ("H-003 dynamic", h003_dyn_runs, ["cell_accuracy", "boundary_sharpness", "map_entropy"]),
        ("H-003 noisy", h003_noisy_runs, ["cell_accuracy", "boundary_sharpness", "map_entropy"]),
    ]

    labels = []
    ds = []
    ci_lows = []
    ci_highs = []

    for exp_name, runs, metrics in experiments:
        if not runs:
            continue
        for metric in metrics:
            bc = np.array(_extract_values(runs, "bayesian_count", metric))
            tbm = np.array(_extract_values(runs, "dstbm", metric))
            if len(bc) == 0 or len(tbm) == 0:
                continue
            es = cohens_d(bc, tbm)
            labels.append(f"{exp_name}\n{_METRIC_LABELS.get(metric, metric)}")
            ds.append(es.d)
            ci_lows.append(es.ci_lower)
            ci_highs.append(es.ci_upper)

    n_items = len(labels)
    y_pos = list(range(n_items))

    fig, ax = plt.subplots(figsize=(8, max(4, n_items * 0.6)))

    for i in range(n_items):
        color = "#55A868" if ds[i] > 0 else "#C44E52"
        ax.barh(y_pos[i], ds[i], height=0.5, color=color, alpha=0.7)
        ax.plot([ci_lows[i], ci_highs[i]], [y_pos[i], y_pos[i]], "k-", linewidth=1.5)
        ax.plot(ds[i], y_pos[i], "ko", markersize=5)

    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(x=0.2, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(x=-0.2, color="gray", linewidth=0.5, linestyle=":")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d (Bayesian - DS/TBM)")
    ax.set_title("Effect Sizes Across Experiments")
    ax.grid(axis="x", alpha=0.3)

    # Annotations
    ax.text(0.2, -0.8, "small effect", ha="center", va="top", fontsize=7, color="gray")
    ax.text(-0.2, -0.8, "small effect", ha="center", va="top", fontsize=7, color="gray")

    fig.tight_layout()
    fig.savefig(_FIG_DIR / "fig3_forest_plot.png")
    fig.savefig(_FIG_DIR / "fig3_forest_plot.pdf")
    plt.close(fig)
    print("  Figure 3: Forest plot saved")


# ---------------------------------------------------------------------------
# Figure 4: Mechanism test
# ---------------------------------------------------------------------------

def figure_mechanism_test() -> None:
    results_path = Path("results/mechanism-test/results.json")
    if not results_path.exists():
        print("  Figure 4: SKIPPED (no mechanism test results)")
        return

    with results_path.open() as fh:
        data = json.load(fh)

    robots = [r["num_robots"] for r in data]
    bc_sharp = [r["bayesian_count_sharpness"] for r in data]
    ds_sharp = [r["dstbm_sharpness"] for r in data]
    bc_acc = [r["bayesian_count_accuracy"] for r in data]
    ds_acc = [r["dstbm_accuracy"] for r in data]

    # Per-run std (may be absent in old results)
    bc_sharp_std = [r.get("bayesian_count_sharpness_std", 0) for r in data]
    ds_sharp_std = [r.get("dstbm_sharpness_std", 0) for r in data]
    bc_acc_std = [r.get("bayesian_count_accuracy_std", 0) for r in data]
    ds_acc_std = [r.get("dstbm_accuracy_std", 0) for r in data]

    has_std = any(s > 0 for s in bc_sharp_std + ds_sharp_std)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(robots, bc_sharp, "o-", color=_ARM_COLORS["bayesian_count"],
             label=_ARM_LABELS["bayesian_count"], markersize=8)
    ax1.plot(robots, ds_sharp, "s-", color=_ARM_COLORS["dstbm"],
             label=_ARM_LABELS["dstbm"], markersize=8)
    if has_std:
        ax1.fill_between(robots,
                         [m - s for m, s in zip(bc_sharp, bc_sharp_std)],
                         [m + s for m, s in zip(bc_sharp, bc_sharp_std)],
                         color=_ARM_COLORS["bayesian_count"], alpha=0.15)
        ax1.fill_between(robots,
                         [m - s for m, s in zip(ds_sharp, ds_sharp_std)],
                         [m + s for m, s in zip(ds_sharp, ds_sharp_std)],
                         color=_ARM_COLORS["dstbm"], alpha=0.15)
    ax1.set_xlabel("Number of Robots")
    ax1.set_ylabel("Boundary Sharpness")
    ax1.set_title("Boundary Sharpness vs Robot Count")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xticks(robots)

    ax2.plot(robots, bc_acc, "o-", color=_ARM_COLORS["bayesian_count"],
             label=_ARM_LABELS["bayesian_count"], markersize=8)
    ax2.plot(robots, ds_acc, "s-", color=_ARM_COLORS["dstbm"],
             label=_ARM_LABELS["dstbm"], markersize=8)
    if has_std:
        ax2.fill_between(robots,
                         [m - s for m, s in zip(bc_acc, bc_acc_std)],
                         [m + s for m, s in zip(bc_acc, bc_acc_std)],
                         color=_ARM_COLORS["bayesian_count"], alpha=0.15)
        ax2.fill_between(robots,
                         [m - s for m, s in zip(ds_acc, ds_acc_std)],
                         [m + s for m, s in zip(ds_acc, ds_acc_std)],
                         color=_ARM_COLORS["dstbm"], alpha=0.15)
    ax2.set_xlabel("Number of Robots")
    ax2.set_ylabel("Cell Accuracy")
    ax2.set_title("Cell Accuracy vs Robot Count")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xticks(robots)

    fig.suptitle("Mechanism Test: Fusion vs Update Effect", fontsize=_FONT_SIZE + 2, y=1.02)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "fig4_mechanism_test.png")
    fig.savefig(_FIG_DIR / "fig4_mechanism_test.pdf")
    plt.close(fig)
    print("  Figure 4: Mechanism test saved")


# ---------------------------------------------------------------------------
# Figure 5: Intel Lab comparison
# ---------------------------------------------------------------------------

def figure_intel_lab() -> None:
    results_path = Path("results/intel-lab/results.json")
    if not results_path.exists():
        print("  Figure 5: SKIPPED (no Intel Lab results)")
        return

    with results_path.open() as fh:
        data = json.load(fh)

    setups = ["single_robot", "split_2_robots", "split_4_robots"]
    setup_labels = ["Single Robot", "2-Robot Split", "4-Robot Split"]
    metrics = ["cell_accuracy", "boundary_sharpness", "brier_score"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, metric in zip(axes, metrics):
        bc_vals = [data[s]["bayesian_count"][metric] for s in setups]
        ds_vals = [data[s]["dstbm"][metric] for s in setups]

        x = np.arange(len(setups))
        width = 0.35

        ax.bar(x - width/2, bc_vals, width, color=_ARM_COLORS["bayesian_count"],
               label=_ARM_LABELS["bayesian_count"], alpha=0.8)
        ax.bar(x + width/2, ds_vals, width, color=_ARM_COLORS["dstbm"],
               label=_ARM_LABELS["dstbm"], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(setup_labels, rotation=15, ha="right")
        ax.set_title(_METRIC_LABELS.get(metric, metric))
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Intel Lab Dataset Validation", fontsize=_FONT_SIZE + 2, y=1.02)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "fig5_intel_lab.png")
    fig.savefig(_FIG_DIR / "fig5_intel_lab.pdf")
    plt.close(fig)
    print("  Figure 5: Intel Lab comparison saved")


# ---------------------------------------------------------------------------
# Figure 6: Freiburg Building 079 comparison
# ---------------------------------------------------------------------------

def figure_freiburg() -> None:
    results_path = Path("results/freiburg-079/results.json")
    if not results_path.exists():
        print("  Figure 6: SKIPPED (no Freiburg results)")
        return

    with results_path.open() as fh:
        data = json.load(fh)

    setups = ["single_robot", "split_2_robots", "split_4_robots"]
    setup_labels = ["Single Robot", "2-Robot Split", "4-Robot Split"]
    metrics = ["cell_accuracy", "boundary_sharpness", "brier_score"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, metric in zip(axes, metrics):
        bc_vals = [data[s]["bayesian_count"][metric] for s in setups]
        ds_vals = [data[s]["dstbm"][metric] for s in setups]

        x = np.arange(len(setups))
        width = 0.35

        ax.bar(x - width/2, bc_vals, width, color=_ARM_COLORS["bayesian_count"],
               label=_ARM_LABELS["bayesian_count"], alpha=0.8)
        ax.bar(x + width/2, ds_vals, width, color=_ARM_COLORS["dstbm"],
               label=_ARM_LABELS["dstbm"], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(setup_labels, rotation=15, ha="right")
        ax.set_title(_METRIC_LABELS.get(metric, metric))
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Freiburg Building 079 Dataset Validation", fontsize=_FONT_SIZE + 2, y=1.02)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "fig6_freiburg.png")
    fig.savefig(_FIG_DIR / "fig6_freiburg.pdf")
    plt.close(fig)
    print("  Figure 6: Freiburg comparison saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    arm_names = ["bayesian", "bayesian_count", "dstbm"]

    print("Loading data...")
    h002_runs = _load_h002_runs(Path("results/single-agent-tbm/publication"))
    h003_dyn_runs = _load_h003_runs(Path("results/hybrid-pgo-ds/pub-dynamic-baseline"), arm_names)
    h003_noisy_runs = _load_h003_runs(Path("results/hybrid-pgo-ds/pub-noisy-sensor"), arm_names)

    print(f"  H-002: {len(h002_runs)} runs")
    print(f"  H-003 dynamic: {len(h003_dyn_runs)} runs")
    print(f"  H-003 noisy: {len(h003_noisy_runs)} runs")
    print()

    print("Generating figures...")
    if h002_runs:
        figure_h002_violins(h002_runs)
    if h003_dyn_runs and h003_noisy_runs:
        figure_h003_violins(h003_dyn_runs, h003_noisy_runs)
    if h002_runs and h003_dyn_runs and h003_noisy_runs:
        figure_forest_plot(h002_runs, h003_dyn_runs, h003_noisy_runs)
    figure_mechanism_test()
    figure_intel_lab()
    figure_freiburg()

    print(f"\nAll figures saved to {_FIG_DIR}/")


if __name__ == "__main__":
    main()
