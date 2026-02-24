#!/usr/bin/env python3
"""Publication analysis: TOST equivalence, Cohen's d, Bayes factor.

Loads per-run results from H-002 and H-003 publication experiments,
computes equivalence statistics, and generates analysis report.

Usage::

    python scripts/analyze-publication.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.publication.stats import (
    bayes_factor_equivalence,
    cohens_d,
    holm_bonferroni,
    tost_equivalence,
    tost_margin_sensitivity,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_H002_DIR = Path("results/single-agent-tbm/publication")
_H003_DYN_DIR = Path("results/hybrid-pgo-ds/pub-dynamic-baseline")
_H003_NOISY_DIR = Path("results/hybrid-pgo-ds/pub-noisy-sensor")
_OUTPUT_DIR = Path("results/publication")

# Primary metrics for equivalence testing (B+count vs DS/TBM)
_H002_METRICS = ["cell_accuracy", "boundary_sharpness", "brier_score"]
_H003_METRICS = ["cell_accuracy", "boundary_sharpness", "map_entropy"]

# Equivalence margins (same as power-analysis.py)
_MARGINS = {
    "cell_accuracy": 0.02,
    "boundary_sharpness": 0.03,
    "brier_score": 0.01,
    "map_entropy": 0.02,
}

# Direction: True if higher is better
_HIGHER_IS_BETTER = {
    "cell_accuracy": True,
    "boundary_sharpness": True,
    "brier_score": False,
    "map_entropy": False,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_h002_runs(results_dir: Path) -> list[dict[str, Any]]:
    """Load H-002 per-run metrics_run*.json files."""
    runs: list[dict[str, Any]] = []
    idx = 0
    while True:
        path = results_dir / f"metrics_run{idx}.json"
        if not path.exists():
            break
        with path.open() as fh:
            runs.append(json.load(fh))
        idx += 1
    return runs


def _load_h003_runs(
    results_dir: Path,
    arm_names: list[str],
) -> list[dict[str, dict[str, float]]]:
    """Load H-003 per-run metrics from run_{idx}/{arm}/metrics.json."""
    runs: list[dict[str, dict[str, float]]] = []
    idx = 0
    while True:
        run_dir = results_dir / f"run_{idx}"
        if not run_dir.exists():
            break
        run_data: dict[str, dict[str, float]] = {}
        for arm in arm_names:
            metrics_path = run_dir / arm / "metrics.json"
            if metrics_path.exists():
                with metrics_path.open() as fh:
                    run_data[arm] = json.load(fh)
        if run_data:
            runs.append(run_data)
        idx += 1
    return runs


def _extract_paired(
    runs: list[dict[str, Any]],
    metric: str,
    arm_a: str = "bayesian_count",
    arm_b: str = "dstbm",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract paired metric arrays for two arms."""
    x = np.array([r[arm_a][metric] for r in runs], dtype=np.float64)
    y = np.array([r[arm_b][metric] for r in runs], dtype=np.float64)
    return x, y


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _analyze_experiment(
    name: str,
    runs: list[dict[str, Any]],
    metrics: list[str],
    lines: list[str],
) -> list[tuple[str, str, float]]:
    """Run full statistical analysis on one experiment.

    Returns
    -------
    list of (label, metric, raw_p_value)
        One entry per TOST test, for use in multiplicity correction.
    """
    n = len(runs)

    lines.append(f"## {name} (n={n})")
    lines.append("")

    # Summary statistics
    lines.append("### Descriptive Statistics")
    lines.append("")
    lines.append(f"| Metric | Mean B+count | Mean DS/TBM | Mean diff | SD(diff) |")
    lines.append("|--------|-------------|------------|-----------|----------|")

    for metric in metrics:
        x, y = _extract_paired(runs, metric)
        diff = x - y
        lines.append(
            f"| {metric} | {np.mean(x):.6f} | {np.mean(y):.6f} | "
            f"{np.mean(diff):+.6f} | {np.std(diff, ddof=1):.6f} |"
        )
    lines.append("")

    # TOST equivalence
    lines.append("### TOST Equivalence Testing")
    lines.append("")
    lines.append(
        f"| Metric | Margin | TOST p (raw) | 90% CI | Equiv? |"
    )
    lines.append("|--------|--------|--------------|--------|--------|")

    tost_entries: list[tuple[str, str, float]] = []
    for metric in metrics:
        x, y = _extract_paired(runs, metric)
        margin = _MARGINS.get(metric, 0.02)
        result = tost_equivalence(x, y, margin=margin)
        equiv_str = "YES" if result.equivalent else "no"
        lines.append(
            f"| {metric} | {margin} | {result.p_value:.4f} | "
            f"[{result.ci_lower:+.6f}, {result.ci_upper:+.6f}] | {equiv_str} |"
        )
        tost_entries.append((f"{name} / {metric}", metric, result.p_value))
    lines.append("")

    # Effect sizes
    lines.append("### Effect Sizes (Cohen's d)")
    lines.append("")
    lines.append(f"| Metric | d | 95% CI | |d| < 0.2? |")
    lines.append("|--------|---|--------|-----------|")

    for metric in metrics:
        x, y = _extract_paired(runs, metric)
        es = cohens_d(x, y)
        small = "yes" if abs(es.d) < 0.2 else f"NO ({abs(es.d):.2f})"
        lines.append(
            f"| {metric} | {es.d:+.4f} | [{es.ci_lower:+.4f}, {es.ci_upper:+.4f}] | "
            f"{small} |"
        )
    lines.append("")

    # Bayes factor
    lines.append("### Bayes Factor (BF01 for equivalence)")
    lines.append("")
    lines.append(f"| Metric | BF01 | Evidence |")
    lines.append("|--------|------|----------|")

    for metric in metrics:
        x, y = _extract_paired(runs, metric)
        margin = _MARGINS.get(metric, 0.02)
        bf = bayes_factor_equivalence(x, y, margin=margin)
        if bf.bf01 > 100:
            strength = "very strong"
        elif bf.bf01 > 10:
            strength = "strong"
        elif bf.bf01 > 3:
            strength = "moderate"
        elif bf.bf01 > 1:
            strength = "anecdotal"
        else:
            strength = "against equiv"
        if bf.bf01 == float("inf") or bf.bf01 > 1e6:
            bf_str = "> 10^6"
        else:
            bf_str = f"{bf.bf01:.2f}"
        lines.append(f"| {metric} | {bf_str} | {strength} |")
    lines.append("")

    # Directional analysis
    lines.append("### Directional Analysis (Is DS/TBM ever better?)")
    lines.append("")
    lines.append(f"| Metric | Mean diff (B-C) | Direction | Consistent? |")
    lines.append("|--------|----------------|-----------|-------------|")

    for metric in metrics:
        x, y = _extract_paired(runs, metric)
        diff = x - y
        mean_diff = float(np.mean(diff))
        higher_better = _HIGHER_IS_BETTER.get(metric, True)

        if higher_better:
            direction = "B+count better" if mean_diff > 0 else "DS/TBM better"
        else:
            direction = "B+count better" if mean_diff < 0 else "DS/TBM better"

        consistent = all(d > 0 for d in diff) or all(d < 0 for d in diff)
        lines.append(
            f"| {metric} | {mean_diff:+.6f} | {direction} | "
            f"{'yes ({0}/{0})'.format(n, n) if consistent else 'no'} |"
        )
    lines.append("")

    return tost_entries


def _multiplicity_section(
    all_tost_entries: list[tuple[str, str, float]],
    lines: list[str],
) -> None:
    """Append Holm-Bonferroni multiplicity correction section."""
    if not all_tost_entries:
        return

    labels = [label for label, _, _ in all_tost_entries]
    raw_p = [p for _, _, p in all_tost_entries]
    correction = holm_bonferroni(raw_p, labels=labels)

    lines.append("## Multiplicity Correction (Holm-Bonferroni)")
    lines.append("")
    lines.append(f"Correcting for {correction.n_tests} TOST comparisons.")
    lines.append("")
    lines.append("| Test | Raw p | Adjusted p | Equiv (raw)? | Equiv (adj)? |")
    lines.append("|------|-------|-----------|-------------|--------------|")

    for label, raw, adj in zip(
        correction.labels, correction.raw_p_values, correction.adjusted_p_values
    ):
        equiv_raw = "YES" if raw < 0.05 else "no"
        equiv_adj = "YES" if adj < 0.05 else "no"
        lines.append(
            f"| {label} | {raw:.4f} | {adj:.4f} | {equiv_raw} | {equiv_adj} |"
        )
    lines.append("")

    n_equiv_raw = sum(1 for p in correction.raw_p_values if p < 0.05)
    n_equiv_adj = sum(1 for p in correction.adjusted_p_values if p < 0.05)
    lines.append(
        f"Equivalent tests: {n_equiv_raw}/{correction.n_tests} (raw), "
        f"{n_equiv_adj}/{correction.n_tests} (adjusted)."
    )
    lines.append("")


def _sensitivity_section(
    all_experiments: list[tuple[str, list[dict[str, Any]], list[str]]],
    lines: list[str],
) -> None:
    """Append TOST margin sensitivity analysis section."""
    lines.append("## TOST Margin Sensitivity Analysis")
    lines.append("")
    lines.append(
        "Margins swept from 0.5x to 2x nominal. "
        "Breakpoint = smallest margin where equivalence fails (p >= 0.05)."
    )
    lines.append("")
    lines.append(
        "| Experiment | Metric | Nominal margin | Holds at 0.5x? | Breakpoint |"
    )
    lines.append("|-----------|--------|---------------|----------------|-----------|")

    for exp_name, runs, metrics in all_experiments:
        if not runs:
            continue
        for metric in metrics:
            x, y = _extract_paired(runs, metric)
            nominal = _MARGINS.get(metric, 0.02)
            sens = tost_margin_sensitivity(x, y, nominal_margin=nominal, metric=metric)
            holds = "YES" if sens.holds_at_half else "no"
            if sens.breakpoint_margin is None:
                bp_str = "none (holds to 2x)"
            else:
                bp_str = f"{sens.breakpoint_margin:.4f} ({sens.breakpoint_margin / nominal:.2f}x)"
            lines.append(
                f"| {exp_name} | {metric} | {nominal} | {holds} | {bp_str} |"
            )
    lines.append("")

    # Detailed sweep for key metric: cell_accuracy across experiments
    lines.append("### Detailed Sweep: cell_accuracy")
    lines.append("")
    lines.append("| Scale | Margin | H-002 p | H-003 dyn p | H-003 noisy p |")
    lines.append("|-------|--------|---------|------------|--------------|")

    # Collect per-experiment data for cell_accuracy
    exp_data: list[tuple[str, np.ndarray, np.ndarray] | None] = []
    for exp_name, runs, metrics in all_experiments:
        if runs and "cell_accuracy" in metrics:
            x, y = _extract_paired(runs, "cell_accuracy")
            exp_data.append((exp_name, x, y))
        else:
            exp_data.append(None)

    nominal_ca = _MARGINS["cell_accuracy"]
    for i in range(16):
        scale = 0.5 + 1.5 * i / 15
        margin = nominal_ca * scale
        row_parts = [f"| {scale:.2f}x | {margin:.4f}"]
        for item in exp_data:
            if item is None:
                row_parts.append("n/a")
            else:
                _, x, y = item
                r = tost_equivalence(x, y, margin=margin)
                row_parts.append(f"{r.p_value:.4f}")
        lines.append(" | ".join(row_parts) + " |")
    lines.append("")


def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Publication Experiment Analysis")
    lines.append("")
    lines.append("Equivalence of Bayesian+count vs DS/TBM across conditions.")
    lines.append("")

    all_tost_entries: list[tuple[str, str, float]] = []

    # Load H-002
    h002_runs = _load_h002_runs(_H002_DIR)
    if h002_runs:
        print(f"H-002: loaded {len(h002_runs)} runs")
        entries = _analyze_experiment(
            "H-002: Single-Agent (publication.yaml)", h002_runs, _H002_METRICS, lines
        )
        all_tost_entries.extend(entries)
    else:
        print(f"WARNING: No H-002 runs found in {_H002_DIR}")
        lines.append(f"**H-002 data not found in {_H002_DIR}**\n")

    # Load H-003 dynamic baseline
    h003_dyn_runs = _load_h003_runs(_H003_DYN_DIR, ["bayesian", "bayesian_count", "dstbm"])
    if h003_dyn_runs:
        print(f"H-003 dynamic: loaded {len(h003_dyn_runs)} runs")
        entries = _analyze_experiment(
            "H-003: Multi-Robot Dynamic Baseline (pub-dynamic-baseline.yaml)",
            h003_dyn_runs, _H003_METRICS, lines,
        )
        all_tost_entries.extend(entries)
    else:
        print(f"WARNING: No H-003 dynamic runs found in {_H003_DYN_DIR}")
        lines.append(f"**H-003 dynamic data not found in {_H003_DYN_DIR}**\n")

    # Load H-003 noisy sensor
    h003_noisy_runs = _load_h003_runs(_H003_NOISY_DIR, ["bayesian", "bayesian_count", "dstbm"])
    if h003_noisy_runs:
        print(f"H-003 noisy: loaded {len(h003_noisy_runs)} runs")
        entries = _analyze_experiment(
            "H-003: Multi-Robot Noisy Sensor (pub-noisy-sensor.yaml)",
            h003_noisy_runs, _H003_METRICS, lines,
        )
        all_tost_entries.extend(entries)
    else:
        print(f"WARNING: No H-003 noisy runs found in {_H003_NOISY_DIR}")
        lines.append(f"**H-003 noisy data not found in {_H003_NOISY_DIR}**\n")

    all_experiments = [
        ("H-002 single", h002_runs, _H002_METRICS),
        ("H-003 dynamic", h003_dyn_runs, _H003_METRICS),
        ("H-003 noisy", h003_noisy_runs, _H003_METRICS),
    ]

    # Cross-experiment comparison
    lines.append("## Cross-Experiment Summary")
    lines.append("")
    lines.append("| Experiment | Metric | TOST equiv? | d | Direction |")
    lines.append("|-----------|--------|-------------|---|-----------|")

    for exp_name, runs, metrics in all_experiments:
        if not runs:
            continue
        for metric in metrics:
            x, y = _extract_paired(runs, metric)
            margin = _MARGINS.get(metric, 0.02)
            tost = tost_equivalence(x, y, margin=margin)
            es = cohens_d(x, y)
            diff = x - y
            mean_diff = float(np.mean(diff))
            higher_better = _HIGHER_IS_BETTER.get(metric, True)
            if higher_better:
                direction = "B >= C" if mean_diff >= 0 else "C > B"
            else:
                direction = "B <= C" if mean_diff <= 0 else "C < B"

            lines.append(
                f"| {exp_name} | {metric} | "
                f"{'YES' if tost.equivalent else 'no'} | "
                f"{es.d:+.3f} | {direction} |"
            )
    lines.append("")

    # Multiplicity correction (Step 1.5)
    if all_tost_entries:
        _multiplicity_section(all_tost_entries, lines)

    # Margin sensitivity analysis (Step 1.6)
    _sensitivity_section(all_experiments, lines)

    # Write output
    output_path = _OUTPUT_DIR / "analysis.md"
    output_path.write_text("\n".join(lines) + "\n")
    print(f"\nAnalysis written to {output_path}")


if __name__ == "__main__":
    main()
