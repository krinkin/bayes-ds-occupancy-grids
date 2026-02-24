"""CLI entry point for H-002 single-agent TBM vs Bayesian experiment.

Usage::

    python src/experiments/single_agent_tbm/run.py --config configs/single-agent-tbm/small.yaml

Supports multi-run experiments via the ``num_runs`` config key.  Each run
uses ``seed + run_index`` so results are deterministic and independent.
Per-run metrics are saved as ``metrics_run{i}.json``; a combined
``metrics.json`` contains the final run's results for backward compat.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path when invoked as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.hybrid_pgo_ds.experiment.config import load_config
from src.experiments.single_agent_tbm.runner import run_experiment


_METRIC_KEYS = [
    "cell_accuracy",
    "boundary_sharpness",
    "brier_score",
    "frontier_quality",
    "dynamic_detection",
    "mean_unc_dynamic",
    "mean_unc_stable",
    "detection_p50",
    "detection_p75",
    "detection_p90",
    "mean_uncertainty",
    "wall_clock_seconds",
]

_ARMS = ["bayesian", "bayesian_count", "dstbm", "yager"]


def _print_table(results: dict[str, Any]) -> None:
    """Print a metric comparison table for one run."""
    header = f"{'Metric':<22s}" + "".join(f"{a:>18s}" for a in _ARMS)
    print(f"\n{header}")
    print("-" * len(header))
    for key in _METRIC_KEYS:
        row = f"{key:<22s}"
        for arm in _ARMS:
            val = results[arm].get(key, float("nan"))
            row += f"{val:>18.4f}"
        print(row)


def _save_run_json(
    output_dir: Path,
    run_idx: int,
    seed: int,
    config: dict[str, Any],
    results: dict[str, Any],
) -> Path:
    """Save per-run metrics to JSON and return the file path."""
    summary: dict[str, Any] = {
        "experiment": config["experiment"]["name"],
        "run": run_idx,
        "seed": seed,
    }
    for arm in _ARMS:
        summary[arm] = {k: float(results[arm][k]) for k in _METRIC_KEYS}
    summary["num_steps"] = results["num_steps"]

    out_path = output_dir / f"metrics_run{run_idx}.json"
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    return out_path


def _save_summary_csv(
    output_dir: Path,
    all_results: list[dict[str, Any]],
) -> Path:
    """Write a flat CSV with one row per (run, arm) for easy analysis."""
    csv_path = output_dir / "summary.csv"
    fieldnames = ["run", "arm"] + _METRIC_KEYS
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for run_idx, results in enumerate(all_results):
            for arm in _ARMS:
                row: dict[str, Any] = {"run": run_idx, "arm": arm}
                for k in _METRIC_KEYS:
                    row[k] = results[arm].get(k, float("nan"))
                writer.writerow(row)
    return csv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="H-002 single-agent experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args(argv)

    config: dict[str, Any] = load_config(args.config)
    num_runs: int = config.get("num_runs", 1)
    base_seed: int = config["experiment"]["seed"]

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running H-002 experiment: {config['experiment']['name']}")
    print(f"  Runs: {num_runs}, base seed: {base_seed}")

    all_results: list[dict[str, Any]] = []

    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        config["experiment"]["seed"] = seed
        print(f"\n=== Run {run_idx + 1}/{num_runs} (seed={seed}) ===")

        results = run_experiment(config)
        all_results.append(results)

        _print_table(results)
        print(f"Steps: {results['num_steps']}")

        out_path = _save_run_json(output_dir, run_idx, seed, config, results)
        print(f"Saved: {out_path}")

    # Save combined CSV
    csv_path = _save_summary_csv(output_dir, all_results)
    print(f"\nSummary CSV: {csv_path}")

    # Backward-compat: also save last run as metrics.json
    compat_summary: dict[str, Any] = {
        "experiment": config["experiment"]["name"],
    }
    last = all_results[-1]
    for arm in _ARMS:
        compat_summary[arm] = {k: float(last[arm][k]) for k in _METRIC_KEYS}
    compat_summary["num_steps"] = last["num_steps"]
    compat_path = output_dir / "metrics.json"
    with compat_path.open("w") as fh:
        json.dump(compat_summary, fh, indent=2)
    print(f"Backward-compat metrics: {compat_path}")


if __name__ == "__main__":
    main()
