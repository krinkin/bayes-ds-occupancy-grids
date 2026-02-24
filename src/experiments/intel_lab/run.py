#!/usr/bin/env python3
"""Run Intel Lab validation experiment.

Usage::

    python src/experiments/intel_lab/run.py --config configs/intel-lab/default.yaml
    python src/experiments/intel_lab/run.py --data data/intel-lab/intel.gfs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.intel_lab.adapter import IntelLabAdapter
from src.experiments.intel_lab.loader import load_intel_lab
from src.experiments.intel_lab.runner import run_multi_robot_split, run_single_robot


def _load_config(config_path: str | None) -> dict:
    """Load config from YAML file or return defaults."""
    if config_path is not None:
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Intel Lab validation experiment")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--data", type=str, help="Path to Intel Lab data file")
    parser.add_argument("--resolution", type=float, help="Grid resolution (m/cell)")
    parser.add_argument("--max-range", type=float, help="Max usable laser range (m)")
    args = parser.parse_args(argv)

    # Load config, then override with CLI args
    cfg = _load_config(args.config)
    data_path = args.data or cfg.get("data", {}).get("path", "data/intel-lab/intel.gfs")
    resolution = args.resolution or cfg.get("grid", {}).get("resolution", 0.1)
    max_range = args.max_range or cfg.get("grid", {}).get("max_range", 8.0)
    train_fraction = cfg.get("grid", {}).get("train_fraction", 0.8)
    arm_names = cfg.get("experiment", {}).get("arms", ["bayesian", "bayesian_count", "dstbm"])
    metrics = cfg.get("experiment", {}).get("metrics", ["cell_accuracy", "boundary_sharpness", "brier_score"])
    multi_splits = cfg.get("experiment", {}).get("multi_robot_splits", [2, 4])
    output_dir = Path(cfg.get("output", {}).get("dir", "results/intel-lab"))
    l_occ = float(cfg.get("sensor", {}).get("l_occ", 2.0))
    l_free = float(cfg.get("sensor", {}).get("l_free", -0.5))
    clamp = float(cfg.get("bayesian", {}).get("l_max", 10.0))
    m_of_min = float(cfg.get("ds", {}).get("m_of_min", 0.0))

    print("Intel Lab Validation Experiment")
    print(f"  Data: {data_path}")
    print(f"  Resolution: {resolution} m/cell")
    print(f"  Max range: {max_range} m")
    print(f"  Train fraction: {train_fraction}")
    print()

    # Load dataset
    t0 = time.time()
    scans = load_intel_lab(data_path)
    print(f"  Loaded {len(scans)} laser scans ({time.time()-t0:.1f}s)")

    # Create adapter
    adapter = IntelLabAdapter(
        scans=scans,
        resolution=resolution,
        max_range=max_range,
        train_fraction=train_fraction,
    )
    gp = adapter.grid_params
    print(f"  Grid: {gp.rows} x {gp.cols} cells ({gp.rows * gp.cols} total)")
    print(f"  Train: {adapter.num_train_scans}, Test: {adapter.num_test_scans}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    # --- Single robot ---
    print("=== Single-Robot (all training scans) ===")
    t0 = time.time()
    single_results, single_paired = run_single_robot(adapter, arm_names, metrics, l_occ=l_occ, l_free=l_free, clamp=clamp, m_of_min=m_of_min)
    print(f"  ({time.time()-t0:.1f}s)")
    all_results["single_robot"] = {}
    for arm_name, result in single_results.items():
        print(f"  {arm_name}:")
        for m, v in result.metrics.items():
            val_str = f"{v:.6f}" if v == v else "NaN"
            print(f"    {m}: {val_str}")
        all_results["single_robot"][arm_name] = result.metrics
    if single_paired:
        all_results["single_robot"]["_paired_deltas"] = single_paired
        print("  paired_deltas (effective-mask):")
        for m, v in single_paired.items():
            print(f"    {m}: {v:+.6f}")

    # --- Multi-robot splits ---
    for num_robots in multi_splits:
        print(f"\n=== {num_robots}-Robot Split ===")
        t0 = time.time()
        multi_results, multi_paired = run_multi_robot_split(adapter, num_robots, arm_names, metrics, l_occ=l_occ, l_free=l_free, clamp=clamp, m_of_min=m_of_min)
        print(f"  ({time.time()-t0:.1f}s)")
        key = f"split_{num_robots}_robots"
        all_results[key] = {}
        for arm_name, result in multi_results.items():
            print(f"  {arm_name}:")
            for m, v in result.metrics.items():
                val_str = f"{v:.6f}" if v == v else "NaN"
                print(f"    {m}: {val_str}")
            all_results[key][arm_name] = result.metrics
        if multi_paired:
            all_results[key]["_paired_deltas"] = multi_paired
            print("  paired_deltas (effective-mask):")
            for m, v in multi_paired.items():
                print(f"    {m}: {v:+.6f}")

    # Save results
    with (output_dir / "results.json").open("w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")

    # Analysis summary
    _print_analysis(all_results, metrics)


def _print_analysis(
    all_results: dict[str, dict],
    metrics: list[str],
) -> None:
    """Print comparison analysis.

    Uses paired deltas (effective-mask intersection) when available,
    falling back to per-arm difference otherwise.
    """
    print("\n=== Analysis: DS/TBM vs Bayesian+count ===")
    print(f"{'Setup':<25} {'Metric':<25} {'B+count':>10} {'DS/TBM':>10} {'Diff':>12} {'Source':>8}")
    print("-" * 95)

    for setup_name, setup_results in all_results.items():
        bc = setup_results.get("bayesian_count", {})
        ds = setup_results.get("dstbm", {})
        paired = setup_results.get("_paired_deltas", {})
        if bc and ds:
            for metric in metrics:
                bc_val = bc.get(metric, float("nan"))
                ds_val = ds.get(metric, float("nan"))
                if bc_val == bc_val and ds_val == ds_val:
                    if metric in paired:
                        diff = paired[metric]
                        source = "paired"
                    else:
                        diff = ds_val - bc_val
                        source = "indep"
                    print(f"  {setup_name:<23} {metric:<25} {bc_val:>10.4f} {ds_val:>10.4f} {diff:>+12.4f} {source:>8}")


if __name__ == "__main__":
    main()
