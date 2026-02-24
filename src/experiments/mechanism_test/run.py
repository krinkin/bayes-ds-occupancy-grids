#!/usr/bin/env python3
"""Run the fusion-vs-update mechanism test.

Varies number of robots (1, 2, 3, 5) with fixed total observations
to show how Dempster normalization affects boundary sharpness.

Usage::

    python src/experiments/mechanism_test/run.py [--config configs/hybrid-pgo-ds/small.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.hybrid_pgo_ds.experiment.config import load_config
from src.experiments.mechanism_test.runner import (
    format_results_table,
    run_mechanism_test,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mechanism test: fusion vs update")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hybrid-pgo-ds/small.yaml",
        help="Base config file",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=500,
        help="Total observation steps (divided among robots)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of independent runs per robot count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    robot_counts = [1, 2, 3, 5]

    print("Mechanism Test: Fusion vs Update (Dempster Sharpening)")
    print(f"  Config: {args.config}")
    print(f"  Total steps: {args.total_steps}")
    print(f"  Runs per config: {args.num_runs}")
    print(f"  Robot counts: {robot_counts}")
    print()

    results = run_mechanism_test(
        base_config=config,
        robot_counts=robot_counts,
        total_steps=args.total_steps,
        seed=args.seed,
        num_runs=args.num_runs,
    )

    print(format_results_table(results))
    print()

    # Analysis
    if len(results) >= 2:
        r1 = results[0]  # 1 robot
        r_last = results[-1]  # most robots

        print("Analysis:")
        print(f"  1 robot: DS-B sharpness diff = {r1.sharpness_diff:+.4f}")
        print(f"  {r_last.num_robots} robots: DS-B sharpness diff = {r_last.sharpness_diff:+.4f}")

        if r1.sharpness_diff <= 0 and r_last.sharpness_diff > 0:
            print("  PREDICTED CROSSOVER OBSERVED: DS/TBM goes from worse to better")
            print("  with more robots, confirming the fusion-vs-update mechanism.")
        elif r1.sharpness_diff <= 0 and r_last.sharpness_diff <= 0:
            print("  NO CROSSOVER: DS/TBM boundary sharpness remains <= Bayesian")
            print("  even with multiple robots. Mechanism may not apply at this scale.")
        else:
            print("  UNEXPECTED: DS/TBM already better at 1 robot.")

    # Save results
    output_dir = Path("results/mechanism-test")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = [
        {
            "num_robots": r.num_robots,
            "steps_per_robot": r.steps_per_robot,
            "bayesian_count_sharpness": r.bayesian_count_sharpness,
            "dstbm_sharpness": r.dstbm_sharpness,
            "sharpness_diff": r.sharpness_diff,
            "bayesian_count_accuracy": r.bayesian_count_accuracy,
            "dstbm_accuracy": r.dstbm_accuracy,
            "accuracy_diff": r.accuracy_diff,
            "bayesian_count_sharpness_std": r.bayesian_count_sharpness_std,
            "dstbm_sharpness_std": r.dstbm_sharpness_std,
            "bayesian_count_accuracy_std": r.bayesian_count_accuracy_std,
            "dstbm_accuracy_std": r.dstbm_accuracy_std,
        }
        for r in results
    ]

    with (output_dir / "results.json").open("w") as fh:
        json.dump(results_data, fh, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")

    # Save markdown
    with (output_dir / "analysis.md").open("w") as fh:
        fh.write("# Mechanism Test: Fusion vs Update\n\n")
        fh.write(format_results_table(results))
        fh.write("\n")
    print(f"Table saved to {output_dir / 'analysis.md'}")


if __name__ == "__main__":
    main()
