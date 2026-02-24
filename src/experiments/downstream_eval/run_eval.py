"""Downstream path planning evaluation runner.

Usage
-----
Standalone (two .npy files):

    python src/experiments/downstream_eval/run_eval.py \\
        --bayesian results/hybrid-pgo-ds/smoke/bayesian_prob.npy \\
        --ds       results/hybrid-pgo-ds/smoke/ds_prob.npy \\
        --output   results/downstream-eval/smoke/ \\
        --n-pairs  100 \\
        --seed     0

Integrated (generate grids from hybrid_pgo_ds experiment inline):

    python src/experiments/downstream_eval/run_eval.py \\
        --from-experiment \\
        --config   configs/hybrid-pgo-ds/small.yaml \\
        --output   results/downstream-eval/smoke/ \\
        --n-pairs  100 \\
        --seed     0

When --from-experiment is given, the script runs a single-seed H-003 simulation
(small config by default), extracts the final Bayesian and DS probability maps,
and evaluates path planning on them.

If neither --bayesian/--ds nor --from-experiment are given, the script
generates synthetic test grids with known small differences (useful for
verifying the evaluation pipeline without a prior experiment run).

Output
------
{output}/
  eval_results.json    -- aggregated metrics as JSON
  pair_results.csv     -- per-pair detail
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.experiments.downstream_eval.path_planning import (
    EvalResult,
    run_downstream_eval,
)


# ---------------------------------------------------------------------------
# Synthetic grid generation
# ---------------------------------------------------------------------------

def _make_synthetic_grids(
    rows: int = 50,
    cols: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a pair of synthetic Bayesian/DS probability maps.

    Simulates the small empirical differences reported in the paper
    (cell accuracy difference 0.001-0.011):

    - Both maps share an underlying room layout (walls as obstacles).
    - Bayesian map is built from log-odds with clamped saturation.
    - DS map is built from pignistic probabilities with added small noise
      representing the documented difference in boundary sharpness.
    - Interior and frontier cells are set to slightly different probabilities
      (difference <= 0.015) to model the reported metric gap.

    The two maps are intentionally similar so path planning results are
    representative of the real experimental conditions.

    Parameters
    ----------
    rows, cols:
        Grid dimensions.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (bayesian_prob, ds_prob)
        Two occupancy probability arrays of shape (rows, cols), dtype float32.
    """
    rng = np.random.default_rng(seed)

    # Build a simple room with outer walls and two interior obstacles
    ground_truth = np.zeros((rows, cols), dtype=np.float32)

    # Outer walls (2-cell thick)
    ground_truth[:2, :] = 1.0
    ground_truth[-2:, :] = 1.0
    ground_truth[:, :2] = 1.0
    ground_truth[:, -2:] = 1.0

    # Interior obstacle 1: horizontal bar
    r0, r1 = rows // 3, rows // 3 + 2
    c0, c1 = cols // 5, int(cols * 0.6)
    ground_truth[r0:r1, c0:c1] = 1.0

    # Interior obstacle 2: vertical bar
    r0, r1 = int(rows * 0.55), int(rows * 0.85)
    c0, c1 = int(cols * 0.4), int(cols * 0.4) + 2
    ground_truth[r0:r1, c0:c1] = 1.0

    # Bayesian: log-odds built from ground truth
    # Known cells: wall cells get high occupancy (0.95), free cells get low (0.05)
    # Unknown/frontier cells get 0.5
    bayesian_log = np.where(ground_truth == 1.0, 4.0, -2.0).astype(np.float32)
    # Add a 10% uncertain ring (1-cell border of observed area inside walls)
    # by setting some cells near walls to 0 (unknown)
    inner_mask = np.zeros((rows, cols), dtype=bool)
    inner_mask[2:-2, 2:-2] = True
    # Cells just inside walls are 'frontier' -- set to 0.0 log-odds
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(inner_mask)
    frontier_mask = inner_mask & ~eroded
    bayesian_log[frontier_mask] = 0.0

    bayesian_prob = (1.0 - 1.0 / (1.0 + np.exp(bayesian_log.astype(np.float64)))).astype(np.float32)

    # DS/TBM: pignistic probabilities with small systematic offset
    # BetP(O) = m_O + m_OF/2; produces slightly smoother boundaries.
    # We model this by adding calibrated Gaussian noise to the Bayesian prob
    # (sigma = 0.005, bounded away from 0/1) and applying a small boundary blur.
    noise = rng.normal(0.0, 0.005, size=(rows, cols)).astype(np.float32)
    ds_prob = np.clip(bayesian_prob + noise, 0.01, 0.99)

    # DS tends to have slightly less sharp boundaries: apply minimal smoothing
    # at wall-adjacent cells only (models the pignistic transform's m_OF/2 term)
    from scipy.ndimage import uniform_filter
    ds_smooth = uniform_filter(ds_prob, size=2).astype(np.float32)
    # Apply smooth only at frontier / near-wall cells
    blend_mask = (np.abs(bayesian_prob - 0.5) < 0.3)
    ds_prob = np.where(blend_mask, ds_smooth, ds_prob).astype(np.float32)

    return bayesian_prob, ds_prob


# ---------------------------------------------------------------------------
# Grid loading from experiment run
# ---------------------------------------------------------------------------

def _load_grids_from_npy(
    bayesian_path: Path,
    ds_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-saved occupancy probability grids from .npy files."""
    bayesian_prob = np.load(str(bayesian_path)).astype(np.float32)
    ds_prob = np.load(str(ds_path)).astype(np.float32)
    return bayesian_prob, ds_prob


def _load_grids_from_experiment(
    config_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single-seed H-003 experiment and return Bayesian / DS final maps.

    Parameters
    ----------
    config_path:
        Path to a hybrid-pgo-ds YAML config file.  The output directory
        from the config is used but can be overridden by --output.

    Returns
    -------
    (bayesian_prob, ds_prob)
    """
    import yaml  # type: ignore[import-untyped]
    from src.experiments.hybrid_pgo_ds.experiment.config import load_config
    from src.experiments.hybrid_pgo_ds.experiment.runner import (
        run_experiment,
        setup_simulation,
        create_fusion_method,
        run_arm,
        pignistic_masses,
    )
    import math

    print(f"[downstream_eval] Loading experiment config: {config_path}", flush=True)

    with config_path.open() as fh:
        raw_cfg = yaml.safe_load(fh)

    config = load_config(str(config_path))

    # We only need a single run to get the final maps
    base_seed = int(config["experiment"]["seed"])
    rng_seed = np.random.default_rng(base_seed)
    run_seed = int(rng_seed.integers(0, 2**31))

    print("[downstream_eval] Setting up simulation...", flush=True)

    (
        gt_env, rows, cols, sim_trajectories,
        gt_trajectories, drifted_trajectories, corrected_trajectories,
        rendezvous_count, lidar_seeds, dynamic_env,
    ) = setup_simulation(config, run_seed)

    env_cfg = config["environment"]
    robots_cfg = config["robots"]
    lidar_cfg = config["lidar"]

    num_robots = int(robots_cfg["count"])
    num_steps = int(robots_cfg["trajectory_steps"])
    num_rays = int(lidar_cfg["num_rays"])
    max_range = float(lidar_cfg["max_range"])
    noise_stddev = float(lidar_cfg["noise_stddev"])
    fp_rate = float(lidar_cfg.get("false_positive_rate", 0.0))
    fn_rate = float(lidar_cfg.get("false_negative_rate", 0.0))
    resolution = float(env_cfg["resolution"])
    height = float(env_cfg["height"])

    _L_OCC = 2.0
    _L_FREE = -0.5
    _m_occ = np.array(pignistic_masses(_L_OCC), dtype=np.float64)
    _m_free = np.array(pignistic_masses(_L_FREE), dtype=np.float64)
    _l_max = float(config.get("bayesian", {}).get("l_max", 10.0))
    _m_of_min = float(config.get("ds", {}).get("m_of_min", 0.0))

    results: dict[str, np.ndarray] = {}
    for arm_name in ["bayesian", "dstbm"]:
        method = create_fusion_method(
            arm_name,
            m_occ=_m_occ,
            m_free=_m_free,
            clamp=_l_max,
            m_of_min=_m_of_min,
        )
        print(f"[downstream_eval] Running arm: {arm_name}", flush=True)
        lidar_rngs = [np.random.default_rng(lidar_seeds[r]) for r in range(num_robots)]
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fused_prob, _, _, _, _ = run_arm(
                method=method,
                gt_env=gt_env,
                sim_trajectories=sim_trajectories,
                num_robots=num_robots,
                rows=rows,
                cols=cols,
                num_steps=num_steps,
                num_rays=num_rays,
                max_range=max_range,
                noise_stddev=noise_stddev,
                resolution=resolution,
                height=height,
                metrics_requested=["cell_accuracy"],
                log_path=Path(tmp) / f"{arm_name}.jsonl",
                grid_snapshot_interval=0,
                lidar_rngs=lidar_rngs,
                active_robots=list(range(num_robots)),
                dynamic_env=dynamic_env,
                fp_rate=fp_rate,
                fn_rate=fn_rate,
            )
        results[arm_name] = fused_prob

    return results["bayesian"], results["dstbm"]


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _write_results(output_dir: Path, eval_result: EvalResult) -> None:
    """Write eval_results.json and pair_results.csv to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregated JSON
    summary: dict = {
        "n_pairs": eval_result.n_pairs,
        "n_both_succeeded": eval_result.n_both_succeeded,
        "path_equivalence_rate": eval_result.path_equivalence_rate,
        "mean_clearance_diff_cells": eval_result.mean_clearance_diff,
        "mean_length_diff_cells": eval_result.mean_length_diff,
        "planning_success_rate_bayesian": eval_result.planning_success_rate_bayesian,
        "planning_success_rate_ds": eval_result.planning_success_rate_ds,
        "safety_critical_disagreement_rate": eval_result.safety_critical_disagreement_rate,
    }

    def _safe(v):
        if isinstance(v, float) and (v != v):  # NaN
            return None
        return v

    summary = {k: _safe(v) for k, v in summary.items()}

    json_path = output_dir / "eval_results.json"
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[downstream_eval] Wrote {json_path}", flush=True)

    # Per-pair CSV
    csv_path = output_dir / "pair_results.csv"
    fieldnames = [
        "pair_idx",
        "start_row", "start_col",
        "goal_row", "goal_col",
        "bayesian_found",
        "ds_found",
        "path_equivalent",
        "clearance_diff",
        "length_diff",
        "hausdorff",
        "safety_critical_disagreement",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, pr in enumerate(eval_result.pair_results):
            writer.writerow({
                "pair_idx": idx,
                "start_row": pr.start[0],
                "start_col": pr.start[1],
                "goal_row": pr.goal[0],
                "goal_col": pr.goal[1],
                "bayesian_found": pr.bayesian_found,
                "ds_found": pr.ds_found,
                "path_equivalent": pr.path_equivalent,
                "clearance_diff": pr.clearance_diff,
                "length_diff": pr.length_diff,
                "hausdorff": pr.hausdorff,
                "safety_critical_disagreement": pr.safety_critical_disagreement,
            })
    print(f"[downstream_eval] Wrote {csv_path}", flush=True)

    # Print summary table
    print("\n=== Downstream Path Planning Evaluation Summary ===", flush=True)
    print(f"  Pairs evaluated:                  {eval_result.n_pairs}", flush=True)
    print(f"  Pairs where both found path:      {eval_result.n_both_succeeded}", flush=True)
    rate = eval_result.path_equivalence_rate
    print(f"  Path equivalence rate:            {rate:.4f}" if rate == rate else
          "  Path equivalence rate:            N/A", flush=True)
    clr = eval_result.mean_clearance_diff
    print(f"  Mean clearance diff (cells):      {clr:.4f}" if clr == clr else
          "  Mean clearance diff (cells):      N/A", flush=True)
    ld = eval_result.mean_length_diff
    print(f"  Mean length diff (cells):         {ld:.4f}" if ld == ld else
          "  Mean length diff (cells):         N/A", flush=True)
    print(f"  Success rate (Bayesian):          {eval_result.planning_success_rate_bayesian:.4f}",
          flush=True)
    print(f"  Success rate (DS):                {eval_result.planning_success_rate_ds:.4f}",
          flush=True)
    scd = eval_result.safety_critical_disagreement_rate
    print(f"  Safety-critical disagreement rate:{scd:.4f}" if scd == scd else
          "  Safety-critical disagreement rate: N/A", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downstream path planning evaluation: Bayesian vs DS occupancy grids."
    )

    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument(
        "--bayesian",
        type=Path,
        metavar="PATH",
        help="Path to Bayesian occupancy probability .npy file.",
    )
    src_group.add_argument(
        "--from-experiment",
        action="store_true",
        help="Run a single H-003 simulation and extract grids inline.",
    )

    parser.add_argument(
        "--ds",
        type=Path,
        metavar="PATH",
        help="Path to DS/TBM occupancy probability .npy file (used with --bayesian).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        default=Path("configs/hybrid-pgo-ds/small.yaml"),
        help="Config file for --from-experiment mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="DIR",
        default=Path("results/downstream-eval/default"),
        help="Output directory for eval_results.json and pair_results.csv.",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=100,
        help="Number of (start, goal) pairs to evaluate. Default: 100.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for pair sampling. Default: 0.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Occupancy probability threshold for obstacle classification. Default: 0.5.",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=5.0,
        help="Minimum cells between start and goal. Default: 5.0.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-pair progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.from_experiment:
        bayesian_prob, ds_prob = _load_grids_from_experiment(args.config)
    elif args.bayesian is not None:
        if args.ds is None:
            print("Error: --ds is required when --bayesian is given.", file=sys.stderr)
            return 1
        print(f"[downstream_eval] Loading grids from .npy files...", flush=True)
        bayesian_prob, ds_prob = _load_grids_from_npy(args.bayesian, args.ds)
    else:
        print("[downstream_eval] No input specified -- generating synthetic test grids.", flush=True)
        bayesian_prob, ds_prob = _make_synthetic_grids()

    print(
        f"[downstream_eval] Grid shape: {bayesian_prob.shape}  "
        f"n_pairs={args.n_pairs}  seed={args.seed}",
        flush=True,
    )

    eval_result = run_downstream_eval(
        bayesian_prob=bayesian_prob,
        ds_prob=ds_prob,
        n_pairs=args.n_pairs,
        seed=args.seed,
        occ_threshold=args.threshold,
        min_start_goal_distance=args.min_distance,
        verbose=not args.quiet,
    )

    _write_results(args.output, eval_result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
