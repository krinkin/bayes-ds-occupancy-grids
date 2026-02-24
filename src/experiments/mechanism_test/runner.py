"""Mechanism test: vary number of robots to show fusion-vs-update effect.

The key mechanism:
- In single-agent UPDATE, Dempster conflict K ~ 0 because one operand is a
  weak sensor observation. The normalization 1/(1-K) ~ 1, so Dempster behaves
  like Bayesian.
- In multi-robot FUSION, both operands are strong (well-observed grids).
  Conflict K grows at boundary cells where one grid says occupied and another
  says free. Normalization 1/(1-K) >> 1 amplifies agreed-upon evidence,
  sharpening boundaries.

The experiment holds TOTAL OBSERVATIONS constant while varying the number of
robots. With 1 robot doing 500 steps, we get pure update. With 5 robots doing
100 steps each, we get 4 fusions of partial maps -- and Dempster normalization
kicks in.

The predicted result:
- 1 robot: DS/TBM boundary_sharpness <= Bayesian (matches H-002)
- 3+ robots: DS/TBM boundary_sharpness > Bayesian (matches H-003)
"""

from __future__ import annotations

import copy
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.hybrid_pgo_ds.experiment.runner import (
    ArmResult,
    create_fusion_method,
    pignistic_masses,
    run_arm,
    setup_simulation,
)
from src.experiments.hybrid_pgo_ds.experiment.metrics import (
    compute_metrics,
)

# Sensor parameters matching BayesianFusion defaults.
# DS/TBM masses are derived from these via pignistic transform so that
# BetP(O) = sigmoid(l) exactly, ensuring a fair per-observation comparison.
_L_OCC: float = 2.0
_L_FREE: float = -0.5


@dataclass(frozen=True)
class MechanismResult:
    """Result for one robot-count configuration."""

    num_robots: int
    steps_per_robot: int
    bayesian_count_sharpness: float
    dstbm_sharpness: float
    sharpness_diff: float  # dstbm - bayesian_count
    bayesian_count_accuracy: float
    dstbm_accuracy: float
    accuracy_diff: float
    bayesian_count_sharpness_std: float = 0.0
    dstbm_sharpness_std: float = 0.0
    bayesian_count_accuracy_std: float = 0.0
    dstbm_accuracy_std: float = 0.0


def run_mechanism_test(
    base_config: dict[str, Any],
    robot_counts: list[int],
    total_steps: int = 500,
    seed: int = 42,
    num_runs: int = 5,
) -> list[MechanismResult]:
    """Run the mechanism test across different robot counts.

    Parameters
    ----------
    base_config : dict
        Base configuration (typically from a small.yaml).
    robot_counts : list[int]
        Number of robots to test (e.g., [1, 2, 3, 5]).
    total_steps : int
        Total observation steps, divided among robots.
    seed : int
        Base random seed.
    num_runs : int
        Number of independent runs to average over.

    Returns
    -------
    list[MechanismResult]
        One result per robot count.
    """
    results: list[MechanismResult] = []

    for num_robots in robot_counts:
        steps_per_robot = total_steps // num_robots

        config = _make_config(base_config, num_robots, steps_per_robot)

        bc_sharp_runs: list[float] = []
        ds_sharp_runs: list[float] = []
        bc_acc_runs: list[float] = []
        ds_acc_runs: list[float] = []

        for run_idx in range(num_runs):
            print(
                f"  [{num_robots} robots] run {run_idx + 1}/{num_runs}",
                flush=True,
            )
            run_seed = seed + run_idx * 1000 + num_robots

            (
                gt_env, rows, cols, sim_trajectories,
                _gt_traj, _drift_traj, _corr_traj,
                rendezvous_count, lidar_seeds, dynamic_env,
            ) = setup_simulation(config, run_seed)

            _m_occ = np.array(pignistic_masses(_L_OCC), dtype=np.float64)
            _m_free = np.array(pignistic_masses(_L_FREE), dtype=np.float64)

            for arm_name in ["bayesian_count", "dstbm"]:
                method = create_fusion_method(
                    arm_name,
                    m_occ=_m_occ if arm_name == "dstbm" else None,
                    m_free=_m_free if arm_name == "dstbm" else None,
                )
                lidar_rngs = [
                    np.random.default_rng(lidar_seeds[r])
                    for r in range(num_robots)
                ]

                with tempfile.TemporaryDirectory() as tmpdir:
                    log_path = Path(tmpdir) / f"{arm_name}.jsonl"

                    fused_prob, fused_grid, mem_bytes, avg_fus_ms, peak_mb = run_arm(
                        method=method,
                        gt_env=gt_env,
                        sim_trajectories=sim_trajectories,
                        num_robots=num_robots,
                        rows=rows,
                        cols=cols,
                        num_steps=steps_per_robot,
                        num_rays=int(config["lidar"]["num_rays"]),
                        max_range=float(config["lidar"]["max_range"]),
                        noise_stddev=float(config["lidar"]["noise_stddev"]),
                        resolution=float(config["environment"]["resolution"]),
                        height=float(config["environment"]["height"]),
                        metrics_requested=["cell_accuracy", "boundary_sharpness"],
                        log_path=log_path,
                        grid_snapshot_interval=0,
                        lidar_rngs=lidar_rngs,
                        active_robots=list(range(num_robots)),
                    )

                metrics = compute_metrics(
                    occupancy_prob=fused_prob,
                    ground_truth=gt_env,
                    requested=["cell_accuracy", "boundary_sharpness"],
                    raw_grid=fused_grid,
                    fusion_name=arm_name,
                )

                acc = metrics.get("cell_accuracy", float("nan"))
                sharp = metrics.get("boundary_sharpness", float("nan"))

                if arm_name == "bayesian_count":
                    if acc == acc:
                        bc_acc_runs.append(acc)
                    if sharp == sharp:
                        bc_sharp_runs.append(sharp)
                else:
                    if acc == acc:
                        ds_acc_runs.append(acc)
                    if sharp == sharp:
                        ds_sharp_runs.append(sharp)

        bc_sharp_mean = float(np.mean(bc_sharp_runs)) if bc_sharp_runs else float("nan")
        ds_sharp_mean = float(np.mean(ds_sharp_runs)) if ds_sharp_runs else float("nan")
        bc_acc_mean = float(np.mean(bc_acc_runs)) if bc_acc_runs else float("nan")
        ds_acc_mean = float(np.mean(ds_acc_runs)) if ds_acc_runs else float("nan")

        bc_sharp_std = float(np.std(bc_sharp_runs, ddof=1)) if len(bc_sharp_runs) > 1 else 0.0
        ds_sharp_std = float(np.std(ds_sharp_runs, ddof=1)) if len(ds_sharp_runs) > 1 else 0.0
        bc_acc_std = float(np.std(bc_acc_runs, ddof=1)) if len(bc_acc_runs) > 1 else 0.0
        ds_acc_std = float(np.std(ds_acc_runs, ddof=1)) if len(ds_acc_runs) > 1 else 0.0

        results.append(MechanismResult(
            num_robots=num_robots,
            steps_per_robot=steps_per_robot,
            bayesian_count_sharpness=bc_sharp_mean,
            dstbm_sharpness=ds_sharp_mean,
            sharpness_diff=ds_sharp_mean - bc_sharp_mean,
            bayesian_count_accuracy=bc_acc_mean,
            dstbm_accuracy=ds_acc_mean,
            accuracy_diff=ds_acc_mean - bc_acc_mean,
            bayesian_count_sharpness_std=bc_sharp_std,
            dstbm_sharpness_std=ds_sharp_std,
            bayesian_count_accuracy_std=bc_acc_std,
            dstbm_accuracy_std=ds_acc_std,
        ))

    return results


def _make_config(
    base_config: dict[str, Any],
    num_robots: int,
    steps_per_robot: int,
) -> dict[str, Any]:
    """Create a config variant for a specific robot count."""
    config = copy.deepcopy(base_config)

    config["robots"]["count"] = num_robots
    config["robots"]["trajectory_steps"] = steps_per_robot

    if num_robots == 1:
        config["robots"]["start_fractions"] = [0.0]
    else:
        config["robots"]["start_fractions"] = [
            i / num_robots for i in range(num_robots)
        ]

    config["pgo"]["enabled"] = num_robots > 1

    config["fusion"]["methods"] = ["bayesian_count", "dstbm"]
    config["metrics"] = ["cell_accuracy", "boundary_sharpness"]
    config["logging"]["grid_snapshot_interval"] = 0

    return config


def format_results_table(results: list[MechanismResult]) -> str:
    """Format results as a markdown table."""
    lines: list[str] = []
    lines.append("| Robots | Steps/robot | B+count sharp | DS/TBM sharp | "
                  "Diff (DS-B) | B+count acc | DS/TBM acc | Diff |")
    lines.append("|--------|-------------|---------------|-------------|"
                  "------------|------------|------------|------|")

    for r in results:
        lines.append(
            f"| {r.num_robots} | {r.steps_per_robot} | "
            f"{r.bayesian_count_sharpness:.4f} | {r.dstbm_sharpness:.4f} | "
            f"{r.sharpness_diff:+.4f} | "
            f"{r.bayesian_count_accuracy:.4f} | {r.dstbm_accuracy:.4f} | "
            f"{r.accuracy_diff:+.4f} |"
        )

    return "\n".join(lines)
