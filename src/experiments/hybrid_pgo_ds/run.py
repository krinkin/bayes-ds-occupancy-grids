"""Main entry point for H-003 experiment.

Usage (from project root):
    python src/experiments/hybrid_pgo_ds/run.py --config configs/hybrid-pgo-ds/small.yaml
    python src/experiments/hybrid_pgo_ds/run.py --config configs/hybrid-pgo-ds/default.yaml

This script is a thin CLI wrapper.  All simulation and metric logic lives in
src/experiments/hybrid_pgo_ds/experiment/runner.py.

Increment 4: runs all configured fusion arms, computes all 7 metrics, writes
per-arm subdirectories + summary.csv.  Adds matplotlib trajectory and map plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when invoked as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # headless backend; safe on Pi 4 and CI
import matplotlib.pyplot as plt
import numpy as np

from src.experiments.hybrid_pgo_ds.experiment.config import load_config
from src.experiments.hybrid_pgo_ds.experiment.runner import (
    ArmResult,
    run_experiment,
    setup_simulation,
)

# Colour palette for per-robot trajectory lines
_ROBOT_COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def _save_map_plot(
    output_path: Path,
    ground_truth: np.ndarray,
    occupancy_prob: np.ndarray,
    arm_name: str,
    trajectories_gt: list,
    resolution: float,
    height: float,
) -> None:
    """Save a static matplotlib figure comparing ground truth and fused map."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax_gt = axes[0]
    ax_gt.imshow(ground_truth, cmap="gray_r", vmin=0, vmax=1, origin="upper")
    ax_gt.set_title("Ground Truth")
    ax_gt.set_xlabel("col")
    ax_gt.set_ylabel("row")

    ax_est = axes[1]
    im = ax_est.imshow(occupancy_prob, cmap="gray_r", vmin=0, vmax=1, origin="upper")
    ax_est.set_title(f"Fused Map ({arm_name})")
    ax_est.set_xlabel("col")
    ax_est.set_ylabel("row")
    fig.colorbar(im, ax=ax_est, label="P(occupied)")

    for idx, traj in enumerate(trajectories_gt):
        if not traj:
            continue
        colour = _ROBOT_COLOURS[idx % len(_ROBOT_COLOURS)]
        xs = [p.x / resolution for p in traj]
        ys = [(height - p.y) / resolution for p in traj]
        ax_est.plot(xs, ys, color=colour, linewidth=0.8, alpha=0.6,
                    label=f"robot {idx}")
        ax_est.plot(xs[0], ys[0], "o", color=colour, markersize=5)

    ax_est.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _save_trajectory_plot(
    output_path: Path,
    trajectories_gt: list,
    trajectories_drifted: list,
    trajectories_corrected: list | None,
    width: float,
    height: float,
) -> None:
    """Save trajectory overlay: ground truth, drifted, and PGO-corrected."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Multi-robot trajectories: true / drifted / PGO-corrected")
    ax.set_aspect("equal")

    for idx in range(len(trajectories_gt)):
        colour = _ROBOT_COLOURS[idx % len(_ROBOT_COLOURS)]

        def _xy(traj: list) -> tuple[list, list]:
            return [p.x for p in traj], [p.y for p in traj]

        gx, gy = _xy(trajectories_gt[idx])
        ax.plot(gx, gy, "-", color=colour, linewidth=1.5, alpha=0.9,
                label=f"R{idx} true")
        ax.plot(gx[0], gy[0], "o", color=colour, markersize=6)

        dx_, dy_ = _xy(trajectories_drifted[idx])
        ax.plot(dx_, dy_, "--", color=colour, linewidth=1.0, alpha=0.6,
                label=f"R{idx} drifted")

        if trajectories_corrected is not None:
            cx, cy = _xy(trajectories_corrected[idx])
            ax.plot(cx, cy, ":", color=colour, linewidth=1.2, alpha=0.85,
                    label=f"R{idx} corrected")

    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> int:
    """Run H-003 experiment: all configured fusion arms, all 7 metrics."""
    parser = argparse.ArgumentParser(
        description="H-003: Hybrid PGO + Cell Fusion Comparison"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    env_cfg = config["environment"]
    robots_cfg = config["robots"]
    fusion_cfg = config["fusion"]
    metrics_cfg = config["metrics"]

    base_seed = int(exp_cfg["seed"])
    output_dir = Path(exp_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    num_runs = int(exp_cfg.get("num_runs", 1))

    width = float(env_cfg["width"])
    height = float(env_cfg["height"])
    resolution = float(env_cfg["resolution"])
    num_robots = int(robots_cfg["count"])
    num_steps = int(robots_cfg["trajectory_steps"])
    arm_names: list[str] = list(fusion_cfg["methods"])
    metrics_requested: list[str] = list(metrics_cfg)
    pgo_enabled = bool(config["pgo"]["enabled"])

    print("H-003 Increment 4 -- Full Metrics + Experiment Runner")
    print(f"  Config     : {args.config}")
    print(f"  Room       : {width}m x {height}m @ {resolution}m/cell")
    print(f"  Robots     : {num_robots}  Steps: {num_steps}")
    print(f"  Runs       : {num_runs}")
    print(f"  PGO        : {'enabled' if pgo_enabled else 'disabled'}")
    print(f"  Fusion arms: {arm_names}")
    print(f"  Metrics    : {metrics_requested}")
    print(f"  Output     : {output_dir}")

    # ------------------------------------------------------------------
    # Generate trajectories for plotting (uses first run seed)
    # ------------------------------------------------------------------
    seed_rng = np.random.default_rng(base_seed)
    first_run_seed = int(seed_rng.integers(0, 2**31))

    (
        gt_env, rows, cols, sim_trajectories,
        gt_trajectories, drifted_trajectories, corrected_trajectories,
        rendezvous_count, _, _,
    ) = setup_simulation(config, first_run_seed)
    print(f"  Grid       : {rows} x {cols} cells")
    print(f"  Rendezvous : {rendezvous_count}")

    traj_plot_path = output_dir / "trajectories.png"
    _save_trajectory_plot(
        output_path=traj_plot_path,
        trajectories_gt=gt_trajectories,
        trajectories_drifted=drifted_trajectories,
        trajectories_corrected=corrected_trajectories,
        width=width,
        height=height,
    )
    print(f"  Traj plot  : {traj_plot_path}")

    # ------------------------------------------------------------------
    # Run the experiment (all arms, all metrics, JSON/CSV output)
    # ------------------------------------------------------------------
    arm_results: dict[str, ArmResult] = run_experiment(config, output_dir)

    # ------------------------------------------------------------------
    # Save per-arm comparison plots
    # ------------------------------------------------------------------
    for arm_name, result in arm_results.items():
        arm_dir = output_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        plot_path = arm_dir / "comparison.png"
        _save_map_plot(
            output_path=plot_path,
            ground_truth=gt_env,
            occupancy_prob=result.fused_prob,
            arm_name=arm_name,
            trajectories_gt=gt_trajectories,
            resolution=resolution,
            height=height,
        )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\nResults:")
    for arm_name, result in arm_results.items():
        print(f"  {arm_name}:")
        print(f"    memory_bytes : {result.memory_bytes}")
        print(f"    avg_fusion_ms: {result.avg_fusion_ms:.3f}")
        for metric_name, value in result.metrics.items():
            if isinstance(value, float) and value == value:
                print(f"    {metric_name:<22}: {value:.4f}")
            else:
                print(f"    {metric_name:<22}: N/A")

    print(f"\n  Summary CSV : {output_dir / 'summary.csv'}")
    print(f"  Metrics JSON: {output_dir / 'metrics.json'}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
