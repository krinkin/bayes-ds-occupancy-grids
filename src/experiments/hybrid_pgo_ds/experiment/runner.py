# TDP: Experiment runner for H-003 Increment 4
#
# Approach: Separate orchestration logic from the CLI entry point (run.py).
#   The runner module owns: per-arm simulation, multi-arm coordination,
#   multi-seed repetition (num_runs), metric collection, and file output.
#   run.py becomes a thin argparse wrapper that calls run_experiment().
#
# Design:
#   ArmResult -- typed result from one fusion arm
#   run_arm() -- run one fusion arm with one set of trajectories
#   run_experiment() -- full orchestration: shared setup + per-arm loop
#
#   num_runs: repeat the experiment with independently-derived seeds,
#   producing per-run files in {output_dir}/{arm_name}/run_{n}/ and
#   a multi-run mean in summary.csv.  Default 1 (single run).
#
# File layout for num_runs == 1:
#   {output_dir}/
#     trajectories.png
#     metrics.json     (arm -> final metrics)
#     summary.csv      (arm x metric)
#     {arm_name}/
#       run.jsonl
#       run_fault.jsonl
#       metrics.json
#       comparison.png
#
# File layout for num_runs > 1:
#   {output_dir}/
#     metrics.json     (arm -> mean final metrics across runs)
#     summary.csv      (arm x metric, mean values)
#     {arm_name}/
#       run_0/  run_1/  ...
#         run.jsonl, run_fault.jsonl, metrics.json
#       metrics.json   (mean across runs)
#
# Alternatives considered:
#   A class-based Runner -- adds indirection with no benefit for a single
#     experiment function; module-level functions are simpler to import and test.
#   Async per-arm execution -- complexity not justified; arms are IO-bound
#     only via matplotlib and JSONL writes.
"""Experiment orchestration for H-003: multi-arm, multi-run runner.

Provides:
- ArmResult: typed result from one fusion-arm run
- run_arm: simulate one arm and return its result
- run_experiment: full multi-arm, multi-run orchestration
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.hybrid_pgo_ds.experiment.logger import ExperimentLogger
from src.experiments.hybrid_pgo_ds.experiment.metrics import cell_accuracy, compute_metrics
from src.experiments.hybrid_pgo_ds.fusion import BayesianCountFusion, BayesianFusion, DSTBMFusion, YagerFusion
from src.experiments.hybrid_pgo_ds.fusion.base import FusionMethod
from src.experiments.hybrid_pgo_ds.simulation.environment import (
    DynamicEnvironment,
    generate_environment,
)
from src.experiments.hybrid_pgo_ds.simulation.lidar import cast_rays
from src.experiments.hybrid_pgo_ds.simulation.pgo import (
    apply_corrections,
    detect_rendezvous,
    solve_pgo,
)
from src.experiments.hybrid_pgo_ds.simulation.trajectory import (
    Pose,
    apply_drift,
    generate_multi_robot_trajectories,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pignistic_masses(log_odds: float) -> tuple[float, float, float]:
    """Compute DS/TBM mass function consistent with a Bayesian log-odds value.

    Derives (m_O, m_F, m_OF) such that BetP(O) = sigmoid(log_odds) exactly.

    Derivation::

        p = sigmoid(log_odds)
        m_O  = max(0, 2p - 1)
        m_F  = max(0, 1 - 2p)
        m_OF = 1 - |2p - 1|

    Verification: BetP(O) = m_O + m_OF / 2 = p (exact).

    Use this to derive matched DS/TBM sensor masses from Bayesian l_occ / l_free:
    - m_occ = pignistic_masses(l_occ)  -- mass for an occupied hit
    - m_free = pignistic_masses(l_free) -- mass for a free traversal

    Parameters
    ----------
    log_odds:
        Bayesian log-odds value for one observation.

    Returns
    -------
    tuple[float, float, float]
        (m_O, m_F, m_OF) mass values summing to 1.0.
    """
    p = 1.0 / (1.0 + math.exp(-log_odds))
    m_O = max(0.0, 2.0 * p - 1.0)
    m_F = max(0.0, 1.0 - 2.0 * p)
    m_OF = 1.0 - abs(2.0 * p - 1.0)
    return (m_O, m_F, m_OF)


def _get_git_commit() -> str:
    """Return the short git commit hash of HEAD, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ArmResult:
    """Result from one fusion-arm simulation run.

    Parameters
    ----------
    arm_name:
        Name of the fusion method ('bayesian', 'bayesian_count', 'dstbm').
    metrics:
        Final metric values for this arm (all 7 requested metrics).
    memory_bytes:
        Fused grid memory in bytes from FusionMethod.get_memory_bytes().
    avg_fusion_ms:
        Mean wall-clock time per fuse_grids call in milliseconds.
    fused_prob:
        Final occupancy probability map, shape (rows, cols), float32.
    peak_memory_mb:
        Peak memory allocated during fusion calls (tracemalloc), in MB.
        Zero when measure_resources is False.
    comm_bytes:
        Estimated communication bytes across all rendezvous events.
        Each rendezvous: each robot broadcasts its grid to the other robots.
        Zero when rendezvous_count == 0.
    """

    arm_name: str
    metrics: dict[str, float]
    memory_bytes: int
    avg_fusion_ms: float
    fused_prob: np.ndarray
    peak_memory_mb: float = 0.0
    comm_bytes: int = 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_fusion_method(
    name: str,
    m_occ: np.ndarray | None = None,
    m_free: np.ndarray | None = None,
    clamp: float = 10.0,
    m_of_min: float = 0.0,
) -> FusionMethod:
    """Return a FusionMethod for the given arm name.

    Parameters
    ----------
    name:
        One of 'bayesian', 'bayesian_count', 'dstbm', 'yager'.
    m_occ:
        Optional sensor mass for occupied hit passed to DSTBMFusion/YagerFusion.
        Derived from Bayesian l_occ via pignistic_masses() for fair comparison.
        Ignored for Bayesian arms.
    m_free:
        Optional sensor mass for free traversal passed to DSTBMFusion/YagerFusion.
        Derived from Bayesian l_free via pignistic_masses() for fair comparison.
        Ignored for Bayesian arms.
    clamp:
        Maximum absolute log-odds value for Bayesian arms. Default 10.0.
        Ignored for DS/TBM arms.
    m_of_min:
        Minimum ignorance mass floor for DS/TBM arms. Default 0.0 (no
        regularization). Ignored for Bayesian arms.

    Raises
    ------
    ValueError
        If the name is not recognised.
    """
    if name == "bayesian":
        return BayesianFusion(clamp=clamp)
    if name == "bayesian_count":
        return BayesianCountFusion(clamp=clamp)
    if name == "dstbm":
        return DSTBMFusion(m_occ=m_occ, m_free=m_free, m_of_min=m_of_min)
    if name == "yager":
        return YagerFusion(m_occ=m_occ, m_free=m_free, m_of_min=m_of_min)
    raise ValueError(
        f"Unknown fusion method: '{name}'. "
        "Expected one of: bayesian, bayesian_count, dstbm, yager."
    )


# ---------------------------------------------------------------------------
# Single-arm simulation
# ---------------------------------------------------------------------------

def run_arm(
    *,
    method: FusionMethod,
    gt_env: np.ndarray,
    sim_trajectories: list[list[Pose]],
    num_robots: int,
    rows: int,
    cols: int,
    num_steps: int,
    num_rays: int,
    max_range: float,
    noise_stddev: float,
    resolution: float,
    height: float,
    metrics_requested: list[str],
    log_path: Path,
    grid_snapshot_interval: int,
    lidar_rngs: list[np.random.Generator],
    active_robots: list[int] | None = None,
    experiment_metadata: dict[str, Any] | None = None,
    dynamic_env: DynamicEnvironment | None = None,
    fp_rate: float = 0.0,
    fn_rate: float = 0.0,
    remove_robot_at_step: int | None = None,
    remove_robot_idx: int | None = None,
    measure_resources: bool = False,
    dynamic_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Run one fusion-arm simulation and log step-by-step results.

    Parameters
    ----------
    method:
        Fusion method to use.
    gt_env:
        Binary ground-truth environment, shape (rows, cols).
    sim_trajectories:
        Pose trajectories for every robot (PGO-corrected when available).
    num_robots:
        Total number of robots (including those not in active_robots).
    active_robots:
        Indices of robots that participate.  None = all robots.
    lidar_rngs:
        Per-robot random generators for LiDAR noise.  Must be pre-seeded.

    Returns
    -------
    tuple
        (fused_prob, fused_grid, memory_bytes, avg_fusion_ms, peak_memory_mb)
        fused_prob: occupancy probability array (rows, cols), float32
        fused_grid: raw internal grid (for method-specific metrics)
        memory_bytes: fused_grid.nbytes
        avg_fusion_ms: mean time per fuse_grids call (0 if single robot)
        peak_memory_mb: peak tracemalloc allocation during fusion (0 if measure_resources=False)
    """
    # Work on a local copy so mid-run removal does not affect the caller's list
    active_robots = list(active_robots) if active_robots is not None else list(range(num_robots))

    grids = {r: method.create_grid(rows, cols) for r in active_robots}

    total_fusion_s = 0.0
    fusion_count = 0
    tracemalloc_peak_bytes = 0

    _per_step = [
        m for m in metrics_requested
        if m not in ("resource_cost", "fault_tolerance")
    ]

    fused_grid: np.ndarray = method.create_grid(rows, cols)
    fused_prob: np.ndarray = method.get_occupancy_probability(fused_grid)

    with ExperimentLogger(log_path, grid_snapshot_interval=grid_snapshot_interval) as logger:
        arm_start_data: dict[str, Any] = {
            "method": method.name,
            "active_robots": active_robots,
        }
        if experiment_metadata:
            arm_start_data.update(experiment_metadata)
        logger.log_event("arm_start", arm_start_data)

        for step in range(num_steps):
            # Mid-run fault tolerance: drop a robot at the configured step
            if (
                remove_robot_at_step is not None
                and step == remove_robot_at_step
                and remove_robot_idx is not None
                and remove_robot_idx in active_robots
                and len(active_robots) > 1
            ):
                active_robots = [r for r in active_robots if r != remove_robot_idx]

            current_gt = dynamic_env.get_grid(step) if dynamic_env is not None else gt_env
            for r in active_robots:
                pose = sim_trajectories[r][step]
                ranges = cast_rays(
                    pose_x=pose.x,
                    pose_y=pose.y,
                    pose_theta=pose.theta,
                    grid=current_gt,
                    resolution=resolution,
                    height=height,
                    num_rays=num_rays,
                    max_range=max_range,
                    noise_stddev=noise_stddev,
                    rng=lidar_rngs[r],
                    false_positive_rate=fp_rate,
                    false_negative_rate=fn_rate,
                )
                method.update_scan(
                    grids[r],
                    pose_x=pose.x,
                    pose_y=pose.y,
                    pose_theta=pose.theta,
                    ranges=ranges,
                    resolution=resolution,
                    height=height,
                    num_rays=num_rays,
                    max_range=max_range,
                )

            if measure_resources:
                tracemalloc.start()
            t0 = time.perf_counter()
            fused_grid = grids[active_robots[0]]
            for r in active_robots[1:]:
                fused_grid = method.fuse_grids(fused_grid, grids[r])
            t1 = time.perf_counter()
            if measure_resources:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                tracemalloc_peak_bytes = max(tracemalloc_peak_bytes, peak)
            if len(active_robots) > 1:
                total_fusion_s += t1 - t0
                fusion_count += 1

            fused_prob = method.get_occupancy_probability(fused_grid)

            step_metrics = compute_metrics(
                occupancy_prob=fused_prob,
                ground_truth=current_gt,
                requested=_per_step,
                raw_grid=fused_grid,
                fusion_name=method.name,
                dynamic_mask=dynamic_mask,
            )

            all_poses = [
                (
                    r,
                    sim_trajectories[r][step].x,
                    sim_trajectories[r][step].y,
                    sim_trajectories[r][step].theta,
                )
                for r in active_robots
            ]
            logger.log_step(
                step=step,
                pose=(
                    sim_trajectories[active_robots[0]][step].x,
                    sim_trajectories[active_robots[0]][step].y,
                    sim_trajectories[active_robots[0]][step].theta,
                ),
                poses=all_poses,
                metrics=step_metrics,
                grid_data=fused_prob,
            )

        logger.log_event("arm_complete", {
            "method": method.name,
            "active_robots": active_robots,
            "steps": num_steps,
        })

    memory_bytes = method.get_memory_bytes(fused_grid)
    avg_fusion_ms = (
        total_fusion_s * 1000.0 / fusion_count
        if fusion_count > 0
        else 0.0
    )
    peak_memory_mb = tracemalloc_peak_bytes / (1024.0 * 1024.0)
    return fused_prob, fused_grid, memory_bytes, avg_fusion_ms, peak_memory_mb


# ---------------------------------------------------------------------------
# Shared simulation setup
# ---------------------------------------------------------------------------

def setup_simulation(
    config: dict[str, Any],
    run_seed: int,
) -> tuple[
    np.ndarray,        # gt_env
    int, int,          # rows, cols
    list[list[Pose]],  # sim_trajectories (PGO-corrected or drifted)
    list[list[Pose]],  # gt_trajectories
    list[list[Pose]],  # drifted_trajectories
    list[list[Pose]] | None,  # corrected_trajectories
    int,               # rendezvous_count
    list[int],         # lidar_seeds
    DynamicEnvironment,  # dynamic_env
]:
    """Generate environment, trajectories, drift, and PGO alignment.

    Returns shared simulation state used by all fusion arms in one run.

    Parameters
    ----------
    config:
        Merged experiment configuration dictionary.
    run_seed:
        Seed for this particular run (derived from base seed by run_experiment).

    Returns
    -------
    tuple
        (gt_env, rows, cols, sim_trajectories, gt_trajectories,
         drifted_trajectories, corrected_trajectories, rendezvous_count,
         lidar_seeds, dynamic_env)
        sim_trajectories: PGO-corrected if PGO ran, else drifted.
        corrected_trajectories: None if PGO was skipped.
        dynamic_env: DynamicEnvironment wrapping gt_env + any moving objects.
    """
    env_cfg = config["environment"]
    robots_cfg = config["robots"]
    lidar_cfg = config["lidar"]
    pgo_cfg = config["pgo"]

    rng = np.random.default_rng(run_seed)
    env_seed = int(rng.integers(0, 2**31))
    traj_seed = int(rng.integers(0, 2**31))

    num_robots = int(robots_cfg["count"])
    drift_seeds = [int(rng.integers(0, 2**31)) for _ in range(num_robots)]
    lidar_seeds = [int(rng.integers(0, 2**31)) for _ in range(num_robots)]

    width = float(env_cfg["width"])
    height = float(env_cfg["height"])
    resolution = float(env_cfg["resolution"])
    num_obstacles = int(env_cfg["obstacles"])
    num_rooms = int(env_cfg.get("rooms", 1))
    num_corridors = int(env_cfg.get("corridors", 0))
    num_dynamic_objects = int(env_cfg.get("dynamic_objects", 0))
    dynamic_speed = float(env_cfg.get("dynamic_speed", 0.5))
    num_steps = int(robots_cfg["trajectory_steps"])
    drift_stddev = float(robots_cfg["drift_stddev"])
    angular_drift_stddev = float(robots_cfg["angular_drift_stddev"])
    rendezvous_distance = float(pgo_cfg["rendezvous_distance"])
    pgo_enabled = bool(pgo_cfg["enabled"])

    dynamic_env = generate_environment(
        width=width,
        height=height,
        resolution=resolution,
        num_rooms=num_rooms,
        num_corridors=num_corridors,
        num_obstacles=num_obstacles,
        num_dynamic_objects=num_dynamic_objects,
        dynamic_speed=dynamic_speed,
        rng=np.random.default_rng(env_seed),
    )
    gt_env = dynamic_env.base_grid
    rows, cols = gt_env.shape

    margin = max(resolution * 2.0, 0.5)
    start_fractions_raw = robots_cfg.get("start_fractions")
    start_fractions: list[float] | None = (
        [float(f) for f in start_fractions_raw] if start_fractions_raw is not None else None
    )
    gt_trajectories = generate_multi_robot_trajectories(
        num_robots=num_robots,
        width=width,
        height=height,
        num_steps=num_steps,
        margin=margin,
        start_fractions=start_fractions,
        rng=np.random.default_rng(traj_seed),
    )

    drifted_trajectories = [
        apply_drift(
            gt_trajectories[r],
            drift_stddev=drift_stddev,
            angular_drift_stddev=angular_drift_stddev,
            rng=np.random.default_rng(drift_seeds[r]),
        )
        for r in range(num_robots)
    ]

    corrected_trajectories: list[list[Pose]] | None = None
    rendezvous_count = 0
    if pgo_enabled and num_robots > 1:
        constraints = detect_rendezvous(
            ground_truth=gt_trajectories,
            rendezvous_distance=rendezvous_distance,
        )
        rendezvous_count = len(constraints)
        if constraints:
            corrections = solve_pgo(drifted=drifted_trajectories, constraints=constraints)
            corrected_trajectories = apply_corrections(drifted_trajectories, corrections)

    sim_trajectories = (
        corrected_trajectories if corrected_trajectories is not None
        else drifted_trajectories
    )
    return (
        gt_env, rows, cols, sim_trajectories,
        gt_trajectories, drifted_trajectories, corrected_trajectories,
        rendezvous_count, lidar_seeds, dynamic_env,
    )


# ---------------------------------------------------------------------------
# Per-run arm execution
# ---------------------------------------------------------------------------

def _run_all_arms(
    *,
    config: dict[str, Any],
    gt_env: np.ndarray,
    rows: int,
    cols: int,
    sim_trajectories: list[list[Pose]],
    lidar_seeds: list[int],
    arm_names: list[str],
    metrics_requested: list[str],
    run_output_dir: Path,
    grid_snapshot_interval: int,
    rendezvous_count: int = 0,
    run_seed: int = 0,
    dynamic_env: DynamicEnvironment | None = None,
) -> dict[str, ArmResult]:
    """Execute all fusion arms for one simulation run.

    Parameters
    ----------
    run_output_dir:
        Directory for this run's per-arm files.

    Returns
    -------
    dict[str, ArmResult]
        Mapping arm_name -> ArmResult.
    """
    env_cfg = config["environment"]
    robots_cfg = config["robots"]
    lidar_cfg = config["lidar"]
    ft_cfg = config.get("fault_tolerance", {})
    log_cfg = config.get("logging", {})

    num_robots = int(robots_cfg["count"])
    num_steps = int(robots_cfg["trajectory_steps"])
    num_rays = int(lidar_cfg["num_rays"])
    max_range = float(lidar_cfg["max_range"])
    noise_stddev = float(lidar_cfg["noise_stddev"])
    fp_rate = float(lidar_cfg.get("false_positive_rate", 0.0))
    fn_rate = float(lidar_cfg.get("false_negative_rate", 0.0))
    resolution = float(env_cfg["resolution"])
    height = float(env_cfg["height"])
    width = float(env_cfg["width"])
    pgo_enabled = bool(config["pgo"]["enabled"])
    measure_resources = bool(log_cfg.get("measure_resources", False))

    # Fault tolerance configuration
    remove_after_raw = ft_cfg.get("remove_robot_after_step")
    remove_after: int | None = int(remove_after_raw) if remove_after_raw is not None else None
    robot_to_remove_spec = ft_cfg.get("robot_to_remove", 0)
    if str(robot_to_remove_spec).lower() == "random":
        _ft_rng = np.random.default_rng(run_seed + 1)
        remove_robot_idx = int(_ft_rng.integers(0, num_robots))
    else:
        remove_robot_idx = int(robot_to_remove_spec)
    # Clamp to valid range
    remove_robot_idx = remove_robot_idx % num_robots if num_robots > 0 else 0

    experiment_metadata: dict[str, Any] = {
        "seed": run_seed,
        "room_dims": [width, height],
        "resolution": resolution,
        "num_robots": num_robots,
        "num_steps": num_steps,
        "pgo_enabled": pgo_enabled,
        "rendezvous_count": rendezvous_count,
        "fusion_methods": arm_names,
        "git_commit": _get_git_commit(),
    }

    # Compute dynamic cell mask if environment has moving objects
    dynamic_mask: np.ndarray | None = None
    if dynamic_env is not None and dynamic_env.has_dynamic_objects:
        dynamic_mask = dynamic_env.get_dynamic_mask(num_steps)

    # Compute pignistic-matched masses for DS/TBM fair comparison (LL-001).
    # Uses module constants matching bayesian.py defaults.
    _L_OCC = 2.0
    _L_FREE = -0.5
    _m_occ_matched = np.array(pignistic_masses(_L_OCC), dtype=np.float64)
    _m_free_matched = np.array(pignistic_masses(_L_FREE), dtype=np.float64)
    _l_max: float = config.get("bayesian", {}).get("l_max", 10.0)
    _m_of_min: float = config.get("ds", {}).get("m_of_min", 0.0)

    results: dict[str, ArmResult] = {}

    for arm_name in arm_names:
        method = create_fusion_method(arm_name, m_occ=_m_occ_matched, m_free=_m_free_matched, clamp=_l_max, m_of_min=_m_of_min)
        arm_dir = run_output_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)

        lidar_rngs_full = [np.random.default_rng(lidar_seeds[r]) for r in range(num_robots)]

        fused_prob, fused_grid, memory_bytes, avg_fusion_ms, peak_mb = run_arm(
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
            metrics_requested=metrics_requested,
            log_path=arm_dir / "run.jsonl",
            grid_snapshot_interval=grid_snapshot_interval,
            lidar_rngs=lidar_rngs_full,
            active_robots=list(range(num_robots)),
            experiment_metadata=experiment_metadata,
            dynamic_env=dynamic_env,
            fp_rate=fp_rate,
            fn_rate=fn_rate,
            measure_resources=measure_resources,
            dynamic_mask=dynamic_mask,
        )

        # Fault tolerance: average over ALL possible single-robot drops.
        # For N robots: ft_score = mean(ft(drop_0), ft(drop_1), ..., ft(drop_N-1)).
        # This gives a more robust estimate than always dropping robot 0, and
        # captures asymmetric coverage patterns across robots.
        # DS non-invertibility note: once a robot's data is fused into the shared
        # grid, DS cannot subtract it -- fault runs therefore start from scratch
        # with N-1 individual grids, not by undoing fused state.
        ft_score: float = float("nan")
        if num_robots > 1:
            full_acc = cell_accuracy(fused_prob, gt_env)
            ft_scores: list[float] = []
            for drop_r in range(num_robots):
                lidar_rngs_ft = [np.random.default_rng(lidar_seeds[r]) for r in range(num_robots)]
                if remove_after is not None:
                    # Mid-run removal: all robots present until remove_after, then drop_r fails
                    ft_prob, _, _, _, _ = run_arm(
                        method=create_fusion_method(arm_name, m_occ=_m_occ_matched, m_free=_m_free_matched, clamp=_l_max, m_of_min=_m_of_min),
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
                        log_path=arm_dir / f"run_fault_{drop_r}.jsonl",
                        grid_snapshot_interval=0,
                        lidar_rngs=lidar_rngs_ft,
                        active_robots=list(range(num_robots)),
                        experiment_metadata=experiment_metadata,
                        dynamic_env=dynamic_env,
                        fp_rate=fp_rate,
                        fn_rate=fn_rate,
                        remove_robot_at_step=remove_after,
                        remove_robot_idx=drop_r,
                    )
                else:
                    # From-scratch: run with every robot except drop_r
                    ft_prob, _, _, _, _ = run_arm(
                        method=create_fusion_method(arm_name, m_occ=_m_occ_matched, m_free=_m_free_matched, clamp=_l_max, m_of_min=_m_of_min),
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
                        log_path=arm_dir / f"run_fault_{drop_r}.jsonl",
                        grid_snapshot_interval=0,
                        lidar_rngs=lidar_rngs_ft,
                        active_robots=[r for r in range(num_robots) if r != drop_r],
                        experiment_metadata=experiment_metadata,
                        dynamic_env=dynamic_env,
                        fp_rate=fp_rate,
                        fn_rate=fn_rate,
                    )
                ft_acc = cell_accuracy(ft_prob, gt_env)
                if full_acc > 1e-6:
                    ft_scores.append(ft_acc / full_acc)
            if ft_scores:
                ft_score = float(np.mean(ft_scores))

        # Communication bytes: each rendezvous = 2-robot exchange (each sends its grid)
        single_grid_bytes = method.get_memory_bytes(method.create_grid(rows, cols))
        comm_bytes = rendezvous_count * 2 * single_grid_bytes

        final_metrics = compute_metrics(
            occupancy_prob=fused_prob,
            ground_truth=gt_env,
            requested=metrics_requested,
            raw_grid=fused_grid,
            fusion_name=arm_name,
            memory_bytes=memory_bytes,
            fault_tolerance_score=ft_score,
            dynamic_mask=dynamic_mask,
        )

        results[arm_name] = ArmResult(
            arm_name=arm_name,
            metrics=final_metrics,
            memory_bytes=memory_bytes,
            avg_fusion_ms=avg_fusion_ms,
            fused_prob=fused_prob,
            peak_memory_mb=peak_mb,
            comm_bytes=comm_bytes,
        )

        # Save per-arm metrics.json (includes resource metrics)
        arm_data: dict[str, Any] = {
            k: (v if v == v else None) for k, v in final_metrics.items()
        }
        arm_data["peak_memory_mb"] = peak_mb
        arm_data["comm_bytes"] = comm_bytes
        arm_data["avg_fusion_ms"] = avg_fusion_ms
        with (arm_dir / "metrics.json").open("w") as fh:
            json.dump(arm_data, fh, indent=2)

    return results


# ---------------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------------

_RESOURCE_COLS = ["avg_fusion_ms", "peak_memory_mb", "comm_bytes"]


def _write_summary_csv(
    output_dir: Path,
    all_results: list[dict[str, ArmResult]],
    arm_names: list[str],
    metrics_requested: list[str],
) -> None:
    """Write summary.csv: one row per arm, mean across all runs.

    For num_runs == 1 this is just the single-run values.
    Columns: arm, <metrics_requested>, avg_fusion_ms, peak_memory_mb, comm_bytes.
    """
    summary_path = output_dir / "summary.csv"
    fieldnames = ["arm"] + metrics_requested + _RESOURCE_COLS

    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for arm_name in arm_names:
            run_values: dict[str, list[float]] = {
                m: [] for m in metrics_requested + _RESOURCE_COLS
            }
            for run_results in all_results:
                arm_result = run_results[arm_name]
                for m in metrics_requested:
                    v = arm_result.metrics.get(m, float("nan"))
                    if isinstance(v, float) and v == v:  # not NaN
                        run_values[m].append(v)
                # Resource columns come from ArmResult fields directly
                for rc in _RESOURCE_COLS:
                    v = getattr(arm_result, rc, float("nan"))
                    fv = float(v)
                    if fv == fv:  # not NaN
                        run_values[rc].append(fv)

            row: dict[str, str] = {"arm": arm_name}
            for col in metrics_requested + _RESOURCE_COLS:
                vals = run_values[col]
                mean_val = float(np.mean(vals)) if vals else float("nan")
                row[col] = str(mean_val) if mean_val == mean_val else ""
            writer.writerow(row)


def _write_metrics_json(
    output_dir: Path,
    all_results: list[dict[str, ArmResult]],
    arm_names: list[str],
    metrics_requested: list[str],
) -> None:
    """Write top-level metrics.json: arm -> mean metric values."""
    summary: dict[str, dict[str, Any]] = {}
    for arm_name in arm_names:
        arm_metrics: dict[str, Any] = {}
        for m in metrics_requested:
            vals = []
            for run_results in all_results:
                v = run_results[arm_name].metrics.get(m, float("nan"))
                if isinstance(v, float) and v == v:
                    vals.append(v)
            mean_val = float(np.mean(vals)) if vals else None
            arm_metrics[m] = mean_val
        summary[arm_name] = arm_metrics

    with (output_dir / "metrics.json").open("w") as fh:
        json.dump(summary, fh, indent=2)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _write_comparison_table(
    output_dir: Path,
    all_results: list[dict[str, ArmResult]],
    arm_names: list[str],
    metrics_requested: list[str],
) -> str:
    """Write a formatted method x metric comparison table.

    Saves to ``{output_dir}/comparison_table.txt`` and returns the table
    as a string (for printing by the caller).
    """
    # Compute mean values per arm per metric (same logic as _write_metrics_json)
    means: dict[str, dict[str, float]] = {}
    for arm_name in arm_names:
        arm_means: dict[str, float] = {}
        for m in metrics_requested:
            vals = []
            for run_results in all_results:
                v = run_results[arm_name].metrics.get(m, float("nan"))
                if isinstance(v, float) and v == v:
                    vals.append(v)
            arm_means[m] = float(np.mean(vals)) if vals else float("nan")
        # Also include resource extras
        extra_keys = ["avg_fusion_ms", "peak_memory_mb", "comm_bytes"]
        for key in extra_keys:
            vals = []
            for run_results in all_results:
                r = run_results[arm_name]
                v = getattr(r, key, float("nan"))
                if isinstance(v, (int, float)) and float(v) == float(v):
                    vals.append(float(v))
            arm_means[key] = float(np.mean(vals)) if vals else float("nan")
        means[arm_name] = arm_means

    all_cols = metrics_requested + ["avg_fusion_ms", "peak_memory_mb", "comm_bytes"]
    # Column widths
    col_w = max(14, *(len(c) for c in all_cols))
    arm_w = max(16, *(len(a) for a in arm_names))

    sep = "+" + "-" * (arm_w + 2) + "+" + ("+".join("-" * (col_w + 2) for _ in all_cols)) + "+"
    header = (
        "| " + "Method".ljust(arm_w) + " |"
        + "".join(" " + c.ljust(col_w) + " |" for c in all_cols)
    )

    rows_str = [sep, header, sep]
    for arm_name in arm_names:
        row = "| " + arm_name.ljust(arm_w) + " |"
        for c in all_cols:
            v = means[arm_name].get(c, float("nan"))
            if v == v:  # not NaN
                cell = f"{v:.4f}" if isinstance(v, float) else str(v)
            else:
                cell = "N/A"
            row += " " + cell.ljust(col_w) + " |"
        rows_str.append(row)
    rows_str.append(sep)

    table = "\n".join(rows_str)
    (output_dir / "comparison_table.txt").write_text(table + "\n")
    return table


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_experiment(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, ArmResult]:
    """Run the full H-003 experiment: all arms, all metrics.

    Reads ``num_runs`` from ``config['experiment']`` (default 1).  When
    num_runs > 1, each run uses a derived seed and outputs go into
    ``{output_dir}/{arm_name}/run_{n}/``.  Final ``metrics.json`` and
    ``summary.csv`` report mean values across runs.

    Parameters
    ----------
    config:
        Merged configuration dictionary from ``load_config``.
    output_dir:
        Root directory for all outputs.

    Returns
    -------
    dict[str, ArmResult]
        Results from the *last* run (or first run if num_runs == 1).
        Use metrics.json / summary.csv for aggregated multi-run results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = config["experiment"]
    log_cfg = config["logging"]

    base_seed = int(exp_cfg["seed"])
    num_runs = int(exp_cfg.get("num_runs", 1))
    arm_names: list[str] = list(config["fusion"]["methods"])
    metrics_requested: list[str] = list(config["metrics"])
    grid_snapshot_interval = int(log_cfg["grid_snapshot_interval"])

    # Derive per-run seeds from the base seed
    seed_rng = np.random.default_rng(base_seed)
    run_seeds = [int(seed_rng.integers(0, 2**31)) for _ in range(num_runs)]

    all_run_results: list[dict[str, ArmResult]] = []
    last_run_results: dict[str, ArmResult] = {}

    for run_idx in range(num_runs):
        run_seed = run_seeds[run_idx]

        (
            gt_env, rows, cols, sim_trajectories,
            gt_trajectories, drifted_trajectories, corrected_trajectories,
            rendezvous_count, lidar_seeds, dynamic_env,
        ) = setup_simulation(config, run_seed)

        # Per-arm output subdirectory
        if num_runs == 1:
            run_output_dir = output_dir
        else:
            run_output_dir = output_dir / f"run_{run_idx}"
            run_output_dir.mkdir(parents=True, exist_ok=True)

        run_results = _run_all_arms(
            config=config,
            gt_env=gt_env,
            rows=rows,
            cols=cols,
            sim_trajectories=sim_trajectories,
            lidar_seeds=lidar_seeds,
            arm_names=arm_names,
            metrics_requested=metrics_requested,
            run_output_dir=run_output_dir,
            grid_snapshot_interval=grid_snapshot_interval,
            rendezvous_count=rendezvous_count,
            run_seed=run_seed,
            dynamic_env=dynamic_env,
        )

        all_run_results.append(run_results)
        last_run_results = run_results

    # Write aggregated outputs
    _write_summary_csv(output_dir, all_run_results, arm_names, metrics_requested)
    _write_metrics_json(output_dir, all_run_results, arm_names, metrics_requested)
    _write_comparison_table(output_dir, all_run_results, arm_names, metrics_requested)

    return last_run_results
