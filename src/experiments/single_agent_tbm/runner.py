"""Single-robot scanning runner for H-002 experiment.

Runs a single robot through a generated environment, updating occupancy
grids using a non-homogeneous sensor model where log-odds / mass values
vary per cell based on distance from robot and angle of incidence.

Inc 1: Bayesian arm only (run_single_robot_scan).
Inc 2: All 3 arms on identical data (run_experiment).
Inc 4: Dynamic objects via DynamicEnvironment, dynamic_detection metric.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.experiments.hybrid_pgo_ds.simulation.environment import (
    DynamicEnvironment,
    generate_environment,
)
from src.experiments.hybrid_pgo_ds.simulation.trajectory import (
    Pose,
    generate_perimeter_trajectory,
)
from src.experiments.hybrid_pgo_ds.simulation.lidar import cast_rays
from src.experiments.hybrid_pgo_ds.experiment.metrics import (
    boundary_sharpness,
    cell_accuracy,
)
from src.experiments.hybrid_pgo_ds.fusion.base import _bresenham_line
from src.experiments.hybrid_pgo_ds.fusion.dstbm import _dempster_combine_scalar
from src.experiments.hybrid_pgo_ds.fusion.yager import _yager_combine_scalar
from src.experiments.single_agent_tbm.sensor_model import NonHomogeneousSensorModel
from src.experiments.single_agent_tbm.metrics import (
    bayesian_uncertainty,
    brier_score,
    dynamic_detection_score,
    frontier_quality,
    pignistic_uncertainty,
)

_ARM_NAMES: list[str] = ["bayesian", "bayesian_count", "dstbm", "yager"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _compute_surface_normals(
    gt_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute per-cell surface normal direction from ground-truth gradient.

    Returns ``(normal_x, normal_y)`` unit-normal arrays of shape
    ``(rows, cols)``.  Zero-gradient cells get (0, 0).
    """
    gy_grid, gx_grid = np.gradient(gt_grid.astype(np.float64))
    normal_x = gx_grid
    normal_y = -gy_grid
    mag = np.sqrt(normal_x**2 + normal_y**2)
    nonzero = mag > 1e-10
    normal_x[nonzero] /= mag[nonzero]
    normal_y[nonzero] /= mag[nonzero]
    return normal_x, normal_y


def _cell_center_world(
    row: int, col: int, resolution: float, height: float
) -> tuple[float, float]:
    """Return world (x, y) of a grid cell centre."""
    x = (col + 0.5) * resolution
    y = height - (row + 0.5) * resolution
    return x, y


def _angle_of_incidence(
    ray_dx: float,
    ray_dy: float,
    normal_x: np.ndarray,
    normal_y: np.ndarray,
    cr: int,
    cc: int,
) -> float:
    """Angle between incoming ray and surface normal at (cr, cc)."""
    nx = float(normal_x[cr, cc])
    ny = float(normal_y[cr, cc])
    if abs(nx) < 1e-10 and abs(ny) < 1e-10:
        return 0.0
    cos_aoi = min(1.0, abs(ray_dx * nx + ray_dy * ny))
    return math.acos(cos_aoi)


# ---------------------------------------------------------------------------
# Shared simulation context
# ---------------------------------------------------------------------------

@dataclass
class _ScanContext:
    """Precomputed simulation data shared by all experimental arms."""

    gt_grid: np.ndarray
    dynamic_mask: np.ndarray | None
    rows: int
    cols: int
    height: float
    resolution: float
    max_range: float
    num_rays: int
    num_steps: int
    trajectory: list[Pose]
    all_ranges: list[np.ndarray] = field(repr=False)
    normal_x: np.ndarray = field(repr=False)
    normal_y: np.ndarray = field(repr=False)
    sensor_model: NonHomogeneousSensorModel = field(repr=False)
    l_max: float = 10.0
    m_of_min: float = 0.0


def _build_scan_context(config: dict[str, Any]) -> _ScanContext:
    """Generate environment, trajectory, and precompute all LiDAR scans.

    The random generator is advanced in a fixed order (environment, then
    trajectory, then cast_rays per step) so all arms see identical data.

    When dynamic objects are present, ``cast_rays`` uses the per-step grid
    from ``DynamicEnvironment.get_grid(step)`` so that rays interact with
    moving obstacles.  Surface normals and final-metric ground truth are
    computed from the static ``base_grid`` (walls don't move).
    """
    env_cfg = config["environment"]
    width: float = env_cfg["width"]
    height: float = env_cfg["height"]
    resolution: float = env_cfg["resolution"]
    num_obstacles: int = env_cfg.get("obstacles", 0)
    num_rooms: int = env_cfg.get("rooms", 1)
    num_corridors: int = env_cfg.get("corridors", 0)
    num_dynamic_objects: int = env_cfg.get("dynamic_objects", 0)
    dynamic_speed: float = env_cfg.get("dynamic_speed", 0.5)

    robot_cfg = config["robots"]
    num_steps: int = robot_cfg["trajectory_steps"]

    lidar_cfg = config["lidar"]
    max_range: float = lidar_cfg["max_range"]
    num_rays: int = lidar_cfg["num_rays"]
    noise_stddev: float = lidar_cfg["noise_stddev"]

    seed: int = config["experiment"]["seed"]
    rng = np.random.default_rng(seed)

    dynamic_env = generate_environment(
        width=width,
        height=height,
        resolution=resolution,
        num_obstacles=num_obstacles,
        num_rooms=num_rooms,
        num_corridors=num_corridors,
        num_dynamic_objects=num_dynamic_objects,
        dynamic_speed=dynamic_speed,
        rng=rng,
    )

    gt_grid = dynamic_env.base_grid  # static base for metrics + normals
    rows, cols = gt_grid.shape

    trajectory = generate_perimeter_trajectory(
        width=width,
        height=height,
        num_steps=num_steps,
        rng=rng,
    )

    sensor_cfg: dict[str, Any] = config.get("sensor_model", {})
    sensor_model = NonHomogeneousSensorModel(
        base_l_occ=sensor_cfg.get("base_l_occ", 2.0),
        base_l_free=sensor_cfg.get("base_l_free", -0.5),
        distance_decay=sensor_cfg.get("distance_decay", 0.1),
        angle_decay=sensor_cfg.get("angle_decay", 0.5),
        max_range=max_range,
        matching=sensor_cfg.get("matching", "betp"),
    )

    # Precompute ranges for all steps so every arm sees the same sensor data.
    # Use per-step grid so rays interact with moving obstacles.
    all_ranges: list[np.ndarray] = []
    for step_idx in range(num_steps):
        pose = trajectory[step_idx]
        step_grid = dynamic_env.get_grid(step_idx)
        ranges = cast_rays(
            pose_x=pose.x,
            pose_y=pose.y,
            pose_theta=pose.theta,
            grid=step_grid,
            resolution=resolution,
            height=height,
            num_rays=num_rays,
            max_range=max_range,
            noise_stddev=noise_stddev,
            rng=rng,
        )
        all_ranges.append(ranges)

    # Surface normals from static base grid (dynamic objects are small
    # circles that don't define wall normals).
    normal_x, normal_y = _compute_surface_normals(gt_grid)

    # Dynamic mask: cells ever occupied by a moving object
    dynamic_mask: np.ndarray | None = None
    if dynamic_env.has_dynamic_objects:
        dynamic_mask = dynamic_env.get_dynamic_mask(num_steps)

    l_max: float = config.get("bayesian", {}).get("l_max", 10.0)
    m_of_min: float = config.get("ds", {}).get("m_of_min", 0.0)

    return _ScanContext(
        gt_grid=gt_grid,
        dynamic_mask=dynamic_mask,
        rows=rows,
        cols=cols,
        height=height,
        resolution=resolution,
        max_range=max_range,
        num_rays=num_rays,
        num_steps=num_steps,
        trajectory=trajectory,
        all_ranges=all_ranges,
        normal_x=normal_x,
        normal_y=normal_y,
        sensor_model=sensor_model,
        l_max=l_max,
        m_of_min=m_of_min,
    )


# ---------------------------------------------------------------------------
# Per-arm grid update helpers
# ---------------------------------------------------------------------------

def _update_free_bayesian(
    grid: np.ndarray, cr: int, cc: int, l_free: float, clamp: float
) -> None:
    grid[cr, cc] = float(np.clip(grid[cr, cc] + l_free, -clamp, clamp))


def _update_hit_bayesian(
    grid: np.ndarray, cr: int, cc: int, l_occ: float, clamp: float
) -> None:
    grid[cr, cc] = float(np.clip(grid[cr, cc] + l_occ, -clamp, clamp))


def _update_free_bayesian_count(
    grid: np.ndarray, cr: int, cc: int, l_free: float, clamp: float
) -> None:
    grid[cr, cc, 0] = float(np.clip(grid[cr, cc, 0] + l_free, -clamp, clamp))
    grid[cr, cc, 1] += 1.0


def _update_hit_bayesian_count(
    grid: np.ndarray, cr: int, cc: int, l_occ: float, clamp: float
) -> None:
    grid[cr, cc, 0] = float(np.clip(grid[cr, cc, 0] + l_occ, -clamp, clamp))
    grid[cr, cc, 1] += 1.0


def _enforce_m_of_min(m: np.ndarray, m_of_min: float) -> np.ndarray:
    """Enforce minimum ignorance mass and re-normalise (scalar)."""
    if m_of_min <= 0.0:
        return m
    result = m.copy()
    result[2] = max(float(result[2]), m_of_min)
    total = result.sum()
    if total > 1e-10:
        result = result / total
    return result


def _update_free_dstbm(
    grid: np.ndarray,
    cr: int,
    cc: int,
    sensor_model: NonHomogeneousSensorModel,
    dist: float,
    m_of_min: float = 0.0,
) -> None:
    m_O, m_F, m_OF = sensor_model.compute_mass(dist, 0.0, "free")
    m_obs = np.array([m_O, m_F, m_OF], dtype=np.float64)
    m_cur = grid[cr, cc, :].astype(np.float64)
    grid[cr, cc, :] = _enforce_m_of_min(
        _dempster_combine_scalar(m_cur, m_obs), m_of_min
    )


def _update_hit_dstbm(
    grid: np.ndarray,
    cr: int,
    cc: int,
    sensor_model: NonHomogeneousSensorModel,
    dist: float,
    aoi: float,
    m_of_min: float = 0.0,
) -> None:
    m_O, m_F, m_OF = sensor_model.compute_mass(dist, aoi, "occupied")
    m_obs = np.array([m_O, m_F, m_OF], dtype=np.float64)
    m_cur = grid[cr, cc, :].astype(np.float64)
    grid[cr, cc, :] = _enforce_m_of_min(
        _dempster_combine_scalar(m_cur, m_obs), m_of_min
    )


def _update_free_yager(
    grid: np.ndarray,
    cr: int,
    cc: int,
    sensor_model: NonHomogeneousSensorModel,
    dist: float,
    m_of_min: float = 0.0,
) -> None:
    m_O, m_F, m_OF = sensor_model.compute_mass(dist, 0.0, "free")
    m_obs = np.array([m_O, m_F, m_OF], dtype=np.float64)
    m_cur = grid[cr, cc, :].astype(np.float64)
    grid[cr, cc, :] = _enforce_m_of_min(
        _yager_combine_scalar(m_cur, m_obs), m_of_min
    )


def _update_hit_yager(
    grid: np.ndarray,
    cr: int,
    cc: int,
    sensor_model: NonHomogeneousSensorModel,
    dist: float,
    aoi: float,
    m_of_min: float = 0.0,
) -> None:
    m_O, m_F, m_OF = sensor_model.compute_mass(dist, aoi, "occupied")
    m_obs = np.array([m_O, m_F, m_OF], dtype=np.float64)
    m_cur = grid[cr, cc, :].astype(np.float64)
    grid[cr, cc, :] = _enforce_m_of_min(
        _yager_combine_scalar(m_cur, m_obs), m_of_min
    )


# ---------------------------------------------------------------------------
# Per-arm scan driver
# ---------------------------------------------------------------------------

def _scan_arm(arm_name: str, ctx: _ScanContext) -> dict[str, Any]:
    """Run one arm's scan loop using precomputed shared data.

    Parameters
    ----------
    arm_name:
        ``"bayesian"``, ``"bayesian_count"``, or ``"dstbm"``.
    ctx:
        Shared simulation context from ``_build_scan_context``.

    Returns
    -------
    dict with ``prob_grid``, ``raw_grid``, ``cell_accuracy``,
    ``boundary_sharpness``, ``mean_uncertainty``.
    """
    # Create arm-specific grid
    if arm_name == "bayesian":
        grid = np.zeros((ctx.rows, ctx.cols), dtype=np.float64)
    elif arm_name == "bayesian_count":
        grid = np.zeros((ctx.rows, ctx.cols, 2), dtype=np.float64)
    elif arm_name in ("dstbm", "yager"):
        grid = np.zeros((ctx.rows, ctx.cols, 3), dtype=np.float64)
        grid[:, :, 2] = 1.0  # vacuous belief
    else:
        raise ValueError(f"Unknown arm: {arm_name}")

    sm = ctx.sensor_model

    for step_idx in range(ctx.num_steps):
        pose = ctx.trajectory[step_idx]
        ranges = ctx.all_ranges[step_idx]

        r_robot = int(np.clip(
            int((ctx.height - pose.y) / ctx.resolution), 0, ctx.rows - 1
        ))
        c_robot = int(np.clip(
            int(pose.x / ctx.resolution), 0, ctx.cols - 1
        ))

        angles = pose.theta + np.linspace(
            0.0, 2.0 * np.pi, ctx.num_rays, endpoint=False
        )

        for i in range(ctx.num_rays):
            r_meas = float(ranges[i])
            angle = float(angles[i])

            endpoint_x = pose.x + r_meas * math.cos(angle)
            endpoint_y = pose.y + r_meas * math.sin(angle)

            r_end = int(np.clip(
                int((ctx.height - endpoint_y) / ctx.resolution), 0, ctx.rows - 1
            ))
            c_end = int(np.clip(
                int(endpoint_x / ctx.resolution), 0, ctx.cols - 1
            ))

            cells = _bresenham_line(r_robot, c_robot, r_end, c_end)
            is_hit = r_meas < ctx.max_range - 1e-3

            # --- Free-space cells ---
            for cr, cc in cells[:-1]:
                if 0 <= cr < ctx.rows and 0 <= cc < ctx.cols:
                    cx, cy = _cell_center_world(
                        cr, cc, ctx.resolution, ctx.height
                    )
                    dist = math.hypot(cx - pose.x, cy - pose.y)

                    if arm_name == "bayesian":
                        _, l_free = sm.compute_log_odds(dist, 0.0)
                        _update_free_bayesian(grid, cr, cc, l_free, ctx.l_max)
                    elif arm_name == "bayesian_count":
                        _, l_free = sm.compute_log_odds(dist, 0.0)
                        _update_free_bayesian_count(grid, cr, cc, l_free, ctx.l_max)
                    elif arm_name == "dstbm":
                        _update_free_dstbm(grid, cr, cc, sm, dist, ctx.m_of_min)
                    else:  # yager
                        _update_free_yager(grid, cr, cc, sm, dist, ctx.m_of_min)

            # --- Endpoint cell ---
            if cells:
                cr, cc = cells[-1]
                if 0 <= cr < ctx.rows and 0 <= cc < ctx.cols:
                    cx, cy = _cell_center_world(
                        cr, cc, ctx.resolution, ctx.height
                    )
                    dist = math.hypot(cx - pose.x, cy - pose.y)

                    if is_hit:
                        ray_dx = math.cos(angle)
                        ray_dy = math.sin(angle)
                        aoi = _angle_of_incidence(
                            ray_dx, ray_dy,
                            ctx.normal_x, ctx.normal_y, cr, cc,
                        )
                        if arm_name == "bayesian":
                            l_occ, _ = sm.compute_log_odds(dist, aoi)
                            _update_hit_bayesian(grid, cr, cc, l_occ, ctx.l_max)
                        elif arm_name == "bayesian_count":
                            l_occ, _ = sm.compute_log_odds(dist, aoi)
                            _update_hit_bayesian_count(grid, cr, cc, l_occ, ctx.l_max)
                        elif arm_name == "dstbm":
                            _update_hit_dstbm(grid, cr, cc, sm, dist, aoi, ctx.m_of_min)
                        else:  # yager
                            _update_hit_yager(grid, cr, cc, sm, dist, aoi, ctx.m_of_min)
                    else:
                        # Max-range: treat as free
                        if arm_name == "bayesian":
                            _, l_free = sm.compute_log_odds(dist, 0.0)
                            _update_free_bayesian(grid, cr, cc, l_free, ctx.l_max)
                        elif arm_name == "bayesian_count":
                            _, l_free = sm.compute_log_odds(dist, 0.0)
                            _update_free_bayesian_count(grid, cr, cc, l_free, ctx.l_max)
                        elif arm_name == "dstbm":
                            _update_free_dstbm(grid, cr, cc, sm, dist, ctx.m_of_min)
                        else:  # yager
                            _update_free_yager(grid, cr, cc, sm, dist, ctx.m_of_min)

    # --- Convert to probability ---
    if arm_name in ("bayesian", "bayesian_count"):
        lo = grid if arm_name == "bayesian" else grid[:, :, 0]
        prob_grid = (1.0 / (1.0 + np.exp(-lo))).astype(np.float32)
    else:  # dstbm or yager: pignistic transform BetP(O) = m_O + m_OF / 2
        m_O = grid[:, :, 0].astype(np.float64)
        m_OF = grid[:, :, 2].astype(np.float64)
        prob_grid = np.clip(m_O + m_OF / 2.0, 0.0, 1.0).astype(np.float32)

    # --- Metrics ---
    acc = cell_accuracy(prob_grid, ctx.gt_grid)
    sharpness = boundary_sharpness(prob_grid, ctx.gt_grid)
    brier = brier_score(prob_grid, ctx.gt_grid)

    if arm_name in ("dstbm", "yager"):
        unc_grid = pignistic_uncertainty(grid.astype(np.float32))
    else:
        unc_grid = bayesian_uncertainty(prob_grid)
    mean_unc = float(np.mean(unc_grid))

    raw_for_frontier = grid.astype(np.float32) if grid.ndim == 3 else None
    f1 = frontier_quality(prob_grid, raw_for_frontier, ctx.gt_grid, arm_name)

    dyn_score = dynamic_detection_score(unc_grid, ctx.dynamic_mask)

    # Absolute uncertainty at dynamic vs stable cells + percentile detection
    mean_unc_dynamic = float("nan")
    mean_unc_stable = float("nan")
    detection_p50 = float("nan")
    detection_p75 = float("nan")
    detection_p90 = float("nan")
    if ctx.dynamic_mask is not None and np.any(ctx.dynamic_mask):
        stable_mask = ~ctx.dynamic_mask
        dynamic_unc = unc_grid[ctx.dynamic_mask]
        mean_unc_dynamic = float(np.mean(dynamic_unc))
        if np.any(stable_mask):
            stable_unc = unc_grid[stable_mask]
            mean_unc_stable = float(np.mean(stable_unc))
            # Percentile detection: fraction of dynamic cells above P-th
            # percentile of stable cells.  High = good detection.
            for pct, name in [(50, "p50"), (75, "p75"), (90, "p90")]:
                thresh = float(np.percentile(stable_unc, pct))
                rate = float(np.mean(dynamic_unc > thresh))
                if name == "p50":
                    detection_p50 = rate
                elif name == "p75":
                    detection_p75 = rate
                else:
                    detection_p90 = rate

    return {
        "prob_grid": prob_grid,
        "raw_grid": grid,
        "cell_accuracy": acc,
        "boundary_sharpness": sharpness,
        "mean_uncertainty": mean_unc,
        "brier_score": brier,
        "frontier_quality": f1,
        "dynamic_detection": dyn_score,
        "mean_unc_dynamic": mean_unc_dynamic,
        "mean_unc_stable": mean_unc_stable,
        "detection_p50": detection_p50,
        "detection_p75": detection_p75,
        "detection_p90": detection_p90,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_single_robot_scan(config: dict[str, Any]) -> dict[str, Any]:
    """Run single-robot Bayesian scan (Inc 1 backward-compatible API).

    Parameters
    ----------
    config:
        Experiment configuration dictionary.

    Returns
    -------
    dict
        ``log_odds_grid``, ``prob_grid``, ``gt_grid``, ``cell_accuracy``,
        ``num_steps``.
    """
    ctx = _build_scan_context(config)
    arm = _scan_arm("bayesian", ctx)
    return {
        "log_odds_grid": arm["raw_grid"],
        "prob_grid": arm["prob_grid"],
        "gt_grid": ctx.gt_grid,
        "cell_accuracy": arm["cell_accuracy"],
        "num_steps": ctx.num_steps,
    }


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run all 3 arms on identical simulation data with timing.

    Parameters
    ----------
    config:
        Experiment configuration dictionary.

    Returns
    -------
    dict
        Per-arm results keyed by arm name (``"bayesian"``,
        ``"bayesian_count"``, ``"dstbm"``), each containing
        ``cell_accuracy``, ``boundary_sharpness``, ``mean_uncertainty``,
        ``wall_clock_seconds``, ``prob_grid``, ``raw_grid``.
        Also ``"gt_grid"`` and ``"num_steps"`` at the top level.
    """
    ctx = _build_scan_context(config)

    results: dict[str, Any] = {
        "gt_grid": ctx.gt_grid,
        "num_steps": ctx.num_steps,
    }

    for arm_name in _ARM_NAMES:
        t0 = time.perf_counter()
        arm_result = _scan_arm(arm_name, ctx)
        t1 = time.perf_counter()
        arm_result["wall_clock_seconds"] = t1 - t0
        results[arm_name] = arm_result

    return results
