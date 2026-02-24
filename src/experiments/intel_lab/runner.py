"""Intel Lab experiment runner: 3-arm comparison on real data.

Runs Bayesian, Bayesian+count, and DS/TBM on the Intel Lab dataset
and computes metrics against ground truth built from the full dataset.

Supports both single-robot (all scans) and multi-robot split (scans
divided among K simulated robots) configurations.

Bootstrap CI support
--------------------
``build_single_robot_prob_maps`` and ``build_multi_robot_prob_maps``
return raw probability arrays (not masked) for all arms, suitable for
``bootstrap_delta_ci`` and ``spatial_block_bootstrap_delta_ci``.

``bootstrap_delta_ci`` resamples evaluated cells with replacement and
computes 95% percentile confidence intervals for each metric delta
(DS/TBM minus Bayesian+count).  Supported metrics: cell_accuracy,
brier_score, boundary_sharpness.

``spatial_block_bootstrap_delta_ci`` uses non-overlapping 2D spatial
blocks to preserve spatial autocorrelation within blocks.  Blocks of
B x B cells are resampled with replacement; cells within each block
are kept together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.experiments.hybrid_pgo_ds.experiment.metrics import compute_metrics
from src.experiments.hybrid_pgo_ds.experiment.runner import pignistic_masses
from src.experiments.hybrid_pgo_ds.fusion.base import FusionMethod
from src.experiments.hybrid_pgo_ds.fusion.bayesian import BayesianFusion
from src.experiments.hybrid_pgo_ds.fusion.bayesian_count import BayesianCountFusion
from src.experiments.hybrid_pgo_ds.fusion.dstbm import DSTBMFusion
from src.experiments.hybrid_pgo_ds.fusion.yager import YagerFusion
from src.experiments.single_agent_tbm.metrics import brier_score
from src.experiments.intel_lab.adapter import IntelLabAdapter
from src.experiments.intel_lab.loader import LaserScan


@dataclass(frozen=True)
class IntelLabResult:
    """Result for one fusion method on Intel Lab data."""

    arm_name: str
    metrics: dict[str, float]
    num_scans_processed: int


def _compute_paired_deltas(
    prob_maps: dict[str, np.ndarray],
    gt: np.ndarray,
    gt_mask: np.ndarray,
    metrics_requested: list[str],
    arm_bc: str = "bayesian_count",
    arm_ds: str = "dstbm",
) -> dict[str, float]:
    """Compute paired metric deltas using the effective-mask intersection.

    For each metric, computes mean(vals_ds) - mean(vals_bc) over the cells
    selected by ``_effective_mask`` (intersection of both arms' observed
    cells).  This ensures the point-estimate delta uses the SAME cell
    population as ``bootstrap_delta_ci``, preventing the point estimate
    from falling outside the bootstrap CI.

    Parameters
    ----------
    prob_maps:
        Raw (unmasked) occupancy probability maps keyed by arm name.
    gt:
        Binary ground truth grid.
    gt_mask:
        Boolean mask of cells with valid GT labels.
    metrics_requested:
        Metrics for which to compute paired deltas.
    arm_bc:
        Name of the Bayesian+count arm in *prob_maps*.
    arm_ds:
        Name of the DS/TBM arm in *prob_maps*.

    Returns
    -------
    dict[str, float]
        Metric name -> paired delta (DS/TBM minus Bayesian+count).
    """
    if arm_bc not in prob_maps or arm_ds not in prob_maps:
        return {}

    prob_bc = prob_maps[arm_bc]
    prob_ds = prob_maps[arm_ds]

    # Only compute for metrics supported by bootstrap
    bootstrap_metrics = {"cell_accuracy", "brier_score", "boundary_sharpness"}
    paired = {}
    for metric in metrics_requested:
        if metric not in bootstrap_metrics:
            continue
        eff_mask = _effective_mask(gt_mask, prob_bc, prob_ds, metric, gt)
        vals_bc = _cell_values_for_metric(prob_bc, gt, eff_mask, metric)
        vals_ds = _cell_values_for_metric(prob_ds, gt, eff_mask, metric)
        paired[metric] = float(np.mean(vals_ds) - np.mean(vals_bc))

    return paired


def _create_method(
    name: str,
    l_occ: float = 2.0,
    l_free: float = -0.5,
    clamp: float = 10.0,
    m_of_min: float = 0.0,
) -> FusionMethod:
    """Create a fusion method by name.

    For DS/TBM, derives sensor masses from l_occ/l_free via pignistic transform
    so that BetP(O) = sigmoid(l) exactly, ensuring a fair comparison with Bayesian.
    """
    if name == "bayesian":
        return BayesianFusion(l_occ=l_occ, l_free=l_free, clamp=clamp)
    elif name == "bayesian_count":
        return BayesianCountFusion(l_occ=l_occ, l_free=l_free, clamp=clamp)
    elif name == "dstbm":
        m_occ = np.array(pignistic_masses(l_occ), dtype=np.float64)
        m_free = np.array(pignistic_masses(l_free), dtype=np.float64)
        return DSTBMFusion(m_occ=m_occ, m_free=m_free, m_of_min=m_of_min)
    elif name == "yager":
        m_occ = np.array(pignistic_masses(l_occ), dtype=np.float64)
        m_free = np.array(pignistic_masses(l_free), dtype=np.float64)
        return YagerFusion(m_occ=m_occ, m_free=m_free, m_of_min=m_of_min)
    else:
        msg = f"Unknown fusion method: {name}"
        raise ValueError(msg)


def _update_grid_from_scans(
    method: FusionMethod,
    grid: np.ndarray,
    scans: list[LaserScan],
    adapter: IntelLabAdapter,
) -> np.ndarray:
    """Update a grid with scans using the given fusion method."""
    for scan in scans:
        rays = adapter.scan_to_ray_cells(scan)
        for ray_cells, hit_cell, is_max in rays:
            method.update_cell(grid, ray_cells, hit_cell, is_max)
    return grid


def run_single_robot(
    adapter: IntelLabAdapter,
    arm_names: list[str],
    metrics_requested: list[str],
    l_occ: float = 2.0,
    l_free: float = -0.5,
    clamp: float = 10.0,
    m_of_min: float = 0.0,
) -> tuple[dict[str, IntelLabResult], dict[str, float]]:
    """Run single-robot experiment: all training scans processed sequentially.

    Parameters
    ----------
    adapter : IntelLabAdapter
        Configured adapter with loaded scans.
    arm_names : list[str]
        Fusion methods to test.
    metrics_requested : list[str]
        Metrics to compute.
    l_occ : float
        Bayesian log-odds for occupied hit. DS/TBM matched masses are derived
        from this value via pignistic transform.
    l_free : float
        Bayesian log-odds for free traversal. DS/TBM matched masses are derived
        from this value via pignistic transform.
    clamp : float
        Maximum absolute log-odds for Bayesian arms. Default 10.0.

    Returns
    -------
    tuple[dict[str, IntelLabResult], dict[str, float]]
        First element: results keyed by arm name.
        Second element: paired deltas (DS/TBM minus Bayesian+count)
        computed using the effective-mask intersection, keyed by metric
        name.  Empty if both ``bayesian_count`` and ``dstbm`` arms are
        not present.
    """
    gp = adapter.grid_params
    gt, gt_mask = adapter.get_ground_truth()
    effective_area = float(gt_mask.mean())
    train_scans = adapter.get_train_scans()
    results: dict[str, IntelLabResult] = {}
    prob_maps: dict[str, np.ndarray] = {}

    for arm_name in arm_names:
        method = _create_method(arm_name, l_occ=l_occ, l_free=l_free, clamp=clamp, m_of_min=m_of_min)
        grid = method.create_grid(gp.rows, gp.cols)

        grid = _update_grid_from_scans(method, grid, train_scans, adapter)
        prob = method.get_occupancy_probability(grid)
        prob_maps[arm_name] = prob

        # Apply GT mask: cells without valid GT labels are marked unknown so
        # they are excluded from all metric computations.
        prob_eval = prob.copy()
        prob_eval[~gt_mask] = 0.5
        gt_eval = gt.copy()
        gt_eval[~gt_mask] = 0.0

        # Use H-003 metrics for cell_accuracy and boundary_sharpness
        h003_metrics = [m for m in metrics_requested if m != "brier_score"]
        metrics = compute_metrics(
            occupancy_prob=prob_eval,
            ground_truth=gt_eval,
            requested=h003_metrics,
            raw_grid=grid,
            fusion_name=arm_name,
        )

        # Add brier_score from H-002 metrics if requested
        if "brier_score" in metrics_requested:
            metrics["brier_score"] = brier_score(prob_eval, gt_eval)

        metrics["effective_area_fraction"] = effective_area

        results[arm_name] = IntelLabResult(
            arm_name=arm_name,
            metrics=metrics,
            num_scans_processed=len(train_scans),
        )

    # Compute paired deltas using effective-mask intersection so that the
    # point estimate matches the bootstrap CI cell population (C3 fix).
    paired_deltas = _compute_paired_deltas(
        prob_maps, gt, gt_mask, metrics_requested,
    )

    return results, paired_deltas


def run_multi_robot_split(
    adapter: IntelLabAdapter,
    num_robots: int,
    arm_names: list[str],
    metrics_requested: list[str],
    l_occ: float = 2.0,
    l_free: float = -0.5,
    clamp: float = 10.0,
    m_of_min: float = 0.0,
) -> tuple[dict[str, IntelLabResult], dict[str, float]]:
    """Run multi-robot split experiment.

    Splits training scans into K segments (simulated robots), builds
    individual grids, then fuses them.

    Parameters
    ----------
    adapter : IntelLabAdapter
        Configured adapter.
    num_robots : int
        Number of simulated robots.
    arm_names : list[str]
        Fusion methods to test.
    metrics_requested : list[str]
        Metrics to compute.
    l_occ : float
        Bayesian log-odds for occupied hit. DS/TBM matched masses are derived
        from this value via pignistic transform.
    l_free : float
        Bayesian log-odds for free traversal. DS/TBM matched masses are derived
        from this value via pignistic transform.
    clamp : float
        Maximum absolute log-odds for Bayesian arms. Default 10.0.

    Returns
    -------
    tuple[dict[str, IntelLabResult], dict[str, float]]
        First element: results keyed by arm name.
        Second element: paired deltas (DS/TBM minus Bayesian+count)
        computed using the effective-mask intersection, keyed by metric
        name.  Empty if both ``bayesian_count`` and ``dstbm`` arms are
        not present.
    """
    gp = adapter.grid_params
    gt, gt_mask = adapter.get_ground_truth()
    effective_area = float(gt_mask.mean())
    segments = adapter.split_for_multi_robot(num_robots)
    total_scans = sum(len(seg) for seg in segments)
    results: dict[str, IntelLabResult] = {}
    prob_maps: dict[str, np.ndarray] = {}

    for arm_name in arm_names:
        method = _create_method(arm_name, l_occ=l_occ, l_free=l_free, clamp=clamp, m_of_min=m_of_min)

        # Build individual grids
        robot_grids: list[np.ndarray] = []
        for segment in segments:
            grid = method.create_grid(gp.rows, gp.cols)
            grid = _update_grid_from_scans(method, grid, segment, adapter)
            robot_grids.append(grid)

        # Fuse all grids
        fused = robot_grids[0]
        for i in range(1, len(robot_grids)):
            fused = method.fuse_grids(fused, robot_grids[i])

        prob = method.get_occupancy_probability(fused)
        prob_maps[arm_name] = prob

        # Apply GT mask: cells without valid GT labels are marked unknown so
        # they are excluded from all metric computations.
        prob_eval = prob.copy()
        prob_eval[~gt_mask] = 0.5
        gt_eval = gt.copy()
        gt_eval[~gt_mask] = 0.0

        h003_metrics = [m for m in metrics_requested if m != "brier_score"]
        metrics = compute_metrics(
            occupancy_prob=prob_eval,
            ground_truth=gt_eval,
            requested=h003_metrics,
            raw_grid=fused,
            fusion_name=arm_name,
        )

        if "brier_score" in metrics_requested:
            metrics["brier_score"] = brier_score(prob_eval, gt_eval)

        metrics["effective_area_fraction"] = effective_area

        results[arm_name] = IntelLabResult(
            arm_name=arm_name,
            metrics=metrics,
            num_scans_processed=total_scans,
        )

    # Compute paired deltas using effective-mask intersection so that the
    # point estimate matches the bootstrap CI cell population (C3 fix).
    paired_deltas = _compute_paired_deltas(
        prob_maps, gt, gt_mask, metrics_requested,
    )

    return results, paired_deltas


# ---------------------------------------------------------------------------
# Probability-map builders (for bootstrap CI computation)
# ---------------------------------------------------------------------------

def build_single_robot_prob_maps(
    adapter: IntelLabAdapter,
    arm_names: list[str],
    l_occ: float = 2.0,
    l_free: float = -0.5,
    clamp: float = 10.0,
    m_of_min: float = 0.0,
) -> dict[str, np.ndarray]:
    """Build raw occupancy probability maps for each arm (single-robot).

    Returns the unmasked probability arrays keyed by arm name.
    Intended for use with ``bootstrap_delta_ci``.
    """
    gp = adapter.grid_params
    train_scans = adapter.get_train_scans()
    prob_maps: dict[str, np.ndarray] = {}
    for arm_name in arm_names:
        method = _create_method(arm_name, l_occ=l_occ, l_free=l_free, clamp=clamp, m_of_min=m_of_min)
        grid = method.create_grid(gp.rows, gp.cols)
        grid = _update_grid_from_scans(method, grid, train_scans, adapter)
        prob_maps[arm_name] = method.get_occupancy_probability(grid)
    return prob_maps


def build_multi_robot_prob_maps(
    adapter: IntelLabAdapter,
    num_robots: int,
    arm_names: list[str],
    l_occ: float = 2.0,
    l_free: float = -0.5,
    clamp: float = 10.0,
    m_of_min: float = 0.0,
) -> dict[str, np.ndarray]:
    """Build raw occupancy probability maps for each arm (multi-robot split).

    Returns the unmasked probability arrays keyed by arm name.
    Intended for use with ``bootstrap_delta_ci``.
    """
    gp = adapter.grid_params
    segments = adapter.split_for_multi_robot(num_robots)
    prob_maps: dict[str, np.ndarray] = {}
    for arm_name in arm_names:
        method = _create_method(arm_name, l_occ=l_occ, l_free=l_free, clamp=clamp, m_of_min=m_of_min)
        robot_grids: list[np.ndarray] = []
        for segment in segments:
            grid = method.create_grid(gp.rows, gp.cols)
            grid = _update_grid_from_scans(method, grid, segment, adapter)
            robot_grids.append(grid)
        fused = robot_grids[0]
        for i in range(1, len(robot_grids)):
            fused = method.fuse_grids(fused, robot_grids[i])
        prob_maps[arm_name] = method.get_occupancy_probability(fused)
    return prob_maps


# ---------------------------------------------------------------------------
# Bootstrap CI computation
# ---------------------------------------------------------------------------

_UNKNOWN_EPSILON: float = 1e-4  # must match brier_score / cell_accuracy convention


def _effective_mask(
    gt_mask: np.ndarray,
    prob_bc: np.ndarray,
    prob_ds: np.ndarray,
    metric_name: str,
    gt: np.ndarray,
) -> np.ndarray:
    """Return the boolean mask of cells used to evaluate *metric_name*.

    Matches the cell selection logic of the original metric functions:

    cell_accuracy / brier_score
        Only cells that are "observed" by BOTH arms are included:
        ``|prob - 0.5| > UNKNOWN_EPSILON`` for each arm, intersected with
        gt_mask.  Using the intersection makes the paired bootstrap symmetric.

    boundary_sharpness
        GT wall cells within gt_mask (independent of arm probability values).
    """
    if metric_name == "boundary_sharpness":
        return gt_mask & (gt >= 0.5)

    # cell_accuracy and brier_score: use observed cells in gt_mask
    obs_bc = np.abs(prob_bc - 0.5) > _UNKNOWN_EPSILON
    obs_ds = np.abs(prob_ds - 0.5) > _UNKNOWN_EPSILON
    return gt_mask & obs_bc & obs_ds


def _cell_values_for_metric(
    prob: np.ndarray,
    gt: np.ndarray,
    eff_mask: np.ndarray,
    metric_name: str,
) -> np.ndarray:
    """Precompute per-cell scalar values for bootstrap resampling.

    For cell_accuracy: boolean correctness at each effective-mask cell.
    For brier_score: squared error at each effective-mask cell.
    For boundary_sharpness: gradient magnitude at GT wall cells.

    All arrays are float64 for consistent mean computation.

    Parameters
    ----------
    prob:
        Occupancy probability map (unmasked).
    gt:
        Binary ground truth grid.
    eff_mask:
        Pre-computed effective mask from ``_effective_mask()``.
    metric_name:
        One of 'cell_accuracy', 'brier_score', 'boundary_sharpness'.
    """
    if metric_name == "cell_accuracy":
        pred = (prob[eff_mask] > 0.5)
        ref = (gt[eff_mask] >= 0.5)
        return (pred == ref).astype(np.float64)

    if metric_name == "brier_score":
        p = prob[eff_mask].astype(np.float64)
        g = (gt[eff_mask].astype(np.float64) >= 0.5).astype(np.float64)
        return (p - g) ** 2

    if metric_name == "boundary_sharpness":
        p_full = prob.astype(np.float64)
        gy, gx = np.gradient(p_full)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        if not np.any(eff_mask):
            return np.full(1, float("nan"))
        return grad_mag[eff_mask]

    msg = f"Unsupported metric for bootstrap: {metric_name}"
    raise ValueError(msg)


def bootstrap_delta_ci(
    prob_bc: np.ndarray,
    prob_ds: np.ndarray,
    gt: np.ndarray,
    gt_mask: np.ndarray,
    metrics: list[str],
    n_iter: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CIs for metric deltas: DS/TBM minus Bayesian+count.

    Resamples evaluated cells with replacement using a paired bootstrap:
    the same cell indices are used for both arms in each iteration.

    Cell selection matches the original metric functions:
    - ``cell_accuracy`` / ``brier_score``: cells in gt_mask observed by both
      arms (``|prob - 0.5| > 1e-4``).
    - ``boundary_sharpness``: GT wall cells in gt_mask.

    Parameters
    ----------
    prob_bc:
        Occupancy probability map for Bayesian+count arm.
    prob_ds:
        Occupancy probability map for DS/TBM arm.
    gt:
        Binary ground truth grid (1.0=occupied, 0.0=free).
    gt_mask:
        Boolean mask of cells with valid GT labels.
    metrics:
        Metrics to compute. Supported: cell_accuracy, brier_score,
        boundary_sharpness.
    n_iter:
        Number of bootstrap resamples (default 1000).
    confidence:
        Confidence level (default 0.95 for 95% CI).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[str, tuple[float, float]]
        Metric name -> (lower_bound, upper_bound) for the delta CI.
        Negative delta = DS/TBM worse (for accuracy; for brier, higher=worse).
    """
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    cis: dict[str, tuple[float, float]] = {}

    for metric in metrics:
        eff_mask = _effective_mask(gt_mask, prob_bc, prob_ds, metric, gt)
        vals_bc = _cell_values_for_metric(prob_bc, gt, eff_mask, metric)
        vals_ds = _cell_values_for_metric(prob_ds, gt, eff_mask, metric)
        n = len(vals_bc)

        deltas = np.empty(n_iter, dtype=np.float64)
        for i in range(n_iter):
            idx = rng.integers(0, n, size=n)
            deltas[i] = np.mean(vals_ds[idx]) - np.mean(vals_bc[idx])

        cis[metric] = (
            float(np.percentile(deltas, lo_pct)),
            float(np.percentile(deltas, hi_pct)),
        )

    return cis


def spatial_block_bootstrap_delta_ci(
    prob_bc: np.ndarray,
    prob_ds: np.ndarray,
    gt: np.ndarray,
    gt_mask: np.ndarray,
    metrics: list[str],
    block_size: int = 10,
    n_iter: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict]:
    """Spatial block bootstrap CIs for metric deltas (DS/TBM minus Bayesian+count).

    Divides the grid into non-overlapping B x B blocks, then resamples
    blocks with replacement.  This preserves spatial autocorrelation
    within blocks and yields conservative (wider) CIs compared to
    independent-cell resampling.

    Parameters
    ----------
    prob_bc:
        Occupancy probability map for Bayesian+count arm (rows, cols).
    prob_ds:
        Occupancy probability map for DS/TBM arm (rows, cols).
    gt:
        Binary ground truth grid (1.0=occupied, 0.0=free).
    gt_mask:
        Boolean mask of cells with valid GT labels.
    metrics:
        Metrics to compute. Supported: cell_accuracy, brier_score,
        boundary_sharpness.
    block_size:
        Side length of each square block in cells.  Default 10 = 1.0 m
        at 0.1 m resolution, exceeding typical spatial autocorrelation
        range in occupancy grids (2-5 cells for lidar at 0.1 m).
    n_iter:
        Number of bootstrap resamples (default 10000).
    confidence:
        Confidence level (default 0.95 for 95% CI).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[str, dict]
        Metric name -> dict with keys: lo, hi, n_blocks, block_size.
    """
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    rows, cols = prob_bc.shape

    cis: dict[str, dict] = {}

    for metric in metrics:
        eff_mask = _effective_mask(gt_mask, prob_bc, prob_ds, metric, gt)
        vals_bc = _cell_values_for_metric(prob_bc, gt, eff_mask, metric)
        vals_ds = _cell_values_for_metric(prob_ds, gt, eff_mask, metric)

        # Build 2D coordinate arrays for effective cells
        eff_rows, eff_cols = np.where(eff_mask)

        # Assign each effective cell to a block
        block_row = eff_rows // block_size
        block_col = eff_cols // block_size

        # Group cell indices by block
        n_col_blocks = (cols + block_size - 1) // block_size
        block_ids = block_row * n_col_blocks + block_col
        unique_blocks = np.unique(block_ids)
        block_cell_indices: list[np.ndarray] = []
        for bid in unique_blocks:
            mask = block_ids == bid
            block_cell_indices.append(np.where(mask)[0])

        n_blocks = len(block_cell_indices)
        if n_blocks == 0:
            cis[metric] = {
                "lo": float("nan"),
                "hi": float("nan"),
                "n_blocks": 0,
                "block_size": block_size,
            }
            continue

        deltas = np.empty(n_iter, dtype=np.float64)
        for i in range(n_iter):
            sampled = rng.integers(0, n_blocks, size=n_blocks)
            idx = np.concatenate([block_cell_indices[s] for s in sampled])
            deltas[i] = np.mean(vals_ds[idx]) - np.mean(vals_bc[idx])

        cis[metric] = {
            "lo": float(np.percentile(deltas, lo_pct)),
            "hi": float(np.percentile(deltas, hi_pct)),
            "n_blocks": n_blocks,
            "block_size": block_size,
        }

    return cis
