# TDP: Full 8-metric suite for H-003 Increment 4 (revised after Inc 4 review)
#
# All metrics take occupancy_prob (float32 array, values in [0,1]) as primary
# input. This is the canonical output of FusionMethod.get_occupancy_probability()
# and works uniformly across all three fusion arms. Method-specific raw grids
# are accepted optionally for metrics that can exploit them (dynamic_detection
# uses DS m_OF when available; uncertainty_awareness uses count or m_OF).
#
# Metrics:
#   1. cell_accuracy:         fraction of observed cells classified correctly
#   2. map_entropy:           mean binary entropy H(p) -- higher = more uncertain
#   3. frontier_quality:      fraction of free cells bordering unknown cells
#   4. boundary_sharpness:    mean gradient magnitude at GT wall cells
#   5. dynamic_detection:     mean DS ignorance mass m_OF (or uncertainty proxy)
#   6. resource_cost:         fused grid memory in MB (from get_memory_bytes)
#   7. fault_tolerance:       cell_accuracy(N-1 robots) / cell_accuracy(N robots)
#                             (computed by runner and passed as fault_tolerance_score)
#   8. uncertainty_awareness: fraction of near-0.5 cells that can be identified
#                             as truly unknown (count=0 or m_OF>threshold).
#                             Bayesian=0 (cannot distinguish unknown/conflicting).
#                             Bayesian+count and DS/TBM score > 0 when unobserved
#                             cells exist. Directly tests the research question:
#                             does count/mass tracking provide epistemic benefit?
#
# Alternatives considered:
#   cell_accuracy on log_odds (old interface) -- not method-agnostic, rejected.
#   boundary_sharpness via Canny edge detection -- requires scikit-image, simpler
#     gradient (np.gradient) preferred to avoid extra dependency.
#   dynamic_detection as fraction of cells with m_OF > threshold -- mean is
#     smoother and avoids threshold tuning.
#   resource_cost as composite score (memory + time) -- two separate floats are
#     cleaner; resource_cost reports memory_mb, timing is logged separately.
#   uncertainty_awareness alternatives: frontier_quality_v2 (weighting frontiers
#     by epistemic state) or information_gain (mean count) -- chosen current
#     approach because it directly answers "can the method distinguish states?"
"""Experiment metrics for H-003: all 7 metrics from the research verdict.

Each metric takes ``occupancy_prob`` (shape ``(rows, cols)``, dtype float32,
values in [0, 1]) as its primary input so it works uniformly across all three
fusion arms.  Method-specific data is accepted as optional extra arguments.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Metric 1: Cell classification accuracy
# ---------------------------------------------------------------------------

def cell_accuracy(
    occupancy_prob: np.ndarray,
    ground_truth: np.ndarray,
    occ_threshold: float = 0.5,
    unknown_epsilon: float = 1e-4,
) -> float:
    """Fraction of observed cells classified correctly versus ground truth.

    Cells with ``|prob - 0.5| <= unknown_epsilon`` are treated as unobserved
    and excluded from the calculation.

    Parameters
    ----------
    occupancy_prob:
        Occupancy probability grid, shape (rows, cols), values in [0, 1].
        0.5 encodes unknown / no information.
    ground_truth:
        Binary occupancy map, shape (rows, cols). 1.0 = occupied, 0.0 = free.
    occ_threshold:
        Probability above which a cell is classified as occupied.
    unknown_epsilon:
        Half-width of the "unknown" band around 0.5.  Cells within this band
        are excluded from accuracy computation.

    Returns
    -------
    float
        Classification accuracy in [0, 1]. NaN if no cells observed.
    """
    observed_mask = np.abs(occupancy_prob - 0.5) > unknown_epsilon
    if not np.any(observed_mask):
        return float("nan")

    estimated = (occupancy_prob[observed_mask] > occ_threshold).astype(np.float32)
    reference = (ground_truth[observed_mask] >= 0.5).astype(np.float32)

    correct = float(np.sum(estimated == reference))
    total = float(observed_mask.sum())
    return correct / total


# ---------------------------------------------------------------------------
# Metric 2: Map entropy
# ---------------------------------------------------------------------------

def map_entropy(occupancy_prob: np.ndarray) -> float:
    """Mean binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p) over all cells.

    Higher entropy indicates more uncertain/unexplored cells.  DS/TBM retains
    higher m_OF (BetP(O) = 0.5) in unobserved cells, which in static scenes
    should keep entropy near 1.0 there.

    Parameters
    ----------
    occupancy_prob:
        Occupancy probability grid, shape (rows, cols), values in [0, 1].

    Returns
    -------
    float
        Mean binary entropy in [0, 1].
    """
    p = np.clip(occupancy_prob.astype(np.float64), 1e-10, 1.0 - 1e-10)
    H = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    return float(np.mean(H))


# ---------------------------------------------------------------------------
# Metric 3: Frontier quality
# ---------------------------------------------------------------------------

def _adjacent_to(mask: np.ndarray) -> np.ndarray:
    """Boolean array: True where at least one 4-neighbor is True in *mask*."""
    adj = np.zeros_like(mask, dtype=bool)
    adj[:-1, :] |= mask[1:, :]   # north neighbor
    adj[1:, :] |= mask[:-1, :]   # south neighbor
    adj[:, :-1] |= mask[:, 1:]   # east neighbor
    adj[:, 1:] |= mask[:, :-1]   # west neighbor
    return adj


def frontier_quality(
    occupancy_prob: np.ndarray,
    unknown_low: float = 0.45,
    unknown_high: float = 0.55,
    free_threshold: float = 0.4,
) -> float:
    """Fraction of free cells that border unknown cells (exploration frontiers).

    A higher score means more of the free space is adjacent to unexplored
    territory, indicating an active exploration frontier.  DS/TBM retains
    more unknown cells (BetP(O) = 0.5 until first observation) so its
    frontier tends to be larger relative to the explored area.

    Parameters
    ----------
    occupancy_prob:
        Occupancy probability grid, shape (rows, cols).
    unknown_low, unknown_high:
        Cells with probability in [unknown_low, unknown_high] are unknown.
    free_threshold:
        Cells with probability < free_threshold are classified as free.

    Returns
    -------
    float
        Frontier fraction in [0, 1]. NaN if no free cells observed.
    """
    p = occupancy_prob
    unknown_mask = (p >= unknown_low) & (p <= unknown_high)
    free_mask = p < free_threshold

    unknown_adj = _adjacent_to(unknown_mask)
    frontier_mask = free_mask & unknown_adj

    n_free = int(np.sum(free_mask))
    if n_free == 0:
        return float("nan")
    return float(np.sum(frontier_mask)) / float(n_free)


# ---------------------------------------------------------------------------
# Metric 4: Boundary sharpness
# ---------------------------------------------------------------------------

def boundary_sharpness(
    occupancy_prob: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """Mean gradient magnitude of the occupancy map at ground-truth wall cells.

    A sharp map has high gradient at occupied/free transitions.  Residual
    alignment error smears the probability across nearby cells, lowering the
    gradient.

    Parameters
    ----------
    occupancy_prob:
        Occupancy probability grid, shape (rows, cols).
    ground_truth:
        Binary occupancy map, shape (rows, cols).

    Returns
    -------
    float
        Mean gradient magnitude at wall cells. NaN if no wall cells.
    """
    wall_mask = ground_truth >= 0.5
    if not np.any(wall_mask):
        return float("nan")

    p = occupancy_prob.astype(np.float64)
    gy, gx = np.gradient(p)
    gradient_mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(gradient_mag[wall_mask]))


# ---------------------------------------------------------------------------
# Metric 5: Dynamic cell detection
# ---------------------------------------------------------------------------

def _uncertainty_grid(
    occupancy_prob: np.ndarray,
    raw_grid: np.ndarray | None,
    fusion_name: str,
) -> np.ndarray:
    """Return per-cell uncertainty as a float64 array.

    DS/TBM: m_OF (ignorance mass) from raw_grid channel 2.
    Bayesian/BayesianCount: 1 - |2p - 1| (epistemic uncertainty from probability).
    """
    if (
        fusion_name == "dstbm"
        and raw_grid is not None
        and raw_grid.ndim == 3
        and raw_grid.shape[2] >= 3
    ):
        return raw_grid[:, :, 2].astype(np.float64)
    return 1.0 - np.abs(2.0 * occupancy_prob.astype(np.float64) - 1.0)


def dynamic_detection(
    occupancy_prob: np.ndarray,
    raw_grid: np.ndarray | None = None,
    fusion_name: str = "",
    dynamic_mask: np.ndarray | None = None,
) -> float:
    """Uncertainty ratio between dynamic and stable cells (detection power).

    When *dynamic_mask* is provided and contains at least one True cell,
    computes::

        ratio = mean_uncertainty(dynamic_cells) / mean_uncertainty(stable_cells)

    A ratio > 1 means the method shows elevated uncertainty in cells affected
    by dynamic objects, indicating detection capability.

    When *dynamic_mask* is None or all-False, falls back to the global mean
    uncertainty (backward compatible with previous behaviour).

    For DS/TBM:
        Uncertainty = m_OF (ignorance mass) from raw_grid channel 2.

    For Bayesian / BayesianCount:
        Uncertainty = 1 - |2*p - 1|.

    Parameters
    ----------
    occupancy_prob:
        Occupancy probability grid, shape (rows, cols).
    raw_grid:
        Raw method-specific grid, used only for DS/TBM.
    fusion_name:
        Name of the fusion method (e.g., 'dstbm').
    dynamic_mask:
        Boolean mask, True for cells affected by dynamic objects.
        When None or all-False, falls back to global mean.

    Returns
    -------
    float
        Uncertainty ratio (targeted) or global mean (fallback).
    """
    unc = _uncertainty_grid(occupancy_prob, raw_grid, fusion_name)

    if dynamic_mask is not None and np.any(dynamic_mask):
        stable_mask = ~dynamic_mask
        mean_dynamic = float(np.mean(unc[dynamic_mask]))
        mean_stable = float(np.mean(unc[stable_mask])) if np.any(stable_mask) else 1e-10
        if mean_stable < 1e-10:
            mean_stable = 1e-10
        return mean_dynamic / mean_stable

    return float(np.mean(unc))


# ---------------------------------------------------------------------------
# Metric 8: Uncertainty awareness
# ---------------------------------------------------------------------------

def uncertainty_awareness(
    occupancy_prob: np.ndarray,
    raw_grid: np.ndarray | None = None,
    fusion_name: str = "",
    unknown_low: float = 0.45,
    unknown_high: float = 0.55,
    ignorance_threshold: float = 0.5,
) -> float:
    """Fraction of near-0.5 cells identifiable as truly unknown (never observed).

    Distinguishes Bayesian+count from plain Bayesian by measuring how many
    near-0.5 cells are genuinely unexplored rather than conflicting-evidence
    cells (count>0 but prob still ~0.5 due to mixed measurements).

    Bayesian: 0.0 -- no count tracking, cannot distinguish unknown from conflicting.
    Bayesian+count: fraction of near-0.5 cells with observation count == 0.
    DS/TBM: fraction of near-0.5 cells with m_OF > ignorance_threshold.

    This directly tests the research question from the H-003 verdict:
    does count / mass tracking provide an epistemic advantage over plain
    Bayesian log-odds when cells have near-0.5 probability?

    Parameters
    ----------
    occupancy_prob:
        Occupancy probability grid, shape (rows, cols), values in [0, 1].
    raw_grid:
        Raw internal grid.  Channel 1 = observation count for bayesian_count;
        channel 2 = m_OF for dstbm.
    fusion_name:
        Name of the fusion method.
    unknown_low, unknown_high:
        Band around 0.5 defining "near-0.5" cells.
    ignorance_threshold:
        m_OF threshold above which a DS/TBM cell is classified as truly unknown.

    Returns
    -------
    float
        Fraction in [0, 1], or NaN if no near-0.5 cells exist.
        Plain Bayesian always returns 0.0.
    """
    p = occupancy_prob
    near_half_mask = (p >= unknown_low) & (p <= unknown_high)
    n_near_half = int(np.sum(near_half_mask))
    if n_near_half == 0:
        return float("nan")

    if (
        fusion_name == "bayesian_count"
        and raw_grid is not None
        and raw_grid.ndim == 3
        and raw_grid.shape[2] >= 2
    ):
        count = raw_grid[:, :, 1]
        truly_unknown = count[near_half_mask] == 0
        return float(np.sum(truly_unknown)) / float(n_near_half)

    if (
        fusion_name == "dstbm"
        and raw_grid is not None
        and raw_grid.ndim == 3
        and raw_grid.shape[2] >= 3
    ):
        m_OF = raw_grid[:, :, 2]
        truly_unknown = m_OF[near_half_mask] > ignorance_threshold
        return float(np.sum(truly_unknown)) / float(n_near_half)

    # Bayesian (or unknown method): no count / mass info -> cannot identify
    return 0.0


# ---------------------------------------------------------------------------
# Metric 6: Resource cost
# ---------------------------------------------------------------------------

def resource_cost(memory_bytes: int | None) -> float:
    """Fused grid memory in megabytes.

    Measures the peak memory cost of the fused occupancy grid.
    Bayesian: 1 float32/cell = 4 bytes.
    BayesianCount: 2 float32/cell = 8 bytes.
    DS/TBM: 3 float32/cell = 12 bytes.

    Parameters
    ----------
    memory_bytes:
        Grid byte count from FusionMethod.get_memory_bytes().

    Returns
    -------
    float
        Memory in megabytes. NaN if not provided.
    """
    if memory_bytes is None:
        return float("nan")
    return float(memory_bytes) / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def compute_metrics(
    *,
    occupancy_prob: np.ndarray,
    ground_truth: np.ndarray,
    requested: list[str],
    raw_grid: np.ndarray | None = None,
    fusion_name: str = "",
    memory_bytes: int | None = None,
    fault_tolerance_score: float | None = None,
    dynamic_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all requested metrics and return a name->value dict.

    Parameters
    ----------
    occupancy_prob:
        Occupancy probability grid from FusionMethod.get_occupancy_probability().
    ground_truth:
        Binary ground-truth occupancy map.
    requested:
        List of metric names to compute.  Unrecognised names produce NaN.
    raw_grid:
        Raw internal grid for method-specific metrics (optional).
    fusion_name:
        Name of the fusion method (e.g., 'bayesian', 'dstbm').
    memory_bytes:
        Grid memory in bytes for resource_cost metric.
    fault_tolerance_score:
        Pre-computed fault-tolerance score from the runner (optional).
        If None, fault_tolerance is reported as NaN.
    dynamic_mask:
        Boolean mask of cells affected by dynamic objects (optional).
        Passed through to dynamic_detection for targeted ratio computation.

    Returns
    -------
    dict[str, float]
        Metric name -> value.  NaN for unavailable metrics.
    """
    results: dict[str, float] = {}
    for name in requested:
        if name == "cell_accuracy":
            results[name] = cell_accuracy(occupancy_prob, ground_truth)
        elif name == "map_entropy":
            results[name] = map_entropy(occupancy_prob)
        elif name == "frontier_quality":
            results[name] = frontier_quality(occupancy_prob)
        elif name == "boundary_sharpness":
            results[name] = boundary_sharpness(occupancy_prob, ground_truth)
        elif name == "dynamic_detection":
            results[name] = dynamic_detection(
                occupancy_prob, raw_grid, fusion_name, dynamic_mask=dynamic_mask,
            )
        elif name == "uncertainty_awareness":
            results[name] = uncertainty_awareness(occupancy_prob, raw_grid, fusion_name)
        elif name == "resource_cost":
            results[name] = resource_cost(memory_bytes)
        elif name == "fault_tolerance":
            results[name] = (
                fault_tolerance_score
                if fault_tolerance_score is not None
                else float("nan")
            )
        else:
            results[name] = float("nan")
    return results
