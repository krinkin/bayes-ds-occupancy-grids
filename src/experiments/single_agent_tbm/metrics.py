"""H-002 experiment metrics.

Inc 2: pignistic / Bayesian uncertainty.
Inc 3: Brier score, frontier quality (F1).

Uncertainty measures distinguish the three experimental arms:
- Plain Bayesian and Bayesian+count use u_Bayes = 1 - |2*P(occ) - 1|
- DS/TBM uses u_pign = 1 - |2*BetP(O) - 1|  where BetP(O) = m_O + m_OF/2

Brier score measures calibration: mean((prob - gt)^2) over observed cells.
Frontier quality measures F1 of frontier detection vs ground truth.
"""

from __future__ import annotations

import numpy as np


def pignistic_uncertainty(grid: np.ndarray) -> np.ndarray:
    """Per-cell pignistic uncertainty for a TBM grid.

    u_pign = 1 - |2 * BetP(O) - 1|  where  BetP(O) = m_O + m_OF / 2.

    Parameters
    ----------
    grid:
        TBM grid, shape ``(rows, cols, 3)``.
        Channels: ``[m_O, m_F, m_OF]``.

    Returns
    -------
    np.ndarray, shape ``(rows, cols)``, dtype float64
        Per-cell uncertainty in [0, 1].
        1.0 = maximally uncertain (BetP = 0.5).
        0.0 = fully confident (BetP = 0 or 1).
    """
    m_O = grid[:, :, 0].astype(np.float64)
    m_OF = grid[:, :, 2].astype(np.float64)
    betp = m_O + m_OF / 2.0
    return 1.0 - np.abs(2.0 * betp - 1.0)


def bayesian_uncertainty(prob_grid: np.ndarray) -> np.ndarray:
    """Per-cell uncertainty for a Bayesian probability grid.

    u_Bayes = 1 - |2 * P(occ) - 1|.

    Parameters
    ----------
    prob_grid:
        Occupancy probability grid, shape ``(rows, cols)``, values in [0, 1].

    Returns
    -------
    np.ndarray, shape ``(rows, cols)``, dtype float64
        Per-cell uncertainty in [0, 1].
    """
    p = prob_grid.astype(np.float64)
    return 1.0 - np.abs(2.0 * p - 1.0)


def dynamic_detection_score(
    uncertainty_grid: np.ndarray,
    gt_dynamic_mask: np.ndarray | None,
) -> float:
    """Ratio of mean uncertainty at dynamic cells to mean uncertainty at stable cells.

    A ratio > 1 means the method detects dynamics (elevated uncertainty at
    cells that changed state due to moving obstacles).

    Parameters
    ----------
    uncertainty_grid:
        Per-cell uncertainty, shape ``(rows, cols)``.
    gt_dynamic_mask:
        Boolean mask of cells affected by dynamic objects, or None.
        True = cell was occupied by a moving object at some step.

    Returns
    -------
    float
        Ratio ``mean_unc(dynamic) / mean_unc(stable)``.
        NaN if no dynamic cells, no stable cells, or mask is None.
    """
    if gt_dynamic_mask is None or not np.any(gt_dynamic_mask):
        return float("nan")

    stable_mask = ~gt_dynamic_mask
    if not np.any(stable_mask):
        return float("nan")

    mean_dynamic = float(np.mean(uncertainty_grid[gt_dynamic_mask]))
    mean_stable = float(np.mean(uncertainty_grid[stable_mask]))

    if mean_stable < 1e-10:
        return float("nan")

    return mean_dynamic / mean_stable


# ---------------------------------------------------------------------------
# Brier score (calibration)
# ---------------------------------------------------------------------------

def brier_score(
    prob_grid: np.ndarray,
    gt_binary: np.ndarray,
    unknown_epsilon: float = 1e-4,
) -> float:
    """Mean squared error between predicted probability and ground truth.

    Only observed cells (where ``|prob - 0.5| > unknown_epsilon``) are
    included, consistent with the ``cell_accuracy`` convention.

    Lower is better.  Perfect predictions score 0.0; uniform 0.5 predictions
    on balanced data score 0.25.

    Parameters
    ----------
    prob_grid:
        Occupancy probability grid, shape ``(rows, cols)``, values in [0, 1].
        For Bayesian: sigmoid(log_odds).
        For TBM: BetP(O) = m_O + m_OF / 2.
    gt_binary:
        Ground-truth binary occupancy, shape ``(rows, cols)``.
        1.0 = occupied, 0.0 = free.
    unknown_epsilon:
        Half-width of the "unknown" band around 0.5.

    Returns
    -------
    float
        Brier score in [0, 1].  NaN if no observed cells.
    """
    observed = np.abs(prob_grid.astype(np.float64) - 0.5) > unknown_epsilon
    if not np.any(observed):
        return float("nan")

    p = prob_grid.astype(np.float64)[observed]
    gt = (gt_binary.astype(np.float64) >= 0.5).astype(np.float64)[observed]
    return float(np.mean((p - gt) ** 2))


# ---------------------------------------------------------------------------
# Frontier quality (F1)
# ---------------------------------------------------------------------------

def _adjacent_to(mask: np.ndarray) -> np.ndarray:
    """Boolean array: True where at least one 4-neighbor is True in *mask*."""
    adj = np.zeros_like(mask, dtype=bool)
    adj[:-1, :] |= mask[1:, :]
    adj[1:, :] |= mask[:-1, :]
    adj[:, :-1] |= mask[:, 1:]
    adj[:, 1:] |= mask[:, :-1]
    return adj


def _unobserved_mask(
    prob_grid: np.ndarray,
    raw_grid: np.ndarray | None,
    method_name: str,
    unknown_epsilon: float = 1e-4,
    ignorance_threshold: float = 0.5,
) -> np.ndarray:
    """Boolean mask of cells identified as unobserved by a given method.

    - Bayesian: prob in [0.5 - eps, 0.5 + eps] (cannot distinguish
      unobserved from conflicted).
    - BayesianCount: observation count == 0.
    - DSTBM: m_OF > ignorance_threshold.
    """
    if (
        method_name == "bayesian_count"
        and raw_grid is not None
        and raw_grid.ndim == 3
        and raw_grid.shape[2] >= 2
    ):
        return raw_grid[:, :, 1] == 0.0

    if (
        method_name == "dstbm"
        and raw_grid is not None
        and raw_grid.ndim == 3
        and raw_grid.shape[2] >= 3
    ):
        return raw_grid[:, :, 2] > ignorance_threshold

    # Plain bayesian: only criterion is near-0.5 probability
    return np.abs(prob_grid.astype(np.float64) - 0.5) <= unknown_epsilon


def frontier_quality(
    prob_grid: np.ndarray,
    raw_grid: np.ndarray | None,
    gt_grid: np.ndarray,
    method_name: str,
    unknown_epsilon: float = 1e-4,
    ignorance_threshold: float = 0.5,
) -> float:
    """F1 score of frontier detection versus ground-truth frontiers.

    A *frontier cell* is an observed cell adjacent to at least one
    unobserved cell.  The definition of "unobserved" differs by method:

    - Bayesian: ``|prob - 0.5| <= epsilon`` (weak -- conflicted cells
      look like frontiers).
    - BayesianCount: ``count == 0``.
    - DSTBM: ``m_OF > threshold``.

    Ground-truth frontiers are observed cells (``|prob - 0.5| > epsilon``)
    adjacent to cells that are genuinely unobserved in the estimated map
    AND are free in the ground truth (i.e., territory yet to be confirmed).

    Parameters
    ----------
    prob_grid:
        Occupancy probability grid, shape ``(rows, cols)``.
    raw_grid:
        Method-specific raw grid (BayesianCount: channel 1 = count;
        DSTBM: channel 2 = m_OF).  None for plain Bayesian.
    gt_grid:
        Ground-truth binary occupancy grid.
    method_name:
        ``"bayesian"``, ``"bayesian_count"``, or ``"dstbm"``.
    unknown_epsilon:
        Half-width of the near-0.5 band.
    ignorance_threshold:
        m_OF threshold for DSTBM unobserved classification.

    Returns
    -------
    float
        F1 score in [0, 1].  NaN if no ground-truth frontier cells exist.
    """
    p = prob_grid.astype(np.float64)

    # Observed cells: the method has formed an opinion
    observed = np.abs(p - 0.5) > unknown_epsilon

    # Unobserved cells per method
    unobs = _unobserved_mask(
        prob_grid, raw_grid, method_name,
        unknown_epsilon=unknown_epsilon,
        ignorance_threshold=ignorance_threshold,
    )

    # Predicted frontier: observed AND adjacent to unobserved
    pred_frontier = observed & _adjacent_to(unobs)

    # Ground-truth frontier: observed AND adjacent to (unobserved AND gt-free)
    # "genuinely unexplored" = unobserved in estimated map AND free in GT
    genuinely_unexplored = unobs & (gt_grid < 0.5)
    gt_frontier = observed & _adjacent_to(genuinely_unexplored)

    n_gt = int(np.sum(gt_frontier))
    n_pred = int(np.sum(pred_frontier))

    if n_gt == 0:
        return float("nan")
    if n_pred == 0:
        return 0.0

    tp = int(np.sum(pred_frontier & gt_frontier))
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0

    if precision + recall < 1e-10:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)
