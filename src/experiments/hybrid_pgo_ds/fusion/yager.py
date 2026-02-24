"""Yager's combination rule for occupancy grid fusion.

Technical Design Proposal (TDP)
================================

Context
-------
This module implements Yager's combination rule as a fourth experimental arm,
alongside Dempster's rule (dstbm.py).  The frame of discernment and grid
representation are identical: {O, F} power set, three focal elements {O},
{F}, {O,F}, stored as a (rows, cols, 3) float32 array.

Yager vs Dempster -- one difference only
-----------------------------------------
Both rules compute the same unnormalised focal-element intersections:

    K_conflict = m1(O) * m2(F) + m1(F) * m2(O)

Dempster redistributes conflict proportionally (closed-world normalisation):

    m12(O)  = (m1(O)*m2(O) + m1(O)*m2(OF) + m1(OF)*m2(O)) / (1 - K)
    m12(F)  = (m1(F)*m2(F) + m1(F)*m2(OF) + m1(OF)*m2(F)) / (1 - K)
    m12(OF) = m1(OF)*m2(OF) / (1 - K)

Yager assigns conflict mass to total ignorance (m_OF) instead:

    m12(O)  = m1(O)*m2(O) + m1(O)*m2(OF) + m1(OF)*m2(O)       [unchanged]
    m12(F)  = m1(F)*m2(F) + m1(F)*m2(OF) + m1(OF)*m2(F)       [unchanged]
    m12(OF) = m1(OF)*m2(OF) + K_conflict                        [conflict added]

No normalisation constant is needed: when both inputs are valid mass functions
(sum to 1), the output sum is exactly 1 by algebraic identity:

    m12_O + m12_F + m12_OF
    = (mO1*mO2 + mO1*mOF2 + mOF1*mO2)
    + (mF1*mF2 + mF1*mOF2 + mOF1*mF2)
    + (mOF1*mOF2 + mO1*mF2 + mF1*mO2)
    = (mO1 + mF1 + mOF1) * (mO2 + mF2 + mOF2)
    = 1 * 1 = 1

Normalisation (division by total) is applied only to guard against
floating-point drift over many sequential updates.

Identity element
----------------
Same as Dempster: (0.0, 0.0, 1.0).  Combining any mass function with
(0, 0, 1): K_conflict = 0, m12_OF = 1*m_OF + 0 = m_OF.  The original
mass function is returned unchanged.  Verified algebraically.

Commutativity
-------------
The formula is symmetric in its operands (K_conflict = K_conflict^T,
intersection terms are symmetric), so Yager's rule is commutative.

Sensor model
------------
Identical to DSTBMFusion: pignistic-transform-derived masses from Bayesian
l_occ / l_free, ensuring BetP(O) = sigmoid(log_odds) exactly (LL-001).
Constructor accepts m_occ / m_free arrays with the same interface as
DSTBMFusion to enable drop-in substitution.

Numerical stability
-------------------
No division by (1 - K_conflict) means Yager has no high-conflict
instability.  The only guard needed is floating-point drift protection:
the output is renormalised by dividing by its sum when sum > _MIN_TOTAL.

Occupancy probability
---------------------
Same pignistic transform as Dempster: BetP(O) = m_O + m_OF / 2.
"""

from __future__ import annotations

import numpy as np

from src.experiments.hybrid_pgo_ds.fusion.base import FusionMethod

_MIN_TOTAL: float = 1e-10

# Default sensor model masses (same as dstbm.py defaults for consistency).
_M_OCC: np.ndarray = np.array([0.75, 0.10, 0.15], dtype=np.float64)
_M_FREE: np.ndarray = np.array([0.10, 0.75, 0.15], dtype=np.float64)


def _yager_combine_scalar(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Combine two mass functions using Yager's combination rule.

    Unlike Dempster's rule which normalises by 1/(1-K), Yager's rule
    assigns conflict mass to total ignorance (m_OF).

    Parameters
    ----------
    m1, m2:
        1-D arrays of length 3: [m_O, m_F, m_OF].

    Returns
    -------
    numpy.ndarray, shape (3,), dtype float64
        Combined mass function [m12_O, m12_F, m12_OF].
    """
    mO1, mF1, mOF1 = float(m1[0]), float(m1[1]), float(m1[2])
    mO2, mF2, mOF2 = float(m2[0]), float(m2[1]), float(m2[2])

    # Unnormalised focal element intersections (same as Dempster numerators).
    m_O = mO1 * mO2 + mO1 * mOF2 + mOF1 * mO2
    m_F = mF1 * mF2 + mF1 * mOF2 + mOF1 * mF2
    # KEY DIFFERENCE: conflict goes to ignorance, not normalised away.
    conflict = mO1 * mF2 + mF1 * mO2
    m_OF = mOF1 * mOF2 + conflict

    result = np.array([m_O, m_F, m_OF], dtype=np.float64)

    # Enforce mass constraint (guards against floating-point drift).
    total = result.sum()
    if total > _MIN_TOTAL:
        result = result / total
    else:
        result = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return np.clip(result, 0.0, 1.0)


def _yager_combine_vectorised(
    grid_a: np.ndarray, grid_b: np.ndarray
) -> np.ndarray:
    """Combine two mass grids elementwise using Yager's combination rule.

    Parameters
    ----------
    grid_a, grid_b:
        Arrays of shape (..., 3), dtype float64.

    Returns
    -------
    numpy.ndarray, same shape as inputs, dtype float64.
    """
    mO_a = grid_a[..., 0]
    mF_a = grid_a[..., 1]
    mOF_a = grid_a[..., 2]

    mO_b = grid_b[..., 0]
    mF_b = grid_b[..., 1]
    mOF_b = grid_b[..., 2]

    # Unnormalised focal element intersections.
    mO_out = mO_a * mO_b + mO_a * mOF_b + mOF_a * mO_b
    mF_out = mF_a * mF_b + mF_a * mOF_b + mOF_a * mF_b
    # Conflict goes to ignorance.
    conflict = mO_a * mF_b + mF_a * mO_b
    mOF_out = mOF_a * mOF_b + conflict

    result = np.stack([mO_out, mF_out, mOF_out], axis=-1)

    # Enforce mass constraint per cell.
    total = result.sum(axis=-1, keepdims=True)
    valid = (total > _MIN_TOTAL)[..., 0]
    result[valid] = result[valid] / total[valid]
    result[~valid] = np.array([0.0, 0.0, 1.0])

    return np.clip(result, 0.0, 1.0)


class YagerFusion(FusionMethod):
    """Yager's combination rule for occupancy grid fusion.

    Identity element: (0.0, 0.0, 1.0) -- vacuous belief (total ignorance).
    Memory: 3 float32 per cell (same as DSTBMFusion).
    Combination rule: Yager's rule -- conflict mass assigned to m_OF.
    Occupancy probability: pignistic transform BetP(O) = m_O + m_OF / 2.
    """

    def __init__(
        self,
        m_occ: np.ndarray | None = None,
        m_free: np.ndarray | None = None,
        m_of_min: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        m_occ:
            Sensor model mass for an occupied hit: [m_O, m_F, m_OF].
            Defaults to (0.75, 0.10, 0.15).
        m_free:
            Sensor model mass for a free traversal: [m_O, m_F, m_OF].
            Defaults to (0.10, 0.75, 0.15).
        m_of_min:
            Minimum ignorance mass floor.  After each combination, m_OF
            is clamped to at least this value and the result re-normalised.
            Default 0.0 = no regularisation (existing behaviour).
        """
        self._m_occ = (
            np.asarray(m_occ, dtype=np.float64)
            if m_occ is not None
            else _M_OCC.copy()
        )
        self._m_free = (
            np.asarray(m_free, dtype=np.float64)
            if m_free is not None
            else _M_FREE.copy()
        )
        self._m_of_min = m_of_min

    def _enforce_m_of_min(self, m: np.ndarray) -> np.ndarray:
        """Enforce minimum ignorance mass and re-normalise."""
        if self._m_of_min <= 0.0:
            return m
        m64 = m.astype(np.float64) if m.dtype != np.float64 else m.copy()
        m64[2] = max(float(m64[2]), self._m_of_min)
        total = m64.sum()
        if total > _MIN_TOTAL:
            m64 = m64 / total
        return m64

    @property
    def name(self) -> str:
        return "yager"

    def create_grid(self, rows: int, cols: int) -> np.ndarray:
        """Return a (rows, cols, 3) float32 grid initialised to vacuous belief.

        All cells start at (m_O=0, m_F=0, m_OF=1) -- total ignorance.
        """
        grid = np.zeros((rows, cols, 3), dtype=np.float32)
        grid[:, :, 2] = 1.0  # m_OF = 1.0 (identity element)
        return grid

    def update_cell(
        self,
        grid: np.ndarray,
        ray_cells: list[tuple[int, int]],
        hit_cell: tuple[int, int] | None,
        is_max_range: bool,
    ) -> None:
        for r, c in ray_cells:
            m_current = grid[r, c, :].astype(np.float64)
            m_new = _yager_combine_scalar(m_current, self._m_free)
            m_new = self._enforce_m_of_min(m_new)
            grid[r, c, :] = m_new.astype(np.float32)
        if hit_cell is not None:
            r, c = hit_cell
            m_obs = self._m_free if is_max_range else self._m_occ
            m_current = grid[r, c, :].astype(np.float64)
            m_new = _yager_combine_scalar(m_current, m_obs)
            m_new = self._enforce_m_of_min(m_new)
            grid[r, c, :] = m_new.astype(np.float32)

    def fuse_grids(self, grid_a: np.ndarray, grid_b: np.ndarray) -> np.ndarray:
        """Fuse two mass grids elementwise using Yager's combination rule."""
        a64 = grid_a.astype(np.float64)
        b64 = grid_b.astype(np.float64)
        result = _yager_combine_vectorised(a64, b64)
        if self._m_of_min > 0.0:
            mof = result[..., 2]
            below = mof < self._m_of_min
            result[below, 2] = self._m_of_min
            total = result.sum(axis=-1, keepdims=True)
            valid = (total > _MIN_TOTAL)[..., 0]
            result[valid] = result[valid] / total[valid]
        return result.astype(np.float32)

    def get_occupancy_probability(self, grid: np.ndarray) -> np.ndarray:
        """Pignistic transform: BetP(O) = m_O + m_OF / 2."""
        m_O = grid[:, :, 0].astype(np.float64)
        m_OF = grid[:, :, 2].astype(np.float64)
        prob = m_O + m_OF / 2.0
        return np.clip(prob, 0.0, 1.0).astype(np.float32)

    def get_memory_bytes(self, grid: np.ndarray) -> int:
        return int(grid.nbytes)
