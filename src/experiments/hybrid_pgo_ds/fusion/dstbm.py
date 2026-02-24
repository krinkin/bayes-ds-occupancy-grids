"""Dempster-Shafer / Transferable Belief Model (DS/TBM) fusion method.

Technical Design Proposal (TDP)
================================

Context
-------
This module implements DS/TBM cell fusion as the third experimental arm of
H-003.  The frame of discernment is {O, F} (Occupied, Free).  The power set
(excluding the empty set under closed-world assumption) has three focal
elements: {O}, {F}, and {O,F}.

Each grid cell stores a mass triplet (m_O, m_F, m_OF) where:
    m_O   = m({O})   -- evidence for occupied
    m_F   = m({F})   -- evidence for free
    m_OF  = m({O,F}) -- ignorance / no information
    m_O + m_F + m_OF = 1.0  (normalised mass function)

Identity element: (0.0, 0.0, 1.0) -- total ignorance.  Combining any mass
function with the vacuous belief function (m_OF = 1) returns the original
mass function unchanged (verified algebraically).

1. Mass normalisation strategy (closed-world Dempster's rule)
--------------------------------------------------------------
We use the standard closed-world Dempster's combination rule:

    K_conflict = m1(O) * m2(F) + m1(F) * m2(O)

    K_norm = 1 / (1 - K_conflict)    [normalisation constant]

    m12(O)  = (m1(O)*m2(O)  + m1(O)*m2(OF) + m1(OF)*m2(O))  * K_norm
    m12(F)  = (m1(F)*m2(F)  + m1(F)*m2(OF) + m1(OF)*m2(F))  * K_norm
    m12(OF) = (m1(OF)*m2(OF))                                 * K_norm

Rationale for closed-world: occupancy grids assume the world is fully
represented by {Occupied, Free}; there is no need for an open-world empty
set.  The normalisation redistributes conflict proportionally.

Alternative considered (open-world TBM): conflict goes to m(empty_set)
rather than being normalised away.  This preserves the Zadeh-paradox-safe
semantics (m_empty signals "the model is wrong here").  However, tracking
m_empty adds a 4th value per cell and was rejected as unnecessary for the
basic occupancy grid use case.  The H-003 research verdict notes that the
key TBM advantage (m_OF distinguishing ignorance from conflict) is preserved
under the closed-world formulation.

2. Conflict mass handling
--------------------------
K_conflict in (0, 1) is the fraction of probability mass that cancels due
to contradictory evidence.  The combination rule redistributes this mass.

When two maps with high-conflict cells are merged (e.g., one robot sees
Occupied, the other sees Free for the same cell), K_conflict approaches 1,
and K_norm approaches infinity.  The unnormalised numerator also approaches
zero at the same rate, so the ratio is well-defined mathematically.  However,
floating-point arithmetic introduces instability when K_conflict > 1 - epsilon.

Empirically, conflict > 0.99 can arise when two sensors make strongly
contradictory observations of the same cell (e.g., dynamic obstacles).

3. Numerical stability
------------------------
a. Conflict clamping: K_conflict is clipped to [0, 1 - _MIN_DENOMINATOR]
   so that the denominator (1 - K_conflict) never falls below _MIN_DENOMINATOR
   = 1e-10.  When conflict is near 1 the result is an almost-uninformative
   mass function rather than division by zero.

b. Mass constraint enforcement: after each combination we ensure
   m_O + m_F + m_OF = 1 by dividing by their sum (handles floating-point
   drift over many updates).

c. Component clamping: each component is clipped to [0, 1] before renorm.

4. Sensor model
----------------
The sensor model assigns a mass function to each LiDAR observation:

    Occupied hit:    m_obs = (M_OCC_O, M_OCC_F, M_OCC_OF) = (0.75, 0.10, 0.15)
    Free traversal:  m_obs = (M_FREE_O, M_FREE_F, M_FREE_OF) = (0.10, 0.75, 0.15)

These values are analogous to the Bayesian parameters l_occ = 2.0 and
l_free = -0.5 and provide a moderate level of evidence per observation with
15% residual ignorance mass.  The m_OF = 0.15 > 0 ensures that even a
direct observation does not fully collapse the ignorance mass in a single
step, which is more realistic than a purely deterministic sensor model.

5. Occupancy probability: pignistic transform
---------------------------------------------
To convert the mass triplet to a scalar occupancy probability for downstream
use (e.g., plotting, metrics), we apply the pignistic transformation (BetP):

    BetP(O) = m_O + m_OF / 2

This distributes ignorance mass equally between O and F.  For an unobserved
cell: BetP(O) = 0 + 1/2 = 0.5 (neutral). ✓
For a strongly occupied cell: BetP(O) > 0.5. ✓

Internal representation
-----------------------
numpy.ndarray, shape (rows, cols, 3), dtype float32.
Channel 0: m_O   = m({O})
Channel 1: m_F   = m({F})
Channel 2: m_OF  = m({O,F})
Initial state: all cells = (0.0, 0.0, 1.0).
"""

from __future__ import annotations

import numpy as np

from src.experiments.hybrid_pgo_ds.fusion.base import FusionMethod

# Minimum denominator to avoid division by near-zero in Dempster's rule.
_MIN_DENOMINATOR: float = 1e-10

# Sensor model mass functions.
# Occupied hit: strong evidence for O, minimal evidence for F, some ignorance.
_M_OCC: np.ndarray = np.array([0.75, 0.10, 0.15], dtype=np.float64)
# Free traversal: strong evidence for F, minimal evidence for O, some ignorance.
_M_FREE: np.ndarray = np.array([0.10, 0.75, 0.15], dtype=np.float64)


def _dempster_combine_scalar(
    m1: np.ndarray, m2: np.ndarray
) -> np.ndarray:
    """Combine two mass functions using Dempster's closed-world rule.

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

    conflict = mO1 * mF2 + mF1 * mO2
    conflict = float(np.clip(conflict, 0.0, 1.0 - _MIN_DENOMINATOR))
    K_norm = 1.0 / (1.0 - conflict)

    mO_out = (mO1 * mO2 + mO1 * mOF2 + mOF1 * mO2) * K_norm
    mF_out = (mF1 * mF2 + mF1 * mOF2 + mOF1 * mF2) * K_norm
    mOF_out = mOF1 * mOF2 * K_norm

    result = np.array([mO_out, mF_out, mOF_out], dtype=np.float64)

    # Enforce mass constraint (guards against floating-point drift).
    total = result.sum()
    if total > _MIN_DENOMINATOR:
        result = result / total
    else:
        result = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return np.clip(result, 0.0, 1.0)


def _dempster_combine_vectorised(
    grid_a: np.ndarray, grid_b: np.ndarray
) -> np.ndarray:
    """Combine two mass grids elementwise using Dempster's closed-world rule.

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

    conflict = mO_a * mF_b + mF_a * mO_b
    conflict = np.clip(conflict, 0.0, 1.0 - _MIN_DENOMINATOR)
    K_norm = 1.0 / (1.0 - conflict)

    mO_out = (mO_a * mO_b + mO_a * mOF_b + mOF_a * mO_b) * K_norm
    mF_out = (mF_a * mF_b + mF_a * mOF_b + mOF_a * mF_b) * K_norm
    mOF_out = mOF_a * mOF_b * K_norm

    result = np.stack([mO_out, mF_out, mOF_out], axis=-1)

    # Enforce mass constraint per cell.
    total = result.sum(axis=-1, keepdims=True)
    # Where total is too small, fall back to vacuous belief.
    valid = (total > _MIN_DENOMINATOR)[..., 0]
    result[valid] = result[valid] / total[valid]
    result[~valid] = np.array([0.0, 0.0, 1.0])

    return np.clip(result, 0.0, 1.0)


class DSTBMFusion(FusionMethod):
    """DS/TBM occupancy grid fusion.

    Identity element: (0.0, 0.0, 1.0) -- vacuous belief (total ignorance).
    Memory: 3 float32 per cell.
    Combination rule: closed-world Dempster's rule with conflict clamping.
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
            Analogous to Bayesian L_max clamping: prevents mass from
            concentrating entirely on O or F.
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

    @property
    def name(self) -> str:
        return "dstbm"

    def create_grid(self, rows: int, cols: int) -> np.ndarray:
        """Return a (rows, cols, 3) float32 grid initialised to vacuous belief.

        All cells start at (m_O=0, m_F=0, m_OF=1) -- total ignorance.
        """
        grid = np.zeros((rows, cols, 3), dtype=np.float32)
        grid[:, :, 2] = 1.0  # m_OF = 1.0 (identity element)
        return grid

    def _enforce_m_of_min(self, m: np.ndarray) -> np.ndarray:
        """Enforce minimum ignorance mass and re-normalise."""
        if self._m_of_min <= 0.0:
            return m
        m64 = m.astype(np.float64) if m.dtype != np.float64 else m.copy()
        m64[2] = max(float(m64[2]), self._m_of_min)
        total = m64.sum()
        if total > _MIN_DENOMINATOR:
            m64 = m64 / total
        return m64

    def update_cell(
        self,
        grid: np.ndarray,
        ray_cells: list[tuple[int, int]],
        hit_cell: tuple[int, int] | None,
        is_max_range: bool,
    ) -> None:
        for r, c in ray_cells:
            m_current = grid[r, c, :].astype(np.float64)
            m_new = _dempster_combine_scalar(m_current, self._m_free)
            m_new = self._enforce_m_of_min(m_new)
            grid[r, c, :] = m_new.astype(np.float32)
        if hit_cell is not None:
            r, c = hit_cell
            m_obs = self._m_free if is_max_range else self._m_occ
            m_current = grid[r, c, :].astype(np.float64)
            m_new = _dempster_combine_scalar(m_current, m_obs)
            m_new = self._enforce_m_of_min(m_new)
            grid[r, c, :] = m_new.astype(np.float32)

    def fuse_grids(self, grid_a: np.ndarray, grid_b: np.ndarray) -> np.ndarray:
        """Fuse two mass grids elementwise using Dempster's combination rule."""
        a64 = grid_a.astype(np.float64)
        b64 = grid_b.astype(np.float64)
        result = _dempster_combine_vectorised(a64, b64)
        if self._m_of_min > 0.0:
            mof = result[..., 2]
            below = mof < self._m_of_min
            result[below, 2] = self._m_of_min
            total = result.sum(axis=-1, keepdims=True)
            valid = (total > _MIN_DENOMINATOR)[..., 0]
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
