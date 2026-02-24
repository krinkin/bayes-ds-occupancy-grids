"""Bayesian log-odds + observation count fusion method.

Two float32 values per cell: log-odds and observation count.

Motivation
----------
Standard Bayesian log-odds cannot distinguish "never observed" (count = 0,
log-odds = 0) from "equal conflicting evidence" (count > 0, log-odds ~ 0).
Tracking the observation count separates these two epistemic states and
provides an alternative to DS/TBM's m({O,F}) ignorance mass -- at 2x
memory cost instead of 3x.

Internal representation
-----------------------
numpy.ndarray, shape (rows, cols, 2), dtype float32.
Channel 0: log-odds (identity = 0.0).
Channel 1: observation count (identity = 0.0).

Fusion rule
-----------
Log-odds sum (identical to BayesianFusion), with observation count metadata:
    fused_logodds = clip(lo_a + lo_b, -clamp, clamp)
    fused_count   = count_a + count_b

This keeps the cell fusion rule identical to the Bayesian arm so that
experimental differences are caused solely by the availability of the
observation count -- not by a change in the fusion rule itself.
"""

from __future__ import annotations

import numpy as np

from src.experiments.hybrid_pgo_ds.fusion.base import FusionMethod

_L_OCC: float = 2.0
_L_FREE: float = -0.5
_CLAMP: float = 10.0


class BayesianCountFusion(FusionMethod):
    """Bayesian log-odds + observation count occupancy grid fusion.

    Identity element: (log-odds = 0.0, count = 0.0).
    Memory: 2 float32 per cell.
    """

    def __init__(
        self,
        l_occ: float = _L_OCC,
        l_free: float = _L_FREE,
        clamp: float = _CLAMP,
    ) -> None:
        self._l_occ = l_occ
        self._l_free = l_free
        self._clamp = clamp

    @property
    def name(self) -> str:
        return "bayesian_count"

    def create_grid(self, rows: int, cols: int) -> np.ndarray:
        """Return a (rows, cols, 2) float32 grid: all-zero log-odds and counts."""
        return np.zeros((rows, cols, 2), dtype=np.float32)

    def update_cell(
        self,
        grid: np.ndarray,
        ray_cells: list[tuple[int, int]],
        hit_cell: tuple[int, int] | None,
        is_max_range: bool,
    ) -> None:
        for r, c in ray_cells:
            grid[r, c, 0] = float(
                np.clip(grid[r, c, 0] + self._l_free, -self._clamp, self._clamp)
            )
            grid[r, c, 1] += 1.0
        if hit_cell is not None:
            r, c = hit_cell
            delta = self._l_free if is_max_range else self._l_occ
            grid[r, c, 0] = float(
                np.clip(grid[r, c, 0] + delta, -self._clamp, self._clamp)
            )
            grid[r, c, 1] += 1.0

    def fuse_grids(self, grid_a: np.ndarray, grid_b: np.ndarray) -> np.ndarray:
        """Fuse by log-odds sum (same rule as BayesianFusion), sum counts."""
        logodds_a = grid_a[:, :, 0].astype(np.float64)
        logodds_b = grid_b[:, :, 0].astype(np.float64)
        count_a = grid_a[:, :, 1].astype(np.float64)
        count_b = grid_b[:, :, 1].astype(np.float64)

        fused_logodds = np.clip(logodds_a + logodds_b, -self._clamp, self._clamp)

        result = np.zeros((*grid_a.shape[:2], 2), dtype=np.float32)
        result[:, :, 0] = fused_logodds.astype(np.float32)
        result[:, :, 1] = (count_a + count_b).astype(np.float32)
        return result

    def get_occupancy_probability(self, grid: np.ndarray) -> np.ndarray:
        log = grid[:, :, 0].astype(np.float64)
        prob = 1.0 - 1.0 / (1.0 + np.exp(log))
        return prob.astype(np.float32)

    def get_memory_bytes(self, grid: np.ndarray) -> int:
        return int(grid.nbytes)
