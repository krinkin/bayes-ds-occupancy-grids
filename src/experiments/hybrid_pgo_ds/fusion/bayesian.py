"""Bayesian log-odds fusion method.

One float32 per cell.  Ported from simulation/occupancy_grid.py and wrapped
in the FusionMethod interface.

Internal representation
-----------------------
numpy.ndarray, shape (rows, cols), dtype float32.
Value 0.0 = unknown (P = 0.5); >0 = occupied; <0 = free.

Fusion rule
-----------
Adding log-odds from two grids: L_fused = L_a + L_b.
Both grids start at 0.0 (identity element), so fusing with an unobserved
grid changes nothing.
"""

from __future__ import annotations

import numpy as np

from src.experiments.hybrid_pgo_ds.fusion.base import FusionMethod

_L_OCC: float = 2.0     # log-odds increment for an occupied hit
_L_FREE: float = -0.5   # log-odds update for a free-traversal cell
_CLAMP: float = 10.0    # maximum absolute log-odds value


class BayesianFusion(FusionMethod):
    """Bayesian log-odds occupancy grid fusion.

    Identity element: 0.0 (p = 0.5, no information).
    Memory: 1 float32 per cell.
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
        return "bayesian"

    def create_grid(self, rows: int, cols: int) -> np.ndarray:
        """Return a (rows, cols) float32 log-odds grid initialised to 0."""
        return np.zeros((rows, cols), dtype=np.float32)

    def update_cell(
        self,
        grid: np.ndarray,
        ray_cells: list[tuple[int, int]],
        hit_cell: tuple[int, int] | None,
        is_max_range: bool,
    ) -> None:
        for r, c in ray_cells:
            grid[r, c] = float(
                np.clip(grid[r, c] + self._l_free, -self._clamp, self._clamp)
            )
        if hit_cell is not None:
            r, c = hit_cell
            delta = self._l_free if is_max_range else self._l_occ
            grid[r, c] = float(
                np.clip(grid[r, c] + delta, -self._clamp, self._clamp)
            )

    def fuse_grids(self, grid_a: np.ndarray, grid_b: np.ndarray) -> np.ndarray:
        """Return log-odds sum, clamped."""
        return np.clip(
            grid_a.astype(np.float64) + grid_b.astype(np.float64),
            -self._clamp,
            self._clamp,
        ).astype(np.float32)

    def get_occupancy_probability(self, grid: np.ndarray) -> np.ndarray:
        log = grid.astype(np.float64)
        prob = 1.0 - 1.0 / (1.0 + np.exp(log))
        return prob.astype(np.float32)

    def get_memory_bytes(self, grid: np.ndarray) -> int:
        return int(grid.nbytes)
