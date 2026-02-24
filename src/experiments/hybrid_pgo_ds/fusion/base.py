"""Abstract interface for occupancy grid fusion methods.

All three experimental arms (Bayesian, Bayesian+count, DS/TBM) implement this
interface so the experiment runner can swap them transparently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def _bresenham_line(
    r0: int, c0: int, r1: int, c1: int
) -> list[tuple[int, int]]:
    """Return all (row, col) cells on the Bresenham line from (r0,c0) to (r1,c1).

    The endpoint (r1, c1) is included.  Identical to the implementation in
    simulation/occupancy_grid.py; reproduced here so the fusion module has
    no import dependency on simulation internals.
    """
    cells: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    r, c = r0, c0
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    if dc >= dr:
        err = 2 * dr - dc
        for _ in range(dc + 1):
            cells.append((r, c))
            if err >= 0:
                r += sr
                err -= 2 * dc
            c += sc
            err += 2 * dr
    else:
        err = 2 * dc - dr
        for _ in range(dr + 1):
            cells.append((r, c))
            if err >= 0:
                c += sc
                err -= 2 * dr
            r += sr
            err += 2 * dc
    return cells


class FusionMethod(ABC):
    """Abstract occupancy grid fusion method.

    A FusionMethod manages the internal representation of one occupancy grid
    and defines how:
    - Individual cells are updated from LiDAR ray observations.
    - Two grids from different robots are merged (multi-robot map fusion).

    Subclasses choose their own internal grid layout (shape and dtype) but
    must expose occupancy probabilities in [0, 1] through
    get_occupancy_probability().
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this fusion method (e.g., 'bayesian')."""

    @abstractmethod
    def create_grid(self, rows: int, cols: int) -> np.ndarray:
        """Allocate and initialise a new empty grid.

        Parameters
        ----------
        rows, cols:
            Grid dimensions in cells.

        Returns
        -------
        numpy.ndarray
            Method-specific grid representation initialised to the identity
            (no information) state.  The leading two dimensions are always
            (rows, cols); additional dimensions depend on the method.
        """

    @abstractmethod
    def update_cell(
        self,
        grid: np.ndarray,
        ray_cells: list[tuple[int, int]],
        hit_cell: tuple[int, int] | None,
        is_max_range: bool,
    ) -> None:
        """Update grid cells along a single LiDAR ray in-place.

        Parameters
        ----------
        grid:
            Grid returned by create_grid; modified in-place.
        ray_cells:
            List of (row, col) cells traversed by the ray before the
            endpoint (free-space cells).  Each receives a FREE observation.
        hit_cell:
            (row, col) of the ray endpoint cell, or None if the Bresenham
            line produced no cells.
        is_max_range:
            If True the ray reached maximum range: the endpoint cell
            receives a FREE update (no obstacle detected).
            If False the endpoint cell receives an OCCUPIED update.
        """

    @abstractmethod
    def fuse_grids(self, grid_a: np.ndarray, grid_b: np.ndarray) -> np.ndarray:
        """Merge two grids from different robots into a single new grid.

        Parameters
        ----------
        grid_a, grid_b:
            Grids to merge; must have the same shape.

        Returns
        -------
        numpy.ndarray
            New merged grid of the same shape.
        """

    @abstractmethod
    def get_occupancy_probability(self, grid: np.ndarray) -> np.ndarray:
        """Convert internal representation to occupancy probability array.

        Returns
        -------
        numpy.ndarray, shape (rows, cols), dtype float32
            Occupancy probability for each cell in [0, 1].
            0.5 encodes unknown / no information.
        """

    @abstractmethod
    def get_memory_bytes(self, grid: np.ndarray) -> int:
        """Return the number of bytes consumed by the grid data array."""

    # ------------------------------------------------------------------
    # Concrete helper: process a full LiDAR scan
    # ------------------------------------------------------------------

    def update_scan(
        self,
        grid: np.ndarray,
        *,
        pose_x: float,
        pose_y: float,
        pose_theta: float,
        ranges: np.ndarray,
        resolution: float,
        height: float,
        num_rays: int,
        max_range: float,
    ) -> None:
        """Update the grid from a full LiDAR scan.

        Computes Bresenham ray cells for each measurement and delegates to
        update_cell().  This is a convenience wrapper for the simulation
        loop; subclasses do not need to override it.

        Parameters
        ----------
        grid:
            Grid to update in-place.
        pose_x, pose_y:
            Robot position in world coordinates (metres).
        pose_theta:
            Robot heading in radians.
        ranges:
            Measured range per ray, shape (num_rays,).
        resolution:
            Metres per grid cell.
        height:
            World height in metres (for row = int((height - y) / resolution)).
        num_rays:
            Number of rays in the scan.
        max_range:
            Sensor maximum range.  Rays at max_range are treated as free-space.
        """
        rows, cols = grid.shape[:2]
        r_robot = int((height - pose_y) / resolution)
        c_robot = int(pose_x / resolution)

        angles = pose_theta + np.linspace(
            0.0, 2.0 * np.pi, num_rays, endpoint=False
        )

        for i in range(num_rays):
            r_meas = float(ranges[i])
            angle = float(angles[i])
            endpoint_x = pose_x + r_meas * np.cos(angle)
            endpoint_y = pose_y + r_meas * np.sin(angle)

            r_end = int((height - endpoint_y) / resolution)
            c_end = int(endpoint_x / resolution)

            r_end = int(np.clip(r_end, 0, rows - 1))
            c_end = int(np.clip(c_end, 0, cols - 1))
            r_start = int(np.clip(r_robot, 0, rows - 1))
            c_start = int(np.clip(c_robot, 0, cols - 1))

            cells = _bresenham_line(r_start, c_start, r_end, c_end)
            hit = r_meas < max_range - 1e-3

            ray_cells = [
                (cr, cc)
                for cr, cc in cells[:-1]
                if 0 <= cr < rows and 0 <= cc < cols
            ]

            hit_cell: tuple[int, int] | None = None
            if cells:
                cr, cc = cells[-1]
                if 0 <= cr < rows and 0 <= cc < cols:
                    hit_cell = (cr, cc)

            self.update_cell(grid, ray_cells, hit_cell, not hit)
