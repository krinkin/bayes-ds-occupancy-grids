"""Adapter: wraps Intel Lab dataset into our experiment interface.

Provides grid construction from laser scans, train/test splitting,
and multi-robot splitting for fusion mechanism validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.experiments.intel_lab.loader import LaserScan


@dataclass(frozen=True)
class GridParams:
    """Parameters for the occupancy grid."""

    rows: int
    cols: int
    resolution: float
    origin_x: float
    origin_y: float


class IntelLabAdapter:
    """Adapts Intel Lab scans into the experiment interface."""

    def __init__(
        self,
        scans: list[LaserScan],
        resolution: float = 0.1,
        max_range: float = 15.0,
        train_fraction: float = 0.8,
    ) -> None:
        """
        Parameters
        ----------
        scans : list[LaserScan]
            Loaded laser scans.
        resolution : float
            Grid resolution in meters per cell.
        max_range : float
            Maximum usable range (meters). Readings beyond this are discarded.
        train_fraction : float
            Fraction of scans to use for mapping (rest for evaluation).
        """
        self._scans = scans
        self._resolution = resolution
        self._max_range = max_range

        split_idx = int(len(scans) * train_fraction)
        self._train_scans = scans[:split_idx]
        self._test_scans = scans[split_idx:]

        # Compute bounds from all scan poses
        all_x = [s.pose_x for s in scans]
        all_y = [s.pose_y for s in scans]

        # Add margin for laser reach
        margin = max_range + 2.0
        self._origin_x = min(all_x) - margin
        self._origin_y = min(all_y) - margin
        self._width = max(all_x) - min(all_x) + 2 * margin
        self._height = max(all_y) - min(all_y) + 2 * margin

        self._cols = int(math.ceil(self._width / resolution))
        self._rows = int(math.ceil(self._height / resolution))

    @property
    def grid_params(self) -> GridParams:
        return GridParams(
            rows=self._rows,
            cols=self._cols,
            resolution=self._resolution,
            origin_x=self._origin_x,
            origin_y=self._origin_y,
        )

    @property
    def num_train_scans(self) -> int:
        return len(self._train_scans)

    @property
    def num_test_scans(self) -> int:
        return len(self._test_scans)

    def get_ground_truth(
        self, min_obs: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a ground truth occupancy grid from TEST scans ONLY.

        Uses only the held-out test scans to avoid data leakage: the mapping
        algorithms are trained on train_scans, so GT must be constructed from
        independent data.

        A cell is included in the valid GT mask only if it accumulates at least
        min_obs observations from the test set (reliability filter).

        Parameters
        ----------
        min_obs : int
            Minimum number of test-scan observations required for a cell to
            be included in the GT mask.  Default is 3.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (gt, mask):
            - gt: float32 array of shape (rows, cols).  1.0 = occupied,
              0.0 = free.  Values at mask==False are 0.0 (unused placeholders).
            - mask: bool array of shape (rows, cols).  True for cells with
              valid GT (>= min_obs test-scan observations).  Only these cells
              should be used for metric evaluation.
        """
        counts_occ = np.zeros((self._rows, self._cols), dtype=np.float32)
        counts_free = np.zeros((self._rows, self._cols), dtype=np.float32)

        # Use test scans only -- no data leakage from training set
        for scan in self._test_scans:
            self._accumulate_scan(scan, counts_occ, counts_free)

        total = counts_occ + counts_free

        # Valid GT mask: cells with sufficient observations
        mask = total >= min_obs

        gt = np.zeros((self._rows, self._cols), dtype=np.float32)
        gt[mask] = (counts_occ[mask] / total[mask]) > 0.5

        return gt, mask

    def get_train_scans(self) -> list[LaserScan]:
        """Return training scans (first train_fraction of dataset)."""
        return self._train_scans

    def get_test_scans(self) -> list[LaserScan]:
        """Return test scans (last 1-train_fraction of dataset)."""
        return self._test_scans

    def split_for_multi_robot(self, num_robots: int) -> list[list[LaserScan]]:
        """Split training scans into segments simulating multiple robots.

        Each "robot" gets a contiguous chunk of scans. This simulates
        multiple robots that each explore a portion of the environment.

        Parameters
        ----------
        num_robots : int
            Number of simulated robots.

        Returns
        -------
        list[list[LaserScan]]
            One list of scans per robot.
        """
        chunk_size = len(self._train_scans) // num_robots
        segments: list[list[LaserScan]] = []
        for i in range(num_robots):
            start = i * chunk_size
            end = start + chunk_size if i < num_robots - 1 else len(self._train_scans)
            segments.append(self._train_scans[start:end])
        return segments

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to grid indices."""
        col = int((x - self._origin_x) / self._resolution)
        row = int((y - self._origin_y) / self._resolution)
        # Flip row so that increasing y goes up
        row = self._rows - 1 - row
        return (
            max(0, min(row, self._rows - 1)),
            max(0, min(col, self._cols - 1)),
        )

    def scan_to_ray_cells(
        self,
        scan: LaserScan,
    ) -> list[tuple[list[tuple[int, int]], tuple[int, int] | None, bool]]:
        """Convert a laser scan to ray cell lists for grid update.

        Returns a list of (ray_cells, hit_cell, is_max_range) tuples,
        one per beam, compatible with FusionMethod.update_cell().
        """
        results: list[tuple[list[tuple[int, int]], tuple[int, int] | None, bool]] = []

        for i, r in enumerate(scan.ranges):
            angle = scan.pose_theta + scan.angle_min + i * scan.angle_increment

            if r <= 0 or r > self._max_range:
                # Invalid reading
                continue

            is_max = r >= self._max_range * 0.99

            # Trace ray from robot position to hit point
            x0 = scan.pose_x
            y0 = scan.pose_y
            x1 = x0 + r * math.cos(angle)
            y1 = y0 + r * math.sin(angle)

            ray_cells = self._bresenham(x0, y0, x1, y1)

            if is_max:
                hit_cell = None
            else:
                hit_cell = self.world_to_grid(x1, y1)

            results.append((ray_cells, hit_cell, is_max))

        return results

    def _accumulate_scan(
        self,
        scan: LaserScan,
        counts_occ: np.ndarray,
        counts_free: np.ndarray,
    ) -> None:
        """Add one scan's evidence to occupancy/free counts."""
        for i, r in enumerate(scan.ranges):
            angle = scan.pose_theta + scan.angle_min + i * scan.angle_increment

            if r <= 0 or r > self._max_range:
                continue

            is_max = r >= self._max_range * 0.99

            x0 = scan.pose_x
            y0 = scan.pose_y
            x1 = x0 + r * math.cos(angle)
            y1 = y0 + r * math.sin(angle)

            # Free cells along the ray
            for row, col in self._bresenham(x0, y0, x1, y1):
                counts_free[row, col] += 1.0

            # Hit cell (occupied)
            if not is_max:
                hit_r, hit_c = self.world_to_grid(x1, y1)
                counts_occ[hit_r, hit_c] += 1.0

    def _bresenham(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> list[tuple[int, int]]:
        """Bresenham line from (x0,y0) to (x1,y1) in grid coordinates.

        Returns list of (row, col) along the ray, EXCLUDING the endpoint.
        """
        r0, c0 = self.world_to_grid(x0, y0)
        r1, c1 = self.world_to_grid(x1, y1)

        cells: list[tuple[int, int]] = []

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1

        err = dc - dr
        cr, cc = r0, c0

        max_cells = dr + dc + 1

        for _ in range(max_cells):
            if cr == r1 and cc == c1:
                break  # exclude endpoint

            if 0 <= cr < self._rows and 0 <= cc < self._cols:
                cells.append((cr, cc))

            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                cc += sc
            if e2 < dc:
                err += dc
                cr += sr

        return cells
