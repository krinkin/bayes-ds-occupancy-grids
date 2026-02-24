# TDP: Bayesian log-odds occupancy grid update (Increment 1 -- direct in simulation)
# Approach: Standard log-odds inverse sensor model. For each ray:
#   - Cells along the ray (before the endpoint) are updated as FREE.
#   - The endpoint cell (if within max_range) is updated as OCCUPIED.
#   Bresenham line algorithm is used to enumerate cells along each ray
#   efficiently without floating-point stepping.
#
#   Log-odds representation:
#     L(x) = log(p / (1-p))
#     0.0  = unknown (p = 0.5)
#     >0   = occupied
#     <0   = free
#   Clamped to [-10, 10] to prevent numerical overflow.
#
#   The inverse sensor model uses two parameters:
#     l_occ: log-odds for an occupied cell hit by a ray endpoint
#     l_free: log-odds update for a free cell traversed by a ray
#
# Alternatives considered:
#   Floating-point step traversal -- simpler but misses cells; Bresenham is
#   exact and has well-understood properties.
# Risks: Bresenham does not account for partial cell occupancy at ray start/end.
#   Acceptable for smoke test; not a concern for this increment.
"""Bayesian log-odds occupancy grid.

Maintains a 2D log-odds grid and provides an update method that ingests
a set of LiDAR ray measurements from a single robot pose.

Fusion interfaces (Increment 3) will wrap or replace this class.
"""

from __future__ import annotations

import numpy as np


_LOG_ODDS_CLAMP = 10.0  # log-odds saturation value
_L_OCC = 2.0            # log-odds increment for a cell hit by a ray
_L_FREE = -0.5          # log-odds update for a free cell traversed by a ray


def _bresenham_line(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Return all (row, col) cells on the Bresenham line from (r0,c0) to (r1,c1).

    The endpoint (r1, c1) is included.
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


class BayesianOccupancyGrid:
    """Log-odds Bayesian occupancy grid.

    Parameters
    ----------
    rows, cols:
        Grid dimensions.
    l_occ:
        Log-odds increment applied to a cell at a ray endpoint.
    l_free:
        Log-odds update applied to cells traversed by a ray (before endpoint).
    clamp:
        Maximum absolute log-odds value (prevents saturation).
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        l_occ: float = _L_OCC,
        l_free: float = _L_FREE,
        clamp: float = _LOG_ODDS_CLAMP,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.l_occ = l_occ
        self.l_free = l_free
        self.clamp = clamp
        # Log-odds grid: 0.0 = unknown, >0 = occupied, <0 = free
        self.log_odds: np.ndarray = np.zeros((rows, cols), dtype=np.float32)

    def update(
        self,
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
        """Update the grid with a single LiDAR scan.

        Parameters
        ----------
        pose_x, pose_y:
            Robot position in world coordinates (metres).
        pose_theta:
            Robot heading in radians.
        ranges:
            Measured range for each ray, shape (num_rays,).
        resolution:
            Metres per grid cell.
        height:
            Room height in metres (for coordinate conversion).
        num_rays:
            Number of rays.
        max_range:
            Sensor maximum range. Rays at max_range are treated as free-space
            rays (no occupied endpoint update).
        """
        # Robot cell
        r_robot = int((height - pose_y) / resolution)
        c_robot = int(pose_x / resolution)

        angles = pose_theta + np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)

        for i in range(num_rays):
            r = ranges[i]
            angle = angles[i]
            endpoint_x = pose_x + r * np.cos(angle)
            endpoint_y = pose_y + r * np.sin(angle)

            r_end = int((height - endpoint_y) / resolution)
            c_end = int(endpoint_x / resolution)

            # Clamp endpoint to grid
            r_end = int(np.clip(r_end, 0, self.rows - 1))
            c_end = int(np.clip(c_end, 0, self.cols - 1))
            r_start = int(np.clip(r_robot, 0, self.rows - 1))
            c_start = int(np.clip(c_robot, 0, self.cols - 1))

            cells = _bresenham_line(r_start, c_start, r_end, c_end)

            hit = r < max_range - 1e-3  # ray actually hit something

            # All cells except the last are free-space traversed
            for cr, cc in cells[:-1]:
                if 0 <= cr < self.rows and 0 <= cc < self.cols:
                    self.log_odds[cr, cc] = np.clip(
                        self.log_odds[cr, cc] + self.l_free, -self.clamp, self.clamp
                    )
            # Last cell: occupied if hit, otherwise free
            if cells:
                cr, cc = cells[-1]
                if 0 <= cr < self.rows and 0 <= cc < self.cols:
                    if hit:
                        self.log_odds[cr, cc] = np.clip(
                            self.log_odds[cr, cc] + self.l_occ, -self.clamp, self.clamp
                        )
                    else:
                        self.log_odds[cr, cc] = np.clip(
                            self.log_odds[cr, cc] + self.l_free, -self.clamp, self.clamp
                        )

    def to_probability(self) -> np.ndarray:
        """Convert log-odds grid to probability grid in [0, 1].

        Returns
        -------
        numpy.ndarray, shape (rows, cols), dtype float32
            Cell occupancy probability: 0.5 = unknown, >0.5 = occupied,
            <0.5 = free.
        """
        log = self.log_odds.astype(np.float64)
        prob = 1.0 - 1.0 / (1.0 + np.exp(log))
        return prob.astype(np.float32)

    def to_binary(self, threshold: float = 0.0) -> np.ndarray:
        """Convert to binary occupancy map using log-odds threshold.

        Parameters
        ----------
        threshold:
            Log-odds threshold; cells above are classified as occupied.
            Default 0.0 means any positive evidence -> occupied.

        Returns
        -------
        numpy.ndarray, shape (rows, cols), dtype float32
            1.0 = occupied, 0.0 = free/unknown.
        """
        return (self.log_odds > threshold).astype(np.float32)
