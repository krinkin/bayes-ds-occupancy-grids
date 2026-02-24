# TDP: 2D LiDAR ray casting with vectorised numpy operations
# Approach: For each ray, step along the ray in increments of resolution/2 until
#   hitting an occupied cell or exceeding max_range. Vectorise across rays using
#   numpy broadcasting: precompute unit direction vectors for all rays, then
#   step all rays simultaneously until each terminates. This is more efficient
#   than a per-ray Python loop when num_rays is large.
#
#   Implementation detail: "DDA-lite" -- step size is half a cell diagonal to
#   avoid skipping thin walls. Exact DDA (Bresenham) would be faster but
#   significantly more complex. For the smoke-test scenario (60 rays, 10x10m
#   room at 0.1m/cell) the simple approach is fast enough.
#
# Alternatives considered:
#   Exact DDA (Bresenham grid traversal) -- more correct, faster, but ~5x more
#   code; deferred to a future increment when performance is measured.
#   Scipy map_coordinates -- not applicable (we need ray termination, not
#   interpolation).
# Risks: Half-cell-diagonal step may occasionally skip a 1-cell-wide diagonal
#   wall. Acceptable for smoke test; revisit if wall-hit accuracy fails tests.
"""2D LiDAR ray caster.

Given a robot pose and a ground-truth occupancy grid, cast *num_rays* rays
evenly distributed around 360 degrees and return measured ranges.

All operations are vectorised with numpy for speed.
"""

from __future__ import annotations

import numpy as np


def cast_rays(
    *,
    pose_x: float,
    pose_y: float,
    pose_theta: float,
    grid: np.ndarray,
    resolution: float,
    height: float,
    num_rays: int,
    max_range: float,
    noise_stddev: float,
    rng: np.random.Generator,
    false_positive_rate: float = 0.0,
    false_negative_rate: float = 0.0,
) -> np.ndarray:
    """Cast LiDAR rays from a robot pose and return range measurements.

    Parameters
    ----------
    pose_x, pose_y:
        Robot position in world coordinates (metres).
    pose_theta:
        Robot heading in radians.
    grid:
        Ground-truth occupancy grid, shape (rows, cols), float32.
        1.0 = occupied, 0.0 = free.
    resolution:
        Metres per grid cell.
    height:
        Room height in metres (for world-to-grid y-axis flip).
    num_rays:
        Number of rays to cast, evenly spaced over [0, 2*pi).
    max_range:
        Maximum sensor range in metres.
    noise_stddev:
        Standard deviation of Gaussian range noise in metres.
        Set to 0.0 for noiseless measurements.
    rng:
        NumPy random generator (for additive noise and FP/FN draws).
    false_positive_rate:
        Probability that a non-hitting ray (max_range) is replaced by a
        spurious return at a random range in [0, max_range).  Models ghost
        detections.
    false_negative_rate:
        Probability that a hitting ray (range < max_range) is replaced by
        max_range.  Models missed detections.

    Returns
    -------
    numpy.ndarray, shape (num_rays,), dtype float32
        Range measurement for each ray in metres.
        Rays that do not hit any obstacle return *max_range*.
    """
    rows, cols = grid.shape
    # Step size: half a cell diagonal to avoid skipping thin walls
    step = resolution * 0.5 / np.sqrt(2.0)
    max_steps = int(np.ceil(max_range / step)) + 1

    # Ray angles in world frame
    angles = pose_theta + np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)
    cos_a = np.cos(angles)  # shape (num_rays,)
    sin_a = np.sin(angles)

    # Measured ranges initialised to max_range
    ranges = np.full(num_rays, max_range, dtype=np.float64)
    hit = np.zeros(num_rays, dtype=bool)

    for s in range(1, max_steps):
        dist = s * step
        # Current x, y for all rays
        rx = pose_x + dist * cos_a  # (num_rays,)
        ry = pose_y + dist * sin_a

        # Convert to grid indices
        gc = (rx / resolution).astype(int)                  # col
        gr = ((height - ry) / resolution).astype(int)       # row

        # Mask: rays not yet terminated and within grid bounds
        active = (
            ~hit
            & (gc >= 0) & (gc < cols)
            & (gr >= 0) & (gr < rows)
        )

        # Check which active rays hit an occupied cell
        active_idx = np.where(active)[0]
        if active_idx.size == 0:
            break

        cell_vals = grid[gr[active_idx], gc[active_idx]]
        newly_hit = active_idx[cell_vals >= 0.5]

        ranges[newly_hit] = dist
        hit[newly_hit] = True

        # Rays outside bounds are also terminated (no hit)
        out_of_bounds = ~hit & (
            (rx < 0) | (rx >= cols * resolution)
            | (ry < 0) | (ry >= rows * resolution)
        )
        hit[out_of_bounds] = True  # leave their range at max_range

        if hit.all():
            break

    # Add Gaussian noise
    if noise_stddev > 0.0:
        noise = rng.normal(0.0, noise_stddev, size=num_rays)
        ranges = np.clip(ranges + noise, 0.0, max_range)

    # False negatives: some obstacle hits are dropped (returned as max_range)
    if false_negative_rate > 0.0:
        hit_mask = ranges < max_range - 1e-6
        fn_mask = hit_mask & (rng.random(num_rays) < false_negative_rate)
        ranges[fn_mask] = max_range

    # False positives: some non-hitting rays return a spurious short range
    if false_positive_rate > 0.0:
        miss_mask = ranges >= max_range - 1e-6
        fp_mask = miss_mask & (rng.random(num_rays) < false_positive_rate)
        n_fp = int(np.sum(fp_mask))
        if n_fp > 0:
            ranges[fp_mask] = rng.uniform(0.0, max_range, size=n_fp)

    return ranges.astype(np.float32)


def ray_endpoints(
    *,
    pose_x: float,
    pose_y: float,
    pose_theta: float,
    ranges: np.ndarray,
    num_rays: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the world-coordinate endpoints of each ray given measured ranges.

    Parameters
    ----------
    pose_x, pose_y, pose_theta:
        Robot pose.
    ranges:
        Measured range for each ray, shape (num_rays,).
    num_rays:
        Number of rays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ex, ey) -- x and y endpoint coordinates, shape (num_rays,).
    """
    angles = pose_theta + np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)
    ex = pose_x + ranges * np.cos(angles)
    ey = pose_y + ranges * np.sin(angles)
    return ex, ey
