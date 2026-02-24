# TDP: Single-robot trajectory generator + Increment 2 drift model
# Approach: Generate a simple rectangular patrol path inside the room. The robot
#   starts near the bottom-left interior corner and walks counter-clockwise along
#   the inside perimeter of the room. This ensures good LiDAR coverage of all
#   walls without requiring complex path planning.
#   Heading (theta) is updated to point in the direction of travel at each step.
#
#   Increment 2 drift model: cumulative Gaussian odometric drift.
#   At each step the robot's true motion increment (dx, dy, dtheta) is perturbed
#   by independent Gaussian noise. Noise is added to increments (not absolute
#   poses) so drift accumulates as a random walk: sigma ~ drift_stddev * sqrt(t).
#
# Alternatives considered:
#   Random walk -- poor wall coverage, hard to test.
#   Bias drift -- constant offset per step; too simple, doesn't model random walk.
#   Full SE(2) noise model -- more realistic but overcomplicates Increment 2.
# Risks: If the room is very small and resolution coarse, the interior perimeter
#   may degenerate. Mitigated by checking inner dimensions are positive.
"""Ground-truth trajectory generator + Increment 2 odometric drift model.

Provides:
- generate_perimeter_trajectory: single ground-truth perimeter path
- apply_drift: add cumulative Gaussian odometric noise to a trajectory
- generate_multi_robot_trajectories: N independent ground-truth trajectories
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class Pose(NamedTuple):
    """A 2D robot pose."""

    x: float     # metres
    y: float     # metres
    theta: float  # radians, counter-clockwise from positive x axis


def generate_perimeter_trajectory(
    *,
    width: float,
    height: float,
    num_steps: int,
    margin: float = 1.0,
    start_fraction: float = 0.0,
    rng: np.random.Generator,  # reserved for future noise; not used here
) -> list[Pose]:
    """Generate a ground-truth perimeter trajectory inside a rectangular room.

    The robot walks counter-clockwise along the inside perimeter of the room,
    staying *margin* metres away from each wall.  Poses are evenly distributed
    along the four sides of the inner rectangle.

    Parameters
    ----------
    width, height:
        Room dimensions in metres.
    num_steps:
        Total number of trajectory poses to generate.
    margin:
        Distance in metres to keep from each wall.
    start_fraction:
        Starting position as a fraction of the perimeter in [0, 1).
        Default 0.0 = bottom-left corner.  Use different values for different
        robots to give them separate starting positions.
    rng:
        Random number generator (reserved for future noise; unused here).

    Returns
    -------
    list[Pose]
        Ordered list of (x, y, theta) poses.
    """
    inner_width = width - 2.0 * margin
    inner_height = height - 2.0 * margin

    if inner_width <= 0.0 or inner_height <= 0.0:
        raise ValueError(
            f"Room ({width}x{height}m) too small for margin={margin}m. "
            "Reduce margin or increase room size."
        )

    # Perimeter of the inner rectangle
    perimeter = 2.0 * (inner_width + inner_height)
    step_length = perimeter / num_steps

    # Starting point: bottom-left corner of inner rectangle
    x0 = margin
    y0 = margin

    # Four segments: bottom, right, top, left (counter-clockwise when y-up)
    # Segment order:  bottom (left->right), right (bottom->top),
    #                 top (right->left),    left (top->bottom)
    segments = [
        (inner_width, 0.0),    # right along bottom
        (0.0, inner_height),   # up along right side
        (-inner_width, 0.0),   # left along top
        (0.0, -inner_height),  # down along left side
    ]

    poses: list[Pose] = []
    # Accumulate arc-length positions along the perimeter, offset by start_fraction
    arc_positions = (
        np.linspace(0.0, perimeter, num_steps, endpoint=False)
        + start_fraction * perimeter
    ) % perimeter

    segment_lengths = [math.hypot(dx, dy) for dx, dy in segments]
    segment_starts = [0.0]
    for length in segment_lengths[:-1]:
        segment_starts.append(segment_starts[-1] + length)

    seg_start_xy = [
        (x0, y0),
        (x0 + inner_width, y0),
        (x0 + inner_width, y0 + inner_height),
        (x0, y0 + inner_height),
    ]

    for arc in arc_positions:
        # Find which segment this arc position falls in
        seg_idx = 0
        for i in range(len(segments) - 1, -1, -1):
            if arc >= segment_starts[i]:
                seg_idx = i
                break

        t = arc - segment_starts[seg_idx]
        sx, sy = seg_start_xy[seg_idx]
        dx, dy = segments[seg_idx]
        seg_len = segment_lengths[seg_idx]

        x = sx + dx * (t / seg_len)
        y = sy + dy * (t / seg_len)
        theta = math.atan2(dy, dx)

        poses.append(Pose(x=x, y=y, theta=theta))

    return poses


def apply_drift(
    ground_truth: list[Pose],
    *,
    drift_stddev: float,
    angular_drift_stddev: float,
    rng: np.random.Generator,
) -> list[Pose]:
    """Apply cumulative Gaussian odometric drift to a ground-truth trajectory.

    Models a robot that estimates its motion increment at each step but adds
    independent Gaussian noise.  Drift accumulates as a random walk:
    expected position error ~ drift_stddev * sqrt(t) after t steps.

    Parameters
    ----------
    ground_truth:
        Ground-truth trajectory (list of Pose namedtuples).
    drift_stddev:
        Standard deviation of Gaussian noise applied to each (x, y) increment
        in metres.
    angular_drift_stddev:
        Standard deviation of Gaussian noise applied to each theta increment
        in radians.
    rng:
        NumPy random generator with explicit seed.

    Returns
    -------
    list[Pose]
        Drifted trajectory with the same length as *ground_truth*.
        The first pose is identical to the ground-truth first pose.
    """
    if not ground_truth:
        return []

    drifted: list[Pose] = [ground_truth[0]]

    for t in range(1, len(ground_truth)):
        # True motion increment from ground truth
        true_dx = ground_truth[t].x - ground_truth[t - 1].x
        true_dy = ground_truth[t].y - ground_truth[t - 1].y
        true_dtheta = ground_truth[t].theta - ground_truth[t - 1].theta

        # Perturb increment with independent Gaussian noise
        noisy_dx = true_dx + rng.normal(0.0, drift_stddev)
        noisy_dy = true_dy + rng.normal(0.0, drift_stddev)
        noisy_dtheta = true_dtheta + rng.normal(0.0, angular_drift_stddev)

        prev = drifted[-1]
        drifted.append(Pose(
            x=prev.x + noisy_dx,
            y=prev.y + noisy_dy,
            theta=prev.theta + noisy_dtheta,
        ))

    return drifted


def generate_multi_robot_trajectories(
    *,
    num_robots: int,
    width: float,
    height: float,
    num_steps: int,
    margin: float = 1.0,
    start_fractions: list[float] | None = None,
    rng: np.random.Generator,
) -> list[list[Pose]]:
    """Generate N independent ground-truth perimeter trajectories.

    Each robot gets a distinct starting position on the perimeter controlled
    by *start_fractions*.  If not provided, robots are evenly spaced at
    fractions 0/N, 1/N, ..., (N-1)/N of the perimeter.

    Parameters
    ----------
    num_robots:
        Number of robots.
    width, height:
        Room dimensions in metres.
    num_steps:
        Steps per trajectory.
    margin:
        Distance from each wall in metres.
    start_fractions:
        Optional list of length *num_robots* specifying each robot's starting
        position as a fraction of the total perimeter length in [0, 1).
    rng:
        NumPy random generator passed through to each call of
        generate_perimeter_trajectory.

    Returns
    -------
    list[list[Pose]]
        One trajectory list per robot, each of length *num_steps*.
    """
    if start_fractions is None:
        start_fractions = [i / num_robots for i in range(num_robots)]

    if len(start_fractions) != num_robots:
        raise ValueError(
            f"start_fractions length {len(start_fractions)} != num_robots {num_robots}"
        )

    return [
        generate_perimeter_trajectory(
            width=width,
            height=height,
            num_steps=num_steps,
            margin=margin,
            start_fraction=frac,
            rng=rng,
        )
        for frac in start_fractions
    ]
