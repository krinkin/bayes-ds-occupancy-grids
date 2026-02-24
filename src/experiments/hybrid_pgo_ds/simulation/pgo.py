# TDP: 2D Pose Graph Optimisation via scipy.optimize.least_squares
#
# Approach:
#   Model each robot as having an unknown SE(2) frame correction
#   (dx, dy, dtheta) that is applied uniformly to every pose in its trajectory.
#   When two robots rendezvous at step t, we have a relative position measurement
#   between them.  PGO finds corrections that minimise the sum of squared
#   rendezvous residuals across all constraint events.
#
#   State vector: x = [dx_1, dy_1, dtheta_1, dx_2, dy_2, dtheta_2, ...]
#   Robot 0 is the anchor (fixed at (0,0,0)) -- prevents gauge freedom.
#   Total free variables: (n_robots - 1) * 3.
#
#   Residual for rendezvous at step t between robots i and j:
#     T_i = (dx_i, dy_i, dtheta_i),  T_j = (dx_j, dy_j, dtheta_j)
#     corrected_i = R(dtheta_i) * drifted_i[t].xy + (dx_i, dy_i)
#     corrected_j = R(dtheta_j) * drifted_j[t].xy + (dx_j, dy_j)
#     residual_x = corrected_j.x - corrected_i.x - measured_delta_x
#     residual_y = corrected_j.y - corrected_i.y - measured_delta_y
#
#   Angular regularisation: add one residual per non-anchor robot,
#     reg_weight * dtheta_k, to prevent the system being underdetermined when
#     n_constraints < n_robots - 1.  This ensures m >= n required by LM.
#     reg_weight = 1e-3 (<<1m) does not bias the translation corrections.
#
#   Solver: scipy.optimize.least_squares with method='lm' (Levenberg-Marquardt).
#   Initial guess: all corrections zero.
#
#   Sparsity: for small N (2-4 robots), dense Jacobian.  scipy handles
#   N < 10 comfortably without explicit sparsity structure.  Larger N
#   would benefit from scipy.sparse; deferred to a later increment.
#
# Alternatives considered:
#   g2o / GTSAM: standard for production SLAM, but require compilation and
#     are not readily available on Raspberry Pi.  Out of scope for this experiment.
#   scipy.sparse + manual Gauss-Newton: minimal dependencies but significant
#     implementation complexity; LM via scipy.optimize is cleaner and correct.
#   Odometric edges (consecutive-pose constraints): would improve conditioning
#     when rendezvous events are sparse.  Deferred; current increment assumes
#     sufficient rendezvous density.
#
# Risks:
#   Underdetermined system (fewer constraints than DOF): scipy.least_squares
#   returns the minimum-norm solution, which is the zero correction -- correct
#   behaviour when there is no evidence to correct.
#   Large angular drift may cause the linearised approximation to diverge;
#   LM handles moderate nonlinearity well (dtheta < 0.5 rad is safe).
"""2D Pose Graph Optimisation for multi-robot frame alignment.

Provides:
- RendezvousConstraint: data class recording a relative measurement
- detect_rendezvous: scan paired trajectories for close-encounter events
- solve_pgo: find per-robot SE(2) corrections from rendezvous constraints
- apply_corrections: apply corrections to drifted trajectories
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.experiments.hybrid_pgo_ds.simulation.trajectory import Pose


@dataclass(frozen=True)
class RendezvousConstraint:
    """A relative pose measurement between two robots at a given step.

    The measurement states: after correction, robot *robot_j* should be at
    position (robot_i.corrected + delta_x, robot_i.corrected + delta_y).

    *delta_x* and *delta_y* are derived from the ground-truth positions at
    the rendezvous step, optionally perturbed by sensor noise.
    """

    step: int        # trajectory step index
    robot_i: int     # first robot (reference)
    robot_j: int     # second robot (relative to i)
    delta_x: float   # measured x offset of j relative to i (metres)
    delta_y: float   # measured y offset of j relative to i (metres)


def detect_rendezvous(
    ground_truth: list[list[Pose]],
    rendezvous_distance: float,
    measurement_noise_stddev: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[RendezvousConstraint]:
    """Detect rendezvous events and create relative-pose constraints.

    For each time step, checks every pair (i, j) of robots.  When their
    ground-truth positions are within *rendezvous_distance*, a constraint
    is emitted.  A debounce prevents emitting consecutive constraints for a
    single prolonged encounter: one constraint per encounter (first step
    the pair enters range, then suppressed until they leave and re-enter).

    The measured delta is the ground-truth relative position, optionally
    perturbed by zero-mean Gaussian noise with *measurement_noise_stddev*.

    Parameters
    ----------
    ground_truth:
        Ground-truth trajectories, shape (n_robots, n_steps).
    rendezvous_distance:
        Distance threshold in metres.
    measurement_noise_stddev:
        Optional noise added to the measured relative position.
    rng:
        Random generator required if measurement_noise_stddev > 0.

    Returns
    -------
    list[RendezvousConstraint]
    """
    n_robots = len(ground_truth)
    if n_robots < 2 or not ground_truth[0]:
        return []

    n_steps = len(ground_truth[0])
    constraints: list[RendezvousConstraint] = []

    # Track which pairs are currently "in range" for debouncing
    in_range: dict[tuple[int, int], bool] = {}
    for i in range(n_robots):
        for j in range(i + 1, n_robots):
            in_range[(i, j)] = False

    for t in range(n_steps):
        for i in range(n_robots):
            for j in range(i + 1, n_robots):
                pi = ground_truth[i][t]
                pj = ground_truth[j][t]
                dist = math.hypot(pj.x - pi.x, pj.y - pi.y)
                pair = (i, j)

                if dist < rendezvous_distance:
                    if not in_range[pair]:
                        # First step entering rendezvous range: emit constraint
                        in_range[pair] = True
                        dx_gt = pj.x - pi.x
                        dy_gt = pj.y - pi.y
                        if measurement_noise_stddev > 0.0 and rng is not None:
                            dx_gt += rng.normal(0.0, measurement_noise_stddev)
                            dy_gt += rng.normal(0.0, measurement_noise_stddev)
                        constraints.append(RendezvousConstraint(
                            step=t,
                            robot_i=i,
                            robot_j=j,
                            delta_x=dx_gt,
                            delta_y=dy_gt,
                        ))
                else:
                    in_range[pair] = False

    return constraints


def _apply_se2(
    pose: Pose,
    dx: float,
    dy: float,
    dtheta: float,
) -> tuple[float, float]:
    """Apply SE(2) frame correction to a pose position.

    Returns corrected (x, y).  The correction rotates the position vector
    by *dtheta* (around the world origin) then translates by (dx, dy).
    For small *dtheta*, this is approximately a pure translation.
    """
    cos_d = math.cos(dtheta)
    sin_d = math.sin(dtheta)
    cx = cos_d * pose.x - sin_d * pose.y + dx
    cy = sin_d * pose.x + cos_d * pose.y + dy
    return cx, cy


def _build_residuals(
    params: np.ndarray,
    drifted: list[list[Pose]],
    constraints: list[RendezvousConstraint],
    n_robots: int,
    reg_weight: float = 1e-3,
) -> np.ndarray:
    """Compute residuals for scipy.optimize.least_squares.

    Each rendezvous constraint contributes 2 residuals (x and y).
    An angular regularisation term (reg_weight * dtheta_k) is added per
    non-anchor robot to prevent the system being underdetermined when there
    are fewer constraints than free variables.  The regularisation weight is
    small enough not to bias translation corrections.
    """
    # Decode per-robot corrections; robot 0 is anchored at (0, 0, 0)
    corrections: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    for k in range(1, n_robots):
        base = (k - 1) * 3
        corrections.append((float(params[base]), float(params[base + 1]), float(params[base + 2])))

    residuals: list[float] = []
    for c in constraints:
        dx_i, dy_i, dt_i = corrections[c.robot_i]
        dx_j, dy_j, dt_j = corrections[c.robot_j]

        cxi, cyi = _apply_se2(drifted[c.robot_i][c.step], dx_i, dy_i, dt_i)
        cxj, cyj = _apply_se2(drifted[c.robot_j][c.step], dx_j, dy_j, dt_j)

        residuals.append(cxj - cxi - c.delta_x)
        residuals.append(cyj - cyi - c.delta_y)

    # Angular regularisation: penalise large dtheta corrections (prefer zero rotation)
    for k in range(1, n_robots):
        residuals.append(reg_weight * params[(k - 1) * 3 + 2])

    return np.array(residuals, dtype=np.float64)


def solve_pgo(
    drifted: list[list[Pose]],
    constraints: list[RendezvousConstraint],
    anchor_robot: int = 0,
) -> list[tuple[float, float, float]]:
    """Find per-robot SE(2) frame corrections that satisfy rendezvous constraints.

    Parameters
    ----------
    drifted:
        Drifted trajectories, one list per robot.
    constraints:
        Rendezvous constraints from detect_rendezvous.
    anchor_robot:
        Index of the robot whose correction is fixed at (0, 0, 0).
        Currently only robot 0 is supported as anchor.

    Returns
    -------
    list[tuple[float, float, float]]
        Per-robot corrections (dx, dy, dtheta), one tuple per robot.
        The anchor robot's correction is always (0.0, 0.0, 0.0).
    """
    n_robots = len(drifted)

    if n_robots <= 1 or not constraints:
        # Nothing to optimise
        return [(0.0, 0.0, 0.0)] * n_robots

    # Free variables: (n_robots - 1) * 3  (robot 0 anchored)
    x0 = np.zeros((n_robots - 1) * 3, dtype=np.float64)

    result = least_squares(
        _build_residuals,
        x0,
        args=(drifted, constraints, n_robots),
        method="lm",
    )

    corrections: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    for k in range(1, n_robots):
        base = (k - 1) * 3
        corrections.append((
            float(result.x[base]),
            float(result.x[base + 1]),
            float(result.x[base + 2]),
        ))
    return corrections


def apply_corrections(
    drifted: list[list[Pose]],
    corrections: list[tuple[float, float, float]],
) -> list[list[Pose]]:
    """Apply per-robot SE(2) corrections to drifted trajectories.

    Parameters
    ----------
    drifted:
        Drifted trajectories, one list per robot.
    corrections:
        Per-robot (dx, dy, dtheta) corrections from solve_pgo.

    Returns
    -------
    list[list[Pose]]
        Corrected trajectories with the same shape as *drifted*.
    """
    corrected: list[list[Pose]] = []
    for robot_idx, traj in enumerate(drifted):
        dx, dy, dtheta = corrections[robot_idx]
        corrected_traj: list[Pose] = []
        for pose in traj:
            cx, cy = _apply_se2(pose, dx, dy, dtheta)
            corrected_traj.append(Pose(x=cx, y=cy, theta=pose.theta + dtheta))
        corrected.append(corrected_traj)
    return corrected
