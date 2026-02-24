"""A* path planning and downstream evaluation metrics for occupancy grid comparison.

This module provides:
- astar: standard A* path planner on a binary obstacle grid
- min_clearance: minimum distance from a path to the nearest obstacle
- evaluate_pair: compute all downstream metrics for one start-goal pair
- run_downstream_eval: full evaluation over N sampled start-goal pairs

Downstream metrics
------------------
For each randomly sampled (start, goal) pair, both maps are planned on
independently and the following quantities are measured:

1. path_equivalence_rate
   Fraction of pairs where both grids produce paths within 1 cell of each
   other at every waypoint (Hausdorff distance <= 1.0 cells).

2. min_clearance_diff
   Mean absolute difference in minimum obstacle clearance (cells) between
   Bayesian path and DS path, averaged over pairs where both maps succeed.

3. path_length_diff
   Mean absolute difference in path length (cells) between the two maps,
   averaged over pairs where both maps succeed.

4. planning_success_rate_bayesian / planning_success_rate_ds
   Fraction of pairs where the given map found a path.

5. safety_critical_disagreement_rate
   Fraction of pairs where:
   - One map finds a path and the other does not, OR
   - Both maps find paths but clearance differs by more than 1 cell.

Coordinate convention
---------------------
Grids are 2D numpy arrays (rows, cols).  Cells are (row, col) tuples.
Occupancy threshold 0.5: cells with probability >= 0.5 are obstacles.
The robot is modeled as a point (no inflation).

Design decisions
----------------
- No external pathfinding library: pure numpy + heapq A*.
- Deterministic: fixed random seed for start-goal sampling.
- Progress: prints one line per pair with flush.
- Clearance computed via scipy.ndimage.distance_transform_edt on the obstacle
  binary mask (exact Euclidean distance in cells).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------------
# A* path planner
# ---------------------------------------------------------------------------

def astar(
    obstacle_map: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> Optional[list[tuple[int, int]]]:
    """Find a path from start to goal on a binary obstacle grid using A*.

    Parameters
    ----------
    obstacle_map:
        Binary 2D array, shape (rows, cols), dtype bool or uint8.
        True / 1 = obstacle (impassable), False / 0 = free.
    start:
        (row, col) of the start cell.  Must be free.
    goal:
        (row, col) of the goal cell.  Must be free.

    Returns
    -------
    list[tuple[int, int]] or None
        Ordered list of (row, col) cells from start (inclusive) to goal
        (inclusive), or None if no path exists.

    Notes
    -----
    - 8-connected grid (diagonals allowed, cost sqrt(2)).
    - Heuristic: octile distance (admissible for 8-connectivity).
    - Returns None if start or goal is an obstacle.
    """
    rows, cols = obstacle_map.shape

    def _in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    sr, sc = start
    gr, gc = goal

    if obstacle_map[sr, sc] or obstacle_map[gr, gc]:
        return None
    if start == goal:
        return [start]

    def _octile(r: int, c: int) -> float:
        dr = abs(r - gr)
        dc = abs(c - gc)
        return (dr + dc) + (math.sqrt(2.0) - 2.0) * min(dr, dc)

    # (f, g, row, col)
    open_heap: list[tuple[float, float, int, int]] = []
    heapq.heappush(open_heap, (_octile(sr, sc), 0.0, sr, sc))

    g_score: dict[tuple[int, int], float] = {(sr, sc): 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    _neighbors = [
        (-1, -1, math.sqrt(2.0)),
        (-1,  0, 1.0),
        (-1,  1, math.sqrt(2.0)),
        ( 0, -1, 1.0),
        ( 0,  1, 1.0),
        ( 1, -1, math.sqrt(2.0)),
        ( 1,  0, 1.0),
        ( 1,  1, math.sqrt(2.0)),
    ]

    while open_heap:
        f, g, r, c = heapq.heappop(open_heap)
        node = (r, c)

        if node in closed:
            continue
        closed.add(node)

        if node == goal:
            # Reconstruct path
            path: list[tuple[int, int]] = [node]
            cur = node
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        for dr, dc, cost in _neighbors:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)
            if not _in_bounds(nr, nc):
                continue
            if obstacle_map[nr, nc]:
                continue
            if nb in closed:
                continue
            tentative_g = g + cost
            if tentative_g < g_score.get(nb, float("inf")):
                g_score[nb] = tentative_g
                came_from[nb] = node
                heapq.heappush(
                    open_heap,
                    (tentative_g + _octile(nr, nc), tentative_g, nr, nc),
                )

    return None


# ---------------------------------------------------------------------------
# Clearance computation
# ---------------------------------------------------------------------------

def compute_clearance_map(obstacle_map: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance transform from every free cell to the nearest obstacle.

    Parameters
    ----------
    obstacle_map:
        Binary 2D array.  True / 1 = obstacle.

    Returns
    -------
    numpy.ndarray, shape (rows, cols), dtype float64
        Distance in cells from each cell to the nearest obstacle cell.
        Obstacle cells themselves have distance 0.
    """
    free_mask = ~obstacle_map.astype(bool)
    return distance_transform_edt(free_mask)


def min_clearance(
    path: list[tuple[int, int]],
    clearance_map: np.ndarray,
) -> float:
    """Return the minimum clearance along a path.

    Parameters
    ----------
    path:
        Sequence of (row, col) cells.
    clearance_map:
        Output of compute_clearance_map for the same grid.

    Returns
    -------
    float
        Minimum distance to the nearest obstacle along the path, in cells.
        Returns 0.0 for an empty path.
    """
    if not path:
        return 0.0
    return float(min(clearance_map[r, c] for r, c in path))


# ---------------------------------------------------------------------------
# Path equivalence (Hausdorff distance <= 1 cell)
# ---------------------------------------------------------------------------

def _path_length(path: list[tuple[int, int]]) -> float:
    """Sum of step lengths (1.0 for cardinal, sqrt(2) for diagonal)."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        dr = abs(path[i + 1][0] - path[i][0])
        dc = abs(path[i + 1][1] - path[i][1])
        total += math.sqrt(float(dr * dr + dc * dc))
    return total


def _hausdorff_distance(
    path_a: list[tuple[int, int]],
    path_b: list[tuple[int, int]],
) -> float:
    """One-sided (max of min) Hausdorff distance between two paths in cell units.

    Uses the symmetric Hausdorff: max(directed_a_to_b, directed_b_to_a).
    """
    if not path_a or not path_b:
        return float("inf")

    arr_a = np.array(path_a, dtype=np.float64)
    arr_b = np.array(path_b, dtype=np.float64)

    # Directed: for each point in A, min dist to any point in B
    def _directed(src: np.ndarray, dst: np.ndarray) -> float:
        max_min = 0.0
        for pt in src:
            diffs = dst - pt
            dists = np.sqrt((diffs ** 2).sum(axis=1))
            max_min = max(max_min, float(dists.min()))
        return max_min

    return max(_directed(arr_a, arr_b), _directed(arr_b, arr_a))


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    """Result for one start-goal pair.

    Attributes
    ----------
    start, goal:
        Cell coordinates.
    bayesian_found:
        True if Bayesian map produced a valid path.
    ds_found:
        True if DS map produced a valid path.
    path_equivalent:
        True if both paths found and Hausdorff distance <= 1 cell.
        None if at least one map did not find a path.
    clearance_diff:
        Absolute difference in minimum clearance (cells).
        None if at least one map did not find a path.
    length_diff:
        Absolute difference in path length (cells).
        None if at least one map did not find a path.
    safety_critical_disagreement:
        True if one map succeeds and the other fails, OR if both succeed
        but clearance differs by more than 1 cell.
    hausdorff:
        Hausdorff distance between the two paths (cells), or None.
    """

    start: tuple[int, int]
    goal: tuple[int, int]
    bayesian_found: bool
    ds_found: bool
    path_equivalent: Optional[bool]
    clearance_diff: Optional[float]
    length_diff: Optional[float]
    safety_critical_disagreement: bool
    hausdorff: Optional[float]


def evaluate_pair(
    start: tuple[int, int],
    goal: tuple[int, int],
    obs_bayesian: np.ndarray,
    obs_ds: np.ndarray,
    clearance_bayesian: np.ndarray,
    clearance_ds: np.ndarray,
) -> PairResult:
    """Plan on both maps for a single (start, goal) and compute all metrics.

    Parameters
    ----------
    start, goal:
        (row, col) cells.
    obs_bayesian, obs_ds:
        Binary obstacle maps (True = obstacle) derived from each grid.
    clearance_bayesian, clearance_ds:
        Precomputed clearance maps for each binary obstacle map.

    Returns
    -------
    PairResult
    """
    path_b = astar(obs_bayesian, start, goal)
    path_d = astar(obs_ds, start, goal)

    found_b = path_b is not None
    found_d = path_d is not None

    path_equivalent: Optional[bool] = None
    clearance_diff: Optional[float] = None
    length_diff: Optional[float] = None
    hausdorff: Optional[float] = None

    if found_b and found_d:
        assert path_b is not None
        assert path_d is not None
        h = _hausdorff_distance(path_b, path_d)
        hausdorff = h
        path_equivalent = h <= 1.0

        clr_b = min_clearance(path_b, clearance_bayesian)
        clr_d = min_clearance(path_d, clearance_ds)
        clearance_diff = abs(clr_b - clr_d)

        len_b = _path_length(path_b)
        len_d = _path_length(path_d)
        length_diff = abs(len_b - len_d)

        safety_critical = clearance_diff > 1.0
    else:
        # One or both maps failed to find a path
        safety_critical = found_b != found_d

    return PairResult(
        start=start,
        goal=goal,
        bayesian_found=found_b,
        ds_found=found_d,
        path_equivalent=path_equivalent,
        clearance_diff=clearance_diff,
        length_diff=length_diff,
        safety_critical_disagreement=safety_critical,
        hausdorff=hausdorff,
    )


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Aggregate downstream evaluation result over N pairs.

    Attributes
    ----------
    n_pairs:
        Total number of (start, goal) pairs attempted.
    path_equivalence_rate:
        Fraction of pairs where both paths are within 1 cell (Hausdorff <= 1).
        Denominator: pairs where both maps succeeded.
    mean_clearance_diff:
        Mean |clearance_bayesian - clearance_ds| in cells.
        NaN if no successful pairs.
    mean_length_diff:
        Mean |length_bayesian - length_ds| in cells.
        NaN if no successful pairs.
    planning_success_rate_bayesian:
        Fraction of pairs where Bayesian map found a path.
    planning_success_rate_ds:
        Fraction of pairs where DS map found a path.
    safety_critical_disagreement_rate:
        Fraction of pairs with a safety-critical disagreement.
    n_both_succeeded:
        Number of pairs where both maps found a path.
    pair_results:
        Per-pair detail records.
    """

    n_pairs: int
    path_equivalence_rate: float
    mean_clearance_diff: float
    mean_length_diff: float
    planning_success_rate_bayesian: float
    planning_success_rate_ds: float
    safety_critical_disagreement_rate: float
    n_both_succeeded: int
    pair_results: list[PairResult] = field(default_factory=list)


def _sample_free_cells(
    obstacle_map: np.ndarray,
    n: int,
    rng: np.random.Generator,
    min_distance: float = 5.0,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Sample N (start, goal) pairs from free cells in obstacle_map.

    Parameters
    ----------
    obstacle_map:
        Binary obstacle map (True = obstacle).
    n:
        Number of pairs to sample.
    rng:
        Seeded random generator.
    min_distance:
        Minimum Euclidean distance in cells between start and goal.
        Pairs closer than this are rejected and re-sampled.

    Returns
    -------
    list of (start, goal) tuples as (row, col) pairs.
    """
    free_rows, free_cols = np.where(~obstacle_map)
    if len(free_rows) < 2:
        return []

    free_cells = list(zip(free_rows.tolist(), free_cols.tolist()))
    n_free = len(free_cells)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    max_attempts = n * 50

    attempts = 0
    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        idx_start = int(rng.integers(0, n_free))
        idx_goal = int(rng.integers(0, n_free))
        if idx_start == idx_goal:
            continue
        sr, sc = free_cells[idx_start]
        gr, gc = free_cells[idx_goal]
        dist = math.sqrt(float((sr - gr) ** 2 + (sc - gc) ** 2))
        if dist < min_distance:
            continue
        pairs.append(((sr, sc), (gr, gc)))

    return pairs


def occupancy_prob_to_obstacle_map(
    prob: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert occupancy probability grid to binary obstacle map.

    Parameters
    ----------
    prob:
        Occupancy probability in [0, 1], shape (rows, cols).
    threshold:
        Cells with probability >= threshold are obstacles.

    Returns
    -------
    numpy.ndarray, shape (rows, cols), dtype bool
        True = obstacle.
    """
    return prob >= threshold


def run_downstream_eval(
    bayesian_prob: np.ndarray,
    ds_prob: np.ndarray,
    n_pairs: int = 100,
    seed: int = 0,
    occ_threshold: float = 0.5,
    min_start_goal_distance: float = 5.0,
    verbose: bool = True,
) -> EvalResult:
    """Run downstream path planning evaluation comparing two occupancy grids.

    Parameters
    ----------
    bayesian_prob:
        Occupancy probability grid from Bayesian fusion, shape (rows, cols),
        values in [0, 1].  Cells >= occ_threshold are treated as obstacles.
    ds_prob:
        Occupancy probability grid from DS/TBM fusion, same shape and
        convention as bayesian_prob.
    n_pairs:
        Number of random (start, goal) pairs to evaluate.  Default 100.
    seed:
        Random seed for reproducible pair sampling.  Default 0.
    occ_threshold:
        Probability threshold for classifying cells as obstacles.
    min_start_goal_distance:
        Minimum Euclidean distance in cells between start and goal.
    verbose:
        If True, print one progress line per pair with flush.

    Returns
    -------
    EvalResult
        Aggregated metrics and per-pair results.
    """
    if bayesian_prob.shape != ds_prob.shape:
        raise ValueError(
            f"Grid shape mismatch: bayesian {bayesian_prob.shape} vs ds {ds_prob.shape}"
        )

    # Build binary obstacle maps
    obs_b = occupancy_prob_to_obstacle_map(bayesian_prob, occ_threshold)
    obs_d = occupancy_prob_to_obstacle_map(ds_prob, occ_threshold)

    # Precompute clearance maps (expensive, done once)
    if verbose:
        print("[downstream_eval] Computing clearance maps...", flush=True)
    clr_b = compute_clearance_map(obs_b)
    clr_d = compute_clearance_map(obs_d)

    # Sample start-goal pairs from cells free in both maps (ensures fairness)
    obs_union = obs_b | obs_d  # obstacle if either map says so
    rng = np.random.default_rng(seed)
    pairs = _sample_free_cells(obs_union, n_pairs, rng, min_start_goal_distance)

    if not pairs:
        raise ValueError(
            "Could not sample any valid (start, goal) pairs. "
            "The grids may be nearly entirely occupied."
        )

    # Evaluate each pair
    pair_results: list[PairResult] = []
    for idx, (start, goal) in enumerate(pairs):
        result = evaluate_pair(start, goal, obs_b, obs_d, clr_b, clr_d)
        pair_results.append(result)
        if verbose:
            status_b = "OK" if result.bayesian_found else "NO"
            status_d = "OK" if result.ds_found else "NO"
            equiv = (
                f"equiv={result.path_equivalent}"
                if result.path_equivalent is not None
                else "equiv=N/A"
            )
            print(
                f"[downstream_eval] pair {idx + 1:3d}/{len(pairs)} "
                f"start={start} goal={goal} "
                f"bayes={status_b} ds={status_d} {equiv}",
                flush=True,
            )

    # Aggregate
    n_total = len(pair_results)
    found_b_list = [r.bayesian_found for r in pair_results]
    found_d_list = [r.ds_found for r in pair_results]

    success_rate_b = float(sum(found_b_list)) / n_total if n_total > 0 else float("nan")
    success_rate_d = float(sum(found_d_list)) / n_total if n_total > 0 else float("nan")

    both_succeeded = [r for r in pair_results if r.bayesian_found and r.ds_found]
    n_both = len(both_succeeded)

    equiv_values = [r.path_equivalent for r in both_succeeded if r.path_equivalent is not None]
    path_equiv_rate = float(sum(equiv_values)) / len(equiv_values) if equiv_values else float("nan")

    clr_diffs = [r.clearance_diff for r in both_succeeded if r.clearance_diff is not None]
    mean_clr_diff = float(np.mean(clr_diffs)) if clr_diffs else float("nan")

    len_diffs = [r.length_diff for r in both_succeeded if r.length_diff is not None]
    mean_len_diff = float(np.mean(len_diffs)) if len_diffs else float("nan")

    scd_values = [r.safety_critical_disagreement for r in pair_results]
    scd_rate = float(sum(scd_values)) / n_total if n_total > 0 else float("nan")

    return EvalResult(
        n_pairs=n_total,
        path_equivalence_rate=path_equiv_rate,
        mean_clearance_diff=mean_clr_diff,
        mean_length_diff=mean_len_diff,
        planning_success_rate_bayesian=success_rate_b,
        planning_success_rate_ds=success_rate_d,
        safety_critical_disagreement_rate=scd_rate,
        n_both_succeeded=n_both,
        pair_results=pair_results,
    )
