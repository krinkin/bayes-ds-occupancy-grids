# TDP: 2D environment generator -- Increment 5 extension
#
# Increment 1: single rectangular room, rectangular obstacles.
# Increment 5 adds:
#   Multi-room: subdivide space into a grid of rooms separated by internal
#     walls; carve doorway openings ('corridors') in the internal walls.
#     Layout: nC x nR rectangular rooms where nC=ceil(sqrt(N)), nR=ceil(N/nC).
#     Each doorway is a contiguous opening of width ~room_dim/4 cells.
#
#   DynamicObject: circular obstacle with linear back-and-forth motion.
#     Position at step t: back-and-forth along a unit vector with given speed
#     and half-period path_length.  The formula is continuous and deterministic
#     from the initial position and step count (no mutable state).
#
#   DynamicEnvironment: wraps the static base_grid and a list of DynamicObjects.
#     get_grid(step) rasterises objects onto a copy of the base_grid.
#     Keeps the static grid unchanged so metric evaluation uses the stable GT.
#
#   generate_environment(): main factory.  Calls generate_room() for the static
#     grid, then places DynamicObjects in free interior cells.
#
# Alternatives considered:
#   Room graphs via MST for connectivity -- guarantees reachability but overkill
#     for the acceptance criteria; corridors are randomly selected.
#   Moving obstacles via A* paths -- correct, but requires grid search per step;
#     linear back-and-forth is sufficient and O(1) per object.
#   Rasterise with scipy.ndimage.distance_transform -- simpler but adds a
#     dependency; np.meshgrid approach is self-contained.
"""2D environment generator for H-003.

Produces ground-truth occupancy grids (numpy float32 arrays) with:
- Walls along all four borders
- Optional rectangular obstacles inside the room
- Optional multi-room layouts connected by corridor doorways
- Optional dynamic (moving circular) obstacles via DynamicEnvironment

Coordinate convention:
- Grid index (row=0, col=0) corresponds to the top-left corner.
- Physical coordinates (x, y) are measured in metres from the bottom-left
  corner, with x pointing right and y pointing up.
- Conversion: row = int((height - y) / resolution), col = int(x / resolution)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Obstacle placement parameters
# ---------------------------------------------------------------------------

_MIN_OBSTACLE_FRACTION = 0.05
_MAX_OBSTACLE_FRACTION = 0.15
_OBSTACLE_PLACEMENT_RETRIES = 100


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def grid_to_world(row: int, col: int, resolution: float, height: float) -> tuple[float, float]:
    """Convert grid (row, col) indices to world (x, y) coordinates.

    Parameters
    ----------
    row, col:
        Grid indices.
    resolution:
        Metres per cell.
    height:
        Room height in metres (used to flip y axis).

    Returns
    -------
    tuple[float, float]
        (x, y) world coordinates of the cell centre.
    """
    x = (col + 0.5) * resolution
    y = height - (row + 0.5) * resolution
    return x, y


def world_to_grid(x: float, y: float, resolution: float, height: float) -> tuple[int, int]:
    """Convert world (x, y) coordinates to grid (row, col) indices.

    Parameters
    ----------
    x, y:
        World coordinates in metres.
    resolution:
        Metres per cell.
    height:
        Room height in metres.

    Returns
    -------
    tuple[int, int]
        (row, col) grid indices (not bounds-checked).
    """
    col = int(x / resolution)
    row = int((height - y) / resolution)
    return row, col


# ---------------------------------------------------------------------------
# Static obstacle placement
# ---------------------------------------------------------------------------

def _place_obstacles(
    grid: np.ndarray,
    num_obstacles: int,
    rng: np.random.Generator,
) -> None:
    """Place rectangular obstacles in free interior cells of *grid* (in-place)."""
    rows, cols = grid.shape
    interior_rows = rows - 2
    interior_cols = cols - 2

    min_h = max(1, int(_MIN_OBSTACLE_FRACTION * interior_rows))
    max_h = max(min_h, int(_MAX_OBSTACLE_FRACTION * interior_rows))
    min_w = max(1, int(_MIN_OBSTACLE_FRACTION * interior_cols))
    max_w = max(min_w, int(_MAX_OBSTACLE_FRACTION * interior_cols))

    placed = 0
    attempts = 0
    while placed < num_obstacles and attempts < _OBSTACLE_PLACEMENT_RETRIES * num_obstacles:
        attempts += 1
        obs_h = int(rng.integers(min_h, max_h + 1))
        obs_w = int(rng.integers(min_w, max_w + 1))

        r0 = int(rng.integers(1, rows - 1 - obs_h))
        c0 = int(rng.integers(1, cols - 1 - obs_w))

        r1 = r0 + obs_h
        c1 = c0 + obs_w

        if np.any(grid[r0:r1, c0:c1] > 0.0):
            continue

        grid[r0:r1, c0:c1] = 1.0
        placed += 1


# ---------------------------------------------------------------------------
# Multi-room grid generation
# ---------------------------------------------------------------------------

def _generate_multi_room_grid(
    rows: int,
    cols: int,
    nR: int,
    nC: int,
    num_corridors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a grid subdivided into nR x nC rooms with doorway openings.

    Parameters
    ----------
    rows, cols:
        Grid dimensions.
    nR, nC:
        Number of room rows and columns.
    num_corridors:
        Number of doorway openings to carve in the internal walls.
    rng:
        Random generator for doorway placement.

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype float32
        Occupancy grid: 1.0 = wall, 0.0 = free.
    """
    grid = np.zeros((rows, cols), dtype=np.float32)

    # Border walls
    grid[0, :] = 1.0
    grid[-1, :] = 1.0
    grid[:, 0] = 1.0
    grid[:, -1] = 1.0

    # Room boundary positions in grid coordinates
    r_splits = [round(rows * i / nR) for i in range(nR + 1)]
    c_splits = [round(cols * j / nC) for j in range(nC + 1)]

    # Draw internal horizontal walls (between room row bands)
    for i in range(1, nR):
        wr = r_splits[i]
        if 0 < wr < rows - 1:
            grid[wr, :] = 1.0

    # Draw internal vertical walls (between room column bands)
    for j in range(1, nC):
        wc = c_splits[j]
        if 0 < wc < cols - 1:
            grid[:, wc] = 1.0

    # Collect all possible corridor placements
    # Each entry: (kind, wall_coord, span_start, span_end)
    # "H" = horizontal wall at wall_coord row, doorway spans col [span_start, span_end)
    # "V" = vertical wall at wall_coord col, doorway spans row [span_start, span_end)
    connections: list[tuple] = []
    for i in range(nR - 1):
        wr = r_splits[i + 1]
        if 0 < wr < rows - 1:
            for j in range(nC):
                c_lo = c_splits[j] + 1
                c_hi = c_splits[j + 1] - 1
                if c_hi > c_lo + 1:
                    connections.append(("H", wr, c_lo, c_hi))
    for j in range(nC - 1):
        wc = c_splits[j + 1]
        if 0 < wc < cols - 1:
            for i in range(nR):
                r_lo = r_splits[i] + 1
                r_hi = r_splits[i + 1] - 1
                if r_hi > r_lo + 1:
                    connections.append(("V", wc, r_lo, r_hi))

    if not connections:
        return grid

    # Shuffle and carve selected doorways
    perm = rng.permutation(len(connections))
    n_doors = min(num_corridors, len(connections))
    for k in range(n_doors):
        kind, wall_coord, span_lo, span_hi = connections[int(perm[k])]
        span = span_hi - span_lo
        if kind == "H":
            door_w = max(2, span // 4)
            max_c0 = span_hi - door_w
            c0 = int(rng.integers(span_lo, max_c0 + 1)) if max_c0 >= span_lo else span_lo
            grid[wall_coord, c0:c0 + door_w] = 0.0
        else:  # "V"
            door_h = max(2, span // 4)
            max_r0 = span_hi - door_h
            r0 = int(rng.integers(span_lo, max_r0 + 1)) if max_r0 >= span_lo else span_lo
            grid[r0:r0 + door_h, wall_coord] = 0.0

    return grid


# ---------------------------------------------------------------------------
# Static room generator (public)
# ---------------------------------------------------------------------------

def generate_room(
    *,
    width: float,
    height: float,
    resolution: float,
    num_obstacles: int = 0,
    num_rooms: int = 1,
    num_corridors: int = 0,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a 2D environment as a ground-truth occupancy grid.

    For num_rooms=1 this is a single rectangular room (Increment 1 behaviour).
    For num_rooms>1 the space is divided into a grid of rooms connected by
    corridor doorways.

    Parameters
    ----------
    width:
        Room width in metres (x axis).
    height:
        Room height in metres (y axis).
    resolution:
        Metres per grid cell.
    num_obstacles:
        Number of rectangular obstacles to place inside the room(s).
    num_rooms:
        Number of rooms.  When 1 (default), a single rectangular room is
        generated.  When > 1, the space is divided into a nC x nR grid of
        rooms with internal walls.
    num_corridors:
        Number of doorway openings to carve in the internal walls.  Ignored
        when num_rooms <= 1.
    rng:
        NumPy random generator (explicit seed, no global state).

    Returns
    -------
    numpy.ndarray, shape (rows, cols), dtype float32
        Occupancy grid: 1.0 = occupied, 0.0 = free.
        rows = ceil(height / resolution), cols = ceil(width / resolution).
    """
    cols = int(np.ceil(width / resolution))
    rows = int(np.ceil(height / resolution))

    if num_rooms <= 1:
        grid = np.zeros((rows, cols), dtype=np.float32)
        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0
    else:
        nC = max(1, int(math.ceil(math.sqrt(num_rooms))))
        nR = max(1, int(math.ceil(num_rooms / nC)))
        grid = _generate_multi_room_grid(rows, cols, nR, nC, num_corridors, rng)

    _place_obstacles(grid, num_obstacles, rng)
    return grid


# ---------------------------------------------------------------------------
# Dynamic object
# ---------------------------------------------------------------------------

@dataclass
class DynamicObject:
    """A circular moving obstacle with linear back-and-forth motion.

    Motion model:
        - Starts at (cx0, cy0).
        - Moves in direction (dx, dy) (unit vector) at *speed* metres/step.
        - Bounces when it has travelled *path_length* metres from the start.
        - Period = 2 * path_length / speed steps.
        - Position is computed deterministically from step count (no state).

    Attributes
    ----------
    cx0, cy0 : float
        Initial centre position in world coordinates (metres).
    radius : float
        Obstacle radius in metres.
    dx, dy : float
        Unit direction vector.
    speed : float
        Speed in metres per step.
    path_length : float
        Distance from start to end (half-period distance).
    """

    cx0: float
    cy0: float
    radius: float
    dx: float
    dy: float
    speed: float
    path_length: float


def _dynamic_object_position(
    obj: DynamicObject,
    step: int,
) -> tuple[float, float]:
    """Return the world position of *obj* at the given *step*."""
    if obj.speed <= 0.0 or obj.path_length <= 0.0:
        return obj.cx0, obj.cy0
    dist = obj.speed * step
    period = 2.0 * obj.path_length
    t = math.fmod(dist, period)
    if t > obj.path_length:
        t = period - t
    frac = t / obj.path_length
    return (
        obj.cx0 + frac * obj.path_length * obj.dx,
        obj.cy0 + frac * obj.path_length * obj.dy,
    )


def _rasterize_circle(
    grid: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    resolution: float,
    height: float,
) -> None:
    """Rasterise a filled circle onto *grid* (in-place).  Avoids border cells."""
    rows, cols = grid.shape
    cr = int((height - cy) / resolution)
    cc = int(cx / resolution)
    r_cells = int(math.ceil(radius / resolution)) + 1

    r0 = max(1, cr - r_cells)
    r1 = min(rows - 1, cr + r_cells + 1)
    c0 = max(1, cc - r_cells)
    c1 = min(cols - 1, cc + r_cells + 1)

    if r0 >= r1 or c0 >= c1:
        return

    r_idx = np.arange(r0, r1)
    c_idx = np.arange(c0, c1)
    rr, cc_arr = np.meshgrid(r_idx, c_idx, indexing="ij")
    wx = (cc_arr + 0.5) * resolution
    wy = height - (rr + 0.5) * resolution
    dist_sq = (wx - cx) ** 2 + (wy - cy) ** 2
    mask = dist_sq <= radius ** 2
    grid[r0:r1, c0:c1][mask] = 1.0


# ---------------------------------------------------------------------------
# Dynamic environment
# ---------------------------------------------------------------------------

class DynamicEnvironment:
    """Static background grid plus a list of moving circular obstacles.

    Parameters
    ----------
    base_grid:
        Static occupancy grid (walls + static obstacles).  Shape (rows, cols),
        dtype float32.  Copied on construction -- caller's array is not modified.
    objects:
        List of DynamicObject instances.
    width, height, resolution:
        Physical dimensions of the environment (metres and metres/cell).
    """

    def __init__(
        self,
        base_grid: np.ndarray,
        objects: list[DynamicObject],
        width: float,
        height: float,
        resolution: float,
    ) -> None:
        self._base_grid = base_grid.copy()
        self.objects = list(objects)
        self.width = width
        self.height = height
        self.resolution = resolution

    @property
    def base_grid(self) -> np.ndarray:
        """Static background grid (read-only copy)."""
        return self._base_grid

    @property
    def has_dynamic_objects(self) -> bool:
        """True if any moving obstacles are present."""
        return len(self.objects) > 0

    def get_grid(self, step: int = 0) -> np.ndarray:
        """Return the ground-truth occupancy grid at the given simulation step.

        For purely static environments (no DynamicObjects) this returns
        ``base_grid`` without copying.  When dynamic objects are present a
        fresh copy is created with the objects rasterised at their current
        positions.

        Parameters
        ----------
        step:
            Current simulation step (0-indexed).

        Returns
        -------
        np.ndarray, shape (rows, cols), dtype float32
        """
        if not self.objects:
            return self._base_grid
        grid = self._base_grid.copy()
        for obj in self.objects:
            cx, cy = _dynamic_object_position(obj, step)
            _rasterize_circle(
                grid, cx, cy, obj.radius, self.resolution, self.height
            )
        return grid

    def get_dynamic_mask(self, total_steps: int) -> np.ndarray:
        """Boolean mask: True for cells occupied by a dynamic object at any step.

        Iterates over steps in ``[0, total_steps)`` and marks cells where the
        dynamic grid exceeds the static base grid (dynamic objects only ADD
        occupied cells).

        Parameters
        ----------
        total_steps:
            Number of simulation steps to sweep.

        Returns
        -------
        np.ndarray, shape (rows, cols), dtype bool
            True where any dynamic object occupied the cell at any step.
        """
        mask = np.zeros(self._base_grid.shape, dtype=bool)
        if not self.objects:
            return mask
        for step in range(total_steps):
            grid = self.get_grid(step)
            mask |= (grid > self._base_grid)
        return mask


# ---------------------------------------------------------------------------
# Top-level factory
# ---------------------------------------------------------------------------

def generate_environment(
    *,
    width: float,
    height: float,
    resolution: float,
    num_rooms: int = 1,
    num_corridors: int = 0,
    num_obstacles: int = 0,
    num_dynamic_objects: int = 0,
    dynamic_speed: float = 0.5,
    rng: np.random.Generator,
) -> DynamicEnvironment:
    """Generate a complete environment (static + optional dynamic objects).

    Calls ``generate_room`` for the static background, then places
    DynamicObjects in randomly chosen free interior cells.

    Parameters
    ----------
    width, height:
        Dimensions in metres.
    resolution:
        Metres per grid cell.
    num_rooms:
        Number of rooms.  See ``generate_room``.
    num_corridors:
        Number of corridor doorways.  See ``generate_room``.
    num_obstacles:
        Number of static rectangular obstacles.
    num_dynamic_objects:
        Number of moving circular obstacles.
    dynamic_speed:
        Speed of each dynamic object in metres per simulation step.
    rng:
        NumPy random generator (all randomness flows through this).

    Returns
    -------
    DynamicEnvironment
        Wraps the static grid and moving obstacles.
    """
    base_grid = generate_room(
        width=width,
        height=height,
        resolution=resolution,
        num_obstacles=num_obstacles,
        num_rooms=num_rooms,
        num_corridors=num_corridors,
        rng=rng,
    )

    objects: list[DynamicObject] = []
    if num_dynamic_objects > 0:
        rows, cols = base_grid.shape
        # Collect free interior cell indices
        free_r, free_c = np.where(base_grid[1:-1, 1:-1] == 0.0)
        free_r = free_r + 1  # shift for the border slicing offset
        free_c = free_c + 1

        obj_radius = max(resolution * 1.5, 0.3)
        path_length = max(min(width, height) * 0.3, 1.0)

        placed = 0
        attempts = 0
        max_attempts = max(num_dynamic_objects * 50, 100)
        while placed < num_dynamic_objects and attempts < max_attempts:
            attempts += 1
            if len(free_r) == 0:
                break
            idx = int(rng.integers(0, len(free_r)))
            r, c = int(free_r[idx]), int(free_c[idx])
            cx0, cy0 = grid_to_world(r, c, resolution, height)
            # Random direction
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            dx = math.cos(angle)
            dy = math.sin(angle)
            objects.append(
                DynamicObject(
                    cx0=cx0,
                    cy0=cy0,
                    radius=obj_radius,
                    dx=dx,
                    dy=dy,
                    speed=dynamic_speed,
                    path_length=path_length,
                )
            )
            placed += 1

    return DynamicEnvironment(
        base_grid=base_grid,
        objects=objects,
        width=width,
        height=height,
        resolution=resolution,
    )
