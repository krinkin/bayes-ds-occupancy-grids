# TDP: Minimal structured JSON logger
# Approach: Write one JSON object per line (JSONL format) to a log file.
#   Each entry contains step number, robot poses, and metric values.
#   Grid snapshots store occupancy probability (float32, [0,1]) in grid_data --
#   NOT log-odds. Snapshots are written every N steps to avoid large files.
#   The logger streams directly to disk and does not accumulate entries in RAM.
# Schema per step entry:
#   { "step": int, "pose": {x,y,theta},  <- first robot (backward compat)
#     "poses": [{robot, x, y, theta}, ...],  <- all active robots
#     "metrics": {...},
#     "grid_snapshot": [[...]] }  <- occupancy prob, shape (rows, cols)
# Alternatives considered: CSV -- easier to analyse in pandas but cannot
#   accommodate the variable-size grid snapshot; JSONL handles both uniformly.
# Risks: JSON serialisation of large grids (1000x1000) can be slow. Mitigated
#   by configurable snapshot interval; default 10 steps is safe for small configs.
"""Minimal structured JSON (JSONL) logger for H-003 experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ExperimentLogger:
    """Writes experiment events to a JSONL file.

    Parameters
    ----------
    log_path:
        Path to the output JSONL log file.
    grid_snapshot_interval:
        Write a grid snapshot entry every this many steps. Set to 0 to
        disable snapshots.
    """

    def __init__(self, log_path: Path, grid_snapshot_interval: int = 10) -> None:
        self._path = log_path
        self._interval = grid_snapshot_interval
        self._fh = log_path.open("w", encoding="utf-8")

    def log_step(
        self,
        *,
        step: int,
        pose: tuple[float, float, float],
        poses: list[tuple[int, float, float, float]] | None = None,
        metrics: dict[str, float],
        grid_data: np.ndarray | None = None,
    ) -> None:
        """Write a step entry to the log file.

        Parameters
        ----------
        step:
            Current step number (0-indexed).
        pose:
            First active robot's pose as (x, y, theta). Kept for backward
            compatibility; prefer poses for multi-robot logging.
        poses:
            All active robots' poses as [(robot_idx, x, y, theta), ...].
            Stored as ``"poses": [{"robot": r, "x": ..., "y": ..., "theta": ...}]``.
        metrics:
            Dict of metric name -> value for this step.
        grid_data:
            Occupancy probability grid (values in [0,1]). Written as a snapshot
            if the step number is a multiple of the snapshot interval and
            grid_data is not None.
        """
        entry: dict[str, Any] = {
            "step": step,
            "pose": {"x": float(pose[0]), "y": float(pose[1]), "theta": float(pose[2])},
            "metrics": {k: (float(v) if not (isinstance(v, float) and np.isnan(v)) else None)
                        for k, v in metrics.items()},
        }

        if poses is not None:
            entry["poses"] = [
                {"robot": int(r), "x": float(x), "y": float(y), "theta": float(t)}
                for r, x, y, t in poses
            ]

        include_snapshot = (
            grid_data is not None
            and self._interval > 0
            and step % self._interval == 0
        )
        if include_snapshot and grid_data is not None:
            entry["grid_snapshot"] = grid_data.tolist()

        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Write an arbitrary event entry to the log file.

        Parameters
        ----------
        event_type:
            Short string identifying the event kind.
        data:
            Arbitrary JSON-serialisable data.
        """
        entry: dict[str, Any] = {"event": event_type, **data}
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the log file handle."""
        self._fh.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
