"""Load Intel Research Lab dataset from Carmen/GFS log format.

The Intel Research Lab dataset (Dirk Haehnel, 2003) contains 2D laser range
scans with odometry from a single robot navigating an indoor office environment.

Supports multiple formats:

1. **GFS format** (GMapping output):
   - LASER_READING: 180 range values + pose + timestamp
   - SM_UPDATE: scan-matched particle poses (used for corrected pose)
   - We use the best particle from SM_UPDATE as the corrected pose.

2. **Carmen log format**:
   - FLASER: Front laser with pose and odometry
   - ROBOTLASER1: Alternative format with explicit angle parameters

GFS LASER_READING format::

    LASER_READING num_readings r1 r2 ... rN x y theta timestamp

Where x, y, theta is the raw odometry pose and the corrected pose comes
from the preceding SM_UPDATE line (best particle by weight).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LaserScan:
    """A single laser scan with pose information."""

    timestamp: float
    """Scan timestamp in seconds."""

    pose_x: float
    """Corrected x position (meters)."""

    pose_y: float
    """Corrected y position (meters)."""

    pose_theta: float
    """Corrected heading (radians)."""

    odom_x: float
    """Raw odometry x."""

    odom_y: float
    """Raw odometry y."""

    odom_theta: float
    """Raw odometry heading."""

    ranges: list[float]
    """Range values in meters, from -90 to +90 degrees."""

    angle_min: float
    """Start angle of the scan (radians)."""

    angle_increment: float
    """Angular resolution (radians per beam)."""

    max_range: float
    """Maximum valid range (meters)."""


def load_intel_lab(data_path: str | Path) -> list[LaserScan]:
    """Load Intel Lab dataset from a GFS/Carmen log file.

    Parameters
    ----------
    data_path : str or Path
        Path to the .gfs or .log file.

    Returns
    -------
    list[LaserScan]
        Parsed laser scans sorted by timestamp.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    ValueError
        If the file contains no parseable laser scans.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        msg = f"Dataset file not found: {data_path}"
        raise FileNotFoundError(msg)

    scans: list[LaserScan] = []

    # For GFS format: track the best particle pose from SM_UPDATE
    # so we can assign it to the following LASER_READING.
    last_sm_pose: tuple[float, float, float] | None = None

    with data_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            if line.startswith("SM_UPDATE"):
                last_sm_pose = _parse_sm_update_pose(line)
            elif line.startswith("LASER_READING"):
                scan = _parse_laser_reading(line, corrected_pose=last_sm_pose)
                if scan is not None:
                    scans.append(scan)
            elif line.startswith("FLASER"):
                scan = _parse_flaser(line)
                if scan is not None:
                    scans.append(scan)
            elif line.startswith("ROBOTLASER1"):
                scan = _parse_robotlaser1(line)
                if scan is not None:
                    scans.append(scan)

    if not scans:
        msg = f"No laser scans found in {data_path}"
        raise ValueError(msg)

    scans.sort(key=lambda s: s.timestamp)
    return scans


def _parse_sm_update_pose(line: str) -> tuple[float, float, float] | None:
    """Extract the best particle pose from an SM_UPDATE line.

    SM_UPDATE format:
        SM_UPDATE num_particles x1 y1 theta1 weight1 x2 y2 theta2 weight2 ...

    We take the first particle (which is typically the best after resampling).
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        x = float(parts[2])
        y = float(parts[3])
        theta = float(parts[4])
        return (x, y, theta)
    except (ValueError, IndexError):
        return None


def _parse_laser_reading(
    line: str,
    corrected_pose: tuple[float, float, float] | None = None,
) -> LaserScan | None:
    """Parse a LASER_READING line from GFS format.

    Format: LASER_READING num_readings r1..rN x y theta timestamp
    """
    parts = line.split()
    if len(parts) < 5:
        return None

    try:
        num_readings = int(parts[1])
        if len(parts) < 2 + num_readings + 4:
            return None

        ranges = [float(parts[2 + i]) for i in range(num_readings)]

        base = 2 + num_readings
        odom_x = float(parts[base])
        odom_y = float(parts[base + 1])
        odom_theta = float(parts[base + 2])
        timestamp = float(parts[base + 3])

        # Use scan-matched pose if available, otherwise fall back to odometry
        if corrected_pose is not None:
            pose_x, pose_y, pose_theta = corrected_pose
        else:
            pose_x, pose_y, pose_theta = odom_x, odom_y, odom_theta

        # Standard 180-beam LiDAR: -90 to +90 degrees
        angle_min = -math.pi / 2
        angle_increment = math.pi / max(num_readings - 1, 1)

        return LaserScan(
            timestamp=timestamp,
            pose_x=pose_x,
            pose_y=pose_y,
            pose_theta=pose_theta,
            odom_x=odom_x,
            odom_y=odom_y,
            odom_theta=odom_theta,
            ranges=ranges,
            angle_min=angle_min,
            angle_increment=angle_increment,
            max_range=80.0,
        )
    except (ValueError, IndexError):
        return None


def _parse_flaser(line: str) -> LaserScan | None:
    """Parse a FLASER line.

    Format: FLASER num_readings r1..rN x y theta odom_x odom_y odom_theta ts host
    """
    parts = line.split()
    if len(parts) < 10:
        return None

    try:
        num_readings = int(parts[1])
        if len(parts) < 2 + num_readings + 7:
            return None

        ranges = [float(parts[2 + i]) for i in range(num_readings)]

        base = 2 + num_readings
        x = float(parts[base])
        y = float(parts[base + 1])
        theta = float(parts[base + 2])
        odom_x = float(parts[base + 3])
        odom_y = float(parts[base + 4])
        odom_theta = float(parts[base + 5])

        # Timestamp is at base+6 (ipc_timestamp), hostname at base+7,
        # logger_timestamp at base+8.  Fallback: try last numeric field.
        timestamp = 0.0
        for candidate in [base + 6, -1, -2]:
            try:
                timestamp = float(parts[candidate])
                break
            except (ValueError, IndexError):
                continue

        angle_min = -math.pi / 2
        angle_increment = math.pi / max(num_readings - 1, 1)

        return LaserScan(
            timestamp=timestamp,
            pose_x=x,
            pose_y=y,
            pose_theta=theta,
            odom_x=odom_x,
            odom_y=odom_y,
            odom_theta=odom_theta,
            ranges=ranges,
            angle_min=angle_min,
            angle_increment=angle_increment,
            max_range=80.0,
        )
    except (ValueError, IndexError):
        return None


def _parse_robotlaser1(line: str) -> LaserScan | None:
    """Parse a ROBOTLASER1 line.

    Format: ROBOTLASER1 type angle_min angle_range angle_incr max_range acc
            num_readings r1..rN num_remissions [remissions]
            laser_x laser_y laser_theta odom_x odom_y odom_theta ...
    """
    parts = line.split()
    if len(parts) < 15:
        return None

    try:
        angle_min = float(parts[2])
        angle_increment = float(parts[4])
        max_range = float(parts[5])

        num_readings = int(parts[7])
        if len(parts) < 8 + num_readings + 10:
            return None

        ranges = [float(parts[8 + i]) for i in range(num_readings)]

        base = 8 + num_readings
        num_remissions = int(parts[base])
        base += 1 + num_remissions

        if len(parts) < base + 6:
            return None

        odom_x = float(parts[base + 3])
        odom_y = float(parts[base + 4])
        odom_theta = float(parts[base + 5])

        timestamp = float(parts[-1]) if parts[-1].replace(".", "").replace("-", "").isdigit() else 0.0

        return LaserScan(
            timestamp=timestamp,
            pose_x=odom_x,
            pose_y=odom_y,
            pose_theta=odom_theta,
            odom_x=odom_x,
            odom_y=odom_y,
            odom_theta=odom_theta,
            ranges=ranges,
            angle_min=angle_min,
            angle_increment=angle_increment,
            max_range=max_range,
        )
    except (ValueError, IndexError):
        return None
