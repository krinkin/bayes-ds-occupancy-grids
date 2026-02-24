# TDP: Config loader
# Approach: Load YAML file and merge with defaults. Return plain dict rather than
#   a custom class so downstream code can use standard dict access without coupling
#   to a config type. Provide explicit defaults for every parameter so configs can
#   be minimal (smoke configs only override what matters).
# Alternatives considered: dataclasses/attrs -- heavier, adds coupling; pydantic --
#   not in requirements.
# Risks: Silent merging of defaults may hide missing keys; mitigated by
#   documenting all default values here.
"""Config loader for H-003 experiment.

Loads a YAML config file and merges it with built-in defaults so that partial
configs (e.g., smoke test configs) work correctly.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS: dict[str, Any] = {
    "experiment": {
        "name": "default",
        "output_dir": "results/hybrid-pgo-ds/default",
        "seed": 0,
    },
    "environment": {
        "width": 20.0,
        "height": 20.0,
        "resolution": 0.05,
        "rooms": 1,
        "corridors": 0,
        "obstacles": 0,
        "dynamic_objects": 0,
        "dynamic_speed": 0.5,
    },
    "robots": {
        "count": 1,
        "trajectory_steps": 100,
        "drift_stddev": 0.01,
        "angular_drift_stddev": 0.005,
    },
    "lidar": {
        "max_range": 10.0,
        "num_rays": 180,
        "noise_stddev": 0.02,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
    },
    "pgo": {
        "enabled": False,
        "rendezvous_distance": 2.0,
    },
    "fusion": {
        "methods": ["bayesian"],
    },
    "metrics": ["cell_accuracy"],
    "fault_tolerance": {
        "remove_robot_after_step": None,
        "robot_to_remove": 0,
    },
    "logging": {
        "grid_snapshot_interval": 10,
        "format": "json",
        "measure_resources": False,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config from *path* and merge with defaults.

    Parameters
    ----------
    path:
        Path to a YAML config file.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    path = Path(path)
    with path.open("r") as fh:
        user_config = yaml.safe_load(fh) or {}
    return _deep_merge(_DEFAULTS, user_config)
