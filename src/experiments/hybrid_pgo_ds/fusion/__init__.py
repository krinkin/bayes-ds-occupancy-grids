"""Occupancy grid fusion methods for H-003 experiment.

Four experimental arms:
    BayesianFusion      -- 1 float per cell (log-odds)
    BayesianCountFusion -- 2 floats per cell (log-odds + observation count)
    DSTBMFusion         -- 3 floats per cell (DS/TBM mass triplet, Dempster's rule)
    YagerFusion         -- 3 floats per cell (DS/TBM mass triplet, Yager's rule)

All implement the FusionMethod abstract interface.
"""

from src.experiments.hybrid_pgo_ds.fusion.base import FusionMethod
from src.experiments.hybrid_pgo_ds.fusion.bayesian import BayesianFusion
from src.experiments.hybrid_pgo_ds.fusion.bayesian_count import BayesianCountFusion
from src.experiments.hybrid_pgo_ds.fusion.dstbm import DSTBMFusion
from src.experiments.hybrid_pgo_ds.fusion.yager import YagerFusion

__all__ = [
    "FusionMethod",
    "BayesianFusion",
    "BayesianCountFusion",
    "DSTBMFusion",
    "YagerFusion",
]
