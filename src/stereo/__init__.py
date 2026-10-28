"""Classical stereo geometry, I/O, matching, and evaluation utilities."""

from .geometry import depth_sensitivity, disparity_to_depth
from .io import (
    MiddleburyCalibration,
    MiddleburyScene,
    load_middlebury_calibration,
    load_middlebury_scene,
    read_pfm,
)
from .matching import (
    StereoMatcherConfig,
    StereoMatchResult,
    left_right_consistency_mask,
    match_stereo,
)
from .metrics import compute_disparity_metrics, compute_stereo_depth_metrics

__all__ = [
    "MiddleburyCalibration",
    "MiddleburyScene",
    "StereoMatchResult",
    "StereoMatcherConfig",
    "compute_disparity_metrics",
    "compute_stereo_depth_metrics",
    "depth_sensitivity",
    "disparity_to_depth",
    "left_right_consistency_mask",
    "load_middlebury_calibration",
    "load_middlebury_scene",
    "match_stereo",
    "read_pfm",
]
