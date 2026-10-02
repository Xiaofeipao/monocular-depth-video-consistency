"""Unit-aware disparity/depth conversion utilities."""

from __future__ import annotations

import numpy as np


def disparity_to_depth(
    disparity_px: np.ndarray,
    *,
    focal_length_px: float,
    baseline: float,
    disparity_offset_px: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert rectified disparity to depth in the baseline's length unit.

    Middlebury calibration uses ``depth = f * baseline / (disparity + doffs)``.
    Invalid denominators produce NaN and are marked false in the returned mask.
    """

    if focal_length_px <= 0 or baseline <= 0:
        raise ValueError("Focal length and baseline must be strictly positive")
    disparity = np.asarray(disparity_px, dtype=np.float64)
    denominator = disparity + float(disparity_offset_px)
    valid = np.isfinite(denominator) & (denominator > 0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float64)
    depth[valid] = focal_length_px * baseline / denominator[valid]
    return depth, valid


def depth_sensitivity(
    depth: np.ndarray,
    *,
    focal_length_px: float,
    baseline: float,
    disparity_error_px: float,
) -> np.ndarray:
    """First-order absolute depth error induced by a disparity error."""

    if focal_length_px <= 0 or baseline <= 0:
        raise ValueError("Focal length and baseline must be strictly positive")
    depth = np.asarray(depth, dtype=np.float64)
    return np.square(depth) * abs(disparity_error_px) / (focal_length_px * baseline)

