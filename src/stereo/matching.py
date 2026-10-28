"""Reproducible OpenCV BM/SGBM matching and left-right consistency."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoMatcherConfig:
    method: str
    min_disparity: int = 0
    block_size: int = 5
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    pre_filter_cap: int = 63
    disp12_max_diff: int = 1
    texture_threshold: int = 10
    mode: str = "sgbm_3way"


@dataclass(frozen=True)
class StereoMatchResult:
    raw_left_disparity_px: np.ndarray
    left_disparity_px: np.ndarray
    right_disparity_px: np.ndarray
    raw_left_valid: np.ndarray
    raw_right_valid: np.ndarray
    lr_consistent: np.ndarray
    valid_mask: np.ndarray
    num_disparities: int
    runtime_seconds: float


def round_num_disparities(requested: int) -> int:
    """Round a positive search range up to OpenCV's required multiple of 16."""

    if requested <= 0:
        raise ValueError("requested disparities must be positive")
    return int(math.ceil(requested / 16) * 16)


def _validate_block_size(block_size: int) -> None:
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 3")


def _create_matcher(
    config: StereoMatcherConfig,
    *,
    min_disparity: int,
    num_disparities: int,
) -> cv2.StereoMatcher:
    _validate_block_size(config.block_size)
    if config.method == "bm":
        matcher = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=config.block_size,
        )
        matcher.setMinDisparity(min_disparity)
        matcher.setPreFilterCap(config.pre_filter_cap)
        matcher.setTextureThreshold(config.texture_threshold)
        matcher.setUniquenessRatio(config.uniqueness_ratio)
        matcher.setSpeckleWindowSize(config.speckle_window_size)
        matcher.setSpeckleRange(config.speckle_range)
        matcher.setDisp12MaxDiff(config.disp12_max_diff)
        return matcher
    if config.method != "sgbm":
        raise ValueError(f"Unsupported stereo method: {config.method!r}")
    modes = {
        "sgbm": cv2.STEREO_SGBM_MODE_SGBM,
        "hh": cv2.STEREO_SGBM_MODE_HH,
        "sgbm_3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        "hh4": cv2.STEREO_SGBM_MODE_HH4,
    }
    if config.mode not in modes:
        raise ValueError(f"Unsupported SGBM mode: {config.mode!r}")
    block_area = config.block_size**2
    return cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=config.block_size,
        P1=8 * block_area,
        P2=32 * block_area,
        disp12MaxDiff=config.disp12_max_diff,
        preFilterCap=config.pre_filter_cap,
        uniquenessRatio=config.uniqueness_ratio,
        speckleWindowSize=config.speckle_window_size,
        speckleRange=config.speckle_range,
        mode=modes[config.mode],
    )


def decode_opencv_disparity(raw: np.ndarray, *, min_disparity: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode OpenCV's signed 1/16-pixel fixed-point disparity."""

    raw = np.asarray(raw)
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.signedinteger):
        raise ValueError(f"Expected a signed integer HxW disparity, got {raw.shape}/{raw.dtype}")
    invalid_code = (min_disparity - 1) * 16
    valid = raw > invalid_code
    disparity = raw.astype(np.float32) / 16.0
    disparity[~valid] = np.nan
    return disparity, valid


def left_right_consistency_mask(
    left_disparity_px: np.ndarray,
    right_disparity_px: np.ndarray,
    *,
    threshold_px: float,
) -> np.ndarray:
    """Check ``|d_L(x) + d_R(x-d_L)| <= threshold`` in the left view."""

    left = np.asarray(left_disparity_px, dtype=np.float32)
    right = np.asarray(right_disparity_px, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(f"Expected matching HxW disparities, got {left.shape}/{right.shape}")
    if threshold_px < 0:
        raise ValueError("threshold_px must be non-negative")
    height, width = left.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x - left
    geometric_valid = (
        np.isfinite(left)
        & (left > 0)
        & np.isfinite(map_x)
        & (map_x >= 0)
        & (map_x <= width - 1)
    )
    right_finite = np.isfinite(right)
    warped_right = cv2.remap(
        np.where(right_finite, right, 0.0),
        np.where(np.isfinite(map_x), map_x, -1.0).astype(np.float32),
        grid_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_valid_fraction = cv2.remap(
        right_finite.astype(np.float32),
        np.where(np.isfinite(map_x), map_x, -1.0).astype(np.float32),
        grid_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_valid = warped_valid_fraction >= 1.0 - 1e-6
    residual = np.abs(left + warped_right)
    return geometric_valid & warped_valid & np.isfinite(residual) & (residual <= threshold_px)


def match_stereo(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    *,
    calibration_num_disparities: int,
    config: StereoMatcherConfig,
    lr_threshold_px: float,
) -> StereoMatchResult:
    """Compute bidirectional disparity and a left-view consistency mask."""

    left_rgb = np.asarray(left_rgb)
    right_rgb = np.asarray(right_rgb)
    if left_rgb.shape != right_rgb.shape or left_rgb.ndim != 3 or left_rgb.shape[2] != 3:
        raise ValueError(f"Expected matching HxWx3 images, got {left_rgb.shape}/{right_rgb.shape}")
    left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
    num_disparities = round_num_disparities(calibration_num_disparities)
    right_min_disparity = -config.min_disparity - num_disparities + 1
    left_matcher = _create_matcher(
        config,
        min_disparity=config.min_disparity,
        num_disparities=num_disparities,
    )
    right_matcher = _create_matcher(
        config,
        min_disparity=right_min_disparity,
        num_disparities=num_disparities,
    )
    start = time.perf_counter()
    raw_left = left_matcher.compute(left_gray, right_gray)
    raw_right = right_matcher.compute(right_gray, left_gray)
    runtime = time.perf_counter() - start
    left, left_valid = decode_opencv_disparity(
        raw_left, min_disparity=config.min_disparity
    )
    right, right_valid = decode_opencv_disparity(
        raw_right, min_disparity=right_min_disparity
    )
    lr_consistent = left_right_consistency_mask(
        left,
        right,
        threshold_px=lr_threshold_px,
    )
    valid = left_valid & (left > 0) & lr_consistent
    raw_left_disparity = left.copy()
    left = raw_left_disparity.copy()
    left[~valid] = np.nan
    return StereoMatchResult(
        raw_left_disparity_px=raw_left_disparity,
        left_disparity_px=left,
        right_disparity_px=right,
        raw_left_valid=left_valid,
        raw_right_valid=right_valid,
        lr_consistent=lr_consistent,
        valid_mask=valid,
        num_disparities=num_disparities,
        runtime_seconds=runtime,
    )
