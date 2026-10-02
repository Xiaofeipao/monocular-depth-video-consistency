"""Optical-flow warping and temporal depth-consistency metrics."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np


def _validate_flow(flow: np.ndarray) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected flow shape (H, W, 2), got {flow.shape}")
    return flow


def warp_with_backward_flow(
    source: np.ndarray,
    backward_flow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``source`` into the target frame using target-to-source flow.

    ``backward_flow[y, x] == (dx, dy)`` samples source coordinate
    ``(x + dx, y + dy)`` for target pixel ``(x, y)``. The returned mask excludes
    out-of-bounds coordinates, non-finite flow, and non-finite samples.
    """

    source = np.asarray(source, dtype=np.float32)
    flow = _validate_flow(backward_flow)
    if source.ndim != 2 or source.shape != flow.shape[:2]:
        raise ValueError(
            f"Source/flow shapes must be (H, W) and (H, W, 2); got "
            f"{source.shape} and {flow.shape}"
        )

    height, width = source.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    finite_flow = np.isfinite(map_x) & np.isfinite(map_y)
    valid = (
        finite_flow
        & (map_x >= 0)
        & (map_x <= width - 1)
        & (map_y >= 0)
        & (map_y <= height - 1)
    )

    safe_x = np.where(finite_flow, map_x, -1).astype(np.float32)
    safe_y = np.where(finite_flow, map_y, -1).astype(np.float32)
    warped = cv2.remap(
        source,
        safe_x,
        safe_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        # NaN borders contaminate exact last-row/last-column samples in some
        # OpenCV builds even when their interpolation weight is zero. Geometry
        # is masked explicitly above, so a finite sentinel is safer here.
        borderValue=0.0,
    )
    valid &= np.isfinite(warped)
    warped[~valid] = np.nan
    return warped, valid


def forward_backward_consistency_mask(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    *,
    threshold_px: float = 1.0,
) -> np.ndarray:
    """Return target-frame pixels whose forward/backward flow cycle closes."""

    forward = _validate_flow(forward_flow)
    backward = _validate_flow(backward_flow)
    if forward.shape != backward.shape:
        raise ValueError(f"Flow shapes differ: {forward.shape} != {backward.shape}")
    if threshold_px < 0:
        raise ValueError("threshold_px must be non-negative")

    warped_forward_x, valid_x = warp_with_backward_flow(forward[..., 0], backward)
    warped_forward_y, valid_y = warp_with_backward_flow(forward[..., 1], backward)
    cycle_x = backward[..., 0] + warped_forward_x
    cycle_y = backward[..., 1] + warped_forward_y
    cycle_error = np.hypot(cycle_x, cycle_y)
    return valid_x & valid_y & np.isfinite(cycle_error) & (cycle_error <= threshold_px)


def compute_temporal_warp_metrics(
    previous_depth: np.ndarray,
    current_depth: np.ndarray,
    backward_flow: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> Mapping[str, float | int]:
    """Compare current depth with flow-warped previous depth without alignment."""

    previous_depth = np.asarray(previous_depth, dtype=np.float32)
    current_depth = np.asarray(current_depth, dtype=np.float32)
    if previous_depth.shape != current_depth.shape:
        raise ValueError(
            f"Depth shapes differ: {previous_depth.shape} != {current_depth.shape}"
        )
    warped, warp_valid = warp_with_backward_flow(previous_depth, backward_flow)
    valid = (
        warp_valid
        & np.isfinite(current_depth)
        & (warped > 0)
        & (current_depth > 0)
    )
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != current_depth.shape:
            raise ValueError(
                f"Valid mask shape differs: {supplied.shape} != {current_depth.shape}"
            )
        valid &= supplied
    valid_count = int(valid.sum())
    if valid_count == 0:
        raise ValueError("No valid pixels remain for temporal evaluation")

    prior = warped[valid].astype(np.float64)
    current = current_depth[valid].astype(np.float64)
    difference = current - prior
    log_difference = np.log(current) - np.log(prior)
    return {
        "temporal_abs_rel": float(np.mean(np.abs(difference) / current)),
        "temporal_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "temporal_rmse_log": float(np.sqrt(np.mean(np.square(log_difference)))),
        "valid_pixels": valid_count,
    }


def compute_median_scale_drift(
    depth_sequence: np.ndarray,
    *,
    valid_masks: np.ndarray | None = None,
    reference_index: int = 0,
) -> Mapping[str, np.ndarray | float | int]:
    """Measure frame-wise median-depth scale drift relative to one frame.

    This is a sequence-level scale proxy, not a metric-accuracy measure. It is
    most interpretable for static-camera/static-scene clips and must be reported
    alongside the clip category.
    """

    depths = np.asarray(depth_sequence, dtype=np.float64)
    if depths.ndim != 3:
        raise ValueError(f"Expected depth sequence shape (T, H, W), got {depths.shape}")
    if not 0 <= reference_index < depths.shape[0]:
        raise ValueError(f"reference_index {reference_index} is out of range")
    if valid_masks is None:
        supplied_masks = np.ones(depths.shape, dtype=bool)
    else:
        supplied_masks = np.asarray(valid_masks, dtype=bool)
        if supplied_masks.shape != depths.shape:
            raise ValueError(
                f"Valid masks shape differs: {supplied_masks.shape} != {depths.shape}"
            )

    medians = np.empty(depths.shape[0], dtype=np.float64)
    valid_counts = np.empty(depths.shape[0], dtype=np.int64)
    for frame_index, (depth, supplied) in enumerate(zip(depths, supplied_masks)):
        valid = supplied & np.isfinite(depth) & (depth > 0)
        valid_counts[frame_index] = valid.sum()
        if valid_counts[frame_index] == 0:
            raise ValueError(f"Frame {frame_index} has no valid positive depth")
        medians[frame_index] = np.median(depth[valid])

    reference_median = medians[reference_index]
    scale_ratios = medians / reference_median
    absolute_log_drift = np.abs(np.log(scale_ratios))
    return {
        "reference_index": reference_index,
        "frame_medians": medians,
        "scale_ratios": scale_ratios,
        "mean_abs_log_drift": float(absolute_log_drift.mean()),
        "max_abs_log_drift": float(absolute_log_drift.max()),
        "valid_pixels_per_frame": valid_counts,
    }
