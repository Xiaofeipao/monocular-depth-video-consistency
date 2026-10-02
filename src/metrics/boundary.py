"""Depth-boundary extraction and tolerance-aware F1 metrics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.ndimage import binary_dilation


def depth_edge_map(
    depth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    log_threshold: float = 0.05,
) -> np.ndarray:
    """Extract four-connected depth discontinuities in log-depth space.

    Both pixels adjacent to a qualifying discontinuity are marked. Log depth
    makes the threshold a relative-depth threshold; ``0.05`` is roughly a 5%
    jump. This extraction definition remains configurable until the evaluation
    protocol is frozen.
    """

    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D depth map, got shape {depth.shape}")
    if log_threshold <= 0:
        raise ValueError("log_threshold must be positive")

    valid = np.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != depth.shape:
            raise ValueError(
                f"Valid mask shape differs: {valid_mask.shape} != {depth.shape}"
            )
        valid &= valid_mask

    log_depth = np.zeros_like(depth)
    log_depth[valid] = np.log(depth[valid])
    edges = np.zeros(depth.shape, dtype=bool)

    horizontal = (
        valid[:, :-1]
        & valid[:, 1:]
        & (np.abs(log_depth[:, 1:] - log_depth[:, :-1]) >= log_threshold)
    )
    edges[:, :-1] |= horizontal
    edges[:, 1:] |= horizontal

    vertical = (
        valid[:-1, :]
        & valid[1:, :]
        & (np.abs(log_depth[1:, :] - log_depth[:-1, :]) >= log_threshold)
    )
    edges[:-1, :] |= vertical
    edges[1:, :] |= vertical
    return edges


def binary_boundary_f1(
    prediction_edges: np.ndarray,
    target_edges: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    tolerance_px: int = 1,
) -> Mapping[str, float | int]:
    """Compute symmetric, tolerance-aware precision/recall/F1 for edge maps."""

    prediction_edges = np.asarray(prediction_edges, dtype=bool)
    target_edges = np.asarray(target_edges, dtype=bool)
    if prediction_edges.shape != target_edges.shape or prediction_edges.ndim != 2:
        raise ValueError(
            "Prediction and target edge maps must be matching 2-D arrays; "
            f"got {prediction_edges.shape} and {target_edges.shape}"
        )
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be non-negative")

    if valid_mask is None:
        valid = np.ones(prediction_edges.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != prediction_edges.shape:
            raise ValueError(f"Valid mask shape differs: {valid.shape}")

    prediction_edges &= valid
    target_edges &= valid
    prediction_count = int(prediction_edges.sum())
    target_count = int(target_edges.sum())

    if prediction_count == 0 and target_count == 0:
        precision = recall = f1 = 1.0
    elif prediction_count == 0 or target_count == 0:
        precision = recall = f1 = 0.0
    else:
        size = 2 * tolerance_px + 1
        structure = np.ones((size, size), dtype=bool)
        target_neighborhood = binary_dilation(target_edges, structure=structure)
        prediction_neighborhood = binary_dilation(prediction_edges, structure=structure)
        precision = float((prediction_edges & target_neighborhood).sum() / prediction_count)
        recall = float((target_edges & prediction_neighborhood).sum() / target_count)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "prediction_edge_pixels": prediction_count,
        "target_edge_pixels": target_count,
    }


def compute_depth_boundary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    log_threshold: float = 0.05,
    tolerance_px: int = 1,
) -> Mapping[str, float | int]:
    """Extract prediction/target depth edges and compare them."""

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError(f"Depth shapes differ: {prediction.shape} != {target.shape}")
    common_valid = (
        np.isfinite(prediction)
        & np.isfinite(target)
        & (prediction > 0)
        & (target > 0)
    )
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != target.shape:
            raise ValueError(f"Valid mask shape differs: {supplied.shape} != {target.shape}")
        common_valid &= supplied
    prediction_edges = depth_edge_map(
        prediction, valid_mask=common_valid, log_threshold=log_threshold
    )
    target_edges = depth_edge_map(
        target, valid_mask=common_valid, log_threshold=log_threshold
    )
    return binary_boundary_f1(
        prediction_edges,
        target_edges,
        valid_mask=common_valid,
        tolerance_px=tolerance_px,
    )
