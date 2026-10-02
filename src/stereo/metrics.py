"""Disparity and metric-depth evaluation for Middlebury stereo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def compute_disparity_metrics(
    prediction_px: np.ndarray,
    target_px: np.ndarray,
    *,
    evaluation_mask: np.ndarray,
) -> Mapping[str, float | int]:
    """Compute non-occluded disparity errors and prediction coverage.

    MAE/RMSE use pixels with valid predictions. Bad-pixel rates use every GT
    evaluation pixel and therefore count invalid predictions as bad; this
    prevents methods from improving error simply by returning fewer pixels.
    """

    prediction = np.asarray(prediction_px, dtype=np.float64)
    target = np.asarray(target_px, dtype=np.float64)
    mask = np.asarray(evaluation_mask, dtype=bool)
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError(
            f"Disparity/mask shapes differ: {prediction.shape}, {target.shape}, {mask.shape}"
        )
    gt_valid = mask & np.isfinite(target) & (target > 0)
    gt_count = int(gt_valid.sum())
    if gt_count == 0:
        raise ValueError("No valid ground-truth disparity pixels")
    common = gt_valid & np.isfinite(prediction) & (prediction > 0)
    prediction_count = int(common.sum())
    if prediction_count == 0:
        raise ValueError("No valid predicted disparity pixels")
    absolute_error = np.abs(prediction[common] - target[common])
    full_error = np.full(target.shape, np.inf, dtype=np.float64)
    full_error[common] = np.abs(prediction[common] - target[common])
    return {
        "disparity_mae_px": float(absolute_error.mean()),
        "disparity_rmse_px": float(np.sqrt(np.mean(np.square(absolute_error)))),
        "bad1": float(np.mean(full_error[gt_valid] > 1.0)),
        "bad2": float(np.mean(full_error[gt_valid] > 2.0)),
        "bad4": float(np.mean(full_error[gt_valid] > 4.0)),
        "valid_coverage": float(prediction_count / gt_count),
        "evaluation_pixels": gt_count,
        "valid_prediction_pixels": prediction_count,
    }


def compute_stereo_depth_metrics(
    prediction_m: np.ndarray,
    target_m: np.ndarray,
    *,
    evaluation_mask: np.ndarray,
    depth_bin_edges_m: Sequence[float] = (0.0, 2.0, 4.0, float("inf")),
) -> Mapping[str, float | int | None]:
    """Compute metric-depth error and fixed near/mid/far bin diagnostics."""

    prediction = np.asarray(prediction_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    mask = np.asarray(evaluation_mask, dtype=bool)
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError(
            f"Depth/mask shapes differ: {prediction.shape}, {target.shape}, {mask.shape}"
        )
    edges = np.asarray(depth_bin_edges_m, dtype=np.float64)
    if edges.ndim != 1 or len(edges) != 4 or not np.all(np.diff(edges) > 0):
        raise ValueError("depth_bin_edges_m must contain four increasing boundaries")
    gt_valid = mask & np.isfinite(target) & (target > 0)
    common = gt_valid & np.isfinite(prediction) & (prediction > 0)
    if not common.any():
        raise ValueError("No common valid depth pixels")
    difference = prediction[common] - target[common]
    result: dict[str, float | int | None] = {
        "depth_abs_rel": float(np.mean(np.abs(difference) / target[common])),
        "depth_rmse_m": float(np.sqrt(np.mean(np.square(difference)))),
        "depth_valid_coverage": float(common.sum() / gt_valid.sum()),
    }
    labels = ("near", "mid", "far")
    for index, label in enumerate(labels):
        selected = common & (target >= edges[index]) & (target < edges[index + 1])
        count = int(selected.sum())
        result[f"depth_{label}_pixels"] = count
        if count == 0:
            result[f"depth_{label}_abs_rel"] = None
            result[f"depth_{label}_rmse_m"] = None
            continue
        bin_difference = prediction[selected] - target[selected]
        result[f"depth_{label}_abs_rel"] = float(
            np.mean(np.abs(bin_difference) / target[selected])
        )
        result[f"depth_{label}_rmse_m"] = float(
            np.sqrt(np.mean(np.square(bin_difference)))
        )
    return result
