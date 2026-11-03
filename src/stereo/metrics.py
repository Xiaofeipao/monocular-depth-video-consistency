"""Disparity and metric-depth evaluation for Middlebury stereo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def compute_disparity_metrics(
    prediction_px: np.ndarray,
    target_px: np.ndarray,
    *,
    evaluation_mask: np.ndarray | None = None,
    prediction_valid_mask: np.ndarray | None = None,
) -> Mapping[str, float | int | None]:
    """Course MAE and valid fraction on every finite positive GT disparity.

    Invalid predictions remain in the fraction's denominator, but not in MAE.
    No valid prediction means MAE=None and fraction=0, never a perfect score.
    An explicit evaluation mask is supported only for separately labelled subsets.
    """

    prediction = np.asarray(prediction_px, dtype=np.float64)
    target = np.asarray(target_px, dtype=np.float64)
    mask = np.ones(target.shape, bool) if evaluation_mask is None else np.asarray(evaluation_mask, dtype=bool)
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError(
            f"Disparity/mask shapes differ: {prediction.shape}, {target.shape}, {mask.shape}"
        )
    gt_valid = mask & np.isfinite(target) & (target > 0)
    gt_count = int(gt_valid.sum())
    if gt_count == 0:
        raise ValueError("No valid ground-truth disparity pixels")
    common = gt_valid & np.isfinite(prediction) & (prediction > 0)
    if prediction_valid_mask is not None:
        if np.shape(prediction_valid_mask) != target.shape:
            raise ValueError('Prediction validity shape differs')
        common &= np.asarray(prediction_valid_mask, dtype=bool)
    prediction_count = int(common.sum())
    absolute_error = np.abs(prediction[common] - target[common])
    result = {
        "disparity_mae_px": float(absolute_error.mean()) if prediction_count else None,
        "valid_prediction_fraction": float(prediction_count / gt_count),
        "evaluation_pixels": gt_count,
        "valid_prediction_pixels": prediction_count,
        "invalid_prediction_pixels": gt_count - prediction_count,
    }
    # Invalid predictions are failures, including when no prediction is valid.
    # Thresholds are strict (>), and the denominator is the full GT domain.
    for threshold in (1, 2, 4):
        bad_count = gt_count - prediction_count + int((absolute_error > threshold).sum())
        result[f'bad{threshold}'] = float(bad_count / gt_count)
    return result


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
    difference = prediction[common] - target[common]
    result: dict[str, float | int | None] = {
        "depth_abs_rel": float(np.mean(np.abs(difference) / target[common])) if common.any() else None,
        "depth_rmse_m": float(np.sqrt(np.mean(np.square(difference)))) if common.any() else None,
        "depth_valid_prediction_fraction": float(common.sum() / gt_valid.sum()) if gt_valid.any() else None,
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
