"""Depth-estimation metrics with explicit validity handling."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _as_matching_float_arrays(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    if prediction.ndim < 1:
        raise ValueError("Depth inputs must have at least one dimension")
    return prediction, target


def build_valid_mask(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    external_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return the common finite, positive, in-range evaluation mask.

    The range is applied to ground truth. Predictions must be finite and strictly
    positive because the required protocol includes logarithmic and ratio metrics.
    """

    raw_target = np.asarray(target)
    comparison_dtype = (
        raw_target.dtype
        if np.issubdtype(raw_target.dtype, np.floating)
        else np.dtype(np.float64)
    )
    min_bound = float(np.asarray(min_depth, dtype=comparison_dtype))
    max_bound = float(np.asarray(max_depth, dtype=comparison_dtype))
    prediction, target = _as_matching_float_arrays(prediction, target)
    if not 0 < min_depth < max_depth:
        raise ValueError("Expected 0 < min_depth < max_depth")

    valid = (
        np.isfinite(prediction)
        & np.isfinite(target)
        & (prediction > 0)
        & (target > min_bound)
        & (target < max_bound)
    )
    if external_mask is not None:
        external_mask = np.asarray(external_mask, dtype=bool)
        if external_mask.shape != target.shape:
            raise ValueError(
                f"External mask shape differs: {external_mask.shape} != {target.shape}"
            )
        valid &= external_mask
    return valid


def compute_depth_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    external_mask: np.ndarray | None = None,
    clip_prediction: bool = True,
) -> Mapping[str, float | int]:
    """Compute the seven mandatory metric-depth measures on one sample.

    This function performs no scale or shift alignment. Any allowed alignment
    must happen explicitly before this call and must be recorded in run metadata.
    """

    prediction, target = _as_matching_float_arrays(prediction, target)
    valid = build_valid_mask(
        prediction,
        target,
        min_depth=min_depth,
        max_depth=max_depth,
        external_mask=external_mask,
    )
    valid_count = int(valid.sum())
    if valid_count == 0:
        raise ValueError("No valid pixels remain for depth evaluation")

    pred = prediction[valid]
    gt = target[valid]
    if clip_prediction:
        pred = np.clip(pred, min_depth, max_depth)

    difference = pred - gt
    log_difference = np.log(pred) - np.log(gt)
    ratio = np.maximum(pred / gt, gt / pred)

    return {
        "abs_rel": float(np.mean(np.abs(difference) / gt)),
        "sq_rel": float(np.mean(np.square(difference) / gt)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "rmse_log": float(np.sqrt(np.mean(np.square(log_difference)))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
        "valid_pixels": valid_count,
    }
