"""Numerical and first-order disparity-to-depth sensitivity analysis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from .geometry import depth_sensitivity, disparity_to_depth
from .io import MiddleburyScene


def analyze_depth_sensitivity(
    scenes: Iterable[MiddleburyScene],
    *,
    disparity_errors_px: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
    depth_bin_width_m: float = 0.25,
) -> list[dict[str, float | int]]:
    """Aggregate exact ±disparity perturbations and first-order predictions."""

    if depth_bin_width_m <= 0:
        raise ValueError("depth_bin_width_m must be positive")
    records: list[tuple[np.ndarray, np.ndarray, float, float, float]] = []
    maximum_depth = 0.0
    for scene in scenes:
        calibration = scene.calibration
        gt_depth, depth_valid = disparity_to_depth(
            scene.gt_disparity_px,
            focal_length_px=calibration.focal_length_px,
            baseline=calibration.baseline_m,
            disparity_offset_px=calibration.doffs_px,
        )
        valid = scene.non_occluded_mask & depth_valid & np.isfinite(scene.gt_disparity_px)
        if not valid.any():
            continue
        depth_values = gt_depth[valid]
        disparity_values = scene.gt_disparity_px[valid]
        maximum_depth = max(maximum_depth, float(depth_values.max()))
        records.append(
            (
                depth_values,
                disparity_values,
                calibration.focal_length_px,
                calibration.baseline_m,
                calibration.doffs_px,
            )
        )
    if not records:
        raise ValueError("No valid scene pixels for sensitivity analysis")

    bin_edges = np.arange(0.0, maximum_depth + depth_bin_width_m, depth_bin_width_m)
    if bin_edges[-1] <= maximum_depth:
        bin_edges = np.append(bin_edges, bin_edges[-1] + depth_bin_width_m)
    rows: list[dict[str, float | int]] = []
    for error_magnitude in disparity_errors_px:
        if error_magnitude <= 0:
            raise ValueError("disparity_errors_px must be positive")
        per_bin_exact: list[list[np.ndarray]] = [[] for _ in range(len(bin_edges) - 1)]
        per_bin_first: list[list[np.ndarray]] = [[] for _ in range(len(bin_edges) - 1)]
        for depths, disparities, focal_length_px, baseline_m, doffs_px in records:
            first_order = depth_sensitivity(
                depths,
                focal_length_px=focal_length_px,
                baseline=baseline_m,
                disparity_error_px=error_magnitude,
            )
            exact_directions: list[np.ndarray] = []
            for signed_error in (-error_magnitude, error_magnitude):
                perturbed_depth, valid = disparity_to_depth(
                    disparities + signed_error,
                    focal_length_px=focal_length_px,
                    baseline=baseline_m,
                    disparity_offset_px=doffs_px,
                )
                absolute_error = np.full(depths.shape, np.nan, dtype=np.float64)
                absolute_error[valid] = np.abs(perturbed_depth[valid] - depths[valid])
                exact_directions.append(absolute_error)
            exact_mean = np.nanmean(np.stack(exact_directions), axis=0)
            bin_indices = np.digitize(depths, bin_edges, right=False) - 1
            for bin_index in range(len(bin_edges) - 1):
                selected = bin_indices == bin_index
                if selected.any():
                    per_bin_exact[bin_index].append(exact_mean[selected])
                    per_bin_first[bin_index].append(first_order[selected])
        for bin_index, (exact_parts, first_parts) in enumerate(
            zip(per_bin_exact, per_bin_first)
        ):
            if not exact_parts:
                continue
            exact = np.concatenate(exact_parts)
            first = np.concatenate(first_parts)
            rows.append(
                {
                    "disparity_error_px": float(error_magnitude),
                    "depth_bin_start_m": float(bin_edges[bin_index]),
                    "depth_bin_end_m": float(bin_edges[bin_index + 1]),
                    "depth_bin_center_m": float(
                        (bin_edges[bin_index] + bin_edges[bin_index + 1]) / 2
                    ),
                    "mean_exact_abs_depth_error_m": float(np.nanmean(exact)),
                    "mean_first_order_abs_depth_error_m": float(np.nanmean(first)),
                    "pixel_count": int(np.isfinite(exact).sum()),
                }
            )
    return rows


def analyze_prediction_error_by_distance(
    scenes: Iterable[MiddleburyScene],
    predicted_disparities_px: Mapping[str, np.ndarray],
    *,
    depth_bin_width_m: float = 0.25,
) -> list[dict[str, float | int]]:
    """Bin one experiment's observed disparity/depth errors by GT distance.

    Unlike :func:`analyze_depth_sensitivity`, this is matcher-dependent. Invalid
    predictions contribute to Bad-2 and reduce coverage, but are not assigned a
    fabricated numerical MAE.
    """

    if depth_bin_width_m <= 0:
        raise ValueError("depth_bin_width_m must be positive")
    scene_list = list(scenes)
    records: list[
        tuple[MiddleburyScene, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    maximum_depth = 0.0
    for scene in scene_list:
        if scene.name not in predicted_disparities_px:
            raise ValueError(f"Missing prediction for scene: {scene.name}")
        prediction = np.asarray(predicted_disparities_px[scene.name], dtype=np.float32)
        if prediction.shape != scene.gt_disparity_px.shape:
            raise ValueError(
                f"Prediction shape mismatch for {scene.name}: "
                f"{prediction.shape} != {scene.gt_disparity_px.shape}"
            )
        calibration = scene.calibration
        gt_depth, gt_depth_valid = disparity_to_depth(
            scene.gt_disparity_px,
            focal_length_px=calibration.focal_length_px,
            baseline=calibration.baseline_m,
            disparity_offset_px=calibration.doffs_px,
        )
        predicted_depth, predicted_depth_valid = disparity_to_depth(
            prediction,
            focal_length_px=calibration.focal_length_px,
            baseline=calibration.baseline_m,
            disparity_offset_px=calibration.doffs_px,
        )
        evaluation = scene.non_occluded_mask & gt_depth_valid
        if not evaluation.any():
            continue
        maximum_depth = max(maximum_depth, float(gt_depth[evaluation].max()))
        records.append(
            (scene, gt_depth, scene.gt_disparity_px, predicted_depth, prediction)
        )

    if not records:
        raise ValueError("No valid scene pixels for empirical sensitivity analysis")
    bin_edges = np.arange(0.0, maximum_depth + depth_bin_width_m, depth_bin_width_m)
    if bin_edges[-1] <= maximum_depth:
        bin_edges = np.append(bin_edges, bin_edges[-1] + depth_bin_width_m)

    totals = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    valids = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    bad2_counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    depth_errors: list[list[np.ndarray]] = [[] for _ in range(len(bin_edges) - 1)]
    disparity_errors: list[list[np.ndarray]] = [[] for _ in range(len(bin_edges) - 1)]
    for scene, gt_depth, gt_disparity, predicted_depth, prediction in records:
        evaluation = scene.non_occluded_mask & np.isfinite(gt_depth) & (gt_depth > 0)
        prediction_valid = (
            evaluation
            & np.isfinite(prediction)
            & np.isfinite(predicted_depth)
            & (predicted_depth > 0)
        )
        bin_indices = np.digitize(gt_depth, bin_edges, right=False) - 1
        for bin_index in range(len(bin_edges) - 1):
            in_bin = evaluation & (bin_indices == bin_index)
            valid_in_bin = prediction_valid & in_bin
            total = int(in_bin.sum())
            valid = int(valid_in_bin.sum())
            if total == 0:
                continue
            totals[bin_index] += total
            valids[bin_index] += valid
            if valid:
                disparity_error = np.abs(prediction[valid_in_bin] - gt_disparity[valid_in_bin])
                disparity_errors[bin_index].append(disparity_error)
                depth_errors[bin_index].append(
                    np.abs(predicted_depth[valid_in_bin] - gt_depth[valid_in_bin])
                )
                bad2_counts[bin_index] += int((disparity_error > 2.0).sum())
            bad2_counts[bin_index] += total - valid

    rows: list[dict[str, float | int]] = []
    for bin_index, total in enumerate(totals):
        if total == 0:
            continue
        valid = int(valids[bin_index])
        depth_error = (
            np.concatenate(depth_errors[bin_index]) if depth_errors[bin_index] else None
        )
        disparity_error = (
            np.concatenate(disparity_errors[bin_index])
            if disparity_errors[bin_index]
            else None
        )
        rows.append(
            {
                "depth_bin_start_m": float(bin_edges[bin_index]),
                "depth_bin_end_m": float(bin_edges[bin_index + 1]),
                "depth_bin_center_m": float(
                    (bin_edges[bin_index] + bin_edges[bin_index + 1]) / 2
                ),
                "mean_abs_depth_error_m": (
                    float(np.mean(depth_error)) if depth_error is not None else float("nan")
                ),
                "mean_abs_disparity_error_px": (
                    float(np.mean(disparity_error))
                    if disparity_error is not None
                    else float("nan")
                ),
                "bad2": float(bad2_counts[bin_index] / total),
                "valid_coverage": float(valid / total),
                "evaluation_pixel_count": int(total),
                "valid_prediction_count": valid,
            }
        )
    return rows
