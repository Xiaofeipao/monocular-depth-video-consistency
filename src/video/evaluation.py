"""Auditable paired evaluation of arbitrary relative video-depth predictions.

There is deliberately no reciprocal, positivity filter, per-frame rescaling or
ground-truth alignment here.  A fixed reference-frame percentile span is a unit
for reporting changes, not an operation on the saved model predictions.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.metrics.temporal import (
    forward_backward_consistency_mask,
    warp_with_backward_flow,
)


MODELS = ("dav2", "vda")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _path_inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact must be relative and inside {root}: {value}")
    return path


def _checked(path: Path, expected: str, label: str) -> Path:
    if not isinstance(expected, str) or len(expected) != 64 or _sha(path) != expected:
        raise ValueError(f"SHA256 mismatch or missing checksum ({label}): {path}")
    return path


def _complete(payload: dict, label: str) -> None:
    if ("status" in payload and payload["status"] != "complete") or ("status" not in payload and payload.get("complete") is not True):
        raise ValueError(f"Incomplete {label}")


def _array(path: Path, shape: tuple[int, ...], *, dtype: np.dtype | None = np.dtype("float32")) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != shape or (dtype is not None and array.dtype != dtype):
        raise ValueError(f"Unexpected array shape/dtype: {path}: {array.shape}/{array.dtype}; expected {shape}/{dtype}")
    return array


def _model_field(run: dict, name: str, default: Any = None) -> Any:
    model = run["model"]
    return model.get(name, model.get("metadata", {}).get(name, default))


def _load_sources(manifest_path: str | Path, dav2_run: str | Path, vda_run: str | Path,
                  flow_cache: str | Path) -> dict:
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    _complete(manifest, "frame manifest")
    manifest_sha = _sha(manifest_path)
    frames = manifest.get("frames", [])
    if len(frames) < 2 or [item.get("index") for item in frames] != list(range(len(frames))):
        raise ValueError("Paired video evaluation requires at least two consecutive indexed frames")
    height, width = int(manifest["height"]), int(manifest["width"])
    shape = (height, width)
    if min(shape) < 1 or not math.isfinite(float(manifest["fps"])) or float(manifest["fps"]) <= 0:
        raise ValueError("Invalid manifest dimensions/fps")
    timestamps = [float(item["timestamp_seconds"]) for item in frames]
    if not all(math.isfinite(value) and value >= 0 for value in timestamps) or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Manifest timestamps must be finite and strictly increasing")
    source_indices = [item.get("source_index") for item in frames]
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in source_indices) or any(b <= a for a, b in zip(source_indices, source_indices[1:])):
        raise ValueError("Source frame identities must be strictly increasing nonnegative integers")
    if len({item["rgb_path"] for item in frames}) != len(frames):
        raise ValueError("Different cached frames cannot share one RGB path")
    source_files = [{"path": str(manifest_path), "sha256": manifest_sha, "role": "frame_manifest"}]
    for item in frames:
        path = _checked(_path_inside(manifest_path.parent, item["rgb_path"]), item["rgb_sha256"], "RGB frame")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != shape:
            raise ValueError(f"RGB size/read mismatch: {path}")
        source_files.append({"path": str(path), "sha256": item["rgb_sha256"], "role": "rgb"})
    roi = np.ones(shape, dtype=bool)
    if manifest.get("roi_path"):
        path = _checked(_path_inside(manifest_path.parent, manifest["roi_path"]), manifest.get("roi_sha256"), "fixed ROI")
        value = np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if value is None or value.shape != shape or not np.isfinite(value).all():
            raise ValueError("ROI must be a finite two-dimensional mask on the shared output grid")
        roi = value != 0
        if not roi.any():
            raise ValueError("Fixed ROI is empty")
        source_files.append({"path": str(path), "sha256": manifest["roi_sha256"], "role": "fixed_roi"})
    runs, raw = {}, {}
    for name, directory in zip(MODELS, (dav2_run, vda_run)):
        directory = Path(directory).resolve()
        path = directory / "run.json"
        run = _read_json(path)
        _complete(run, f"{name} run")
        if run.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"{name} run is bound to a different frame manifest")
        if run.get("manifest_path") and Path(run["manifest_path"]).resolve() != manifest_path:
            raise ValueError(f"{name} manifest path differs from the evaluated source")
        if _model_field(run, "depth_type") != "relative" or _model_field(run, "representation") != "inverse_depth_like":
            raise ValueError(f"{name}: expected relative inverse_depth_like predictions")
        if _model_field(run, "unit") != "arbitrary":
            raise ValueError(f"{name}: relative comparison requires arbitrary units")
        if _model_field(run, "alignment") != "none" or _model_field(run, "gt_used", False):
            raise ValueError(f"{name}: official relative comparison requires alignment:none and no GT")
        model_name = _model_field(run, "name", name)
        if model_name != name:
            raise ValueError(f"Model identity mismatch: requested {name}, run records {model_name}")
        entries = run.get("frames", [])
        if len(entries) != len(frames) or [entry.get("index") for entry in entries] != list(range(len(frames))):
            raise ValueError(f"{name} output frame identities/count differ")
        arrays = []
        source_files.append({"path": str(path), "sha256": _sha(path), "role": f"{name}_run"})
        for entry, frame in zip(entries, frames):
            if entry.get("shape") != list(shape) or entry.get("dtype") != "float32":
                raise ValueError(f"{name}: raw metadata shape/dtype mismatch")
            for key in ("source_index", "timestamp_seconds", "rgb_sha256"):
                if key in entry and entry[key] != frame.get(key):
                    raise ValueError(f"{name}: raw frame {key} differs from source manifest")
            path = _checked(_path_inside(directory, entry["raw_path"]), entry["raw_sha256"], f"{name} raw")
            arrays.append(_array(path, shape))
            source_files.append({"path": str(path), "sha256": entry["raw_sha256"], "role": f"{name}_raw"})
        raw[name] = arrays
        runs[name] = run
    network_hw = [_model_field(runs[name], "network_input_hw") for name in MODELS]
    if network_hw[0] != network_hw[1] or not isinstance(network_hw[0], (tuple, list)) or len(network_hw[0]) != 2 or min(network_hw[0]) < 1:
        raise ValueError(f"Actual network input H/W differ or are missing: {network_hw}")
    precision = [_model_field(runs[name], "precision") for name in MODELS]
    if not precision[0] or precision[0] != precision[1]:
        raise ValueError(f"Model precision differs or is missing: {precision}")
    for field in ("encoder", "device"):
        values = [_model_field(runs[name], field) for name in MODELS]
        if values[0] is not None and values[1] is not None and values[0] != values[1]:
            raise ValueError(f"Model {field} differs: {values}")
    flow_cache = Path(flow_cache).resolve()
    flow_path = flow_cache / "flow_manifest.json"
    flow_manifest = _read_json(flow_path)
    _complete(flow_manifest, "flow cache")
    if flow_manifest.get("manifest_sha256") != manifest_sha:
        raise ValueError("Flow cache is bound to a different RGB frame manifest")
    if flow_manifest.get("frames_count", flow_manifest.get("frame_count")) != len(frames) or flow_manifest.get("shape") != list(shape):
        raise ValueError("Flow cache grid/frame count mismatch")
    pairs = flow_manifest.get("pairs", [])
    if [(item.get("previous_index"), item.get("current_index")) for item in pairs] != [(i - 1, i) for i in range(1, len(frames))]:
        raise ValueError("Flow cache does not contain exactly the ordered adjacent frame pairs")
    source_files.append({"path": str(flow_path), "sha256": _sha(flow_path), "role": "flow_manifest"})
    protocol_bindings = [payload.get("protocol_sha256") for payload in (*runs.values(), flow_manifest)]
    if any(protocol_bindings):
        if not all(protocol_bindings) or len(set(protocol_bindings)) != 1:
            raise ValueError("Model runs and flow cache use different frozen protocols")
        for payload in (*runs.values(), flow_manifest):
            path = _checked(Path(payload["protocol_path"]).resolve(), payload["protocol_sha256"], "frozen protocol")
            protocol = _read_json(path)
            if protocol.get("status") != "frozen" or protocol.get("manifest_sha256") != manifest_sha:
                raise ValueError("Frozen protocol is not bound to this frame manifest")
        source_files.append({"path": str(path), "sha256": protocol_bindings[0], "role": "frozen_protocol"})
    flows = []
    for pair in pairs:
        arrays = []
        for direction in ("forward", "backward"):
            path = _checked(_path_inside(flow_cache, pair[f"{direction}_path"]), pair[f"{direction}_sha256"], direction)
            arrays.append(_array(path, (*shape, 2)))
            source_files.append({"path": str(path), "sha256": pair[f"{direction}_sha256"], "role": f"{direction}_flow"})
        flows.append(arrays)
    return {"manifest": manifest, "manifest_path": manifest_path, "manifest_sha256": manifest_sha,
            "shape": shape, "roi": roi, "raw": raw, "runs": runs, "flows": flows,
            "flow_manifest": flow_manifest, "source_files": source_files,
            "run_directories": {name: str(Path(directory).resolve()) for name, directory in zip(MODELS, (dav2_run, vda_run))},
            "flow_directory": str(flow_cache)}


def _verify_config(sources: dict, config: dict) -> None:
    """A report option may change presentation; scientific settings stay frozen."""
    requested = {key: value for key, value in config.items() if key != "visualization"}
    for name in MODELS:
        recorded = sources["runs"][name].get("config")
        if not isinstance(recorded, dict):
            raise ValueError(f"{name}: saved configuration is missing")
        recorded = {key: value for key, value in recorded.items() if key != "visualization"}
        if recorded != requested:
            raise ValueError(f"{name}: evaluation configuration differs from the saved prediction protocol")
    expected_backend = config.get("flow", {}).get("backend")
    actual_backend = sources["flow_manifest"].get("flow", {}).get("backend")
    if expected_backend and expected_backend != actual_backend:
        raise ValueError("Flow estimator differs from the declared evaluation protocol")


def _protocol(config: dict, frames_count: int) -> dict:
    evaluation = config.get("evaluation", config)
    result = {"reference_index": int(evaluation.get("reference_index", 0)),
              "percentile_low": float(evaluation.get("percentile_low", 5.0)),
              "percentile_high": float(evaluation.get("percentile_high", 95.0)),
              "fb_threshold_px": float(evaluation.get("fb_threshold_px", 1.0)),
              "minimum_coverage": float(evaluation.get("minimum_coverage", 0.05)),
              "alignment": evaluation.get("alignment", "none")}
    if not 0 <= result["reference_index"] < frames_count:
        raise ValueError("Reference frame index is outside the evaluated clip")
    if not 0 <= result["percentile_low"] < result["percentile_high"] <= 100:
        raise ValueError("Invalid fixed-reference percentile interval")
    if not math.isfinite(result["fb_threshold_px"]) or result["fb_threshold_px"] < 0:
        raise ValueError("FB threshold must be finite and nonnegative")
    if not 0 <= result["minimum_coverage"] <= 1 or result["alignment"] != "none":
        raise ValueError("Invalid minimum coverage or forbidden alignment")
    result.update({"depth_type": "relative", "representation": "inverse_depth_like", "unit": "arbitrary",
                   "zero_values": "valid_if_finite", "negative_values": "valid_if_finite",
                   "per_frame_alignment": False, "gt_used": False,
                   "span_domain": "reference_frame_common_finite_in_fixed_roi_or_full_frame",
                   "percentile_method": "numpy_linear", "roi_application": "target_pixels_only",
                   "median_domain": "one_fixed_roi_or_full_frame_region_finite_in_both_models_in_all_frames",
                   "primary_pair_aggregation": "arithmetic_mean_over_all_nonempty_shared_mask_pairs; low_coverage_is_a_quality_flag_only",
                   "secondary_aggregation": "pixel_weighted_over_the_same_included_pairs",
                   "flow_warp": "current_to_previous_bilinear_with_all_nonzero_weight_support_finite",
                   "interpretation": "stability_proxy_without_depth_GT; scene/camera/flow changes also contribute"})
    return result


def _finite_support(source: np.ndarray, backward: np.ndarray) -> np.ndarray:
    """Finite support of *all* contributing bilinear neighbours, including zeros."""
    height, width = source.shape
    yy, xx = np.indices(source.shape, dtype=np.float32)
    x, y = xx + backward[..., 0], yy + backward[..., 1]
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    safe_x, safe_y = np.where(valid, x, 0), np.where(valid, y, 0)
    x0, y0 = np.floor(safe_x).astype(int), np.floor(safe_y).astype(int)
    x1, y1 = np.ceil(safe_x).astype(int), np.ceil(safe_y).astype(int)
    finite = np.isfinite(source)
    return valid & finite[y0, x0] & finite[y0, x1] & finite[y1, x0] & finite[y1, x1]


def _safe_warp(source: np.ndarray, backward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    warped, geometric = warp_with_backward_flow(np.where(np.isfinite(source), source, 0).astype(np.float32), backward)
    return warped, geometric & _finite_support(source, backward)


def _aggregate(rows: list[dict], name: str, span: float | None) -> dict:
    included = [row for row in rows if row["included_in_aggregate"] and row[f"{name}_normalized_mae"] is not None]
    if not included:
        return {"included_pairs": 0, "included_pixels": 0, "mean_pair_normalized_mae": None,
                "mean_pair_normalized_rmse": None, "pixel_weighted_normalized_mae": None,
                "pixel_weighted_normalized_rmse": None,
                "unavailable_reason": "invalid_reference_span" if span is None else "no_nonempty_shared_mask_pairs"}
    pixels = sum(row["common_pixels"] for row in included)
    return {"included_pairs": len(included), "included_pixels": pixels,
            "mean_pair_normalized_mae": float(np.mean([row[f"{name}_normalized_mae"] for row in included], dtype=np.float64)),
            "mean_pair_normalized_rmse": float(np.mean([row[f"{name}_normalized_rmse"] for row in included], dtype=np.float64)),
            "pixel_weighted_normalized_mae": sum(row[f"{name}_abs_error_sum"] for row in included) / pixels / span,
            "pixel_weighted_normalized_rmse": math.sqrt(sum(row[f"{name}_squared_error_sum"] for row in included) / pixels) / span,
            "unavailable_reason": None}


def _artifact(output: Path, path: Path) -> dict:
    return {"path": str(path.relative_to(output)), "sha256": _sha(path)}


def _array_artifact(output: Path | None, relative: str, array: np.ndarray, role: str,
                    *, verification_dir: Path | None = None, **extra) -> dict:
    """Hash saved bytes; compare recomputed float diagnostics with tolerance."""
    if output is not None:
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array, allow_pickle=False)
        digest = _sha(path)
    elif verification_dir is not None:
        path = _path_inside(verification_dir, relative)
        saved = _array(path, array.shape, dtype=array.dtype)
        equal = np.array_equal(saved, array) if array.dtype == np.bool_ else np.allclose(saved, array, rtol=1e-6, atol=1e-8, equal_nan=True)
        if not equal:
            raise ValueError(f"Verification failed, recomputed shared masks and background std arrays: {relative}")
        digest = _sha(path)
    else:
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        digest = hashlib.sha256(buffer.getbuffer()).hexdigest()
    return {"path": relative, "sha256": digest, "role": role, **extra}


def _measure(sources: dict, protocol: dict, output: Path | None = None,
             *, verification_dir: Path | None = None) -> tuple[dict, list[dict], list[dict], list[dict]]:
    raw, roi, shape = sources["raw"], sources["roi"], sources["shape"]
    manifest, reference_index = sources["manifest"], protocol["reference_index"]
    count = len(manifest["frames"])
    roi_count = int(roi.sum())
    reference_mask = roi.copy()
    spatial = roi.copy()
    for name in MODELS:
        reference_mask &= np.isfinite(raw[name][reference_index])
        for frame in raw[name]:
            spatial &= np.isfinite(frame)
    references = {}
    for name in MODELS:
        values = np.asarray(raw[name][reference_index][reference_mask], dtype=np.float64)
        if values.size:
            low, high = np.percentile(values, [protocol["percentile_low"], protocol["percentile_high"]], method="linear")
            span = float(high - low)
            reason = None if span > 0 and math.isfinite(span) else "reference_percentile_span_is_zero"
            references[name] = {"percentile_low_value": float(low), "percentile_high_value": float(high),
                                "span": span if reason is None else None, "pixels": int(values.size), "unavailable_reason": reason}
        else:
            references[name] = {"percentile_low_value": None, "percentile_high_value": None,
                                "span": None, "pixels": 0, "unavailable_reason": "no_common_finite_reference_pixels"}
    rows, artifacts = [], []
    for i, (forward, backward) in enumerate(sources["flows"], start=1):
        # FB helper checks flow geometry; explicitly require finite interpolation
        # support as well, so NaNs cannot leak through a zero interpolation weight.
        fb = forward_backward_consistency_mask(forward, backward, threshold_px=protocol["fb_threshold_px"])
        fb &= _finite_support(forward[..., 0], backward) & _finite_support(forward[..., 1], backward)
        warped = {}
        previous_valid = np.ones(shape, dtype=bool)
        current_valid = np.ones(shape, dtype=bool)
        for name in MODELS:
            warped[name], valid = _safe_warp(raw[name][i - 1], backward)
            previous_valid &= valid
            current_valid &= np.isfinite(raw[name][i])
        mask = roi & fb & previous_valid & current_valid
        pixels = int(mask.sum())
        coverage = pixels / roi_count
        reason = "empty_common_mask" if not pixels else None
        quality_flag = "below_minimum_coverage" if pixels and coverage < protocol["minimum_coverage"] else None
        row = {"previous_index": i - 1, "current_index": i,
               "timestamp_seconds": float(manifest["frames"][i]["timestamp_seconds"]),
               "grid_pixels": int(np.prod(shape)),
               "roi_pixels": roi_count, "fb_valid_pixels_in_roi": int((roi & fb).sum()),
               "geometric_valid_pixels_in_roi": int((roi & _finite_support(np.ones(shape, dtype=np.float32), backward)).sum()),
               "after_current_finite_pixels": int((roi & fb & current_valid).sum()),
               "common_pixels": pixels, "coverage": coverage, "included_in_aggregate": reason is None,
               "exclusion_reason": reason, "quality_flag": quality_flag}
        row.update({"geometry_rejected_pixels": roi_count - row["geometric_valid_pixels_in_roi"],
                    "fb_rejected_pixels": row["geometric_valid_pixels_in_roi"] - row["fb_valid_pixels_in_roi"],
                    "current_nonfinite_rejected_pixels": row["fb_valid_pixels_in_roi"] - row["after_current_finite_pixels"],
                    "previous_support_rejected_pixels": row["after_current_finite_pixels"] - pixels})
        for name in MODELS:
            span = references[name]["span"]
            difference = np.asarray(raw[name][i][mask], dtype=np.float64) - np.asarray(warped[name][mask], dtype=np.float64)
            absolute = float(np.abs(difference).sum(dtype=np.float64)) if pixels else None
            squared = float(np.square(difference).sum(dtype=np.float64)) if pixels else None
            row.update({f"{name}_abs_error_sum": absolute, f"{name}_squared_error_sum": squared,
                        f"{name}_raw_mae": absolute / pixels if pixels else None,
                        f"{name}_raw_rmse": math.sqrt(squared / pixels) if pixels else None,
                        f"{name}_normalized_mae": absolute / pixels / span if pixels and span else None,
                        f"{name}_normalized_rmse": math.sqrt(squared / pixels) / span if pixels and span else None})
        artifacts.append(_array_artifact(output, f"masks/pair_{i - 1:06d}_{i:06d}.npy", mask,
                                         "pair_mask", verification_dir=verification_dir, current_index=i, pixels=pixels))
        rows.append(row)
    frame_rows = []
    fixed_pixels = int(spatial.sum())
    for i in range(count):
        row = {"index": i, "source_index": manifest["frames"][i].get("source_index"),
               "timestamp_seconds": float(manifest["frames"][i]["timestamp_seconds"]), "fixed_region_pixels": fixed_pixels}
        for name in MODELS:
            median = float(np.median(np.asarray(raw[name][i][spatial], dtype=np.float64))) if fixed_pixels else None
            refmedian = float(np.median(np.asarray(raw[name][reference_index][spatial], dtype=np.float64))) if fixed_pixels else None
            ratio = median / refmedian if median is not None and median > 0 and refmedian > 0 else None
            span = references[name]["span"]
            row.update({f"{name}_raw_median": median,
                        f"{name}_normalized_median": (median - references[name]["percentile_low_value"]) / span if median is not None and span else None,
                        f"{name}_normalized_median_delta": (median - refmedian) / span if median is not None and span else None,
                        f"{name}_median_ratio_to_reference": ratio,
                        f"{name}_abs_log_median_ratio": abs(math.log(ratio)) if ratio is not None else None,
                        f"{name}_ratio_unavailable_reason": None if ratio is not None else ("no_fixed_common_finite_region" if not fixed_pixels else "nonpositive_raw_median_not_ratio_compatible")})
        frame_rows.append(row)
    median_summary = {}
    for name in MODELS:
        values = [row[f"{name}_abs_log_median_ratio"] for row in frame_rows]
        valid = [value for value in values if value is not None]
        median_summary[name] = {"valid_ratio_frames": len(valid), "total_frames": count,
                                "mean_abs_log_median_ratio": float(np.mean(valid)) if valid else None,
                                "max_abs_log_median_ratio": max(valid) if valid else None,
                                "partial_ratio_curve": len(valid) != count,
                                "interpretation": "raw_median_scale_proxy_not_ground_truth_scale_drift"}
    motion = str(manifest.get("camera_motion", "unknown")).lower()
    fixed_camera = motion in {"fixed", "static", "fixed_camera", "static_camera"}
    background = {"applicable": fixed_camera and bool(manifest.get("roi_path")),
                  "camera_motion": motion, "region_pixels": fixed_pixels, "frames": count,
                  "unavailable_reason": None if fixed_camera and manifest.get("roi_path") else "requires_fixed_camera_and_explicit_predeclared_background_roi"}
    if background["applicable"]:
        if not fixed_pixels:
            background["unavailable_reason"] = "no_fixed_common_finite_region"
        for name in MODELS:
            span = references[name]["span"]
            item = {"normalized_std_mean": None, "normalized_std_p95": None, "normalized_std_max": None,
                    "adjacent_normalized_mae_mean": None, "unavailable_reason": None}
            if not fixed_pixels or span is None:
                item["unavailable_reason"] = "no_fixed_common_finite_region" if not fixed_pixels else "invalid_reference_span"
            else:
                mean, m2 = np.zeros(fixed_pixels, dtype=np.float64), np.zeros(fixed_pixels, dtype=np.float64)
                adjacent, prior = [], None
                for i, frame in enumerate(raw[name], start=1):
                    values = np.asarray(frame[spatial], dtype=np.float64) / span
                    delta = values - mean
                    mean += delta / i
                    m2 += delta * (values - mean)
                    if prior is not None:
                        adjacent.append(float(np.mean(np.abs(values - prior), dtype=np.float64)))
                    prior = values
                std = np.sqrt(np.maximum(m2 / count, 0))  # population std, ddof=0
                item.update({"normalized_std_mean": float(std.mean()), "normalized_std_p95": float(np.percentile(std, 95)),
                             "normalized_std_max": float(std.max()), "adjacent_normalized_mae_mean": float(np.mean(adjacent)),
                             "temporal_std_ddof": 0})
                std_map = np.full(shape, np.nan, dtype=np.float32)
                std_map[spatial] = std.astype(np.float32)
                artifacts.append(_array_artifact(output, f"{name}_background_normalized_std.npy", std_map, f"{name}_background_std", verification_dir=verification_dir))
            background[name] = item
    artifacts.append(_array_artifact(output, "fixed_common_region.npy", spatial, "fixed_common_region", verification_dir=verification_dir, pixels=fixed_pixels))
    aggregate = {name: _aggregate(rows, name, references[name]["span"]) for name in MODELS}
    left, right = [aggregate[name]["mean_pair_normalized_mae"] for name in MODELS]
    improvement = (left - right) / left * 100 if left is not None and right is not None and left > 0 else None
    windowing = _model_field(sources["runs"]["vda"], "windowing", {}) or {}
    starts = windowing.get("window_starts", [])
    boundaries = []
    for start in starts:
        if isinstance(start, dict):
            start = start.get("start", start.get("start_index"))
        if isinstance(start, int) and 0 < start < count:
            overlap = int(windowing.get("overlap", 10))
            interpolation = int(windowing.get("interpolation_length", 8))
            blend_start, new_start = start + overlap - interpolation, start + overlap
            relevant = sorted({i for center in (start, blend_start, new_start)
                               for i in range(center - 1, center + 2) if 1 <= i < count})
            boundaries.append({"window_start": start,
                               "interpolation_start": blend_start if blend_start < count else None,
                               "interpolation_stop_exclusive": min(new_start, count) if blend_start < count else None,
                               "new_output_start": new_start if new_start < count else None,
                               "nearby_pair_current_indices": relevant,
                               "pair_scores": [{key: rows[i - 1][key] for key in ("current_index", "coverage", "quality_flag", "dav2_normalized_mae", "vda_normalized_mae")} for i in relevant],
                               "note": "official window-start context and blended output region are distinguished; only unpadded retained output pairs are scored"})
    ranked = sorted((row for row in rows if any(row[f"{name}_normalized_mae"] is not None for name in MODELS)),
                    key=lambda row: max(row[f"{name}_normalized_mae"] or 0 for name in MODELS), reverse=True)
    result = {"reference": references, "aggregate": aggregate,
              "comparison": {"vda_mean_pair_mae_reduction_percent": improvement,
                             "interpretation": "negative means VDA worse; no accuracy claim without GT"},
              "coverage": {"pair_count": len(rows), "included_pairs": sum(row["included_in_aggregate"] for row in rows),
                           "empty_pairs": sum(row["common_pixels"] == 0 for row in rows),
                           "low_coverage_pairs": sum(row["quality_flag"] == "below_minimum_coverage" for row in rows),
                           "mean": float(np.mean([row["coverage"] for row in rows])), "minimum": min(row["coverage"] for row in rows)},
              "fixed_common_region_pixels": fixed_pixels, "median_summary": median_summary,
              "background_flicker": background,
              "worst_pairs": [{key: row[key] for key in ("current_index", "coverage", "exclusion_reason", "dav2_normalized_mae", "vda_normalized_mae")} for row in ranked[:5]],
              "window_boundaries": {"metadata_available": bool(starts), "windowing": windowing, "boundaries": boundaries}}
    return result, rows, frame_rows, artifacts


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        result = []
        for row in csv.DictReader(stream):
            converted = {}
            for key, value in row.items():
                if value == "":
                    converted[key] = None
                elif value in {"True", "False"}:
                    converted[key] = value == "True"
                else:
                    try:
                        number = float(value)
                        if not math.isfinite(number):
                            raise ValueError(f"Non-finite CSV number: {path} {key}")
                        converted[key] = number
                    except ValueError:
                        if value.lower() in {"nan", "inf", "-inf", "+inf"}:
                            raise ValueError(f"Non-finite CSV number: {path} {key}")
                        converted[key] = value
            result.append(converted)
    return result


def _equivalent(actual: Any, expected: Any, label: str = "result") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Verification failed, object keys differ: {label}")
        for key in expected:
            _equivalent(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Verification failed, list differs: {label}")
        for i, (left, right) in enumerate(zip(actual, expected)):
            _equivalent(left, right, f"{label}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"Verification failed, numerical mismatch: {label}: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"Verification failed: {label}: {actual!r} != {expected!r}")


def evaluate_pair(manifest_path: str | Path, dav2_run: str | Path, vda_run: str | Path,
                  flow_cache: str | Path, output_dir: str | Path, *, config: dict) -> Path:
    """Evaluate saved paired runs; persist masks, CSV, plots and source bindings.

    Existing output directories are never overwritten.  A failed invocation
    retains ``evaluation_status.json`` with the actual error and traceback.
    This function performs no model inference, flow inference or download.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "evaluation_status.json"
    _write_json(status_path, {"schema_version": 1, "status": "running"})
    try:
        sources = _load_sources(manifest_path, dav2_run, vda_run, flow_cache)
        _verify_config(sources, config)
        protocol = _protocol(config, len(sources["manifest"]["frames"]))
        result, pair_rows, frame_rows, artifacts = _measure(sources, protocol, output)
        pair_path, frame_path = output / "pair_metrics.csv", output / "frame_metrics.csv"
        _write_csv(pair_path, pair_rows)
        _write_csv(frame_path, frame_rows)
        artifacts += [{**_artifact(output, pair_path), "role": "pair_metrics"}, {**_artifact(output, frame_path), "role": "frame_metrics"}]
        manifest = sources["manifest"]
        summary = {"schema_version": 1, "status": "complete", "scope": "part1b_part2b_relative_paired_video",
                   "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                   "evaluation_environment": {"python": sys.version, "executable": sys.executable,
                                              "platform": platform.platform(), "numpy": np.__version__, "opencv": cv2.__version__},
                   "evaluation_source_sha256": {str(source): _sha(source) for source in
                                                (Path(__file__).resolve(), Path(__file__).with_name("report.py").resolve(),
                                                 Path(__file__).resolve().parents[1] / "metrics/temporal.py")},
                   "clip": {key: manifest.get(key) for key in ("clip_id", "source_kind", "camera_motion", "split", "fps", "height", "width")},
                   "frames_count": len(manifest["frames"]), "protocol": protocol, "config": config,
                   "sources": {"manifest_path": str(sources["manifest_path"]), "manifest_sha256": sources["manifest_sha256"],
                               "runs": sources["run_directories"], "flow_cache": sources["flow_directory"],
                               "files": sources["source_files"]},
                   "models": {name: sources["runs"][name]["model"] for name in MODELS},
                   "timing": {name: sources["runs"][name].get("timing", {}) for name in MODELS},
                   "flow": sources["flow_manifest"].get("flow", {}), "artifacts": artifacts, **result}
        path = output / "summary.json"
        _write_json(path, summary)
        if config.get("visualization", {}).get("generate_report", True):
            from .report import generate_report

            report = generate_report(path, config=config)
            summary["report"] = report
            _write_json(path, summary)
        _write_json(status_path, {"schema_version": 1, "status": "complete", "summary": "summary.json", "summary_sha256": _sha(path)})
        return path
    except Exception as error:
        _write_json(status_path, {"schema_version": 1, "status": "failed", "error_type": type(error).__name__,
                                 "error": str(error), "traceback": traceback.format_exc()})
        raise


def verify_evaluation(output_dir: str | Path, *, verify_sources: bool = True) -> dict:
    """Reload hashes/masks/CSV and independently recompute summaries from raw.

    ``verify_sources=False`` is an explicitly limited archive integrity check;
    the default also binds every raw/RGB/flow/ROI to its original source and
    recomputes all numerical rows from saved predictions (no model execution).
    """
    output = Path(output_dir).resolve()
    path = output / "summary.json"
    status = _read_json(output / "evaluation_status.json")
    _complete(status, "evaluation")
    _checked(path, status.get("summary_sha256"), "evaluation summary")
    summary = _read_json(path)
    _complete(summary, "evaluation summary")
    artifacts = summary["artifacts"] + summary.get("report", {}).get("assets", [])
    pair_masks = {}
    checked = 1
    for item in artifacts:
        artifact = _checked(_path_inside(output, item["path"]), item["sha256"], item.get("role", "report"))
        checked += 1
        if item.get("role") in {"pair_mask", "fixed_common_region"}:
            array = _array(artifact, (summary["clip"]["height"], summary["clip"]["width"]), dtype=np.dtype("bool"))
            if int(array.sum()) != item["pixels"]:
                raise ValueError("Persisted shared mask pixel count differs")
            if item["role"] == "pair_mask":
                pair_masks[item["current_index"]] = array
    pair_rows, frame_rows = _read_csv(output / "pair_metrics.csv"), _read_csv(output / "frame_metrics.csv")
    if len(pair_rows) != summary["frames_count"] - 1 or len(frame_rows) != summary["frames_count"]:
        raise ValueError("Saved CSV frame/pair count differs")
    for i, row in enumerate(pair_rows, start=1):
        if row["current_index"] != i or row["previous_index"] != i - 1 or i not in pair_masks or row["common_pixels"] != int(pair_masks[i].sum()):
            raise ValueError("Saved pair CSV and mask identities differ")
    for name in MODELS:
        _equivalent(summary["aggregate"][name], _aggregate(pair_rows, name, summary["reference"][name]["span"]), f"aggregate.{name}")
    if verify_sources:
        for source, digest in summary["evaluation_source_sha256"].items():
            _checked(Path(source), digest, "evaluation scientific source")
            checked += 1
        for item in summary["sources"]["files"]:
            _checked(Path(item["path"]), item["sha256"], item["role"])
            checked += 1
        sources = _load_sources(summary["sources"]["manifest_path"], summary["sources"]["runs"]["dav2"],
                                summary["sources"]["runs"]["vda"], summary["sources"]["flow_cache"])
        protocol = _protocol(summary["config"], summary["frames_count"])
        _verify_config(sources, summary["config"])
        _equivalent(summary["protocol"], protocol, "frozen protocol")
        recomputed, expected_pairs, expected_frames, expected_arrays = _measure(sources, protocol, verification_dir=output)
        _equivalent(pair_rows, expected_pairs, "pair CSV from raw")
        _equivalent(frame_rows, expected_frames, "frame CSV from raw")
        for key, value in recomputed.items():
            _equivalent(summary[key], value, key)
        actual_arrays = [item for item in summary["artifacts"] if item["role"] not in {"pair_metrics", "frame_metrics"}]
        _equivalent(actual_arrays, expected_arrays, "recomputed shared masks and background std arrays")
        if "report" in summary:
            from .report import _source_binding

            _equivalent(summary["report"]["binding"], _source_binding(summary), "report visual source binding")
    return {"status": "verified", "source_verification": verify_sources, "checked_files": checked,
            "frames": len(frame_rows), "pairs": len(pair_rows), "models_executed": False,
            "numeric_rows_recomputed_from_raw": verify_sources,
            "numeric_tolerance": {"relative": 1e-6, "absolute": 1e-8},
            "saved_file_integrity": "exact_sha256; recomputed_float_statistics_use_tolerance"}
