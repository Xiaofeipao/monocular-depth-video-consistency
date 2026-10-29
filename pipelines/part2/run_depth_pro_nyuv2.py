#!/usr/bin/env python3
"""Inference-only Depth Pro evaluation on the frozen NYUv2 test split."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.nyuv2 import NYUv2LabeledDataset
from src.data.output_schema import validate_depth_output
from src.metrics.boundary import compute_depth_pro_si_boundary_f1
from src.metrics.depth import build_valid_mask, compute_depth_metrics
from src.models.depth_pro_adapter import DepthProAdapter
from src.visualization.nyuv2 import save_contact_sheet, save_nyuv2_panel


DEPTH_METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["protocol_status"] not in {"candidate_v2", "frozen_v1"}:
        raise ValueError(f"Unsupported protocol status: {config['protocol_status']}")
    if config["evaluation"]["test_time_alignment"] != "none":
        raise ValueError("Native metric evaluation must not use test-time alignment")
    if config["model"]["provide_external_focal_length"]:
        raise ValueError("Primary Depth Pro track must use model-predicted focal length")
    return config


def resolve_and_verify(project_root: Path, relative_path: str, expected_sha256: str) -> Path:
    path = project_root / relative_path
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected_sha256}")
    return path


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def crop_slices(config: dict) -> tuple[slice, slice]:
    crop = config["evaluation"]["crop"]
    return slice(int(crop["top"]), int(crop["bottom"])), slice(
        int(crop["left"]), int(crop["right"])
    )


def prediction_stem(test_position: int, matlab_index_1based: int) -> str:
    return f"test_{test_position:04d}_mat_{matlab_index_1based:04d}"


def resize_prediction(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return np.asarray(depth, dtype=np.float32)
    resized = cv2.resize(
        np.asarray(depth, dtype=np.float32),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.asarray(resized, dtype=np.float32)


def selected_positions(
    dataset_length: int,
    max_samples: int | None,
    explicit_positions: list[int] | None,
) -> list[int]:
    if explicit_positions is not None:
        if max_samples is not None:
            raise ValueError("Use either --positions or --max-samples, not both")
        positions = [int(value) for value in explicit_positions]
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("--positions must contain unique test positions")
        if min(positions) < 0 or max(positions) >= dataset_length:
            raise ValueError(f"--positions must be within [0, {dataset_length})")
        return positions
    count = dataset_length if max_samples is None else min(max_samples, dataset_length)
    if count <= 0:
        raise ValueError("max_samples must be positive")
    return list(range(count))


def infer_predictions(
    *,
    config: dict,
    dataset: NYUv2LabeledDataset,
    positions: list[int],
    output_dir: Path,
    device: str,
    resume: bool,
    checkpoint: Path,
) -> dict[str, object]:
    predictions_dir = output_dir / "predictions"
    metadata_dir = output_dir / "metadata"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    adapter = DepthProAdapter(
        repository=PROJECT_ROOT / config["model"]["repository"],
        checkpoint=checkpoint,
        device=device,
        precision=config["model"]["precision"],
    )
    environment = {
        **adapter.environment(),
        "python": platform.python_version(),
        "model_load_seconds": adapter.load_seconds,
    }
    (output_dir / "inference_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )

    warmup_iterations = int(config["efficiency"]["warmup_iterations"])
    if warmup_iterations and positions:
        warmup_sample = dataset.get_test_sample(positions[0])
        for _ in range(warmup_iterations):
            adapter.infer(warmup_sample.rgb)

    inferred = skipped = 0
    start = time.perf_counter()
    for sequence_index, position in enumerate(positions, start=1):
        sample = dataset.get_test_sample(position)
        stem = prediction_stem(position, sample.matlab_index_1based)
        prediction_path = predictions_dir / f"{stem}.npz"
        metadata_path = metadata_dir / f"{stem}.json"
        if prediction_path.exists() or metadata_path.exists():
            if resume and prediction_path.exists() and metadata_path.exists():
                skipped += 1
                print(f"[{sequence_index:03d}/{len(positions)}] resume {stem}", flush=True)
                continue
            raise FileExistsError(
                f"Refusing to overwrite incomplete/existing output for {stem}; use --resume"
            )
        depth, focal_length_px, runtime, peak_memory = adapter.infer(sample.rgb)
        depth = resize_prediction(depth, sample.depth_m.shape)
        metadata = {
            "sample_id": stem,
            "test_position": position,
            "matlab_index_1based": sample.matlab_index_1based,
            "scene": sample.scene,
            "scene_type": sample.scene_type,
            "model": config["model"]["name"],
            "depth_type": "metric",
            "unit": "meter",
            "alignment": "none",
            "focal_length_source": "model_prediction",
            "focal_length_px": focal_length_px,
            "external_focal_length_provided": False,
            "input_shape": list(sample.rgb.shape[:2]),
            "output_shape": list(depth.shape),
            "inference_seconds": runtime,
            "peak_gpu_memory_bytes": peak_memory,
        }
        validate_depth_output(depth, metadata)
        np.savez_compressed(
            prediction_path,
            depth_m=depth,
            focal_length_px=np.float32(focal_length_px),
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        inferred += 1
        print(
            f"[{sequence_index:03d}/{len(positions)}] {stem}: "
            f"f={focal_length_px:.1f}px, {runtime:.3f}s, "
            f"peak={peak_memory / 2**30:.2f}GiB",
            flush=True,
        )
    wall_seconds = time.perf_counter() - start
    return {
        "positions_requested": len(positions),
        "inferred": inferred,
        "resumed": skipped,
        "wall_seconds": wall_seconds,
        "end_to_end_fps_including_io_and_save": (
            inferred / wall_seconds if inferred else None
        ),
        "environment": environment,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Bootstrap values must be a finite vector")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    batch_size = 250
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def manual_depth_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> dict[str, float]:
    """Independent explicit implementation used only for the 3-sample audit."""

    valid = (
        np.isfinite(prediction)
        & np.isfinite(target)
        & (prediction > 0)
        & (target > minimum)
        & (target < maximum)
    )
    pred = np.clip(prediction[valid].astype(np.float64), minimum, maximum)
    gt = target[valid].astype(np.float64)
    difference = pred - gt
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "abs_rel": float(np.sum(np.abs(difference) / gt) / len(gt)),
        "sq_rel": float(np.sum(difference**2 / gt) / len(gt)),
        "rmse": float(np.sqrt(np.sum(difference**2) / len(gt))),
        "rmse_log": float(np.sqrt(np.sum((np.log(pred) - np.log(gt)) ** 2) / len(gt))),
        "delta1": float(np.count_nonzero(ratio < 1.25) / len(gt)),
        "delta2": float(np.count_nonzero(ratio < 1.25**2) / len(gt)),
        "delta3": float(np.count_nonzero(ratio < 1.25**3) / len(gt)),
    }


def aggregate_scene_types(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["scene_type"]), []).append(row)
    summaries = []
    for scene_type, selected in sorted(grouped.items()):
        summary: dict[str, object] = {"scene_type": scene_type, "image_count": len(selected)}
        for metric in (*DEPTH_METRICS, "si_boundary_f1"):
            summary[metric] = float(np.mean([float(row[metric]) for row in selected]))
        summaries.append(summary)
    return summaries


def select_failure_cases(
    rows: list[dict[str, object]], thresholds: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    """Select every sample outside at least one configured metric threshold."""

    operators = {
        "greater_than": lambda value, threshold: value > threshold,
        "less_than": lambda value, threshold: value < threshold,
    }
    selected: list[dict[str, object]] = []
    for row in rows:
        triggered_metrics: list[str] = []
        for metric, specification in thresholds.items():
            if metric not in row:
                raise KeyError(f"Failure threshold references unknown metric: {metric}")
            operator_name = str(specification["operator"])
            if operator_name not in operators:
                raise ValueError(f"Unsupported failure threshold operator: {operator_name}")
            value = float(row[metric])
            threshold = float(specification["value"])
            if operators[operator_name](value, threshold):
                triggered_metrics.append(metric)
        if triggered_metrics:
            selected.append(
                {
                    "test_position": int(row["test_position"]),
                    "matlab_index_1based": int(row["matlab_index_1based"]),
                    "scene": row["scene"],
                    "scene_type": row["scene_type"],
                    "triggered_metrics": triggered_metrics,
                    "abs_rel": float(row["abs_rel"]),
                    "delta1": float(row["delta1"]),
                    "si_boundary_f1": float(row["si_boundary_f1"]),
                }
            )
    return sorted(
        selected,
        key=lambda item: (
            -len(item["triggered_metrics"]),
            -float(item["abs_rel"]),
            int(item["test_position"]),
        ),
    )


def evaluate_saved_predictions(
    *,
    config: dict,
    dataset: NYUv2LabeledDataset,
    positions: list[int],
    output_dir: Path,
) -> dict[str, object]:
    evaluation = config["evaluation"]
    minimum = float(evaluation["valid_depth_m"]["minimum"])
    maximum = float(evaluation["valid_depth_m"]["maximum"])
    crop = crop_slices(config)
    boundary = config["metrics"]["boundary"]
    camera_output = str(config["model"].get("camera_output", "focal_length_px"))
    if camera_output not in {"focal_length_px", "intrinsics_px"}:
        raise ValueError(f"Unsupported camera output: {camera_output}")
    rows: list[dict[str, object]] = []
    prediction_cache: dict[int, np.ndarray] = {}

    for sequence_index, position in enumerate(positions, start=1):
        sample = dataset.get_test_sample(position)
        stem = prediction_stem(position, sample.matlab_index_1based)
        prediction_path = output_dir / "predictions" / f"{stem}.npz"
        metadata_path = output_dir / "metadata" / f"{stem}.json"
        if not prediction_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Missing saved prediction or metadata for {stem}")
        with np.load(prediction_path) as archive:
            prediction = resize_prediction(archive["depth_m"], sample.depth_m.shape)
            if camera_output == "focal_length_px":
                saved_focal = float(archive["focal_length_px"])
                saved_intrinsics = None
            else:
                saved_focal = None
                saved_intrinsics = np.asarray(
                    archive["intrinsics_px"], dtype=np.float64
                ).copy()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_depth_output(prediction, metadata)
        if camera_output == "focal_length_px":
            if not np.isclose(
                saved_focal, float(metadata["focal_length_px"]), rtol=1e-6
            ):
                raise AssertionError(f"Focal length mismatch for {stem}")
            camera_fields = {"focal_length_px": saved_focal}
        else:
            metadata_intrinsics = np.asarray(
                metadata["intrinsics_px"], dtype=np.float64
            )
            if (
                saved_intrinsics.shape != (3, 3)
                or not np.all(np.isfinite(saved_intrinsics))
                or saved_intrinsics[0, 0] <= 0
                or saved_intrinsics[1, 1] <= 0
                or not np.allclose(
                    saved_intrinsics, metadata_intrinsics, rtol=1e-6, atol=1e-6
                )
            ):
                raise AssertionError(f"Intrinsics mismatch or invalid values for {stem}")
            camera_fields = {
                "fx_px": float(saved_intrinsics[0, 0]),
                "fy_px": float(saved_intrinsics[1, 1]),
                "cx_px": float(saved_intrinsics[0, 2]),
                "cy_px": float(saved_intrinsics[1, 2]),
            }
        prediction_crop = prediction[crop]
        target_crop = sample.depth_m[crop]
        primary = compute_depth_metrics(
            prediction_crop,
            target_crop,
            min_depth=minimum,
            max_depth=maximum,
            clip_prediction=True,
        )
        valid = build_valid_mask(
            prediction_crop,
            target_crop,
            min_depth=minimum,
            max_depth=maximum,
        )
        prediction_clipped = np.clip(prediction_crop, minimum, maximum)
        si_boundary_f1 = compute_depth_pro_si_boundary_f1(
            prediction_clipped,
            target_crop,
            valid_mask=valid,
            threshold_min=float(boundary["threshold_min"]),
            threshold_max=float(boundary["threshold_max"]),
            threshold_count=int(boundary["threshold_count"]),
        )
        row: dict[str, object] = {
            "sample_id": stem,
            "test_position": position,
            "matlab_index_1based": sample.matlab_index_1based,
            "scene": sample.scene,
            "scene_type": sample.scene_type,
            "inference_seconds": float(metadata["inference_seconds"]),
            "peak_gpu_memory_bytes": int(metadata["peak_gpu_memory_bytes"]),
            **camera_fields,
            **primary,
            "si_boundary_f1": si_boundary_f1,
        }
        rows.append(row)
        if position in set(config["visualization"]["audit_test_positions"]):
            prediction_cache[position] = prediction
        print(
            f"[{sequence_index:03d}/{len(positions)}] evaluate {stem}: "
            f"AbsRel={row['abs_rel']:.4f}, δ1={row['delta1']:.4f}, "
            f"SI-F1={si_boundary_f1:.4f}",
            flush=True,
        )

    write_csv(output_dir / "metrics_per_image.csv", rows)
    scene_type_rows = aggregate_scene_types(rows)
    write_csv(output_dir / "metrics_by_scene_type.csv", scene_type_rows)

    stats = config["statistics"]
    report_metrics = (*DEPTH_METRICS, "si_boundary_f1")
    macro_mean: dict[str, float] = {}
    confidence_intervals: dict[str, dict[str, float]] = {}
    for metric_index, metric in enumerate(report_metrics):
        values = np.asarray([float(row[metric]) for row in rows])
        macro_mean[metric] = float(values.mean())
        lower, upper = bootstrap_mean_ci(
            values,
            resamples=int(stats["bootstrap_resamples"]),
            confidence_level=float(stats["confidence_level"]),
            seed=int(stats["seed"]) + metric_index,
        )
        confidence_intervals[metric] = {"lower": lower, "upper": upper}
    runtimes = np.asarray([float(row["inference_seconds"]) for row in rows])
    efficiency = {
        "mean_seconds": float(runtimes.mean()),
        "median_seconds": float(np.median(runtimes)),
        "p95_seconds": float(np.quantile(runtimes, 0.95)),
        "fps_from_mean_inference_time": float(1.0 / runtimes.mean()),
        "peak_gpu_memory_bytes": int(
            max(int(row["peak_gpu_memory_bytes"]) for row in rows)
        ),
    }
    summary: dict[str, object] = {
        "protocol_status": config["protocol_status"],
        "image_count": len(rows),
        "aggregation": "per_image_macro_mean",
        "depth_type": "metric",
        "unit": "meter",
        "alignment": "none",
        "extra_camera_information": "none",
        "camera_output": camera_output,
        "macro_mean": macro_mean,
        "bootstrap_95_ci": confidence_intervals,
        "efficiency": efficiency,
    }
    if camera_output == "focal_length_px":
        focal_lengths = np.asarray([float(row["focal_length_px"]) for row in rows])
        summary["focal_length_source"] = "model_prediction"
        summary["focal_length_px"] = {
            "mean": float(focal_lengths.mean()),
            "minimum": float(focal_lengths.min()),
            "maximum": float(focal_lengths.max()),
        }
    else:
        summary["intrinsics_source"] = "model_prediction"
        summary["intrinsics_px"] = {
            key: {
                "mean": float(np.mean([float(row[key]) for row in rows])),
                "minimum": float(np.min([float(row[key]) for row in rows])),
                "maximum": float(np.max([float(row[key]) for row in rows])),
            }
            for key in ("fx_px", "fy_px", "cx_px", "cy_px")
        }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    failure_thresholds = config["visualization"]["failure_thresholds"]
    failures = select_failure_cases(rows, failure_thresholds)
    failure_report = {
        "selection_rule": "outside_any_threshold",
        "thresholds": failure_thresholds,
        "failure_count": len(failures),
        "cases": failures,
    }
    (output_dir / "failure_cases.json").write_text(
        json.dumps(failure_report, indent=2) + "\n", encoding="utf-8"
    )
    visualization_dir = output_dir / "visualization"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    for stale_panel in visualization_dir.glob("test_*.png"):
        stale_panel.unlink()
    legacy_failure_panel_dir = visualization_dir / "failure_cases"
    if legacy_failure_panel_dir.exists():
        for stale_panel in legacy_failure_panel_dir.glob("test_*.png"):
            stale_panel.unlink()
        legacy_failure_panel_dir.rmdir()
    contact_sheet_path = visualization_dir / "failure_cases_contact_sheet.jpg"
    contact_sheet_path.unlink(missing_ok=True)
    visualization_path_by_position: dict[int, Path] = {}
    row_by_position = {int(row["test_position"]): row for row in rows}
    failure_by_position = {
        int(item["test_position"]): item for item in failures
    }
    panel_positions=list(positions)
    maximum_panels=config['visualization'].get('max_panels')
    if maximum_panels is not None:
        if not isinstance(maximum_panels,int) or maximum_panels<1:
            raise ValueError('visualization.max_panels must be a positive integer')
        # Display selection only; metrics and failure classification still use
        # every requested image. Reserve audits before worst-error examples.
        priority=[int(p) for p in config['visualization']['audit_test_positions'] if int(p) in row_by_position]
        priority.extend(int(row['test_position']) for row in sorted(rows,key=lambda row:float(row['abs_rel']),reverse=True))
        panel_positions=list(dict.fromkeys(priority))[:maximum_panels]
    (visualization_dir/'selection.json').write_text(json.dumps({
        'scored_images':len(rows),'max_panels':maximum_panels,
        'visualized_test_positions':panel_positions,'policy':'audit positions then descending AbsRel; metrics unchanged'},indent=2))
    for position in panel_positions:
        sample = dataset.get_test_sample(position)
        stem = prediction_stem(position, sample.matlab_index_1based)
        if position in prediction_cache:
            prediction = prediction_cache[position]
        else:
            with np.load(output_dir / "predictions" / f"{stem}.npz") as archive:
                prediction = resize_prediction(archive["depth_m"], sample.depth_m.shape)
        row = row_by_position[position]
        panel_path = visualization_dir / f"{stem}.png"
        failure = failure_by_position.get(position)
        status = (
            f"fail={','.join(failure['triggered_metrics'])}"
            if failure is not None
            else "thresholds=pass"
        )
        save_nyuv2_panel(
            panel_path,
            rgb=sample.rgb,
            target_depth_m=sample.depth_m,
            prediction_depth_m=np.clip(prediction, minimum, maximum),
            crop=crop,
            title=(
                f"{stem} | {sample.scene_type} | AbsRel={float(row['abs_rel']):.3f} | "
                f"SI-F1={float(row['si_boundary_f1']):.3f} | {status}"
            ),
            prediction_title=str(
                config["model"].get(
                    "visualization_title", "Depth Pro native metric (m)"
                )
            ),
            depth_range_m=tuple(
                float(value) for value in config["visualization"]["metric_depth_range_m"]
            ),
            error_max_m=float(config["visualization"]["absolute_error_max_m"]),
        )
        visualization_path_by_position[position] = panel_path
    failure_visualization_paths = [
        visualization_path_by_position[int(item["test_position"])]
        for item in failures
        if int(item['test_position']) in visualization_path_by_position
    ]
    if failure_visualization_paths:
        save_contact_sheet(contact_sheet_path, failure_visualization_paths)

    audit_records = []
    tolerance = 1e-10
    for position in config["visualization"]["audit_test_positions"]:
        position = int(position)
        if position not in row_by_position:
            continue
        sample = dataset.get_test_sample(position)
        stem = prediction_stem(position, sample.matlab_index_1based)
        with np.load(output_dir / "predictions" / f"{stem}.npz") as archive:
            prediction = resize_prediction(archive["depth_m"], sample.depth_m.shape)
        manual = manual_depth_metrics(
            prediction[crop],
            sample.depth_m[crop],
            minimum=minimum,
            maximum=maximum,
        )
        recorded = row_by_position[position]
        differences = {
            key: abs(manual[key] - float(recorded[key])) for key in DEPTH_METRICS
        }
        audit_records.append(
            {
                "test_position": position,
                "sample_id": stem,
                "manual_metrics": manual,
                "absolute_differences": differences,
                "passed": max(differences.values()) <= tolerance,
            }
        )
    audit = {
        "description": "Independent explicit NumPy formulas versus shared evaluator",
        "tolerance": tolerance,
        "all_passed": all(item["passed"] for item in audit_records),
        "samples": audit_records,
    }
    if not audit["all_passed"]:
        raise AssertionError("Three-sample independent metric audit failed")
    (output_dir / "three_sample_metric_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "summary": summary,
        "failure_case_count": len(failures),
        "metric_audit_passed": bool(audit["all_passed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/part2/depth_pro_nyuv2.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/part2/depth_pro_nyuv2/frozen_v1",
    )
    parser.add_argument(
        "--mode", choices=("infer-evaluate", "infer", "evaluate"), default="infer-evaluate"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--positions", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = config["dataset"]
    labeled_mat = resolve_and_verify(
        PROJECT_ROOT,
        dataset_config["labeled_mat"],
        dataset_config["labeled_mat_sha256"],
    )
    split_file = resolve_and_verify(
        PROJECT_ROOT,
        dataset_config["split_file"],
        dataset_config["split_file_sha256"],
    )
    checkpoint = resolve_and_verify(
        PROJECT_ROOT,
        config["model"]["checkpoint"],
        config["model"]["checkpoint_sha256"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with NYUv2LabeledDataset(
        labeled_mat,
        split_file,
        split_variable=dataset_config["split_variable"],
    ) as dataset:
        if len(dataset) != int(dataset_config["expected_test_images"]):
            raise ValueError(f"Unexpected NYUv2 test count: {len(dataset)}")
        positions = selected_positions(len(dataset), args.max_samples, args.positions)
        inference = None
        if args.mode in {"infer-evaluate", "infer"}:
            inference = infer_predictions(
                config=config,
                dataset=dataset,
                positions=positions,
                output_dir=args.output_dir,
                device=args.device,
                resume=args.resume,
                checkpoint=checkpoint,
            )
            (args.output_dir / "inference_summary.json").write_text(
                json.dumps(inference, indent=2) + "\n", encoding="utf-8"
            )
        evaluation = None
        if args.mode in {"infer-evaluate", "evaluate"}:
            evaluation = evaluate_saved_predictions(
                config=config,
                dataset=dataset,
                positions=positions,
                output_dir=args.output_dir,
            )

    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "protocol_status": config["protocol_status"],
        "mode": args.mode,
        "sample_count": len(positions),
        "full_test_split": len(positions) == int(dataset_config["expected_test_images"]),
        "formal_training_run": False,
        "test_time_alignment": "none",
        "external_focal_length_provided": False,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"inference": inference, "evaluation": evaluation}, indent=2))


if __name__ == "__main__":
    main()
