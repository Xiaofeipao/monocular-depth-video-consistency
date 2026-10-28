#!/usr/bin/env python3
"""Run reproducible BM/SGBM geometry evaluation on all Middlebury scenes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.output_schema import validate_depth_output
from src.stereo.geometry import disparity_to_depth
from src.stereo.io import MiddleburyScene, load_middlebury_scene
from src.stereo.matching import StereoMatcherConfig, match_stereo
from src.stereo.metrics import compute_disparity_metrics, compute_stereo_depth_metrics
from src.stereo.sensitivity import (
    analyze_depth_sensitivity,
    analyze_prediction_error_by_distance,
)
from src.visualization.stereo import (
    save_contact_sheet,
    save_empirical_depth_error_plot,
    save_sensitivity_plot,
    save_stereo_failure_zoom,
    save_stereo_panel,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    expected = config["dataset"]["expected_scenes"]
    declared = config["split"]["development_scenes"] + config["split"]["evaluation_scenes"]
    if len(declared) != expected or len(set(declared)) != expected:
        raise ValueError("Development/evaluation scene split is incomplete or duplicated")
    return config


def matcher_config(
    config: dict, experiment: str
) -> tuple[str, str, StereoMatcherConfig, float]:
    experiment_config = config["experiments"][experiment]
    algorithm = str(experiment_config["algorithm"])
    parameter_source = str(experiment_config["parameter_source"])
    values = dict(experiment_config["parameters"])
    lr_threshold = float(values.pop("lr_threshold_px"))
    return (
        algorithm,
        parameter_source,
        StereoMatcherConfig(method=algorithm, **values),
        lr_threshold,
    )


def scene_split(config: dict, scene_name: str) -> str:
    if scene_name in config["split"]["development_scenes"]:
        return "development"
    if scene_name in config["split"]["evaluation_scenes"]:
        return "held_out"
    raise ValueError(f"Scene is absent from the frozen split: {scene_name}")


def evaluate_scene_method(
    scene: MiddleburyScene,
    *,
    experiment: str,
    config: dict,
    output_dir: Path,
) -> dict[str, object]:
    algorithm, parameter_source, match_config, lr_threshold = matcher_config(
        config, experiment
    )
    result = match_stereo(
        scene.left_rgb,
        scene.right_rgb,
        calibration_num_disparities=scene.calibration.num_disparities,
        config=match_config,
        lr_threshold_px=lr_threshold,
    )
    calibration = scene.calibration
    predicted_depth, predicted_geometry_valid = disparity_to_depth(
        result.left_disparity_px,
        focal_length_px=calibration.focal_length_px,
        baseline=calibration.baseline_m,
        disparity_offset_px=calibration.doffs_px,
    )
    target_depth, target_geometry_valid = disparity_to_depth(
        scene.gt_disparity_px,
        focal_length_px=calibration.focal_length_px,
        baseline=calibration.baseline_m,
        disparity_offset_px=calibration.doffs_px,
    )
    prediction_valid = result.valid_mask & predicted_geometry_valid
    evaluation_mask = scene.non_occluded_mask & target_geometry_valid
    predicted_depth = predicted_depth.astype(np.float32)
    predicted_depth[~prediction_valid] = np.nan
    target_depth = target_depth.astype(np.float32)
    metadata = {
        "scene": scene.name,
        "experiment": experiment,
        "algorithm": algorithm,
        "parameter_source": parameter_source,
        "depth_type": "metric",
        "unit": "meter",
        "alignment": "none",
        "invalid_representation": "nan_with_boolean_mask",
        "focal_length_px": calibration.focal_length_px,
        "baseline_m": calibration.baseline_m,
        "doffs_px": calibration.doffs_px,
        "opencv_fixed_point_scale": 16.0,
        "num_disparities": result.num_disparities,
        "lr_threshold_px": lr_threshold,
        "matcher_parameters": config["experiments"][experiment]["parameters"],
    }
    validate_depth_output(predicted_depth, metadata, valid_mask=prediction_valid)
    disparity_metrics = compute_disparity_metrics(
        result.left_disparity_px,
        scene.gt_disparity_px,
        evaluation_mask=evaluation_mask,
    )
    depth_metrics = compute_stereo_depth_metrics(
        predicted_depth,
        target_depth,
        evaluation_mask=evaluation_mask,
        depth_bin_edges_m=config["evaluation"]["depth_bin_edges_m"],
    )
    method_dir = output_dir / scene.name / experiment
    method_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        method_dir / "prediction.npz",
        disparity_px=result.left_disparity_px.astype(np.float32),
        raw_disparity_px=result.raw_left_disparity_px.astype(np.float32),
        right_disparity_px=result.right_disparity_px.astype(np.float32),
        depth_m=predicted_depth,
        valid_mask=prediction_valid,
        raw_left_valid=result.raw_left_valid,
        lr_consistent=result.lr_consistent,
    )
    (method_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    save_stereo_panel(
        method_dir / "visualization.png",
        scene_name=scene.name,
        method_name=experiment.upper(),
        rgb=scene.left_rgb,
        target_disparity_px=scene.gt_disparity_px,
        prediction_disparity_px=result.left_disparity_px,
        evaluation_mask=evaluation_mask,
        raw_prediction_valid=result.raw_left_valid,
        lr_consistent=result.lr_consistent,
        disparity_max_px=float(result.num_disparities),
    )
    return {
        "scene": scene.name,
        "split": scene_split(config, scene.name),
        "experiment": experiment,
        "algorithm": algorithm,
        "parameter_source": parameter_source,
        "runtime_seconds": result.runtime_seconds,
        **disparity_metrics,
        **depth_metrics,
    }


def macro_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = [
        "runtime_seconds",
        "disparity_mae_px",
        "disparity_rmse_px",
        "bad1",
        "bad2",
        "bad4",
        "valid_coverage",
        "depth_abs_rel",
        "depth_rmse_m",
        "depth_valid_coverage",
        "depth_near_abs_rel",
        "depth_near_rmse_m",
        "depth_mid_abs_rel",
        "depth_mid_rmse_m",
        "depth_far_abs_rel",
        "depth_far_rmse_m",
    ]
    summaries: dict[str, object] = {}
    for experiment in sorted({str(row["experiment"]) for row in rows}):
        method_rows = [row for row in rows if row["experiment"] == experiment]
        summaries[experiment] = {
            "algorithm": method_rows[0]["algorithm"],
            "parameter_source": method_rows[0]["parameter_source"],
        }
        for split in ("development", "held_out", "all"):
            selected = (
                method_rows
                if split == "all"
                else [row for row in method_rows if row["split"] == split]
            )
            values: dict[str, float | int | None] = {"scene_count": len(selected)}
            for metric in metrics:
                available = [float(row[metric]) for row in selected if row.get(metric) is not None]
                values[metric] = float(np.mean(available)) if available else None
            summaries[experiment][split] = values
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_saved_outputs(
    *,
    output_dir: Path,
    scenes: list[MiddleburyScene],
    rows: list[dict[str, object]],
    config: dict,
) -> dict[str, object]:
    """Reload saved arrays and require recomputed metrics to match in-memory values."""

    scene_by_name = {scene.name: scene for scene in scenes}
    checked_values = 0
    for row in rows:
        scene = scene_by_name[str(row["scene"])]
        experiment = str(row["experiment"])
        with np.load(output_dir / scene.name / experiment / "prediction.npz") as archive:
            disparity = archive["disparity_px"]
            depth = archive["depth_m"]
            saved_valid = archive["valid_mask"]
        calibration = scene.calibration
        target_depth, target_valid = disparity_to_depth(
            scene.gt_disparity_px,
            focal_length_px=calibration.focal_length_px,
            baseline=calibration.baseline_m,
            disparity_offset_px=calibration.doffs_px,
        )
        target_depth = target_depth.astype(np.float32)
        evaluation_mask = scene.non_occluded_mask & target_valid
        metadata = json.loads(
            (output_dir / scene.name / experiment / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        validate_depth_output(depth, metadata, valid_mask=saved_valid)
        recomputed = {
            **compute_disparity_metrics(
                disparity,
                scene.gt_disparity_px,
                evaluation_mask=evaluation_mask,
            ),
            **compute_stereo_depth_metrics(
                depth,
                target_depth,
                evaluation_mask=evaluation_mask,
                depth_bin_edges_m=config["evaluation"]["depth_bin_edges_m"],
            ),
        }
        for key, value in recomputed.items():
            expected = row[key]
            if value is None or expected is None:
                if value is not None or expected is not None:
                    raise AssertionError(
                        f"Reload mismatch for {scene.name}/{experiment}/{key}"
                    )
            elif isinstance(value, int):
                if int(expected) != value:
                    raise AssertionError(
                        f"Reload mismatch for {scene.name}/{experiment}/{key}"
                    )
            else:
                np.testing.assert_allclose(
                    float(value),
                    float(expected),
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"Reload mismatch for {scene.name}/{experiment}/{key}",
                )
            checked_values += 1
    return {
        "passed": True,
        "prediction_files_checked": len(rows),
        "metric_values_checked": checked_values,
        "relative_tolerance": 1e-6,
        "absolute_tolerance": 1e-8,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/part1/middlebury_sgbm.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/part1/middlebury/frozen_v2",
    )
    parser.add_argument("--experiments", nargs="+")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["protocol_status"] != "frozen_v2":
        raise ValueError(
            "Refusing the 15-scene run until protocol_status is frozen_v2 after dev-only tuning"
        )
    experiments = args.experiments or list(config["experiments"])
    unknown = sorted(set(experiments) - set(config["experiments"]))
    if unknown:
        raise ValueError(f"Unknown experiment ids: {unknown}")
    dataset_root = PROJECT_ROOT / config["dataset"]["root"]
    scene_names = config["split"]["development_scenes"] + config["split"]["evaluation_scenes"]
    scenes = [load_middlebury_scene(dataset_root / name) for name in scene_names]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for scene_index, scene in enumerate(scenes, start=1):
        for experiment in experiments:
            row = evaluate_scene_method(
                scene,
                experiment=experiment,
                config=config,
                output_dir=args.output_dir,
            )
            rows.append(row)
            print(
                f"[{scene_index:02d}/{len(scenes)}] {scene.name}/{experiment}: "
                f"MAE={row['disparity_mae_px']:.3f}px, Bad-2={row['bad2']:.3f}, "
                f"coverage={row['valid_coverage']:.3f}",
                flush=True,
            )
    write_csv(args.output_dir / "metrics_per_scene.csv", rows)
    summaries = macro_summary(rows)
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    sensitivity_rows = analyze_depth_sensitivity(
        scenes,
        disparity_errors_px=config["sensitivity"]["disparity_errors_px"],
        depth_bin_width_m=float(config["sensitivity"]["depth_bin_width_m"]),
    )
    write_csv(args.output_dir / "sensitivity.csv", sensitivity_rows)
    save_sensitivity_plot(args.output_dir / "sensitivity.png", sensitivity_rows)
    empirical_rows: list[dict[str, object]] = []
    for experiment in experiments:
        predictions = {}
        for scene in scenes:
            with np.load(
                args.output_dir / scene.name / experiment / "prediction.npz"
            ) as archive:
                predictions[scene.name] = archive["disparity_px"].copy()
        experiment_rows = analyze_prediction_error_by_distance(
            scenes,
            predictions,
            depth_bin_width_m=float(config["sensitivity"]["depth_bin_width_m"]),
        )
        empirical_rows.extend(
            {"experiment": experiment, **row} for row in experiment_rows
        )
    write_csv(args.output_dir / "empirical_depth_error_by_distance.csv", empirical_rows)
    save_empirical_depth_error_plot(
        args.output_dir / "empirical_depth_error_by_distance.png", empirical_rows
    )
    for experiment in experiments:
        save_contact_sheet(
            args.output_dir / f"visualization_contact_sheet_{experiment}.jpg",
            [
                args.output_dir / scene.name / experiment / "visualization.png"
                for scene in scenes
            ],
        )
    zoom_config = config["visualization"]["failure_zoom"]
    zoom_experiment = str(zoom_config["experiment"])
    if zoom_experiment in experiments:
        zoom_scene = next(scene for scene in scenes if scene.name == zoom_config["scene"])
        with np.load(
            args.output_dir / zoom_scene.name / zoom_experiment / "prediction.npz"
        ) as archive:
            zoom_prediction = archive["disparity_px"]
        save_stereo_failure_zoom(
            args.output_dir / f"failure_zoom_{zoom_scene.name}_{zoom_experiment}.png",
            scene_name=zoom_scene.name,
            method_name=zoom_experiment.upper(),
            rgb=zoom_scene.left_rgb,
            target_disparity_px=zoom_scene.gt_disparity_px,
            prediction_disparity_px=zoom_prediction,
            evaluation_mask=zoom_scene.non_occluded_mask,
            crop_xyxy=tuple(int(value) for value in zoom_config["crop_xyxy"]),
            disparity_max_px=float(zoom_scene.calibration.num_disparities),
        )
    reload_validation = validate_saved_outputs(
        output_dir=args.output_dir,
        scenes=scenes,
        rows=rows,
        config=config,
    )
    (args.output_dir / "reload_validation.json").write_text(
        json.dumps(reload_validation, indent=2) + "\n", encoding="utf-8"
    )
    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "protocol_status": config["protocol_status"],
        "experiments": experiments,
        "scene_count": len(scenes),
        "formal_training_run": False,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
