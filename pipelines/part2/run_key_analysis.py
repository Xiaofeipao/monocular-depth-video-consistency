#!/usr/bin/env python3
"""Joint four-part analysis for frozen Depth Pro and UniDepthV2-S outputs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import binary_dilation
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.part2.run_depth_pro_nyuv2 import (
    DEPTH_METRICS,
    bootstrap_mean_ci,
    prediction_stem,
    resize_prediction,
    sha256,
    write_csv,
)
from src.data.nyuv2 import NYUv2LabeledDataset
from src.metrics.boundary import compute_depth_boundary_metrics, depth_edge_map
from src.metrics.depth import build_valid_mask, compute_depth_metrics
from src.models.unidepth_v2_adapter import UniDepthV2Adapter
from src.visualization.nyuv2 import save_contact_sheet


MODEL_LABELS = {
    "depth_pro": "Depth Pro",
    "unidepth_v2": "UniDepthV2-S",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def crop_slices(config: dict) -> tuple[slice, slice]:
    crop = config["evaluation"]["crop"]
    return slice(int(crop["top"]), int(crop["bottom"])), slice(
        int(crop["left"]), int(crop["right"])
    )


def resolve_and_verify(relative_path: str, expected_sha256: str) -> Path:
    path = PROJECT_ROOT / relative_path
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected_sha256}")
    return path


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["protocol_status"] != "analysis_v1":
        raise ValueError("Expected analysis_v1 protocol")
    if config["evaluation"]["test_time_alignment"] != "none":
        raise ValueError("Primary analysis must preserve alignment:none")
    if config["cross_domain_analysis"]["add_another_target_domain"]:
        raise ValueError("This run is intentionally restricted to NYUv2")
    return config


def reference_intrinsics(config: dict) -> np.ndarray:
    matrix = np.asarray(
        config["camera_analysis"]["reference_intrinsics_full_480x640_px"],
        dtype=np.float32,
    )
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Reference intrinsics must be a finite 3x3 matrix")
    return matrix


def validate_baseline_root(root: Path, expected_count: int) -> None:
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    if (
        int(metadata["sample_count"]) != expected_count
        or not metadata["full_test_split"]
        or metadata["test_time_alignment"] != "none"
        or metadata["formal_training_run"]
    ):
        raise ValueError(f"Baseline run metadata is not frozen/complete: {root}")
    config_path = Path(metadata["config"])
    if sha256(config_path) != metadata["config_sha256"]:
        raise ValueError(f"Baseline config changed after inference: {config_path}")
    if len(list((root / "predictions").glob("*.npz"))) != expected_count:
        raise ValueError(f"Baseline prediction count is not {expected_count}: {root}")


def infer_unidepth_with_reference_intrinsics(
    *,
    config: dict,
    dataset: NYUv2LabeledDataset,
    output_root: Path,
    device: str,
    resume: bool,
) -> dict[str, object]:
    track_root = output_root / "camera_ablation" / "unidepth_external_intrinsics"
    predictions_dir = track_root / "predictions"
    metadata_dir = track_root / "metadata"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    model_config = resolve_and_verify(
        config["unidepth_model"]["config"],
        config["unidepth_model"]["config_sha256"],
    )
    checkpoint = resolve_and_verify(
        config["unidepth_model"]["checkpoint"],
        config["unidepth_model"]["checkpoint_sha256"],
    )
    if model_config.parent != checkpoint.parent:
        raise ValueError("UniDepth config and checkpoint must share a directory")
    intrinsics = reference_intrinsics(config)
    adapter = UniDepthV2Adapter(
        repository=PROJECT_ROOT / config["unidepth_model"]["repository"],
        checkpoint_dir=checkpoint.parent,
        device=device,
    )
    warmup = dataset.get_test_sample(0)
    adapter.infer(warmup.rgb, intrinsics_px=intrinsics)

    inferred = skipped = 0
    runtimes: list[float] = []
    peak_memory = 0
    start = time.perf_counter()
    for position in range(len(dataset)):
        sample = dataset.get_test_sample(position)
        stem = prediction_stem(position, sample.matlab_index_1based)
        prediction_path = predictions_dir / f"{stem}.npz"
        metadata_path = metadata_dir / f"{stem}.json"
        if prediction_path.exists() or metadata_path.exists():
            if resume and prediction_path.exists() and metadata_path.exists():
                skipped += 1
                continue
            raise FileExistsError(f"Existing/incomplete external-camera output: {stem}")
        depth, effective_intrinsics, runtime, memory = adapter.infer(
            sample.rgb, intrinsics_px=intrinsics
        )
        depth = resize_prediction(depth, sample.depth_m.shape)
        if not np.allclose(effective_intrinsics, intrinsics, rtol=1e-4, atol=1e-3):
            raise AssertionError(f"Effective external intrinsics changed for {stem}")
        if depth.dtype != np.float32 or not np.all(np.isfinite(depth)) or np.any(depth <= 0):
            raise ValueError(f"Invalid external-camera metric depth for {stem}")
        np.savez_compressed(
            prediction_path,
            depth_m=depth,
            intrinsics_px=effective_intrinsics.astype(np.float32),
        )
        metadata = {
            "sample_id": stem,
            "test_position": position,
            "matlab_index_1based": sample.matlab_index_1based,
            "scene": sample.scene,
            "scene_type": sample.scene_type,
            "model": "UniDepthV2-S (ViT-S/14)",
            "depth_type": "metric",
            "unit": "meter",
            "alignment": "none",
            "intrinsics_source": "nyuv2_calibration_extra_camera_information",
            "intrinsics_px": effective_intrinsics.tolist(),
            "external_intrinsics_provided": True,
            "input_shape": list(sample.rgb.shape[:2]),
            "output_shape": list(depth.shape),
            "inference_seconds": runtime,
            "peak_gpu_memory_bytes": memory,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        runtimes.append(runtime)
        peak_memory = max(peak_memory, memory)
        inferred += 1
        if (position + 1) % 25 == 0 or position in {0, len(dataset) - 1}:
            print(
                f"[{position + 1:03d}/{len(dataset)}] external-K {stem}: "
                f"{runtime:.3f}s, peak={memory / 2**30:.2f}GiB",
                flush=True,
            )
    wall_seconds = time.perf_counter() - start
    summary = {
        "formal_training_run": False,
        "sample_count": len(dataset),
        "inferred_this_invocation": inferred,
        "resumed": skipped,
        "wall_seconds_this_invocation": wall_seconds,
        "mean_inference_seconds_this_invocation": (
            float(np.mean(runtimes)) if runtimes else None
        ),
        "peak_gpu_memory_bytes_this_invocation": peak_memory if runtimes else None,
        "external_intrinsics_px": intrinsics.tolist(),
        "environment": {
            **adapter.environment(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "model_load_seconds": adapter.load_seconds,
        },
    }
    (track_root / "inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def load_prediction(root: Path, stem: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(root / "predictions" / f"{stem}.npz") as archive:
        depth = np.asarray(archive["depth_m"], dtype=np.float32).copy()
        extras = {
            key: np.asarray(archive[key]).copy()
            for key in archive.files
            if key != "depth_m"
        }
    return depth, extras


def mean_ci(values: list[float], config: dict, seed_offset: int = 0) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    stats = config["statistics"]
    lower, upper = bootstrap_mean_ci(
        array,
        resamples=int(stats["bootstrap_resamples"]),
        confidence_level=float(stats["confidence_level"]),
        seed=int(stats["seed"]) + seed_offset,
    )
    return {"mean": float(array.mean()), "lower": lower, "upper": upper}


def correlation(x: list[float], y: list[float]) -> dict[str, float]:
    result = spearmanr(np.asarray(x), np.asarray(y))
    return {"spearman_rho": float(result.statistic), "p_value": float(result.pvalue)}


def edge_region_abs_rel(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    target_edges: np.ndarray,
    band_pixels: int,
) -> tuple[float, float, int, int]:
    size = 2 * band_pixels + 1
    edge_band = binary_dilation(target_edges, structure=np.ones((size, size), bool)) & valid
    nonedge = valid & ~edge_band
    relative_error = np.abs(prediction - target) / target
    return (
        float(relative_error[edge_band].mean()),
        float(relative_error[nonedge].mean()),
        int(edge_band.sum()),
        int(nonedge.sum()),
    )


def save_camera_plots(
    output_root: Path,
    per_image: list[dict[str, object]],
    camera_summary: dict[str, object],
) -> None:
    tracks = [
        ("dp_native", "Depth Pro\npredicted f"),
        ("dp_calibrated", "Depth Pro\ncalibrated f"),
        ("ud_native", "UniDepthV2-S\npredicted K"),
        ("ud_calibrated", "UniDepthV2-S\ncalibrated K"),
    ]
    metrics = [("abs_rel", "AbsRel ↓"), ("rmse", "RMSE (m) ↓"), ("delta1", "δ1 ↑")]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        values = [
            float(np.mean([float(row[f"{prefix}_{metric}"]) for row in per_image]))
            for prefix, _ in tracks
        ]
        axis.bar(range(len(tracks)), values, color=["#4477AA", "#66CCEE", "#CC6677", "#EE99AA"])
        axis.set_xticks(range(len(tracks)), [label for _, label in tracks], fontsize=8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Camera information ablation (external-camera tracks are diagnostic)")
    path = output_root / "camera_ablation" / "camera_ablation_metrics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    definitions = [
        ("dp_focal_abs_relative_error", "dp_abs_log_scale_error", "Depth Pro"),
        ("ud_focal_abs_relative_error", "ud_abs_log_scale_error", "UniDepthV2-S"),
    ]
    for axis, (x_key, y_key, title) in zip(axes, definitions):
        x = np.asarray([float(row[x_key]) for row in per_image])
        y = np.asarray([float(row[y_key]) for row in per_image])
        axis.scatter(x, y, s=8, alpha=0.35)
        rho = camera_summary["correlations"][title]["camera_error_vs_abs_log_scale_error"]["spearman_rho"]
        axis.set_xlabel("absolute focal error (relative)")
        axis.set_ylabel("absolute log scale error")
        axis.set_title(f"{title}: Spearman ρ={rho:.3f}")
        axis.grid(alpha=0.25)
    figure.savefig(output_root / "camera_ablation" / "camera_error_vs_scale.png", dpi=180)
    plt.close(figure)


def save_scale_plots(
    output_root: Path,
    per_image: list[dict[str, object]],
    depth_bin_summary: list[dict[str, object]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    bins = np.linspace(0.5, 1.5, 41)
    for key, label, color in (
        ("dp_scale_ratio", "Depth Pro", "#4477AA"),
        ("ud_scale_ratio", "UniDepthV2-S", "#CC6677"),
    ):
        values = np.asarray([float(row[key]) for row in per_image])
        axes[0].hist(values, bins=bins, alpha=0.55, label=label, color=color)
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("per-image median prediction / GT")
    axes[0].set_ylabel("image count")
    axes[0].set_title("Native metric-scale ratio")
    axes[0].legend()

    for method, label, color in (
        ("depth_pro", "Depth Pro", "#4477AA"),
        ("unidepth_v2", "UniDepthV2-S", "#CC6677"),
    ):
        selected = [row for row in depth_bin_summary if row["method"] == method]
        centers = [(float(row["lower_m"]) + float(row["upper_m"])) / 2 for row in selected]
        values = [float(row["abs_rel_macro"]) for row in selected]
        axes[1].plot(centers, values, marker="o", label=label, color=color)
    axes[1].set_xlabel("GT depth-bin center (m)")
    axes[1].set_ylabel("macro AbsRel")
    axes[1].set_title("Absolute-scale error versus distance")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output_root / "absolute_scale" / "scale_and_depth_bins.png", dpi=180)
    plt.close(figure)


def save_boundary_plots(
    output_root: Path,
    boundary_summary: list[dict[str, object]],
    region_summary: dict[str, dict[str, float]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for method, label, color in (
        ("depth_pro", "Depth Pro", "#4477AA"),
        ("unidepth_v2", "UniDepthV2-S", "#CC6677"),
    ):
        selected = [row for row in boundary_summary if row["method"] == method]
        axes[0].plot(
            [int(row["tolerance_px"]) for row in selected],
            [float(row["f1_mean"]) for row in selected],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xlabel("matching tolerance (px)")
    axes[0].set_ylabel("macro depth-edge F1")
    axes[0].set_title("Section 6.2 depth-edge metric")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    x = np.arange(2)
    width = 0.34
    axes[1].bar(
        x - width / 2,
        [region_summary[m]["edge_abs_rel_mean"] for m in ("depth_pro", "unidepth_v2")],
        width,
        label="within 3 px of GT edge",
        color="#AA3377",
    )
    axes[1].bar(
        x + width / 2,
        [region_summary[m]["nonedge_abs_rel_mean"] for m in ("depth_pro", "unidepth_v2")],
        width,
        label="non-edge",
        color="#66CCEE",
    )
    axes[1].set_xticks(x, ["Depth Pro", "UniDepthV2-S"])
    axes[1].set_ylabel("macro AbsRel")
    axes[1].set_title("Error near depth discontinuities")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(output_root / "boundary" / "boundary_quantitative.png", dpi=180)
    plt.close(figure)


def save_cross_domain_plot(output_root: Path, category_rows: list[dict[str, object]]) -> None:
    categories = sorted(
        {str(row["scene_type"]) for row in category_rows},
        key=lambda category: -sum(
            int(row["image_count"]) for row in category_rows if row["scene_type"] == category
        ),
    )
    y = np.arange(len(categories))
    figure, axis = plt.subplots(
        figsize=(10, max(6, 0.32 * len(categories))), constrained_layout=True
    )
    for offset, method, label, color in (
        (-0.12, "depth_pro", "Depth Pro", "#4477AA"),
        (0.12, "unidepth_v2", "UniDepthV2-S", "#CC6677"),
    ):
        lookup = {
            str(row["scene_type"]): float(row["abs_rel"])
            for row in category_rows
            if row["method"] == method
        }
        axis.barh(y + offset, [lookup[c] for c in categories], height=0.23, label=label, color=color)
    sample_counts = {
        str(row["scene_type"]): int(row["image_count"])
        for row in category_rows
        if row["method"] == "depth_pro"
    }
    axis.set_yticks(
        y,
        [f"{category} (n={sample_counts[category]})" for category in categories],
        fontsize=8,
    )
    axis.invert_yaxis()
    axis.set_xlabel("macro AbsRel")
    axis.set_title("NYUv2 target-domain robustness by scene type")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.savefig(output_root / "cross_domain" / "nyuv2_scene_type_absrel.png", dpi=180)
    plt.close(figure)


def save_boundary_case_panel(
    path: Path,
    *,
    rgb: np.ndarray,
    target: np.ndarray,
    depth_pro: np.ndarray,
    unidepth: np.ndarray,
    log_threshold: float,
    title: str,
) -> None:
    gt_edges = depth_edge_map(target, log_threshold=log_threshold)
    dp_edges = depth_edge_map(depth_pro, log_threshold=log_threshold)
    ud_edges = depth_edge_map(unidepth, log_threshold=log_threshold)
    difference = np.abs(depth_pro - target) - np.abs(unidepth - target)
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    depth_image = axes[0, 1].imshow(target, cmap="turbo_r", vmin=0.5, vmax=10.0)
    axes[0, 1].set_title("GT depth (m)")
    axes[0, 2].imshow(depth_pro, cmap="turbo_r", vmin=0.5, vmax=10.0)
    axes[0, 2].set_title("Depth Pro")
    axes[0, 3].imshow(unidepth, cmap="turbo_r", vmin=0.5, vmax=10.0)
    axes[0, 3].set_title("UniDepthV2-S")
    axes[1, 0].imshow(gt_edges, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("GT depth edges")
    axes[1, 1].imshow(dp_edges, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Depth Pro edges")
    axes[1, 2].imshow(ud_edges, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("UniDepthV2-S edges")
    diff_image = axes[1, 3].imshow(difference, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes[1, 3].set_title("|err DP| − |err UD| (m)")
    for axis in axes.flat:
        axis.axis("off")
    figure.colorbar(depth_image, ax=axes[0, 1:4], shrink=0.7)
    figure.colorbar(diff_image, ax=axes[1, 3], shrink=0.7)
    figure.suptitle(title, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def select_boundary_cases(
    per_image: list[dict[str, object]], config: dict
) -> list[dict[str, object]]:
    requested = config["boundary_analysis"]["qualitative_selection"]
    selected: list[dict[str, object]] = []
    used: set[int] = set()
    rankings = [
        (
            "depth_pro_advantage",
            sorted(
                per_image,
                key=lambda row: float(row["dp_si_boundary_f1"])
                - float(row["ud_si_boundary_f1"]),
                reverse=True,
            ),
        ),
        (
            "unidepth_v2_advantage",
            sorted(
                per_image,
                key=lambda row: float(row["ud_si_boundary_f1"])
                - float(row["dp_si_boundary_f1"]),
                reverse=True,
            ),
        ),
        (
            "both_low_si_f1",
            sorted(
                per_image,
                key=lambda row: (
                    float(row["dp_si_boundary_f1"])
                    + float(row["ud_si_boundary_f1"])
                )
                / 2,
            ),
        ),
    ]
    for reason, ranking in rankings:
        needed = int(requested[reason])
        added = 0
        for row in ranking:
            position = int(row["test_position"])
            if position in used:
                continue
            selected.append(
                {
                    "selection_reason": reason,
                    "test_position": position,
                    "sample_id": row["sample_id"],
                    "scene": row["scene"],
                    "scene_type": row["scene_type"],
                    "depth_pro_si_boundary_f1": float(row["dp_si_boundary_f1"]),
                    "unidepth_v2_si_boundary_f1": float(row["ud_si_boundary_f1"]),
                }
            )
            used.add(position)
            added += 1
            if added == needed:
                break
    return selected


def analyze(config: dict, dataset: NYUv2LabeledDataset, output_root: Path) -> dict[str, object]:
    expected_count = int(config["dataset"]["expected_test_images"])
    baseline_roots = {
        method: PROJECT_ROOT / path
        for method, path in config["baseline_outputs"].items()
    }
    for root in baseline_roots.values():
        validate_baseline_root(root, expected_count)
    external_root = output_root / "camera_ablation" / "unidepth_external_intrinsics"
    if len(list((external_root / "predictions").glob("*.npz"))) != expected_count:
        raise ValueError("UniDepth external-intrinsics track is incomplete")

    primary_rows = {
        method: {
            int(row["test_position"]): row
            for row in read_csv(root / "metrics_per_image.csv")
        }
        for method, root in baseline_roots.items()
    }
    crop = crop_slices(config)
    minimum = float(config["evaluation"]["valid_depth_m"]["minimum"])
    maximum = float(config["evaluation"]["valid_depth_m"]["maximum"])
    intrinsics = reference_intrinsics(config)
    reference_focal = float(math.sqrt(float(intrinsics[0, 0] * intrinsics[1, 1])))
    log_threshold = float(config["boundary_analysis"]["edge_log_threshold"])
    tolerances = [int(value) for value in config["boundary_analysis"]["tolerance_pixels"]]
    edge_band = int(config["boundary_analysis"]["edge_error_band_pixels"])
    scale_lower, scale_upper = (
        float(value) for value in config["absolute_scale_analysis"]["within_scale_ratio"]
    )
    depth_bins = [
        (float(lower), float(upper))
        for lower, upper in config["absolute_scale_analysis"]["depth_bins_m"]
    ]

    per_image: list[dict[str, object]] = []
    boundary_rows: list[dict[str, object]] = []
    depth_bin_rows: list[dict[str, object]] = []
    dp_raw_min = math.inf
    dp_raw_max = -math.inf
    dp_official_lower_clamp_count = 0
    dp_official_upper_clamp_count = 0
    dp_raw_value_count = 0
    for position in range(len(dataset)):
        sample = dataset.get_test_sample(position)
        stem = prediction_stem(position, sample.matlab_index_1based)
        dp_depth, dp_extra = load_prediction(baseline_roots["depth_pro"], stem)
        ud_depth, ud_extra = load_prediction(baseline_roots["unidepth_v2"], stem)
        ud_calibrated_depth, ud_calibrated_extra = load_prediction(external_root, stem)
        dp_focal = float(np.asarray(dp_extra["focal_length_px"]).item())
        dp_raw_min = min(dp_raw_min, float(np.min(dp_depth)))
        dp_raw_max = max(dp_raw_max, float(np.max(dp_depth)))
        dp_official_lower_clamp_count += int(np.count_nonzero(dp_depth <= 1e-4))
        dp_official_upper_clamp_count += int(np.count_nonzero(dp_depth >= 1e4))
        dp_raw_value_count += int(dp_depth.size)
        ud_intrinsics = np.asarray(ud_extra["intrinsics_px"], dtype=np.float64)
        effective_calibrated = np.asarray(
            ud_calibrated_extra["intrinsics_px"], dtype=np.float64
        )
        if not np.allclose(effective_calibrated, intrinsics, rtol=1e-4, atol=1e-3):
            raise AssertionError(f"Unexpected calibrated K in {stem}")
        # This is the exact non-saturated relation in Depth Pro's official
        # infer equation: inverse_depth = canonical_inverse_depth * W / f.
        dp_calibrated_depth = dp_depth * np.float32(reference_focal / dp_focal)

        target = sample.depth_m[crop]
        tracks = {
            "dp_native": dp_depth[crop],
            "dp_calibrated": dp_calibrated_depth[crop],
            "ud_native": ud_depth[crop],
            "ud_calibrated": ud_calibrated_depth[crop],
        }
        row: dict[str, object] = {
            "sample_id": stem,
            "test_position": position,
            "matlab_index_1based": sample.matlab_index_1based,
            "scene": sample.scene,
            "scene_type": sample.scene_type,
        }
        for prefix, prediction in tracks.items():
            metrics = compute_depth_metrics(
                prediction,
                target,
                min_depth=minimum,
                max_depth=maximum,
                clip_prediction=True,
            )
            for metric in DEPTH_METRICS:
                row[f"{prefix}_{metric}"] = float(metrics[metric])

        for short, prediction, intrinsics_pred in (
            ("dp", tracks["dp_native"], None),
            ("ud", tracks["ud_native"], ud_intrinsics),
        ):
            valid = build_valid_mask(
                prediction, target, min_depth=minimum, max_depth=maximum
            )
            clipped = np.clip(prediction, minimum, maximum)
            ratio = clipped[valid] / target[valid]
            scale_ratio = float(np.median(ratio))
            row[f"{short}_scale_ratio"] = scale_ratio
            row[f"{short}_log_scale_bias"] = float(np.log(scale_ratio))
            row[f"{short}_abs_log_scale_error"] = abs(float(np.log(scale_ratio)))
            row[f"{short}_signed_relative_bias"] = float(np.mean(ratio - 1.0))
            row[f"{short}_scale_class"] = (
                "under" if scale_ratio < scale_lower else "over" if scale_ratio > scale_upper else "within_10pct"
            )
            oracle_metrics = compute_depth_metrics(
                prediction / scale_ratio,
                target,
                min_depth=minimum,
                max_depth=maximum,
                clip_prediction=True,
            )
            for metric in DEPTH_METRICS:
                row[f"{short}_oracle_scale_aligned_{metric}"] = float(
                    oracle_metrics[metric]
                )

            target_edges = depth_edge_map(
                target, valid_mask=valid, log_threshold=log_threshold
            )
            edge_absrel, nonedge_absrel, edge_pixels, nonedge_pixels = edge_region_abs_rel(
                clipped, target, valid, target_edges, edge_band
            )
            row[f"{short}_edge_band_abs_rel"] = edge_absrel
            row[f"{short}_nonedge_abs_rel"] = nonedge_absrel
            row[f"{short}_edge_band_pixels"] = edge_pixels
            row[f"{short}_nonedge_pixels"] = nonedge_pixels
            method = "depth_pro" if short == "dp" else "unidepth_v2"
            for tolerance in tolerances:
                metrics = compute_depth_boundary_metrics(
                    clipped,
                    target,
                    valid_mask=valid,
                    log_threshold=log_threshold,
                    tolerance_px=tolerance,
                )
                boundary_rows.append(
                    {
                        "sample_id": stem,
                        "test_position": position,
                        "scene_type": sample.scene_type,
                        "method": method,
                        "tolerance_px": tolerance,
                        **metrics,
                    }
                )
            for lower, upper in depth_bins:
                bin_mask = valid & (target >= lower) & (target < upper)
                if not np.any(bin_mask):
                    continue
                difference = clipped[bin_mask] - target[bin_mask]
                depth_bin_rows.append(
                    {
                        "sample_id": stem,
                        "test_position": position,
                        "method": method,
                        "lower_m": lower,
                        "upper_m": upper,
                        "pixel_count": int(bin_mask.sum()),
                        "abs_rel": float(
                            np.mean(np.abs(difference) / target[bin_mask])
                        ),
                        "signed_relative_bias": float(
                            np.mean(difference / target[bin_mask])
                        ),
                        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                    }
                )

            if intrinsics_pred is None:
                focal_error = dp_focal / reference_focal - 1.0
                row["dp_predicted_focal_px"] = dp_focal
                row["dp_focal_relative_error"] = focal_error
                row["dp_focal_abs_relative_error"] = abs(focal_error)
            else:
                predicted_focal = float(
                    math.sqrt(float(intrinsics_pred[0, 0] * intrinsics_pred[1, 1]))
                )
                focal_error = predicted_focal / reference_focal - 1.0
                row["ud_predicted_fx_px"] = float(intrinsics_pred[0, 0])
                row["ud_predicted_fy_px"] = float(intrinsics_pred[1, 1])
                row["ud_predicted_cx_px"] = float(intrinsics_pred[0, 2])
                row["ud_predicted_cy_px"] = float(intrinsics_pred[1, 2])
                row["ud_predicted_geometric_focal_px"] = predicted_focal
                row["ud_focal_relative_error"] = focal_error
                row["ud_focal_abs_relative_error"] = abs(focal_error)
                diagonal = math.hypot(sample.rgb.shape[1], sample.rgb.shape[0])
                row["ud_principal_point_error_px"] = float(
                    math.hypot(
                        float(intrinsics_pred[0, 2] - intrinsics[0, 2]),
                        float(intrinsics_pred[1, 2] - intrinsics[1, 2]),
                    )
                )
                row["ud_principal_point_error_normalized_diagonal"] = float(
                    row["ud_principal_point_error_px"] / diagonal
                )

        row["dp_si_boundary_f1"] = float(primary_rows["depth_pro"][position]["si_boundary_f1"])
        row["ud_si_boundary_f1"] = float(primary_rows["unidepth_v2"][position]["si_boundary_f1"])
        per_image.append(row)
        if (position + 1) % 50 == 0 or position == len(dataset) - 1:
            print(f"[{position + 1:03d}/{len(dataset)}] four-part analysis", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "per_image_analysis.csv", per_image)
    write_csv(output_root / "boundary" / "boundary_per_image.csv", boundary_rows)
    write_csv(output_root / "absolute_scale" / "depth_bin_per_image.csv", depth_bin_rows)

    camera_summary: dict[str, object] = {
        "reference_intrinsics_px": intrinsics.tolist(),
        "depth_pro_reference_scalar_focal_px": reference_focal,
        "primary_tracks": "model-predicted camera; no extra camera information",
        "external_tracks": "NYUv2 calibrated camera; diagnostic extra information",
        "depth_pro_external_focal_derivation": {
            "equation": "depth_external = depth_predicted * focal_external / focal_predicted",
            "source": "official infer equation inverse_depth = canonical_inverse_depth * W / focal",
            "uses_model_rerun": False,
            "official_depth_clamp_m": [1e-4, 1e4],
            "saved_raw_depth_min_m": dp_raw_min,
            "saved_raw_depth_max_m": dp_raw_max,
            "saved_raw_value_count": dp_raw_value_count,
            "lower_clamp_value_count": dp_official_lower_clamp_count,
            "upper_clamp_value_count": dp_official_upper_clamp_count,
            "closed_form_rescale_validated_without_clamp_saturation": (
                dp_official_lower_clamp_count == 0
                and dp_official_upper_clamp_count == 0
            ),
        },
        "track_metrics": {},
        "external_minus_predicted_paired_delta": {},
        "camera_parameter_error": {},
        "correlations": {},
    }
    for prefix in ("dp_native", "dp_calibrated", "ud_native", "ud_calibrated"):
        camera_summary["track_metrics"][prefix] = {
            metric: mean_ci(
                [float(row[f"{prefix}_{metric}"]) for row in per_image],
                config,
                seed_offset=100 + list(DEPTH_METRICS).index(metric),
            )
            for metric in DEPTH_METRICS
        }
    for model_prefix in ("dp", "ud"):
        camera_summary["external_minus_predicted_paired_delta"][model_prefix] = {}
        for metric_index, metric in enumerate(DEPTH_METRICS):
            deltas = [
                float(row[f"{model_prefix}_calibrated_{metric}"])
                - float(row[f"{model_prefix}_native_{metric}"])
                for row in per_image
            ]
            camera_summary["external_minus_predicted_paired_delta"][model_prefix][metric] = mean_ci(
                deltas, config, seed_offset=200 + metric_index
            )
    camera_summary["camera_parameter_error"] = {
        "Depth Pro": {
            "focal_abs_relative_error": mean_ci(
                [float(row["dp_focal_abs_relative_error"]) for row in per_image],
                config,
                301,
            )
        },
        "UniDepthV2-S": {
            "focal_abs_relative_error": mean_ci(
                [float(row["ud_focal_abs_relative_error"]) for row in per_image],
                config,
                302,
            ),
            "principal_point_error_px": mean_ci(
                [float(row["ud_principal_point_error_px"]) for row in per_image],
                config,
                303,
            ),
        },
    }
    camera_summary["correlations"] = {
        "Depth Pro": {
            "camera_error_vs_abs_log_scale_error": correlation(
                [float(row["dp_focal_abs_relative_error"]) for row in per_image],
                [float(row["dp_abs_log_scale_error"]) for row in per_image],
            ),
            "camera_error_vs_abs_rel": correlation(
                [float(row["dp_focal_abs_relative_error"]) for row in per_image],
                [float(row["dp_native_abs_rel"]) for row in per_image],
            ),
        },
        "UniDepthV2-S": {
            "camera_error_vs_abs_log_scale_error": correlation(
                [float(row["ud_focal_abs_relative_error"]) for row in per_image],
                [float(row["ud_abs_log_scale_error"]) for row in per_image],
            ),
            "camera_error_vs_abs_rel": correlation(
                [float(row["ud_focal_abs_relative_error"]) for row in per_image],
                [float(row["ud_native_abs_rel"]) for row in per_image],
            ),
        },
    }
    (output_root / "camera_ablation" / "summary.json").write_text(
        json.dumps(camera_summary, indent=2) + "\n", encoding="utf-8"
    )
    save_camera_plots(output_root, per_image, camera_summary)

    scale_summary: dict[str, object] = {
        "primary_results_remain_native_metric_without_alignment": True,
        "oracle_scale_alignment_is_diagnostic_only": True,
        "models": {},
    }
    for short, method in (("dp", "depth_pro"), ("ud", "unidepth_v2")):
        ratios = np.asarray([float(row[f"{short}_scale_ratio"]) for row in per_image])
        scale_summary["models"][method] = {
            "scale_ratio_mean": float(ratios.mean()),
            "scale_ratio_median": float(np.median(ratios)),
            "scale_ratio_p05": float(np.quantile(ratios, 0.05)),
            "scale_ratio_p95": float(np.quantile(ratios, 0.95)),
            "images_under_0.9_fraction": float(np.mean(ratios < scale_lower)),
            "images_within_0.9_1.1_fraction": float(
                np.mean((ratios >= scale_lower) & (ratios <= scale_upper))
            ),
            "images_over_1.1_fraction": float(np.mean(ratios > scale_upper)),
            "mean_abs_log_scale_error": float(
                np.mean([float(row[f"{short}_abs_log_scale_error"]) for row in per_image])
            ),
            "mean_signed_relative_bias": float(
                np.mean([float(row[f"{short}_signed_relative_bias"]) for row in per_image])
            ),
            "native_metrics": {
                metric: float(
                    np.mean([float(row[f"{short}_native_{metric}"]) for row in per_image])
                )
                for metric in DEPTH_METRICS
            },
            "oracle_scale_aligned_metrics": {
                metric: float(
                    np.mean(
                        [
                            float(row[f"{short}_oracle_scale_aligned_{metric}"])
                            for row in per_image
                        ]
                    )
                )
                for metric in DEPTH_METRICS
            },
        }
    (output_root / "absolute_scale" / "summary.json").write_text(
        json.dumps(scale_summary, indent=2) + "\n", encoding="utf-8"
    )

    depth_bin_summary: list[dict[str, object]] = []
    for method in MODEL_LABELS:
        for lower, upper in depth_bins:
            selected = [
                row
                for row in depth_bin_rows
                if row["method"] == method
                and float(row["lower_m"]) == lower
                and float(row["upper_m"]) == upper
            ]
            depth_bin_summary.append(
                {
                    "method": method,
                    "lower_m": lower,
                    "upper_m": upper,
                    "image_count": len(selected),
                    "pixel_count": sum(int(row["pixel_count"]) for row in selected),
                    "abs_rel_macro": float(np.mean([float(row["abs_rel"]) for row in selected])),
                    "signed_relative_bias_macro": float(
                        np.mean([float(row["signed_relative_bias"]) for row in selected])
                    ),
                    "rmse_macro": float(np.mean([float(row["rmse"]) for row in selected])),
                }
            )
    write_csv(output_root / "absolute_scale" / "depth_bin_summary.csv", depth_bin_summary)
    save_scale_plots(output_root, per_image, depth_bin_summary)

    boundary_summary: list[dict[str, object]] = []
    for method in MODEL_LABELS:
        for tolerance in tolerances:
            selected = [
                row
                for row in boundary_rows
                if row["method"] == method and int(row["tolerance_px"]) == tolerance
            ]
            p = [float(row["boundary_precision"]) for row in selected]
            r = [float(row["boundary_recall"]) for row in selected]
            f1 = [float(row["boundary_f1"]) for row in selected]
            ci = mean_ci(f1, config, 400 + tolerance)
            boundary_summary.append(
                {
                    "method": method,
                    "tolerance_px": tolerance,
                    "precision_mean": float(np.mean(p)),
                    "recall_mean": float(np.mean(r)),
                    "f1_mean": ci["mean"],
                    "f1_ci_lower": ci["lower"],
                    "f1_ci_upper": ci["upper"],
                }
            )
    write_csv(output_root / "boundary" / "boundary_summary.csv", boundary_summary)
    region_summary = {
        method: {
            "official_si_boundary_f1_mean": float(
                np.mean(
                    [
                        float(row[f"{'dp' if method == 'depth_pro' else 'ud'}_si_boundary_f1"])
                        for row in per_image
                    ]
                )
            ),
            "edge_abs_rel_mean": float(
                np.mean(
                    [
                        float(row[f"{'dp' if method == 'depth_pro' else 'ud'}_edge_band_abs_rel"])
                        for row in per_image
                    ]
                )
            ),
            "nonedge_abs_rel_mean": float(
                np.mean(
                    [
                        float(row[f"{'dp' if method == 'depth_pro' else 'ud'}_nonedge_abs_rel"])
                        for row in per_image
                    ]
                )
            ),
        }
        for method in MODEL_LABELS
    }
    paired_f1: dict[str, object] = {}
    for tolerance in tolerances:
        lookup = {
            (str(row["method"]), int(row["test_position"])): float(row["boundary_f1"])
            for row in boundary_rows
            if int(row["tolerance_px"]) == tolerance
        }
        deltas = [
            lookup[("depth_pro", position)] - lookup[("unidepth_v2", position)]
            for position in range(len(dataset))
        ]
        paired_f1[f"tolerance_{tolerance}px_depth_pro_minus_unidepth"] = mean_ci(
            deltas, config, 450 + tolerance
        )
    boundary_report = {
        "assignment_section_6_2_compliance": {
            "main": "official scale-invariant Boundary F1",
            "supplement": "depth-edge precision/recall/F1",
            "edge_ratio_threshold": config["boundary_analysis"]["edge_ratio_threshold"],
            "tolerance_pixels": tolerances,
        },
        "region_error": region_summary,
        "paired_f1": paired_f1,
    }
    (output_root / "boundary" / "summary.json").write_text(
        json.dumps(boundary_report, indent=2) + "\n", encoding="utf-8"
    )
    save_boundary_plots(output_root, boundary_summary, region_summary)

    selected_cases = select_boundary_cases(per_image, config)
    case_dir = output_root / "boundary" / "cases"
    case_paths: list[Path] = []
    for item in selected_cases:
        position = int(item["test_position"])
        sample = dataset.get_test_sample(position)
        stem = str(item["sample_id"])
        dp_depth, _ = load_prediction(baseline_roots["depth_pro"], stem)
        ud_depth, _ = load_prediction(baseline_roots["unidepth_v2"], stem)
        path = case_dir / f"{stem}_{item['selection_reason']}.png"
        save_boundary_case_panel(
            path,
            rgb=sample.rgb[crop],
            target=sample.depth_m[crop],
            depth_pro=np.clip(dp_depth[crop], minimum, maximum),
            unidepth=np.clip(ud_depth[crop], minimum, maximum),
            log_threshold=log_threshold,
            title=(
                f"{stem} | {item['scene_type']} | {item['selection_reason']} | "
                f"SI-F1 DP={item['depth_pro_si_boundary_f1']:.3f}, "
                f"UD={item['unidepth_v2_si_boundary_f1']:.3f}"
            ),
        )
        case_paths.append(path)
    (output_root / "boundary" / "qualitative_selection.json").write_text(
        json.dumps(
            {
                "selection_is_metric_driven_not_semantic_ground_truth": True,
                "case_count": len(selected_cases),
                "cases": selected_cases,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    save_contact_sheet(
        output_root / "boundary" / "boundary_cases_contact_sheet.jpg",
        case_paths,
        columns=2,
        tile_width=1000,
    )

    category_rows: list[dict[str, object]] = []
    scene_types = sorted({str(row["scene_type"]) for row in per_image})
    for scene_type in scene_types:
        selected = [row for row in per_image if row["scene_type"] == scene_type]
        for short, method in (("dp", "depth_pro"), ("ud", "unidepth_v2")):
            failures = sum(
                float(row[f"{short}_native_abs_rel"]) > 0.25
                or float(row[f"{short}_native_delta1"]) < 0.60
                or float(row[f"{short}_si_boundary_f1"]) < 0.01
                for row in selected
            )
            category_rows.append(
                {
                    "scene_type": scene_type,
                    "method": method,
                    "image_count": len(selected),
                    "abs_rel": float(
                        np.mean([float(row[f"{short}_native_abs_rel"]) for row in selected])
                    ),
                    "rmse": float(
                        np.mean([float(row[f"{short}_native_rmse"]) for row in selected])
                    ),
                    "delta1": float(
                        np.mean([float(row[f"{short}_native_delta1"]) for row in selected])
                    ),
                    "si_boundary_f1": float(
                        np.mean([float(row[f"{short}_si_boundary_f1"]) for row in selected])
                    ),
                    "failure_count": int(failures),
                    "failure_rate": float(failures / len(selected)),
                }
            )
    write_csv(output_root / "cross_domain" / "nyuv2_scene_type_metrics.csv", category_rows)
    model_comparison: dict[str, object] = {}
    for metric_index, metric in enumerate((*DEPTH_METRICS, "si_boundary_f1")):
        dp_key = f"dp_native_{metric}" if metric in DEPTH_METRICS else "dp_si_boundary_f1"
        ud_key = f"ud_native_{metric}" if metric in DEPTH_METRICS else "ud_si_boundary_f1"
        deltas = [float(row[dp_key]) - float(row[ud_key]) for row in per_image]
        model_comparison[f"depth_pro_minus_unidepth_{metric}"] = mean_ci(
            deltas, config, 500 + metric_index
        )
    dp_wins = sum(
        float(row["dp_native_abs_rel"]) < float(row["ud_native_abs_rel"])
        for row in per_image
    )
    ud_wins = sum(
        float(row["ud_native_abs_rel"]) < float(row["dp_native_abs_rel"])
        for row in per_image
    )
    cross_domain_report = {
        "scope": "NYUv2 is the only target domain in this analysis",
        "what_can_be_claimed": (
            "zero-shot target-domain transfer and within-NYUv2 scene-type robustness"
        ),
        "what_cannot_be_claimed": (
            "multi-target-domain generalization; no second target dataset was evaluated"
        ),
        "training_evidence": config["cross_domain_analysis"],
        "per_image_absrel_wins": {
            "depth_pro": dp_wins,
            "unidepth_v2": ud_wins,
            "ties": len(dataset) - dp_wins - ud_wins,
        },
        "paired_model_comparison": model_comparison,
        "scene_type_ranking_minimum_images": int(
            config["cross_domain_analysis"]["minimum_scene_type_images_for_ranking"]
        ),
        "worst_scene_types_by_absrel": {
            method: sorted(
                [
                    row
                    for row in category_rows
                    if row["method"] == method
                    and int(row["image_count"])
                    >= int(
                        config["cross_domain_analysis"][
                            "minimum_scene_type_images_for_ranking"
                        ]
                    )
                ],
                key=lambda row: float(row["abs_rel"]),
                reverse=True,
            )[:5]
            for method in MODEL_LABELS
        },
    }
    (output_root / "cross_domain" / "summary.json").write_text(
        json.dumps(cross_domain_report, indent=2) + "\n", encoding="utf-8"
    )
    save_cross_domain_plot(output_root, category_rows)

    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_status": config["protocol_status"],
        "sample_count": len(dataset),
        "formal_training_run": False,
        "primary_tracks_test_time_alignment": "none",
        "primary_tracks_extra_camera_information": "none",
        "external_camera_tracks_are_diagnostic": True,
        "another_target_domain_added": False,
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "camera": camera_summary,
        "absolute_scale": scale_summary,
        "boundary": boundary_report,
        "cross_domain": cross_domain_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/part2/key_analysis.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/part2/key_analysis_v1",
    )
    parser.add_argument("--mode", choices=("all", "infer-external", "analyze"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    labeled_mat = resolve_and_verify(
        config["dataset"]["labeled_mat"], config["dataset"]["labeled_mat_sha256"]
    )
    split_file = resolve_and_verify(
        config["dataset"]["split_file"], config["dataset"]["split_file_sha256"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    with NYUv2LabeledDataset(
        labeled_mat,
        split_file,
        split_variable=config["dataset"]["split_variable"],
    ) as dataset:
        if len(dataset) != int(config["dataset"]["expected_test_images"]):
            raise ValueError(f"Unexpected NYUv2 test count: {len(dataset)}")
        inference = None
        if args.mode in {"all", "infer-external"}:
            inference = infer_unidepth_with_reference_intrinsics(
                config=config,
                dataset=dataset,
                output_root=args.output_dir,
                device=args.device,
                resume=args.resume,
            )
        analysis = None
        if args.mode in {"all", "analyze"}:
            analysis = analyze(config, dataset, args.output_dir)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "mode": args.mode,
        "formal_training_run": False,
        "inference_performed_this_invocation": inference is not None,
        "analysis_performed_this_invocation": analysis is not None,
        "external_inference_output_complete": len(
            list(
                (
                    args.output_dir
                    / "camera_ablation"
                    / "unidepth_external_intrinsics"
                    / "predictions"
                ).glob("*.npz")
            )
        )
        == int(config["dataset"]["expected_test_images"]),
        "analysis_output_complete": all(
            (args.output_dir / relative_path).exists()
            for relative_path in (
                "camera_ablation/summary.json",
                "absolute_scale/summary.json",
                "boundary/summary.json",
                "cross_domain/summary.json",
            )
        ),
    }
    (args.output_dir / "invocation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "inference": inference,
                "analysis_outputs": str(args.output_dir) if analysis is not None else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
