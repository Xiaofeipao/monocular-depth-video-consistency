#!/usr/bin/env python3
"""Absolute-scale and boundary analysis of frozen native metric predictions."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.part2.run_depth_pro_nyuv2 import (
    DEPTH_METRICS,
    bootstrap_mean_ci,
    prediction_stem,
    sha256,
    write_csv,
)
from src.data.nyuv2 import NYUv2LabeledDataset
from src.metrics.boundary import compute_depth_boundary_metrics, depth_edge_map
from src.metrics.depth import build_valid_mask, compute_depth_metrics
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
    if config["protocol_status"] != "native_depth_error_analysis_v1":
        raise ValueError("Expected native_depth_error_analysis_v1 protocol")
    if config["evaluation"]["test_time_alignment"] != "none":
        raise ValueError("Primary analysis must preserve alignment:none")
    if set(config["baseline_outputs"]) != set(MODEL_LABELS):
        raise ValueError("Both native metric baselines are required")
    evaluation = config["evaluation"]
    if (
        config["dataset"]["split_variable"] != "testNdxs"
        or config["dataset"]["expected_test_images"] != 654
        or [evaluation["crop"][k] for k in ("top", "bottom", "left", "right")]
        != [45, 471, 41, 601]
        or evaluation["valid_depth_m"] != {"minimum": 0.001, "maximum": 10.0, "bounds": "exclusive"}
        or evaluation["clip_prediction_to_valid_range"] is not True
        or evaluation["aggregation"] != "per_image_macro_mean"
    ):
        raise ValueError("Analysis must preserve the standard NYUv2 test protocol")
    oracle = config["absolute_scale_analysis"]["oracle_alignment"]
    if not oracle["enabled_for_diagnosis_only"] or not oracle["forbidden_as_primary_result"]:
        raise ValueError("GT-based oracle alignment is diagnostic only")
    return config


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


def read_baseline_metrics(root: Path, expected_count: int) -> dict[int, dict[str, str]]:
    rows = read_csv(root / "metrics_per_image.csv")
    lookup = {int(row["test_position"]): row for row in rows}
    if len(rows) != expected_count or set(lookup) != set(range(expected_count)):
        raise ValueError(f"Baseline metrics have missing/duplicate positions: {root}")
    return lookup


def validate_sample_identity(row: dict, sample: object, position: int, stem: str) -> None:
    if (
        row["sample_id"] != stem
        or int(row["test_position"]) != position
        or int(row["matlab_index_1based"]) != sample.matlab_index_1based
        or row["scene"] != sample.scene
        or row["scene_type"] != sample.scene_type
        or not np.isfinite(float(row["si_boundary_f1"]))
    ):
        raise ValueError(f"Baseline identity or SI metric mismatch: {stem}")


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
    axes[0].set_title("Depth-edge matching")
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
        if needed <= 0:
            continue
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

    primary_rows = {
        method: read_baseline_metrics(root, expected_count)
        for method, root in baseline_roots.items()
    }
    crop = crop_slices(config)
    minimum = float(config["evaluation"]["valid_depth_m"]["minimum"])
    maximum = float(config["evaluation"]["valid_depth_m"]["maximum"])
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
    for position in range(len(dataset)):
        sample = dataset.get_test_sample(position)
        stem = prediction_stem(position, sample.matlab_index_1based)
        dp_depth, _ = load_prediction(baseline_roots["depth_pro"], stem)
        ud_depth, _ = load_prediction(baseline_roots["unidepth_v2"], stem)
        for method, depth in (("depth_pro", dp_depth), ("unidepth_v2", ud_depth)):
            if depth.shape != sample.depth_m.shape or not np.all(np.isfinite(depth)) or np.any(depth <= 0):
                raise ValueError(f"Invalid full-size native metric prediction: {method}/{stem}")
            validate_sample_identity(primary_rows[method][position], sample, position, stem)

        target = sample.depth_m[crop]
        tracks = {
            "dp_native": dp_depth[crop],
            "ud_native": ud_depth[crop],
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

        for short, prediction in (
            ("dp", tracks["dp_native"]),
            ("ud", tracks["ud_native"]),
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

        for short, method in (("dp", "depth_pro"), ("ud", "unidepth_v2")):
            for metric in DEPTH_METRICS:
                actual = float(row[f"{short}_native_{metric}"])
                expected = float(primary_rows[method][position][metric])
                if not np.isfinite(expected) or not np.isclose(actual, expected, rtol=1e-6, atol=1e-8):
                    raise ValueError(f"Native metrics differ from baseline: {method}/{stem}/{metric}")

        row["dp_si_boundary_f1"] = float(primary_rows["depth_pro"][position]["si_boundary_f1"])
        row["ud_si_boundary_f1"] = float(primary_rows["unidepth_v2"][position]["si_boundary_f1"])
        per_image.append(row)
        if (position + 1) % 50 == 0 or position == len(dataset) - 1:
            print(f"[{position + 1:03d}/{len(dataset)}] absolute-scale and boundary analysis", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "per_image_analysis.csv", per_image)
    write_csv(output_root / "boundary" / "boundary_per_image.csv", boundary_rows)
    write_csv(output_root / "absolute_scale" / "depth_bin_per_image.csv", depth_bin_rows)

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
        "evaluation_protocol": {
            "role": "supplemental_boundary_diagnosis",
            "scale_invariant_metric": "official scale-invariant Boundary F1",
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

    return {
        "absolute_scale": scale_summary,
        "boundary": boundary_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从两条完整native基线分析绝对尺度与边界；不加载模型，不覆盖目录。"
    )
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/part2/key_analysis.yaml')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = (args.output_dir or PROJECT_ROOT / config['outputs']['root']).resolve()
    if output_root.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {output_root}')
    labeled_mat = resolve_and_verify(config['dataset']['labeled_mat'], config['dataset']['labeled_mat_sha256'])
    split_file = resolve_and_verify(config['dataset']['split_file'], config['dataset']['split_file_sha256'])
    source_paths = [
        Path(__file__),
        PROJECT_ROOT / 'pipelines/part2/run_depth_pro_nyuv2.py',
        PROJECT_ROOT / 'src/data/nyuv2.py',
        PROJECT_ROOT / 'src/metrics/depth.py',
        PROJECT_ROOT / 'src/metrics/boundary.py',
        PROJECT_ROOT / 'src/visualization/nyuv2.py',
    ]
    metadata = {
        'status': 'running',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_path': str(args.config.resolve()),
        'config_sha256': sha256(args.config),
        'command': sys.argv,
        'protocol_status': config['protocol_status'],
        'formal_training_run': False,
        'inference_performed_this_invocation': False,
        'primary_tracks_test_time_alignment': 'none',
        'oracle_scale_alignment_is_diagnostic_only': True,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'device': 'cpu'},
        'baseline_sources': {},
        'source_sha256': {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in source_paths},
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / 'config_resolved.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    metadata_path = output_root / 'run_metadata.json'
    def save_metadata() -> None:
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    save_metadata()
    start = time.perf_counter()
    try:
        for method, relative in config['baseline_outputs'].items():
            root = PROJECT_ROOT / relative
            validate_baseline_root(root, int(config['dataset']['expected_test_images']))
            recorded = json.loads((root / 'run_metadata.json').read_text())
            baseline_config = yaml.safe_load(Path(recorded['config']).read_text())
            for key in ('labeled_mat_sha256', 'split_file_sha256', 'split_variable', 'expected_test_images'):
                if baseline_config['dataset'][key] != config['dataset'][key]:
                    raise ValueError(f'Baseline dataset protocol differs: {method}/{key}')
            for key in ('crop', 'valid_depth_m', 'clip_prediction_to_valid_range', 'aggregation', 'test_time_alignment'):
                if baseline_config['evaluation'][key] != config['evaluation'][key]:
                    raise ValueError(f'Baseline evaluation protocol differs: {method}/{key}')
            metadata['baseline_sources'][method] = {
                'root': str(root.resolve()),
                'records_sha256': {name: sha256(root / name) for name in ('run_metadata.json', 'metrics_per_image.csv', 'metrics_summary.json')},
                'prediction_sha256': {p.name: sha256(p) for p in sorted((root / 'predictions').glob('*.npz'))},
            }
        save_metadata()
        with NYUv2LabeledDataset(labeled_mat, split_file, split_variable=config['dataset']['split_variable']) as dataset:
            if len(dataset) != int(config['dataset']['expected_test_images']):
                raise ValueError(f'Unexpected NYUv2 test count: {len(dataset)}')
            analyze(config, dataset, output_root)
            metadata['sample_count'] = len(dataset)
        metadata['status'] = 'complete'
    except Exception as error:
        metadata['status'] = 'failed'
        metadata['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        metadata['wall_seconds'] = time.perf_counter() - start
        save_metadata()
    print(json.dumps({'status': metadata['status'], 'analysis_outputs': str(output_root)}, indent=2))


if __name__ == '__main__':
    main()
