"""Standalone Part 1B diagnostics from one DA-V2 run; no second depth model."""

from __future__ import annotations

import base64
import csv
import math
from pathlib import Path
import sys
import traceback

import cv2
import numpy as np

from src.metrics.temporal import forward_backward_consistency_mask
from .common import (PROJECT_ROOT, create_output_directory, environment_info,
                     read_json, sha256_file, source_fingerprints, utc_now, write_json)
from .evaluation import (_array, _array_artifact, _artifact, _checked, _complete,
                         _equivalent, _finite_support, _path_inside, _read_csv,
                         _safe_warp, _write_csv)
from .frames import validate_manifest
from .runner import load_protocol


def scientific_sources() -> dict:
    sources = source_fingerprints()
    entry = PROJECT_ROOT / "pipelines/part1/framewise_video.py"
    if entry.exists():
        sources[str(entry.relative_to(PROJECT_ROOT))] = sha256_file(entry)
    return sources


def load_single_sources(run_dir: Path, flow_dir: Path) -> dict:
    """Validate frozen single-model identities and load saved arrays only."""
    run_dir, flow_dir = Path(run_dir).resolve(), Path(flow_dir).resolve()
    run_path, flow_path = run_dir / "run.json", flow_dir / "flow_manifest.json"
    run, flow = read_json(run_path), read_json(flow_path)
    _complete(run, "DA-V2 run")
    _complete(flow, "flow cache")
    model = run["model"]
    if (model.get("name") != "dav2" or model.get("depth_type") != "relative"
            or model.get("representation") != "inverse_depth_like"
            or model.get("unit") != "arbitrary" or model.get("alignment") != "none"
            or model.get("gt_used", False)):
        raise ValueError("Part 1B requires native relative DA-V2 output without GT alignment")
    network_shape = model.get("network_input_hw")
    if (not isinstance(network_shape, list) or len(network_shape) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in network_shape)
            or model.get("precision") not in {"float16", "float32"}):
        raise ValueError("Actual network input shape and precision must be recorded")
    protocol_path = _checked(Path(run["protocol_path"]), run["protocol_sha256"], "frozen protocol")
    protocol, manifest, manifest_path = load_protocol(protocol_path)
    if run["config"] != protocol["configuration"]:
        raise ValueError("Run configuration differs from its frozen protocol")
    manifest_sha = sha256_file(manifest_path)
    for payload in (run, flow):
        if payload["manifest_sha256"] != manifest_sha or payload["protocol_sha256"] != run["protocol_sha256"]:
            raise ValueError("Run/flow do not use the same frozen input and protocol")
        if Path(payload["manifest_path"]).resolve() != manifest_path.resolve():
            raise ValueError("Run/flow manifest path differs")
        _checked(Path(payload["protocol_path"]), payload["protocol_sha256"], "protocol")
    count = len(manifest["frames"])
    shape = (manifest["height"], manifest["width"])
    if len(run["frames"]) != count or flow["frames_count"] != count or flow["shape"] != list(shape):
        raise ValueError("Run/flow frame count or geometry differs")
    expected_flow = run["config"]["flow"]
    if flow["flow"]["backend"] != expected_flow["backend"]:
        raise ValueError("Flow backend differs from frozen configuration")
    if flow["flow"].get("iterations") != expected_flow["iterations"]:
        raise ValueError("Flow update count differs from frozen configuration")
    if flow["flow"].get("max_long_edge") != expected_flow.get("max_long_edge"):
        raise ValueError("Flow estimation resolution differs from frozen configuration")
    if expected_flow["backend"] == "raft_small" and flow["flow"].get("checkpoint_sha256") != expected_flow["sha256"]:
        raise ValueError("RAFT weight identity differs")
    bound = []

    def bind(path, digest=None, role="source"):
        path = Path(path).resolve()
        digest = digest or sha256_file(path)
        _checked(path, digest, role)
        bound.append({"path": str(path), "sha256": digest, "role": role})
        return path

    bind(run_path, role="run")
    bind(flow_path, role="flow_manifest")
    bind(protocol_path, run["protocol_sha256"], "protocol")
    bind(manifest_path, manifest_sha, "frame_manifest")
    raw = []
    for i, (saved, frame) in enumerate(zip(run["frames"], manifest["frames"])):
        if (saved["index"] != i or saved["source_index"] != frame["source_index"]
                or saved["timestamp_seconds"] != frame["timestamp_seconds"]
                or saved["shape"] != list(shape) or saved["dtype"] != "float32"):
            raise ValueError("DA-V2 output frame identity/shape/dtype differs")
        bind(_path_inside(manifest_path.parent, frame["rgb_path"]), frame["rgb_sha256"], "rgb")
        path = bind(_path_inside(run_dir, saved["raw_path"]), saved["raw_sha256"], "raw")
        raw.append(_array(path, shape))
    roi = None
    if manifest.get("roi_path"):
        path = bind(_path_inside(manifest_path.parent, manifest["roi_path"]), manifest["roi_sha256"], "background_roi")
        roi = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
    pairs = flow["pairs"]
    if [(p["previous_index"], p["current_index"]) for p in pairs] != [(i - 1, i) for i in range(1, count)]:
        raise ValueError("Flow pair identity/order differs")
    flows = []
    for pair in pairs:
        both = []
        for direction in ("forward", "backward"):
            path = bind(_path_inside(flow_dir, pair[direction + "_path"]), pair[direction + "_sha256"], direction)
            both.append(_array(path, (*shape, 2)))
        flows.append(both)
    return {"run": run, "run_dir": run_dir, "flow_dir": flow_dir, "flow_manifest": flow,
            "protocol": protocol, "manifest": manifest, "manifest_path": manifest_path,
            "raw": raw, "flows": flows, "roi": roi, "shape": shape, "bound_files": bound}


def _aggregate(rows: list[dict], span: float | None) -> dict:
    valid = [r for r in rows if r["normalized_mae"] is not None]
    pixels = sum(r["valid_pixels"] for r in valid)
    return {
        "included_pairs": len(valid), "included_pixels": pixels,
        "mean_pair_normalized_mae": float(np.mean([r["normalized_mae"] for r in valid])) if valid else None,
        "mean_pair_normalized_rmse": float(np.mean([r["normalized_rmse"] for r in valid])) if valid else None,
        "pixel_weighted_normalized_mae": sum(r["abs_error_sum"] for r in valid) / pixels / span if valid else None,
        "pixel_weighted_normalized_rmse": math.sqrt(sum(r["squared_error_sum"] for r in valid) / pixels) / span if valid else None,
        "unavailable_reason": None if valid else ("invalid_reference_span" if span is None else "no_nonempty_pairs"),
    }


def measure_single(sources: dict, output: Path | None = None, *, verification_dir: Path | None = None):
    raw, manifest, roi = sources["raw"], sources["manifest"], sources["roi"]
    evaluation = sources["run"]["config"]["evaluation"]
    ref = evaluation["reference_index"]
    if isinstance(ref, bool) or not isinstance(ref, int):
        raise ValueError("Reference frame index must be an integer")
    if evaluation["alignment"] != "none" or not 0 <= ref < len(raw):
        raise ValueError("Invalid reference or forbidden alignment")
    low_p, high_p = evaluation["percentile_low"], evaluation["percentile_high"]
    if not 0 <= low_p < high_p <= 100 or not math.isfinite(evaluation["fb_threshold_px"]) or evaluation["fb_threshold_px"] < 0:
        raise ValueError("Invalid evaluation parameters")
    shape, count = sources["shape"], len(raw)
    grid_pixels = int(np.prod(shape))
    reference_values = np.asarray(raw[ref][np.isfinite(raw[ref])], dtype=np.float64)
    low, high, span = None, None, None
    if reference_values.size:
        low, high = map(float, np.percentile(reference_values, [low_p, high_p], method="linear"))
        span = high - low if high > low else None
    reference = {"index": ref, "percentile_low_value": low, "percentile_high_value": high,
                 "span": span, "pixels": int(reference_values.size), "percentiles": [low_p, high_p],
                 "region": "full_frame_finite_reference_values; ROI is background diagnostic only",
                 "unavailable_reason": None if span else ("zero_reference_span" if reference_values.size else "no_finite_reference_values")}
    spatial = np.ones(shape, dtype=bool)
    for frame in raw:
        spatial &= np.isfinite(frame)
    background_region = spatial & roi if roi is not None else np.zeros(shape, dtype=bool)
    background_applicable = manifest["camera_motion"] == "fixed" and roi is not None
    background_pixels = int(background_region.sum())
    pairs, frames, artifacts = [], [], []

    def save_array(relative, array, role):
        artifacts.append(_array_artifact(output, relative, array, role, verification_dir=verification_dir))

    for i, (forward, backward) in enumerate(sources["flows"], start=1):
        warped, support = _safe_warp(raw[i - 1], backward)
        fb = forward_backward_consistency_mask(forward, backward, threshold_px=evaluation["fb_threshold_px"])
        fb &= _finite_support(forward[..., 0], backward) & _finite_support(forward[..., 1], backward)
        mask = fb & support & np.isfinite(raw[i])
        pixels = int(mask.sum())
        difference = np.asarray(raw[i][mask], dtype=np.float64) - np.asarray(warped[mask], dtype=np.float64)
        absolute = float(np.abs(difference).sum()) if pixels else None
        squared = float(np.square(difference).sum()) if pixels else None
        coverage = pixels / grid_pixels
        adjacent = float(np.mean(np.abs(np.asarray(raw[i][background_region], dtype=np.float64)
                                     - np.asarray(raw[i - 1][background_region], dtype=np.float64)))) if background_applicable and background_pixels else None
        pairs.append({
            "previous_index": i - 1, "current_index": i,
            "timestamp_seconds": manifest["frames"][i]["timestamp_seconds"],
            "dt_seconds": manifest["frames"][i]["timestamp_seconds"] - manifest["frames"][i - 1]["timestamp_seconds"],
            "grid_pixels": grid_pixels, "fb_valid_pixels": int(fb.sum()),
            "prediction_support_pixels": int((support & np.isfinite(raw[i])).sum()),
            "valid_pixels": pixels, "coverage": coverage,
            "quality_flag": "below_minimum_coverage" if pixels and coverage < evaluation["minimum_coverage"] else None,
            "exclusion_reason": "empty_mask" if not pixels else None,
            "abs_error_sum": absolute, "squared_error_sum": squared,
            "raw_mae": absolute / pixels if pixels else None,
            "raw_rmse": math.sqrt(squared / pixels) if pixels else None,
            "normalized_mae": absolute / pixels / span if pixels and span else None,
            "normalized_rmse": math.sqrt(squared / pixels) / span if pixels and span else None,
            "background_adjacent_raw_mae": adjacent,
            "background_adjacent_normalized_mae": adjacent / span if adjacent is not None and span else None,
        })
        save_array(f"masks/pair_{i - 1:06d}_{i:06d}.npy", mask, "pair_mask")
    medians = [float(np.median(np.asarray(frame[spatial], dtype=np.float64))) if spatial.any() else None for frame in raw]
    m0 = medians[ref]
    for i, median in enumerate(medians):
        ratio = median / m0 if median is not None and m0 is not None and median > 0 and m0 > 0 else None
        frames.append({"index": i, "source_index": manifest["frames"][i]["source_index"],
                       "timestamp_seconds": manifest["frames"][i]["timestamp_seconds"],
                       "fixed_region_pixels": int(spatial.sum()), "raw_median": median,
                       "normalized_median_delta": (median - m0) / span if median is not None and span else None,
                       "median_ratio": ratio, "abs_log_median_ratio": abs(math.log(ratio)) if ratio else None,
                       "ratio_unavailable_reason": None if ratio else "nonpositive_median_or_no_fixed_finite_region"})
    save_array("fixed_median_region.npy", spatial, "fixed_median_region")
    background = {"applicable": background_applicable, "pixels": background_pixels,
                  "unavailable_reason": None if background_applicable else "requires_fixed_camera_and_predeclared_roi",
                  "normalized_std_mean": None, "normalized_std_p95": None,
                  "adjacent_normalized_mae_mean": None, "temporal_std_ddof": 0}
    if background_applicable and span and background_pixels:
        mean, m2 = np.zeros(background_pixels), np.zeros(background_pixels)
        for i, frame in enumerate(raw, 1):
            value = np.asarray(frame[background_region], dtype=np.float64) / span
            delta = value - mean
            mean += delta / i
            m2 += delta * (value - mean)
        std = np.sqrt(np.maximum(m2 / count, 0))
        background.update(normalized_std_mean=float(std.mean()), normalized_std_p95=float(np.percentile(std, 95)),
                          adjacent_normalized_mae_mean=float(np.mean([p["background_adjacent_normalized_mae"] for p in pairs])))
        std_map = np.full(shape, np.nan, dtype=np.float32)
        std_map[background_region] = std.astype(np.float32)
        save_array("background_normalized_std.npy", std_map, "background_std")
    elif background_applicable:
        background["unavailable_reason"] = "invalid_reference_span" if not span else "no_fixed_finite_background_pixels"
    save_array("background_region.npy", background_region, "background_region")
    log_ratios = [r["abs_log_median_ratio"] for r in frames if r["abs_log_median_ratio"] is not None]
    ranked = sorted((p for p in pairs if p["normalized_mae"] is not None), key=lambda p: p["normalized_mae"], reverse=True)
    result = {
        "reference": reference, "aggregate": _aggregate(pairs, span), "background_flicker": background,
        "median_summary": {"mean_abs_log_ratio": float(np.mean(log_ratios)) if log_ratios else None,
                           "max_abs_log_ratio": max(log_ratios) if log_ratios else None,
                           "ratio_frames": len(log_ratios), "total_frames": count,
                           "fixed_region_pixels": int(spatial.sum()), "interpretation": "scale_proxy_not_metric_drift"},
        "coverage": {"pair_count": len(pairs), "nonempty_pairs": sum(p["valid_pixels"] > 0 for p in pairs),
                     "empty_pairs": sum(p["valid_pixels"] == 0 for p in pairs),
                     "low_coverage_pairs": sum(p["quality_flag"] is not None for p in pairs),
                     "mean": float(np.mean([p["coverage"] for p in pairs])), "minimum": min(p["coverage"] for p in pairs)},
        "worst_pairs": [{k: p[k] for k in ("current_index", "timestamp_seconds", "coverage", "normalized_mae", "quality_flag")} for p in ranked[:5]],
        "scientific_protocol": {"alignment": "none", "zero_is_valid": True, "per_frame_normalization": False,
                                "pair_aggregation": "equal_mean_all_nonempty_pairs; low_coverage_flag_only",
                                "reference_region": "finite_reference_full_frame", "median_region": "full_frame_finite_at_all_times",
                                "background_roi_role": "fixed_background_only; does_not_restrict_primary_warp_or_median",
                                "flow_convention": "backward_current_to_previous_bilinear",
                                "score_time_unit": "per_selected_frame_pair_not_per_second", "gt_used": False},
    }
    return result, pairs, frames, artifacts


def analyze_framewise(run_dir: Path, flow_dir: Path, output_dir: Path, *, generate_report=True) -> Path:
    output = create_output_directory(output_dir)
    status = output / "analysis_status.json"
    write_json(status, {"status": "running", "started_at_utc": utc_now(), "command": sys.argv})
    try:
        sources = load_single_sources(run_dir, flow_dir)
        result, pairs, frames, artifacts = measure_single(sources, output)
        for filename, rows in (("pair_metrics.csv", pairs), ("frame_metrics.csv", frames)):
            _write_csv(output / filename, rows)
            artifacts.append({**_artifact(output, output / filename), "role": filename})
        manifest = sources["manifest"]
        scientific = scientific_sources()
        summary = {
            "schema_version": "part1b_analysis_v1", "status": "complete", "scope": "part1b_single_dav2_relative",
            "created_at_utc": utc_now(), "command": sys.argv, "environment": environment_info(),
            "clip": {k: manifest[k] for k in ("clip_id", "source_kind", "camera_motion", "split", "fps", "height", "width")},
            "frames_count": len(frames), "model": sources["run"]["model"], "timing": sources["run"]["timing"],
            "flow": sources["flow_manifest"]["flow"], "config": sources["run"]["config"],
            "sources": {"run_dir": str(sources["run_dir"]), "flow_dir": str(sources["flow_dir"]),
                        "manifest_path": str(sources["manifest_path"]), "files": sources["bound_files"]},
            "analysis_source_sha256": scientific,
            "analysis_source_archive_base64": {k: base64.b64encode((PROJECT_ROOT / k).read_bytes()).decode("ascii") for k in scientific},
            "artifacts": artifacts, **result,
        }
        path = output / "summary.json"
        write_json(path, summary)
        if generate_report and sources["run"]["config"]["visualization"].get("generate_report", True):
            from .framewise_report import generate_single_report
            summary["report"] = generate_single_report(path, sources, pairs, frames)
            write_json(path, summary)
        write_json(status, {"status": "complete", "summary_sha256": sha256_file(path), "finished_at_utc": utc_now()})
        return path
    except Exception as exc:
        write_json(status, {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                            "traceback": traceback.format_exc(), "finished_at_utc": utc_now()})
        raise


def verify_single_analysis(output_dir: Path) -> dict:
    output = Path(output_dir).resolve()
    _complete(read_json(output / "analysis_status.json"), "analysis")
    _checked(output / "summary.json", read_json(output / "analysis_status.json")["summary_sha256"], "summary")
    summary = read_json(output / "summary.json")
    if summary.get("schema_version") != "part1b_analysis_v1" or summary.get("status") != "complete":
        raise ValueError("Not a complete Part 1B standalone analysis")
    for relative, digest in summary["analysis_source_sha256"].items():
        _checked(PROJECT_ROOT / relative, digest, "analysis scientific source")
        archived = base64.b64decode(summary["analysis_source_archive_base64"][relative], validate=True)
        import hashlib
        if hashlib.sha256(archived).hexdigest() != digest:
            raise ValueError("Analysis scientific source archive mismatch")
    for item in summary["artifacts"] + summary.get("report", {}).get("assets", []):
        _checked(_path_inside(output, item["path"]), item["sha256"], "analysis artifact")
    for item in summary["sources"]["files"]:
        _checked(Path(item["path"]), item["sha256"], "source")
    sources = load_single_sources(summary["sources"]["run_dir"], summary["sources"]["flow_dir"])
    if summary["frames_count"] != len(sources["raw"]) or summary["scope"] != "part1b_single_dav2_relative":
        raise ValueError("Analysis frame count or scope differs from raw inputs")
    if Path(summary["sources"]["manifest_path"]).resolve() != sources["manifest_path"].resolve():
        raise ValueError("Analysis manifest path differs from frozen inputs")
    _equivalent(summary["config"], sources["run"]["config"], "saved configuration")
    _equivalent(summary["model"], sources["run"]["model"], "model metadata")
    _equivalent(summary["timing"], sources["run"]["timing"], "timing metadata")
    for key, value in summary["clip"].items():
        _equivalent(value, sources["manifest"][key], "clip." + key)
    result, pairs, frames, _ = measure_single(sources, verification_dir=output)
    for key, value in result.items():
        _equivalent(summary[key], value, key)
    _equivalent(_read_csv(output / "pair_metrics.csv"), pairs, "pair CSV recomputed from raw")
    _equivalent(_read_csv(output / "frame_metrics.csv"), frames, "frame CSV recomputed from raw")
    if "report" in summary:
        from .framewise_report import report_binding
        _equivalent(summary["report"]["binding"], report_binding(summary), "visual source binding")
    return {"status": "verified", "frames": len(frames), "pairs": len(pairs), "models_executed": False,
            "numeric_rows_recomputed_from_raw": True, "masks_and_background_recomputed": True,
            "numeric_tolerance": {"relative": 1e-6, "absolute": 1e-8}}


def summarize_single_analyses(directories: list[Path], output_dir: Path) -> Path:
    if not directories or len({Path(p).resolve() for p in directories}) != len(directories):
        raise ValueError("Provide distinct analysis directories")
    rows, sources, identities, sessions, group_protocols = [], [], set(), {}, {}
    for directory in directories:
        directory = Path(directory).resolve()
        verification = verify_single_analysis(directory)
        summary = read_json(directory / "summary.json")
        manifest = read_json(Path(summary["sources"]["manifest_path"]))
        identity = manifest["source"]["sha256"]
        if identity in identities:
            raise ValueError("Duplicate source recording; adjacent cuts are not independent clips")
        identities.add(identity)
        clip = summary["clip"]
        config = summary["config"]
        group = (clip["source_kind"], clip["split"], clip["camera_motion"])
        signature = {k: config[k] for k in ("runtime", "inference", "flow", "evaluation")}
        signature["depth_model"] = config["models"]["dav2"]
        signature["cache_max_long_edge"] = manifest.get("resize", {}).get("max_long_edge", config["input"]["max_long_edge"])
        signature["target_fps"] = manifest["fps"]
        if group in group_protocols and group_protocols[group] != signature:
            raise ValueError("Different scientific protocols cannot be pooled into one clip group")
        group_protocols[group] = signature
        session = manifest["source"].get("provenance", {}).get("recording_session")
        key = (clip["source_kind"], session)
        if session and key in sessions and sessions[key] != clip["split"]:
            raise ValueError("Recording session appears in development and test")
        sessions[key] = clip["split"]
        rows.append({**clip, "frames": summary["frames_count"],
                     "normalized_mae": summary["aggregate"]["mean_pair_normalized_mae"],
                     "normalized_rmse": summary["aggregate"]["mean_pair_normalized_rmse"],
                     "coverage": summary["coverage"]["mean"],
                     "mean_abs_log_median_ratio": summary["median_summary"]["mean_abs_log_ratio"],
                     "background_normalized_std_mean": summary["background_flicker"]["normalized_std_mean"]})
        sources.append({"path": str(directory / "summary.json"), "sha256": sha256_file(directory / "summary.json"), "verification": verification})
    groups = []
    for key in sorted({(r["source_kind"], r["split"], r["camera_motion"]) for r in rows}):
        selected = [r for r in rows if (r["source_kind"], r["split"], r["camera_motion"]) == key]
        means = {}
        counts = {}
        for metric in ("normalized_mae", "normalized_rmse", "mean_abs_log_median_ratio", "background_normalized_std_mean"):
            values = [r[metric] for r in selected if r[metric] is not None]
            means[metric] = float(np.mean(values)) if values else None
            counts[metric] = len(values)
        groups.append({"source_kind": key[0], "split": key[1], "camera_motion": key[2],
                       "clips": len(selected), "equal_clip_mean": means, "evaluable_clip_counts": counts})
    output = create_output_directory(output_dir)
    _write_csv(output / "clips.csv", rows)
    write_json(output / "summary.json", {"status": "complete", "schema_version": "part1b_collection_v1",
                                         "groups": groups, "sources": sources,
                                         "clips_csv_sha256": sha256_file(output / "clips.csv"),
                                         "aggregation": "equal_clip_within_source_kind_split_camera; N/A not replaced by zero",
                                         "course_completion": "requires_self_recorded_fixed_moving_and_manual_failure_analysis; not_automatically_asserted"})
    return output / "summary.json"
