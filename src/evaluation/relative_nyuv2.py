"""DA-V2-S NYUv2: independent raw inference, GT alignment, and reload audit."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import yaml

from src.data.nyuv2 import NYUv2LabeledDataset
from src.metrics.depth import compute_depth_metrics
from src.metrics.relative_alignment import align_inverse_depth
from src.video.common import (PROJECT_ROOT, environment_info, inside_path, read_json,
                              sha256_file, utc_now, write_json)

METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3")
SOURCE_FILES = ["src/evaluation/relative_nyuv2.py", "src/metrics/relative_alignment.py",
                "src/data/nyuv2.py", "src/metrics/depth.py", "src/video/common.py",
                "src/models/depth_anything_v2_adapter.py", "src/visualization/nyuv2.py",
                "pipelines/part1/dav2_nyuv2.py"]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config["protocol"] != "dav2_nyuv2_aligned_relative_v1":
        raise ValueError("Unsupported relative benchmark protocol")
    e = config["evaluation"]
    expected = {"label": "aligned_relative", "crop": [45, 471, 41, 601],
                "minimum_depth_m": 0.001, "maximum_depth_m": 10.0,
                "alignment": "nonnegative_scale_shift_least_squares",
                "alignment_domain": "inverse_depth",
                "alignment_scope": "per_image_evaluation_crop_valid_gt",
                "degenerate_fit": "constant_mean_inverse_gt",
                "nonpositive_inverse": "clip_to_inverse_depth_bounds_keep_all_valid_gt",
                "aggregation": "per_image_macro_mean"}
    if e != expected or config["dataset"]["expected_test_images"] != 654:
        raise ValueError("NYUv2 split/crop/depth/alignment protocol differs")
    inf, rt = config["inference"], config["runtime"]
    if inf["timing_repeats"] < 1 or inf["warmup"] < 1 or inf["input_size"] < 14:
        raise ValueError("Invalid inference settings")
    if rt["precision"] not in {"float16", "float32"} or rt["torch_threads"] < 1:
        raise ValueError("Invalid runtime settings")
    if config["statistics"]["bootstrap_resamples"] < 1:
        raise ValueError("Invalid bootstrap settings")
    return config


def _new_directory(path):
    path = Path(path).resolve()
    if path.is_relative_to(PROJECT_ROOT) and not path.is_relative_to(PROJECT_ROOT / "outputs"):
        raise ValueError("Project artifacts must be written under outputs/")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _checked(binding):
    path = Path(binding["path"])
    if sha256_file(path) != binding["sha256"]:
        raise ValueError(f"Source SHA256 mismatch: {path}")
    return path


def check_resources(config, *, require_device=True):
    from src.models.depth_anything_v2_adapter import _repository_identity, _runtime
    resources = {}
    for section, key in [("dataset", "labeled_mat"), ("dataset", "split_file"), ("model", "checkpoint")]:
        p = PROJECT_ROOT / config[section][key]
        resources[key] = _binding(p)
        if resources[key]["sha256"] != config[section][key + "_sha256"]:
            raise ValueError(f"Unexpected resource checksum: {p}")
    repository = _repository_identity(PROJECT_ROOT / config["model"]["repository"])
    if repository["upstream_git_sha"] != config["model"]["commit"] or repository["repository_dirty"]:
        raise ValueError("Official DA-V2 repository commit or tracked files differ")
    with _dataset(config) as dataset:
        if len(dataset) != config["dataset"]["expected_test_images"]:
            raise ValueError("Expected standard 654-image test split")
    result = {"resources": resources, "repository": repository, "environment": environment_info()}
    if require_device:
        _runtime(config["runtime"]["device"], config["runtime"]["precision"])
    return result


def _dataset(config):
    return NYUv2LabeledDataset(PROJECT_ROOT / config["dataset"]["labeled_mat"],
                              PROJECT_ROOT / config["dataset"]["split_file"])


def _positions(positions, count):
    values = list(range(count)) if positions is None else list(positions)
    if not values or len(values) != len(set(values)) or any(type(x) is not int or not 0 <= x < count for x in values):
        raise ValueError("Provide unique in-range test positions")
    return values


def _save_array(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _load_run(directory, *, complete=True):
    directory = Path(directory).resolve()
    run = read_json(directory / "run.json")
    if run.get("schema") != "dav2_nyuv2_raw_v1" or (complete and run["status"] != "complete"):
        raise ValueError("Not a complete DA-V2 NYUv2 raw run")
    if run["raw_contract"] != {"depth_type": "relative", "representation": "inverse_depth_like",
                               "unit": "arbitrary", "alignment": "none", "gt_used": False}:
        raise ValueError("Invalid raw prediction contract")
    for relative, digest in run["scientific_sources"].items():
        if sha256_file(PROJECT_ROOT / relative) != digest:
            raise ValueError(f"Scientific source changed: {relative}")
        if sha256_file(directory / "provenance" / relative) != digest:
            raise ValueError("Archived scientific source mismatch")
    for value in run["resources"].values():
        _checked(value)
    repo = run["repository"]
    for relative, digest in repo["source_files_sha256"].items():
        if sha256_file(inside_path(Path(repo["repository"]), relative)) != digest:
            raise ValueError("Official model source changed")
    if complete and [r["test_position"] for r in run["predictions"]] != run["positions"]:
        raise ValueError("Incomplete or duplicate prediction identities")
    with _dataset(run["config"]) as dataset:
        for item in run["predictions"]:
            if int(dataset.test_indices_1based[item["test_position"]]) != item["matlab_index_1based"]:
                raise ValueError("Prediction no longer matches test split")
            path = inside_path(directory, item["path"])
            if sha256_file(path) != item["sha256"]:
                raise ValueError("Raw prediction SHA mismatch")
            with np.load(path, allow_pickle=False) as archive:
                raw = archive["raw_relative"]
                if raw.shape != (480, 640) or raw.dtype != np.float32 or not np.isfinite(raw).all() or np.any(raw < 0):
                    raise ValueError("Invalid raw relative prediction")
    return directory, run


def predict(config, output_dir, *, positions=None, resume=False):
    import torch
    from src.models.depth_anything_v2_adapter import DepthAnythingV2Adapter
    torch.set_num_threads(config["runtime"]["torch_threads"])
    torch.manual_seed(config["runtime"]["seed"])
    resources = check_resources(config)
    selected = _positions(positions, config["dataset"]["expected_test_images"])
    directory = Path(output_dir).resolve()
    if resume:
        directory, run = _load_run(directory, complete=False)
        if run["config"] != config or run["positions"] != selected:
            raise ValueError("Resume config/selection differs from saved run")
        if run["status"] == "complete":
            _load_run(directory)
            return directory / "run.json"
    else:
        directory = _new_directory(directory)
        run = {"schema": "dav2_nyuv2_raw_v1", "status": "running", "created_at_utc": utc_now(),
               "config": config, "positions": selected, "predictions": [],
               "benchmark_scope": "full_654" if selected == list(range(654)) else "subset_smoke_only",
               "raw_contract": {"depth_type": "relative", "representation": "inverse_depth_like",
                                "unit": "arbitrary", "alignment": "none", "gt_used": False},
               **resources,
               "scientific_sources": {p: sha256_file(PROJECT_ROOT / p) for p in SOURCE_FILES}}
        for relative in SOURCE_FILES:
            dest = directory / "provenance" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((PROJECT_ROOT / relative).read_bytes())
        write_json(directory / "run.json", run)
    run["status"] = "running"
    invocation = {"started_at_utc": utc_now(), "resumed": resume, "environment": environment_info(),
                  "command": list(sys.argv), "working_directory": str(Path.cwd())}
    run.setdefault("invocations", []).append(invocation)
    write_json(directory / "run.json", run)
    try:
        start = time.perf_counter()
        adapter = DepthAnythingV2Adapter(repository=PROJECT_ROOT / config["model"]["repository"],
                    checkpoint=PROJECT_ROOT / config["model"]["checkpoint"],
                    device=config["runtime"]["device"], precision=config["runtime"]["precision"],
                    input_size=config["inference"]["input_size"])
        invocation["model_load_seconds"] = time.perf_counter() - start
        device = adapter.device
        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        done = {r["test_position"] for r in run["predictions"]}
        wall = time.perf_counter()
        with _dataset(config) as dataset:
            first = dataset.get_test_sample(selected[0])
            for _ in range(config["inference"]["warmup"]):
                adapter.infer(first.rgb)
            synchronize()
            for position in selected:
                if position in done:
                    continue
                sample = dataset.get_test_sample(position)
                times, raw = [], None
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                for repeat in range(config["inference"]["timing_repeats"]):
                    synchronize()
                    start = time.perf_counter()
                    value = adapter.infer(sample.rgb)
                    synchronize()
                    times.append(time.perf_counter() - start)
                    if repeat == 0:
                        raw = value.copy()
                if raw.shape != (480, 640) or raw.dtype != np.float32 or not np.isfinite(raw).all() or np.any(raw < 0):
                    raise ValueError("Model did not return finite nonnegative full-resolution raw")
                filename = f"predictions/test_{position:04d}_mat_{sample.matlab_index_1based:04d}.npz"
                _save_array(directory / filename, raw_relative=raw)
                record = {"test_position": position, "matlab_index_1based": sample.matlab_index_1based,
                          "scene": sample.scene, "scene_type": sample.scene_type,
                          "rgb_sha256": hashlib.sha256(sample.rgb.tobytes()).hexdigest(),
                          "path": filename, "sha256": sha256_file(directory / filename),
                          "inference_seconds": times, "mean_seconds": float(np.mean(times)),
                          "std_seconds": float(np.std(times)), "network_input_hw": adapter.metadata["network_input_hw"],
                          "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                          "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None}
                run["predictions"].append(record)
                run["model"] = dict(adapter.metadata)
                write_json(directory / "run.json", run)
                print(f"DA-V2 raw {position}: {record['mean_seconds']:.4f}s", flush=True)
        invocation["cached_input_to_saved_raw_seconds"] = time.perf_counter() - wall
        invocation["finished_at_utc"] = utc_now()
        run["status"] = "complete"
        write_json(directory / "run.json", run)
        return directory / "run.json"
    except BaseException:
        run["status"] = "failed"
        invocation["failure"] = traceback.format_exc()
        write_json(directory / "run.json", run)
        raise


def _measure(raw, sample, config):
    e = config["evaluation"]
    depth, mask, alignment = align_inverse_depth(raw, sample.depth_m, crop=e["crop"],
                                minimum=e["minimum_depth_m"], maximum=e["maximum_depth_m"])
    metrics = compute_depth_metrics(depth, sample.depth_m, min_depth=e["minimum_depth_m"],
                max_depth=e["maximum_depth_m"], external_mask=mask, clip_prediction=False)
    if metrics["valid_pixels"] != alignment["valid_pixels"]:
        raise ValueError("A valid GT pixel was excluded after alignment")
    return depth, mask, alignment, metrics


def _aggregate(rows, config):
    rng = np.random.default_rng(config["statistics"]["seed"])
    count = config["statistics"]["bootstrap_resamples"]
    mean, ci = {}, {}
    for metric in METRICS:
        values = np.array([r[metric] for r in rows])
        mean[metric] = float(values.mean())
        means = np.concatenate([values[rng.integers(0, len(values), (min(250, count - i), len(values)))].mean(1)
                                for i in range(0, count, 250)])
        ci[metric] = dict(zip(("lower", "upper"), map(float, np.quantile(means, [0.025, 0.975]))))
    return mean, ci


def evaluate(run_dir, output_dir):
    from src.visualization.nyuv2 import save_nyuv2_panel, save_contact_sheet
    directory, run = _load_run(run_dir)
    output = _new_directory(output_dir)
    status = {"status": "running", "created_at_utc": utc_now(), "models_executed": False,
              "command": list(sys.argv), "working_directory": str(Path.cwd())}
    write_json(output / "status.json", status)
    try:
        config, rows = run["config"], []
        with _dataset(config) as dataset:
            for record in run["predictions"]:
                sample = dataset.get_test_sample(record["test_position"])
                with np.load(directory / record["path"], allow_pickle=False) as archive:
                    raw = archive["raw_relative"]
                depth, mask, alignment, metrics = _measure(raw, sample, config)
                filename = Path(record["path"]).name
                _save_array(output / "aligned" / filename, depth_after_gt_alignment_m=depth.astype(np.float32),
                            evaluation_mask=mask)
                rows.append({"test_position": sample.test_position, "matlab_index_1based": sample.matlab_index_1based,
                             "scene": sample.scene, "scene_type": sample.scene_type,
                             **metrics, **alignment})
            with (output / "metrics_per_image.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            mean, ci = _aggregate(rows, config)
            times = [r["mean_seconds"] for r in run["predictions"]]
            summary = {"label": "aligned_relative", "native_metric": False, "gt_used_for_alignment": True,
                       "benchmark_scope": run["benchmark_scope"], "image_count": len(rows),
                       "evaluation_protocol": config["evaluation"], "macro_mean": mean, "bootstrap_95_ci": ci,
                       "fallback_images": sum(r["fallback_reason"] is not None for r in rows),
                       "inverse_clipped_pixels": sum(r["inverse_below_range_pixels"] + r["inverse_above_range_pixels"] for r in rows),
                       "efficiency": {"mean_seconds": float(np.mean(times)), "model_fps": float(1 / np.mean(times)),
                                      "scope": "model infer incl preprocessing/transfer/resize; excludes alignment, data load and saving",
                                      "parameters": run["model"]["parameters"],
                                      "peak_allocated_bytes": max((r['peak_allocated_bytes'] or 0 for r in run['predictions']), default=0) or None,
                                      "peak_reserved_bytes": max((r['peak_reserved_bytes'] or 0 for r in run['predictions']), default=0) or None},
                       "run_source": _binding(directory / "run.json"), "environment": environment_info()}
            write_json(output / "metrics_summary.json", summary)
            chosen = set(config["visualization"]["audit_positions"]) & set(run["positions"])
            chosen |= {r["test_position"] for r in sorted(rows, key=lambda r: -r["abs_rel"])[:config["visualization"]["worst_count"]]}
            panels = []
            crop = config["evaluation"]["crop"]
            for position in sorted(chosen):
                sample = dataset.get_test_sample(position)
                record = next(r for r in run["predictions"] if r["test_position"] == position)
                with np.load(output / "aligned" / Path(record["path"]).name, allow_pickle=False) as archive:
                    depth = archive["depth_after_gt_alignment_m"]
                    mask = archive["evaluation_mask"]
                depth = np.where(mask, depth, np.nan)
                target = np.where(mask, sample.depth_m, np.nan)
                panel = output / "visualization" / f"test_{position:04d}.png"
                save_nyuv2_panel(panel, rgb=sample.rgb, target_depth_m=target, prediction_depth_m=depth,
                    crop=(slice(crop[0], crop[1]), slice(crop[2], crop[3])),
                    title=f"DA-V2-S aligned relative / test {position} / test GT used",
                    prediction_title="GT-aligned relative (not native metric)",
                    depth_range_m=tuple(config["visualization"]["depth_range_m"]))
                panels.append(panel)
            if panels:
                save_contact_sheet(output / "visualization/contact.jpg", panels)
        manifest = {"schema": "dav2_nyuv2_evaluation_v1", "status": "complete",
                    "run": _binding(directory / "run.json"), "rows": rows,
                    "artifacts": [{"path": str(p.relative_to(output)), "sha256": sha256_file(p)}
                                  for p in sorted(output.rglob('*')) if p.is_file() and p.name != "status.json"]}
        write_json(output / "evaluation.json", manifest)
        write_json(output / "status.json", {**status, "status": "complete", "evaluation_sha256": sha256_file(output / "evaluation.json")})
        return output / "metrics_summary.json"
    except BaseException:
        write_json(output / "status.json", {**status, "status": "failed", "failure": traceback.format_exc()})
        raise


def verify(evaluation_dir):
    output = Path(evaluation_dir).resolve()
    status = read_json(output / "status.json")
    if status["status"] != "complete" or sha256_file(output / "evaluation.json") != status["evaluation_sha256"]:
        raise ValueError("Evaluation incomplete or changed")
    manifest = read_json(output / "evaluation.json")
    directory, run = _load_run(_checked(manifest["run"]).parent)
    for artifact in manifest["artifacts"]:
        if sha256_file(inside_path(output, artifact["path"])) != artifact["sha256"]:
            raise ValueError("Evaluation artifact changed")
    rows = []
    with _dataset(run["config"]) as dataset:
        for record in run["predictions"]:
            sample = dataset.get_test_sample(record["test_position"])
            with np.load(directory / record["path"], allow_pickle=False) as archive:
                raw = archive["raw_relative"]
            depth, mask, alignment, metrics = _measure(raw, sample, run["config"])
            with np.load(output / "aligned" / Path(record["path"]).name, allow_pickle=False) as archive:
                np.testing.assert_allclose(archive["depth_after_gt_alignment_m"], depth, rtol=1e-6, atol=1e-8)
                np.testing.assert_array_equal(archive["evaluation_mask"], mask)
            rows.append({"test_position": sample.test_position, "matlab_index_1based": sample.matlab_index_1based,
                         "scene": sample.scene, "scene_type": sample.scene_type, **metrics, **alignment})
    with (output / "metrics_per_image.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(rows) != len(manifest["rows"]) or len(csv_rows) != len(rows):
        raise ValueError("Row count differs")
    for expected, saved, csv_row in zip(rows, manifest["rows"], csv_rows):
        for key, value in expected.items():
            if isinstance(value, bool):
                if saved[key] is not value or csv_row[key] != str(value):
                    raise ValueError(f"Recomputed boolean differs: {key}")
            elif isinstance(value, (float, int)):
                np.testing.assert_allclose([saved[key], float(csv_row[key])], value, rtol=1e-6, atol=1e-8)
            elif saved[key] != value or csv_row[key] != ("" if value is None else value):
                raise ValueError(f"Recomputed field differs: {key}")
    summary = read_json(output / "metrics_summary.json")
    if summary["label"] != "aligned_relative" or summary["native_metric"] or not summary["gt_used_for_alignment"]:
        raise ValueError("Invalid evaluation label")
    mean, ci = _aggregate(rows, run["config"])
    for metric in METRICS:
        np.testing.assert_allclose(summary["macro_mean"][metric], mean[metric], rtol=1e-6, atol=1e-8)
        for bound in ["lower", "upper"]:
            np.testing.assert_allclose(summary["bootstrap_95_ci"][metric][bound], ci[metric][bound], rtol=1e-6, atol=1e-8)
    return {"status": "verified", "images": len(rows), "benchmark_scope": run["benchmark_scope"],
            "label": "aligned_relative", "models_executed": False,
            "recomputed_from_raw_and_gt": True, "numeric_tolerance": {"relative": 1e-6, "absolute": 1e-8}}
