"""Frozen-protocol inference and shared RGB-flow caching; no training."""

from __future__ import annotations

from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from .common import (
    PROJECT_ROOT, create_output_directory, environment_info, inside_path,
    read_json, repository_identity, sha256_file, source_fingerprints,
    utc_now, verify_source_fingerprints, write_json,
)


def load_config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("shared_protocol"):
        return load_config(PROJECT_ROOT / config["shared_protocol"])
    return validate_config(config)


def validate_config(config: dict) -> dict:
    if config.get("schema_version") != "relative_video_v1":
        raise ValueError("Expected the relative_video_v1 shared configuration")
    ev = config["evaluation"]
    if ev["alignment"] != "none":
        raise ValueError("Relative video evaluation does not use per-frame alignment")
    if not 0 <= ev["percentile_low"] < ev["percentile_high"] <= 100:
        raise ValueError("Invalid reference percentiles")
    if ev["fb_threshold_px"] < 0 or not 0 <= ev["minimum_coverage"] <= 1:
        raise ValueError("Invalid flow threshold or minimum coverage")
    if config["runtime"]["precision"] not in {"float16", "float32"}:
        raise ValueError("Use float16 or float32")
    if config["inference"]["timing_repeats"] < 1 or config["inference"]["warmup"] < 0:
        raise ValueError("Invalid timing repetition or warmup count")
    for section, key, minimum in (("inference", "input_size", 14),
                                  ("inference", "timing_repeats", 1),
                                  ("inference", "warmup", 0),
                                  ("runtime", "torch_threads", 1)):
        value = config[section][key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid {section}.{key}")
    if config["runtime"]["device"] == "cpu" and config["runtime"]["precision"] != "float32":
        raise ValueError("CPU protocol requires float32")
    return config


def resolve_resource(relative: str) -> Path:
    return (PROJECT_ROOT / relative).resolve()


def check_resources(config: dict) -> dict:
    resources = {}
    for name, spec in config["models"].items():
        identity = repository_identity(resolve_resource(spec["repository"]))
        if identity["commit"] != spec["commit"]:
            raise ValueError(f"Unexpected {name} source commit")
        if identity["has_substantive_changes"] or identity["staged_changes"]:
            raise ValueError(f"Modified third-party source for {name}")
        checkpoint = resolve_resource(spec["checkpoint"])
        digest = sha256_file(checkpoint)
        if digest != spec["sha256"]:
            raise ValueError(f"Unexpected {name} checkpoint SHA256")
        resources[name] = {"repository": identity, "checkpoint": str(checkpoint),
                           "checkpoint_sha256": digest,
                           "checkpoint_bytes": checkpoint.stat().st_size}
    if config["flow"]["backend"] == "raft_small":
        checkpoint = resolve_resource(config["flow"]["checkpoint"])
        digest = sha256_file(checkpoint)
        if digest != config["flow"]["sha256"]:
            raise ValueError("Unexpected RAFT checkpoint SHA256")
        resources["flow"] = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest}
    elif config["flow"]["backend"] != "farneback":
        raise ValueError("Choose raft_small or explicit farneback")
    return resources


def freeze_protocol(config: dict, manifest_path: Path, output: Path,
                    *, config_path: Path | None = None) -> Path:
    from .frames import validate_manifest
    validate_config(config)
    manifest_path = Path(manifest_path).resolve()
    manifest = validate_manifest(manifest_path, verify_hashes=True)
    if len(manifest["frames"]) < 2:
        raise ValueError("At least two video frames are required")
    if not 0 <= config["evaluation"]["reference_index"] < len(manifest["frames"]):
        raise ValueError("Reference frame is outside the selected video")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Protocol already exists: {output}")
    resources = check_resources(config)
    value = {
        "schema_version": "frozen_relative_video_v1", "status": "frozen",
        "created_at_utc": utc_now(), "command": sys.argv,
        "configuration": deepcopy(config), "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path), "resources": resources,
        "project_source_sha256": source_fingerprints(),
        "source_config_path": str(config_path.resolve()) if config_path else None,
        "source_config_sha256": sha256_file(config_path) if config_path else None,
        "scope": {"source_kind": manifest["source_kind"], "split": manifest["split"],
                  "camera_motion": manifest["camera_motion"],
                  "is_course_self_recorded": manifest["source_kind"] == "self_recorded",
                  "mandatory_clip_status": "not_inferred_from_filename"},
    }
    value["project_source_archive_base64"] = {
        relative: base64.b64encode((PROJECT_ROOT / relative).read_bytes()).decode("ascii")
        for relative in value["project_source_sha256"]
    }
    value["protocol_content_sha256"] = _protocol_digest(value)
    write_json(output, value)
    return output


def _protocol_digest(value: dict) -> str:
    unsealed = {key: item for key, item in value.items() if key != "protocol_content_sha256"}
    return hashlib.sha256(json.dumps(unsealed, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def load_protocol(path: Path, *, verify_resources: bool = False) -> tuple[dict, dict, Path]:
    from .frames import validate_manifest
    path = Path(path).resolve()
    protocol = read_json(path)
    if protocol.get("schema_version") != "frozen_relative_video_v1" or protocol.get("status") != "frozen":
        raise ValueError("Expected a frozen relative video protocol")
    if protocol.get("protocol_content_sha256") != _protocol_digest(protocol):
        raise ValueError("Frozen protocol content changed")
    for relative, expected in protocol["project_source_sha256"].items():
        content = base64.b64decode(protocol["project_source_archive_base64"][relative], validate=True)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Frozen scientific source archive changed")
    verify_source_fingerprints(protocol["project_source_sha256"])
    manifest_path = Path(protocol["manifest_path"])
    if sha256_file(manifest_path) != protocol["manifest_sha256"]:
        raise ValueError("Frozen input manifest changed")
    manifest = validate_manifest(manifest_path, verify_hashes=True)
    if verify_resources:
        current = check_resources(protocol["configuration"])
        if current != protocol["resources"]:
            raise ValueError("Frozen resource identities changed")
    return protocol, manifest, manifest_path


def _runtime(config: dict):
    import torch
    runtime = config["runtime"]
    if runtime.get("torch_threads"):
        torch.set_num_threads(int(runtime["torch_threads"]))
    device = torch.device(runtime["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only explicit CPU or CUDA devices are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no silent CPU fallback. Freeze a separate CPU protocol for smoke tests.")
        torch.cuda.set_device(device)
    elif runtime["precision"] != "float32":
        raise ValueError("CPU runs require float32")
    torch.manual_seed(int(runtime["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(runtime["seed"]))
    torch.backends.cudnn.benchmark = False
    return torch, device


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peaks(torch, device) -> dict:
    return {"peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None}


def infer_run(protocol_path: Path, model_name: str, output_dir: Path) -> Path:
    if model_name not in {"dav2", "vda"}:
        raise ValueError("Use dav2 or vda")
    protocol, manifest, manifest_path = load_protocol(protocol_path, verify_resources=True)
    config = protocol["configuration"]
    output = create_output_directory(output_dir)
    start_total = time.perf_counter()
    record = {"schema_version": "relative_video_run_v1", "status": "running",
              "started_at_utc": utc_now(), "command": sys.argv,
              "protocol_path": str(Path(protocol_path).resolve()),
              "protocol_sha256": sha256_file(protocol_path),
              "manifest_path": str(manifest_path),
              "manifest_sha256": protocol["manifest_sha256"], "config": config,
              "project_source_sha256": source_fingerprints(), "frames": []}
    write_json(output / "run.json", record)
    try:
        torch, device = _runtime(config)
        record["environment"] = environment_info()
        from .frames import load_frames
        _, frames = load_frames(manifest_path, verify_hashes=True)
        from src.models.depth_anything_v2_adapter import DepthAnythingV2Adapter
        from src.models.video_depth_anything_adapter import VideoDepthAnythingAdapter
        adapter_class = DepthAnythingV2Adapter if model_name == "dav2" else VideoDepthAnythingAdapter
        spec = config["models"][model_name]
        start_load = time.perf_counter()
        adapter = adapter_class(repository=resolve_resource(spec["repository"]),
                                checkpoint=resolve_resource(spec["checkpoint"]),
                                device=str(device), precision=config["runtime"]["precision"],
                                input_size=int(config["inference"]["input_size"]))
        _sync(torch, device)
        load_seconds = time.perf_counter() - start_load
        warmup_count = int(config["inference"]["warmup"])
        warmup_frames = min(32, len(frames)) if model_name == "vda" else 1
        start_warmup = time.perf_counter()
        for _ in range(warmup_count):
            if model_name == "dav2":
                adapter.infer(frames[0])
            else:
                adapter.infer(frames[:warmup_frames], manifest["fps"])
        _sync(torch, device)
        warmup_seconds = time.perf_counter() - start_warmup
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        durations, frame_durations, depths = [], [], None
        for repetition in range(int(config["inference"]["timing_repeats"])):
            _sync(torch, device)
            start = time.perf_counter()
            if model_name == "dav2":
                predictions, current_durations = [], []
                for rgb in frames:
                    _sync(torch, device)
                    frame_start = time.perf_counter()
                    predictions.append(adapter.infer(rgb))
                    _sync(torch, device)
                    current_durations.append(time.perf_counter() - frame_start)
                result = np.stack(predictions)
                frame_durations.append(current_durations)
            else:
                result = adapter.infer(frames, manifest["fps"])
            _sync(torch, device)
            durations.append(time.perf_counter() - start)
            # Preserve the first measured pass; other passes measure throughput,
            # never select predictions according to a stability score.
            if repetition == 0:
                depths = np.array(result, dtype=np.float32, copy=True)
        if depths.shape != frames.shape[:3] or depths.dtype != np.float32:
            raise ValueError(f"Unexpected output shape/dtype: {depths.shape}/{depths.dtype}")
        if not np.isfinite(depths).all():
            raise ValueError("Non-finite model prediction; run remains failed")
        record["model"] = dict(adapter.metadata)
        record["model"].update(name=model_name, depth_type="relative", representation="inverse_depth_like",
                               unit="arbitrary", alignment="none",
                               precision=config["runtime"]["precision"],
                               actual_output_hw=[manifest["height"], manifest["width"]])
        (output / "raw").mkdir()
        for entry, prediction in zip(manifest["frames"], depths):
            relative = f"raw/frame_{entry['index']:06d}.npy"
            np.save(output / relative, prediction, allow_pickle=False)
            record["frames"].append({"index": entry["index"], "source_index": entry["source_index"],
                                     "timestamp_seconds": entry["timestamp_seconds"],
                                     "raw_path": relative, "raw_sha256": sha256_file(output / relative),
                                     "shape": list(prediction.shape), "dtype": str(prediction.dtype)})
        mean_model = float(np.mean(durations))
        record["timing"] = {
            "output_frames": len(frames), "model_load_seconds": load_seconds,
            "warmup_count": warmup_count, "warmup_frames_per_pass": warmup_frames,
            "warmup_seconds": warmup_seconds, "timing_repeats": len(durations),
            "measured_sequence_seconds": durations, "model_seconds": mean_model,
            "model_seconds_std": float(np.std(durations)), "model_fps": len(frames) / mean_model,
            "amortized_ms_per_output_frame": 1000 * mean_model / len(frames),
            "framewise_latency_seconds": frame_durations if model_name == "dav2" else None,
            "offline_sequence_latency_seconds": durations if model_name == "vda" else None,
            "latency_scope": "per-frame DA inference" if model_name == "dav2" else "offline full-sequence latency; amortized cost is not online latency",
            "pipeline_seconds_from_verified_cached_rgb": time.perf_counter() - start_total,
            "pipeline_includes": "cached PNG reads, model load, warmup, all timing repetitions, NPY and SHA saves; excludes original video decode, flow and evaluation",
            "original_video_preparation_seconds": manifest.get("preparation_seconds"),
            **_peaks(torch, device),
        }
        record.update(status="complete", finished_at_utc=utc_now())
        write_json(output / "run.json", record)
        return output / "run.json"
    except Exception as exc:
        record.update(status="failed", finished_at_utc=utc_now(),
                      error={"type": type(exc).__name__, "message": str(exc)},
                      elapsed_seconds=time.perf_counter() - start_total)
        write_json(output / "run.json", record)
        raise


def cache_flow(protocol_path: Path, output_dir: Path) -> Path:
    protocol, manifest, manifest_path = load_protocol(protocol_path, verify_resources=True)
    output = create_output_directory(output_dir)
    config = protocol["configuration"]
    record = {"schema_version": "shared_video_flow_v1", "status": "running",
              "started_at_utc": utc_now(), "command": sys.argv,
              "protocol_path": str(Path(protocol_path).resolve()),
              "protocol_sha256": sha256_file(protocol_path),
              "manifest_path": str(manifest_path), "manifest_sha256": protocol["manifest_sha256"],
              "frames_count": len(manifest["frames"]), "shape": [manifest["height"], manifest["width"]],
              "project_source_sha256": source_fingerprints(), "pairs": []}
    write_json(output / "flow_manifest.json", record)
    start_total = time.perf_counter()
    try:
        torch, device = _runtime(config)
        from .frames import load_frames
        from .flow import FlowEstimator
        _, frames = load_frames(manifest_path)
        flow_spec = config["flow"]
        estimator = FlowEstimator(backend=flow_spec["backend"], device="cpu" if flow_spec["backend"] == "farneback" else str(device),
                                  checkpoint=resolve_resource(flow_spec["checkpoint"]) if flow_spec["backend"] == "raft_small" else None,
                                  max_long_edge=flow_spec.get("max_long_edge"),
                                  iterations=flow_spec["iterations"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        (output / "pairs").mkdir()
        durations = []
        for index in range(1, len(frames)):
            _sync(torch, device)
            start = time.perf_counter()
            forward, backward = estimator.estimate(frames[index - 1], frames[index])
            _sync(torch, device)
            durations.append(time.perf_counter() - start)
            entry = {"previous_index": index - 1, "current_index": index}
            for direction, array in (("forward", forward), ("backward", backward)):
                if array.shape != (*record["shape"], 2) or not np.isfinite(array).all():
                    raise ValueError("Invalid flow geometry or non-finite output")
                relative = f"pairs/{index - 1:06d}_{index:06d}_{direction}.npy"
                np.save(output / relative, np.asarray(array, dtype=np.float32), allow_pickle=False)
                entry[direction + "_path"] = relative
                entry[direction + "_sha256"] = sha256_file(output / relative)
            record["pairs"].append(entry)
        record.update(status="complete", finished_at_utc=utc_now(),
                      flow=dict(estimator.metadata), environment=environment_info(),
                      timing={"pair_bidirectional_seconds": durations,
                              "total_seconds": time.perf_counter() - start_total, **_peaks(torch, device)})
        write_json(output / "flow_manifest.json", record)
        return output / "flow_manifest.json"
    except Exception as exc:
        record.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)},
                      elapsed_seconds=time.perf_counter() - start_total, finished_at_utc=utc_now())
        write_json(output / "flow_manifest.json", record)
        raise


def verify_run(run_dir: Path, *, verify_sources: bool = True) -> dict:
    from .frames import validate_manifest
    run_dir = Path(run_dir).resolve()
    record = read_json(run_dir / "run.json")
    if record.get("status") != "complete":
        raise ValueError("Run is not complete")
    protocol_path = Path(record["protocol_path"])
    if sha256_file(protocol_path) != record["protocol_sha256"]:
        raise ValueError("Run frozen protocol changed")
    protocol, _, _ = load_protocol(protocol_path)
    if protocol["manifest_sha256"] != record["manifest_sha256"] or protocol["configuration"] != record["config"]:
        raise ValueError("Run protocol/input/config binding mismatch")
    manifest_path = Path(record["manifest_path"])
    if sha256_file(manifest_path) != record["manifest_sha256"]:
        raise ValueError("Run input manifest changed")
    manifest = validate_manifest(manifest_path)
    if verify_sources:
        verify_source_fingerprints(record["project_source_sha256"])
    if len(record["frames"]) != len(manifest["frames"]):
        raise ValueError("Prediction frame count mismatch")
    for saved, expected in zip(record["frames"], manifest["frames"]):
        if saved["index"] != expected["index"] or saved["source_index"] != expected["source_index"]:
            raise ValueError("Prediction frame identity mismatch")
        if saved["timestamp_seconds"] != expected["timestamp_seconds"]:
            raise ValueError("Prediction timestamp mismatch")
        path = inside_path(run_dir, saved["raw_path"])
        if sha256_file(path) != saved["raw_sha256"]:
            raise ValueError("Prediction SHA256 mismatch")
        array = np.load(path, allow_pickle=False)
        if array.shape != (manifest["height"], manifest["width"]) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("Prediction shape/dtype/finiteness mismatch")
    return {"status": "passed", "frames": len(record["frames"]),
            "model": record["model"]["name"], "run_sha256": sha256_file(run_dir / "run.json")}
