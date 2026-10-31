"""Prepare one immutable, timestamp-aware RGB cache shared by video models.

The source video is never rewritten.  Frame PTS come from ffprobe, rather than
OpenCV's nominal frame-rate counter, so phone VFR inputs retain their identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np


SCHEMA_VERSION = "video_frames_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "command_argv": list(sys.argv),
        "working_directory": str(Path.cwd()),
    }


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return number


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _probe_video(source: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-show_frames", "-show_entries",
        "stream:frame=best_effort_timestamp_time,pts_time,width,height",
        "-of", "json", str(source),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to preserve real video timestamps; install ffmpeg before preparation") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"ffprobe could not inspect video: {exc.stderr.strip()}") from exc
    metadata = json.loads(result.stdout)
    streams = metadata.get("streams", [])
    frames = metadata.get("frames", [])
    if len(streams) != 1 or not frames:
        raise ValueError("source contains no readable video frames")
    timestamps = []
    for frame in frames:
        value = frame.get("best_effort_timestamp_time", frame.get("pts_time"))
        if value is None:
            raise ValueError("video frame has no PTS; nominal FPS cannot safely recover VFR timestamps")
        timestamps.append(_finite_number(value, "frame timestamp"))
    pts = np.asarray(timestamps, dtype=np.float64)
    if len(pts) > 1 and np.any(np.diff(pts) <= 0):
        raise ValueError("video timestamps must be strictly increasing; duplicate or reordered PTS need an explicit derived source")
    sizes = {(int(frame["height"]), int(frame["width"])) for frame in frames}
    if len(sizes) != 1:
        raise ValueError("variable-resolution video is unsupported; prepare an explicit constant-resolution derived source")
    version = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    return {"stream": streams[0], "frames": frames, "timestamps": pts, "command": command, "version": version, "executable": shutil.which("ffprobe")}


def _selection_indices(
    timestamps: np.ndarray, target_fps: float, start_seconds: float,
    end_seconds: float | None, max_frames: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    relative = timestamps - timestamps[0]
    if len(relative) < 2:
        raise ValueError("at least two source frames with timestamps are required")
    intervals = np.diff(relative)
    median_interval = float(np.median(intervals))
    average_fps = (len(relative) - 1) / float(relative[-1])
    if target_fps > average_fps * (1.0 + 1e-5):
        raise ValueError(f"target_fps {target_fps:g} exceeds available average source FPS {average_fps:g}; upsampling is not allowed")
    stop = float(relative[-1] + median_interval) if end_seconds is None else end_seconds
    if stop <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    if start_seconds > relative[-1] + 1e-9:
        raise ValueError("clip start lies after the last source frame")
    if end_seconds is not None and end_seconds > relative[-1] + median_interval + 1e-5:
        raise ValueError("clip end lies beyond the source video duration")
    # At most one target instant per selected source frame is useful.  Build
    # target instants first, then remove repeated indices for sparse VFR spans.
    grid_count = int(math.ceil((stop - start_seconds) * target_fps - 1e-9))
    grid = start_seconds + np.arange(grid_count, dtype=np.float64) / target_fps
    indices = np.searchsorted(relative + 1e-9, grid, side="left")
    indices = indices[indices < len(relative)]
    indices = np.unique(indices)
    indices = indices[relative[indices] < stop - 1e-9]
    if max_frames is not None:
        indices = indices[:max_frames]
    if len(indices) < 2:
        raise ValueError("selection contains fewer than two distinct frames; video evaluation needs a frame pair")
    tolerance = max(1e-5, median_interval * 1e-3)
    return indices, {
        "rule": "first_source_pts_at_or_after_target_no_duplicates",
        "timestamp_origin": "first_source_frame_pts",
        "target_fps": target_fps,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "max_frames": max_frames,
        "source_average_fps": average_fps,
        "source_median_interval_seconds": median_interval,
        "vfr_detected": bool(np.any(np.abs(intervals - median_interval) > tolerance)),
        "no_upsampling": True,
        "selected_source_indices": indices.tolist(),
        "selected_timestamps_seconds": relative[indices].tolist(),
    }


def _contained_file(base: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("cached file path must be a nonempty relative path")
    target = (base / relative).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"cached path escapes manifest directory: {relative}") from exc
    if not target.is_file():
        raise ValueError(f"cached file is missing: {relative}")
    return target


def prepare_frames(
    source: Path, output_dir: Path, *, clip_id: str,
    source_kind: str = "public", camera_motion: str = "unknown",
    split: str = "development", target_fps: float = 10.,
    max_long_edge: int = 640, max_frames: int | None = None,
    start_seconds: float = 0., end_seconds: float | None = None,
    roi_path: Path | None = None, provenance: dict[str, Any] | None = None,
) -> Path:
    """Decode once and freeze RGB identities/PTS for both video models.

    ``roi_path`` is an optional binary/nonzero PNG already on the final RGB
    canvas.  Errors leave ``failure.json`` and an incomplete manifest, and
    existing output directories (including failed ones) are never overwritten.
    """
    source = Path(source).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite frame cache: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "clip_id": clip_id,
        "source_kind": source_kind, "camera_motion": camera_motion,
        "split": split, "complete": False, "frames": [],
        "source": {"path": str(source)}, "roi_path": None, "roi_sha256": None,
        "environment": _environment(), "preparation_started_unix": started,
        "code_sha256": _sha256(Path(__file__)),
        "source_fingerprints": {"src/video/frames.py": _sha256(Path(__file__))},
    }
    _write_json(manifest_path, manifest)
    capture = None
    try:
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError("clip_id must be a nonempty string")
        if source_kind not in {"public", "self_recorded", "mandatory"}:
            raise ValueError("source_kind must be public, self_recorded, or mandatory")
        if camera_motion not in {"fixed", "moving", "unknown"}:
            raise ValueError("camera_motion must be fixed, moving, or unknown")
        if not isinstance(split, str) or not split.strip():
            raise ValueError("split must be a nonempty string")
        target_fps = _finite_number(target_fps, "target_fps", positive=True)
        max_long_edge = _positive_integer(max_long_edge, "max_long_edge")
        if max_frames is not None:
            max_frames = _positive_integer(max_frames, "max_frames")
        start_seconds = _finite_number(start_seconds, "start_seconds")
        if start_seconds < 0:
            raise ValueError("start_seconds must be nonnegative")
        if end_seconds is not None:
            end_seconds = _finite_number(end_seconds, "end_seconds")
        if not source.is_file():
            raise ValueError(f"source video is missing: {source}")
        # Copy JSON-safe provenance so later caller mutation cannot change it.
        provenance = json.loads(json.dumps(provenance or {}, allow_nan=False))
        probe = _probe_video(source)
        timestamps = probe["timestamps"]
        selected, selection = _selection_indices(timestamps, target_fps, start_seconds, end_seconds, max_frames)
        manifest["source"] = {
            "path": str(source), "sha256": _sha256(source),
            "size_bytes": source.stat().st_size, "stream": probe["stream"],
            "frame_count": len(timestamps), "first_pts_seconds": float(timestamps[0]),
            "last_pts_seconds": float(timestamps[-1]), "provenance": provenance,
            "ffprobe_command": probe["command"],
            "ffprobe_version": probe["version"], "ffprobe_executable": probe["executable"],
        }
        manifest["selection"] = selection
        manifest["fps"] = target_fps
        manifest["resize"] = {"max_long_edge": max_long_edge, "keep_aspect_ratio": True, "interpolation": "opencv_INTER_AREA_downsample"}
        (output_dir / "rgb").mkdir()
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError("OpenCV could not open source video")
        # OpenCV normally applies container orientation.  Record that behavior
        # and demand the decoded dimensions agree with the stream rotation.
        orientation_auto = capture.get(cv2.CAP_PROP_ORIENTATION_AUTO) if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO") else None
        manifest["decoder"] = {"name": "opencv_VideoCapture", "orientation_auto": orientation_auto}
        source_height = int(probe["stream"]["height"])
        source_width = int(probe["stream"]["width"])
        rotation = float(probe["stream"].get("tags", {}).get("rotate", 0.0))
        for data in probe["stream"].get("side_data_list", []):
            if "rotation" in data:
                rotation = float(data["rotation"])
        expected_shape = (source_width, source_height) if int(round(rotation)) % 180 == 90 and orientation_auto else (source_height, source_width)
        manifest["decoder"]["source_rotation_degrees"] = rotation
        manifest["decoder"]["expected_decoded_height_width"] = list(expected_shape)
        selected_positions = {int(index): position for position, index in enumerate(selected)}
        decoded_count = 0
        decoded_shape = None
        for source_index in range(len(timestamps)):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"decoder stopped at frame {source_index}, before ffprobe frame count {len(timestamps)}")
            decoded_count += 1
            shape = tuple(bgr.shape[:2])
            if decoded_shape is None:
                decoded_shape = shape
                if shape != expected_shape:
                    raise ValueError(f"decoded orientation/shape {shape} disagrees with expected {expected_shape}")
                factor = min(1.0, max_long_edge / max(shape))
                height, width = max(1, int(round(shape[0] * factor))), max(1, int(round(shape[1] * factor)))
                manifest.update({"height": height, "width": width})
                manifest["decoder"]["decoded_height_width"] = list(shape)
            elif shape != decoded_shape:
                raise ValueError("decoded frame resolution changed inside source video")
            if source_index not in selected_positions:
                continue
            if shape != (manifest["height"], manifest["width"]):
                bgr = cv2.resize(bgr, (manifest["width"], manifest["height"]), interpolation=cv2.INTER_AREA)
            frame_index = selected_positions[source_index]
            relative_name = f"rgb/{frame_index:06d}.png"
            frame_path = output_dir / relative_name
            if not cv2.imwrite(str(frame_path), bgr):
                raise IOError(f"failed to save RGB PNG: {frame_path}")
            manifest["frames"].append({
                "index": frame_index, "source_index": source_index,
                "timestamp_seconds": float(timestamps[source_index] - timestamps[0]),
                "source_timestamp_seconds": float(timestamps[source_index]),
                "rgb_path": relative_name, "rgb_sha256": _sha256(frame_path),
            })
        extra, _ = capture.read()
        if extra:
            raise ValueError("decoder returned more frames than ffprobe; source frame identity is ambiguous")
        manifest["decoder"]["decoded_frame_count"] = decoded_count
        if roi_path is not None:
            roi_source = Path(roi_path).expanduser().resolve()
            if roi_source.suffix.lower() != ".png":
                raise ValueError("ROI source must be a PNG on the final shared RGB canvas")
            roi = cv2.imread(str(roi_source), cv2.IMREAD_GRAYSCALE)
            if roi is None or roi.shape != (manifest["height"], manifest["width"]):
                raise ValueError("ROI must be a readable PNG with the final shared RGB height and width")
            if not np.any(roi > 0):
                raise ValueError("ROI must contain at least one nonzero pixel")
            target = output_dir / "roi.png"
            if not cv2.imwrite(str(target), (roi > 0).astype(np.uint8) * 255):
                raise IOError("failed to save fixed ROI")
            manifest.update({"roi_path": "roi.png", "roi_sha256": _sha256(target), "roi_source": {"path": str(roi_source), "sha256": _sha256(roi_source)}})
        manifest["complete"] = True
        manifest["preparation_finished_unix"] = time.time()
        manifest["preparation_seconds"] = time.time() - started
        _write_json(manifest_path, manifest)
        validate_manifest(manifest_path)
        return manifest_path
    except Exception as exc:
        manifest["complete"] = False
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        manifest["preparation_finished_unix"] = time.time()
        _write_json(manifest_path, manifest)
        _write_json(output_dir / "failure.json", {"complete": False, "error": manifest["error"], "command_argv": list(sys.argv)})
        raise
    finally:
        if capture is not None:
            capture.release()


def validate_manifest(manifest_path: Path, verify_hashes: bool = True) -> dict[str, Any]:
    """Validate cached identities without requiring the original source file."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("complete") is not True:
        raise ValueError("frame manifest is incomplete or has an unsupported schema")
    height = _positive_integer(manifest.get("height"), "manifest height")
    width = _positive_integer(manifest.get("width"), "manifest width")
    _finite_number(manifest.get("fps"), "manifest fps", positive=True)
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("manifest must contain at least two frames")
    source = manifest.get("source", {})
    source_count = _positive_integer(source.get("frame_count"), "source frame count")
    source_first_pts = _finite_number(source.get("first_pts_seconds"), "source first PTS")
    source_hash = source.get("sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(character not in "0123456789abcdef" for character in source_hash):
        raise ValueError("source SHA256 must be recorded in the manifest")
    previous_source_index = -1
    previous_timestamp = -math.inf
    paths = set()
    for index, frame in enumerate(frames):
        source_index = frame.get("source_index")
        timestamp = _finite_number(frame.get("timestamp_seconds"), "frame timestamp")
        if frame.get("index") != index or not isinstance(source_index, int) or isinstance(source_index, bool) or source_index <= previous_source_index or source_index >= source_count:
            raise ValueError("frame indices must be contiguous and source indices strictly increasing")
        if timestamp < 0 or timestamp <= previous_timestamp:
            raise ValueError("selected frame timestamps must be nonnegative and strictly increasing")
        source_timestamp = _finite_number(frame.get("source_timestamp_seconds"), "frame source PTS")
        if not math.isclose(source_timestamp - source_first_pts, timestamp, rel_tol=0, abs_tol=1e-8):
            raise ValueError("selected relative timestamp disagrees with source PTS")
        previous_source_index, previous_timestamp = source_index, timestamp
        path = _contained_file(manifest_path.parent, frame.get("rgb_path"))
        if path in paths:
            raise ValueError("each selected frame must have a distinct cached path")
        paths.add(path)
        if verify_hashes and _sha256(path) != frame.get("rgb_sha256"):
            raise ValueError(f"RGB SHA256 mismatch: {frame.get('rgb_path')}")
        rgb = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgb is None or rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"cached RGB has invalid dimensions/type: {frame.get('rgb_path')}")
    selection = manifest.get("selection", {})
    if selection.get("target_fps") != manifest["fps"]:
        raise ValueError("selection target FPS disagrees with manifest FPS")
    if selection.get("selected_source_indices") != [frame["source_index"] for frame in frames] or selection.get("selected_timestamps_seconds") != [frame["timestamp_seconds"] for frame in frames]:
        raise ValueError("selection record disagrees with cached frame identities")
    if manifest.get("roi_path") is not None:
        roi = _contained_file(manifest_path.parent, manifest["roi_path"])
        if verify_hashes and _sha256(roi) != manifest.get("roi_sha256"):
            raise ValueError("ROI SHA256 mismatch")
        mask = cv2.imread(str(roi), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != (height, width) or mask.dtype != np.uint8 or not np.isin(mask, [0, 255]).all() or not np.any(mask):
            raise ValueError("fixed ROI must be a nonempty binary PNG on the shared canvas")
    elif manifest.get("roi_sha256") is not None:
        raise ValueError("ROI checksum exists without an ROI path")
    return manifest


def load_frames(manifest_path: Path, verify_hashes: bool = True) -> tuple[dict[str, Any], np.ndarray]:
    """Load exact cached RGB arrays in frozen order (never decode the source)."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = validate_manifest(manifest_path, verify_hashes=verify_hashes)
    frames = [cv2.cvtColor(cv2.imread(str(_contained_file(manifest_path.parent, frame["rgb_path"])), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for frame in manifest["frames"]]
    return manifest, np.stack(frames, axis=0)
