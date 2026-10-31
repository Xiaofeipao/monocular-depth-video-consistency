"""Shared video inputs must preserve source identity, not nominal FPS guesses."""

import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from src.video.frames import _selection_indices, load_frames, prepare_frames, validate_manifest


pytestmark = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required for real PTS")


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15., (96, 64))
    if not writer.isOpened():
        pytest.skip("MJPG encoder unavailable")
    for index in range(12):
        # Deliberately asymmetric RGB/BGR channels catch swapped cached colors.
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:] = [index * 5, 40 + index * 3, 180 - index * 4]
        frame[:, index:index + 8, :] = 220
        writer.write(frame)
    writer.release()
    return path


def test_cfr_freezes_exact_source_frames_rgb_pts_and_provenance(source: Path, tmp_path: Path) -> None:
    result = prepare_frames(source, tmp_path / "cache", clip_id="clip", target_fps=10., max_frames=5, max_long_edge=48, provenance={"session_id": "session-a", "url": "https://example.test/source"})
    manifest, rgb = load_frames(result)
    assert manifest["complete"] is True
    assert manifest["selection"]["selected_source_indices"] == [0, 2, 3, 5, 6]
    np.testing.assert_allclose([frame["timestamp_seconds"] for frame in manifest["frames"]], [0., 2 / 15, 3 / 15, 5 / 15, 6 / 15], atol=1e-6)
    assert rgb.shape == (5, 32, 48, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 20, 30, 0] > rgb[0, 20, 30, 2] + 100
    assert manifest["source"]["provenance"]["session_id"] == "session-a"
    assert manifest["source"]["frame_count"] == manifest["decoder"]["decoded_frame_count"] == 12
    assert len(manifest["source"]["sha256"]) == 64
    assert manifest["selection"]["vfr_detected"] is False
    assert manifest["environment"]["command_argv"]
    assert manifest["source_fingerprints"]["src/video/frames.py"]


def test_start_end_are_source_time_not_reindexed_frame_times(source: Path, tmp_path: Path) -> None:
    manifest_path = prepare_frames(source, tmp_path / "slice", clip_id="slice", target_fps=5., start_seconds=.15, end_seconds=.71, max_long_edge=192)
    manifest = validate_manifest(manifest_path)
    assert [frame["source_index"] for frame in manifest["frames"]] == [3, 6, 9]
    np.testing.assert_allclose([frame["timestamp_seconds"] for frame in manifest["frames"]], [.2, .4, .6], atol=1e-6)
    assert (manifest["height"], manifest["width"]) == (64, 96), "never upscale source RGB"


def test_roi_is_frozen_binary_on_shared_canvas(source: Path, tmp_path: Path) -> None:
    roi = np.zeros((32, 48), dtype=np.uint8)
    roi[4:20, 5:30] = 17
    roi_path = tmp_path / "background.png"
    cv2.imwrite(str(roi_path), roi)
    manifest_path = prepare_frames(source, tmp_path / "roi-cache", clip_id="fixed", camera_motion="fixed", max_long_edge=48, roi_path=roi_path)
    manifest = validate_manifest(manifest_path)
    frozen = cv2.imread(str(manifest_path.parent / manifest["roi_path"]), cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(frozen > 0, roi > 0)
    assert set(np.unique(frozen)) == {0, 255}
    assert manifest["roi_sha256"] != manifest["roi_source"]["sha256"]


@pytest.mark.parametrize("options", [
    {"target_fps": 0}, {"target_fps": float("nan")}, {"target_fps": 30},
    {"max_long_edge": 0}, {"max_long_edge": 2.5}, {"max_frames": 1},
    {"max_frames": True}, {"start_seconds": -1},
    {"start_seconds": 1.}, {"start_seconds": .4, "end_seconds": .3},
    {"end_seconds": 3.}, {"source_kind": "pretend"},
])
def test_invalid_selections_record_failure_without_modifying_source(source: Path, tmp_path: Path, options: dict) -> None:
    before = source.read_bytes()
    target = tmp_path / "failed"
    with pytest.raises(ValueError):
        prepare_frames(source, target, clip_id="bad", **options)
    assert source.read_bytes() == before
    assert json.loads((target / "manifest.json").read_text())["complete"] is False
    assert json.loads((target / "failure.json").read_text())["error"]["type"] == "ValueError"
    with pytest.raises(FileExistsError):
        prepare_frames(source, target, clip_id="cannot-overwrite")


def test_existing_output_is_never_changed(source: Path, tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("user content")
    with pytest.raises(FileExistsError):
        prepare_frames(source, output, clip_id="no-overwrite")
    assert list(output.iterdir()) == [output / "keep.txt"]


@pytest.mark.parametrize("tamper", ["content", "missing", "escape", "identity", "timestamps"])
def test_tamper_or_missing_identity_is_rejected(source: Path, tmp_path: Path, tamper: str) -> None:
    path = prepare_frames(source, tmp_path / "tamper", clip_id="immutable", max_frames=3)
    manifest = json.loads(path.read_text())
    rgb_path = path.parent / manifest["frames"][0]["rgb_path"]
    if tamper == "content":
        array = cv2.imread(str(rgb_path))
        array[0, 0] = 0
        cv2.imwrite(str(rgb_path), array)
    elif tamper == "missing":
        rgb_path.unlink()
    elif tamper == "escape":
        manifest["frames"][0]["rgb_path"] = "../source.avi"
    elif tamper == "identity":
        manifest["frames"][1]["source_index"] = manifest["frames"][0]["source_index"]
    else:
        manifest["selection"]["selected_timestamps_seconds"][0] = 123.
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_frames(path)


def test_relative_cache_can_move_and_load_without_original_source(source: Path, tmp_path: Path) -> None:
    path = prepare_frames(source, tmp_path / "original-cache", clip_id="portable", max_frames=3)
    moved = tmp_path / "elsewhere"
    shutil.move(path.parent, moved)
    source.unlink()
    manifest, arrays = load_frames(moved / "manifest.json")
    assert len(arrays) == 3
    assert manifest["clip_id"] == "portable"


def test_sparse_vfr_timestamp_selection_has_no_duplicates_or_nominal_time_guess() -> None:
    pts = np.array([10., 10.04, 10.08, 10.2, 10.24, 10.6, 10.64])
    selected, metadata = _selection_indices(pts, 8., 0., None, None)
    assert list(selected) == [0, 3, 5, 6]
    np.testing.assert_allclose(metadata["selected_timestamps_seconds"], [0., .2, .6, .64])
    assert metadata["vfr_detected"] is True
    assert len(selected) == len(set(selected))


def test_encoded_vfr_source_uses_real_pts(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to create synthetic VFR fixture")
    listing = []
    for index, duration in enumerate([.08, .16, .12, .28, .08, .12]):
        image = tmp_path / f"input{index}.png"
        cv2.imwrite(str(image), np.full((32, 48, 3), index * 30, dtype=np.uint8))
        listing.extend([f"file '{str(image)}'", f"duration {duration}"])
    listing.append(f"file '{str(tmp_path / 'input5.png')}'")
    concat = tmp_path / "frames.ffconcat"
    concat.write_text("\n".join(listing) + "\n")
    source = tmp_path / "vfr.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-vsync", "vfr", "-c:v", "ffv1", str(source)], check=True, capture_output=True)
    path = prepare_frames(source, tmp_path / "vfr-cache", clip_id="phone-like", target_fps=5.)
    manifest = validate_manifest(path)
    assert manifest["selection"]["vfr_detected"] is True
    times = [frame["timestamp_seconds"] for frame in manifest["frames"]]
    assert times != pytest.approx(np.arange(len(times)) / 5.), "VFR source PTS must not be replaced by nominal output times"
    assert len({frame["source_index"] for frame in manifest["frames"]}) == len(times)


def test_wrong_canvas_roi_is_not_silently_resized(source: Path, tmp_path: Path) -> None:
    roi = tmp_path / "wrong.png"
    cv2.imwrite(str(roi), np.ones((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="final shared RGB"):
        prepare_frames(source, tmp_path / "wrong-roi", clip_id="fixed", roi_path=roi)


def test_nonmonotonic_pts_are_rejected_instead_of_assuming_nominal_fps(source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.video.frames as frames_module

    original_run = frames_module.subprocess.run

    def changed_pts(command, **kwargs):
        result = original_run(command, **kwargs)
        if "-show_frames" in command:
            payload = json.loads(result.stdout)
            payload["frames"][1]["best_effort_timestamp_time"] = payload["frames"][0]["best_effort_timestamp_time"]
            result.stdout = json.dumps(payload)
        return result

    monkeypatch.setattr(frames_module.subprocess, "run", changed_pts)
    with pytest.raises(ValueError, match="strictly increasing"):
        prepare_frames(source, tmp_path / "bad-pts", clip_id="bad-pts")


def test_variable_source_resolution_is_rejected(source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.video.frames as frames_module

    original_run = frames_module.subprocess.run

    def changed_resolution(command, **kwargs):
        result = original_run(command, **kwargs)
        if "-show_frames" in command:
            payload = json.loads(result.stdout)
            payload["frames"][1]["width"] += 1
            result.stdout = json.dumps(payload)
        return result

    monkeypatch.setattr(frames_module.subprocess, "run", changed_resolution)
    with pytest.raises(ValueError, match="variable-resolution"):
        prepare_frames(source, tmp_path / "variable-size", clip_id="variable-size")


def test_cache_with_16bit_rgb_is_rejected_even_when_hash_check_disabled(source: Path, tmp_path: Path) -> None:
    path = prepare_frames(source, tmp_path / "dtype", clip_id="dtype", max_frames=3)
    manifest = json.loads(path.read_text())
    rgb_path = path.parent / manifest["frames"][0]["rgb_path"]
    cv2.imwrite(str(rgb_path), cv2.imread(str(rgb_path)).astype(np.uint16) * 256)
    with pytest.raises(ValueError, match="dimensions/type"):
        validate_manifest(path, verify_hashes=False)


def test_rotated_phone_container_records_actual_decoded_canvas(source: Path, tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to create rotation fixture")
    unrotated, rotated = tmp_path / "plain.mp4", tmp_path / "rotate.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-c:v", "mpeg4", str(unrotated)], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(unrotated), "-c", "copy", "-metadata:s:v:0", "rotate=90", str(rotated)], check=True, capture_output=True)
    path = prepare_frames(rotated, tmp_path / "rotated-cache", clip_id="portrait", max_frames=3, max_long_edge=48)
    manifest, frames = load_frames(path)
    rotation = manifest["decoder"]["source_rotation_degrees"]
    assert abs(rotation) % 180 == 90
    expected = (48, 32) if manifest["decoder"]["orientation_auto"] else (32, 48)
    assert tuple(frames.shape[1:3]) == expected
    assert manifest["decoder"]["decoded_height_width"] == manifest["decoder"]["expected_decoded_height_width"]
