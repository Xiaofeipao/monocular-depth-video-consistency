"""Real-file orchestration tests without loading models or running a GPU.

Only the expensive upstream adapters and repository lookup are replaced.
Manifests, PNGs, checkpoints, frozen protocols, NPYs and their hashes are real.
"""

from copy import deepcopy
import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video import common, runner
from src.video.common import read_json, sha256_file, write_json
from src.video.frames import SCHEMA_VERSION, validate_manifest


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    project = tmp_path / "project"
    for name in ("src/video", "src/models", "src/metrics", "pipelines/video", "weights", "DA", "VDA"):
        (project / name).mkdir(parents=True, exist_ok=True)
    scientific = project / "src/video/scientific.py"
    scientific.write_text("# fixed scientific implementation\n")
    (project / "src/metrics/temporal.py").write_text("# temporal implementation\n")
    monkeypatch.setattr(common, "PROJECT_ROOT", project)
    monkeypatch.setattr(runner, "PROJECT_ROOT", project)

    def repository_identity(path):
        return {"path": str(Path(path).resolve()), "commit": "commit-v1",
                "has_substantive_changes": False, "staged_changes": False}

    monkeypatch.setattr(runner, "repository_identity", repository_identity)
    # Real environment introspection is covered elsewhere; model-run tests
    # deliberately stay CPU-only, independent of the machine's visible GPUs.
    monkeypatch.setattr(runner, "environment_info", lambda: {"test_fixture": "CPU-only"})
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    models = {}
    for name, repository in (("dav2", "DA"), ("vda", "VDA")):
        checkpoint = project / "weights" / f"{name}.bin"
        checkpoint.write_bytes(f"fixture checkpoint {name}".encode())
        models[name] = {"repository": repository, "checkpoint": str(checkpoint.relative_to(project)),
                        "commit": "commit-v1", "sha256": sha256_file(checkpoint)}
    config = {
        "schema_version": "relative_video_v1",
        "models": models,
        "runtime": {"device": "cpu", "precision": "float32", "seed": 7, "torch_threads": 1},
        "inference": {"input_size": 14, "warmup": 2, "timing_repeats": 2},
        "evaluation": {"reference_index": 0, "percentile_low": 5., "percentile_high": 95.,
                       "alignment": "none", "fb_threshold_px": 1., "minimum_coverage": 0.1},
        "flow": {"backend": "farneback", "iterations": 2, "max_long_edge": None},
    }
    cache = project / "cache"
    (cache / "rgb").mkdir(parents=True)
    frame_entries = []
    for index in range(3):
        image = np.full((8, 12, 3), index * 20, dtype=np.uint8)
        path = cache / "rgb" / f"{index:06d}.png"
        assert cv2.imwrite(str(path), image)
        frame_entries.append({"index": index, "source_index": index * 2,
                              "timestamp_seconds": index / 10., "source_timestamp_seconds": 5. + index / 10.,
                              "rgb_path": str(path.relative_to(cache)), "rgb_sha256": sha256_file(path)})
    manifest = {
        "schema_version": SCHEMA_VERSION, "complete": True, "clip_id": "fixture-public",
        "source_kind": "public", "camera_motion": "moving", "split": "development",
        "fps": 10., "height": 8, "width": 12, "frames": frame_entries,
        "source": {"frame_count": 5, "first_pts_seconds": 5., "sha256": "a" * 64,
                   "provenance": {"session_id": "fixture"}},
        "selection": {"target_fps": 10., "selected_source_indices": [0, 2, 4],
                      "selected_timestamps_seconds": [0., .1, .2]},
        "roi_path": None, "roi_sha256": None,
    }
    manifest_path = cache / "manifest.json"
    write_json(manifest_path, manifest)
    validate_manifest(manifest_path)
    return {"project": project, "config": config, "manifest_path": manifest_path,
            "manifest": manifest, "scientific": scientific, "tmp": tmp_path}


def freeze(workspace: dict, *, changes: dict | None = None, filename: str = "protocol.json") -> Path:
    config = deepcopy(workspace["config"])
    if changes:
        for key, value in changes.items():
            config[key].update(value)
    return runner.freeze_protocol(config, workspace["manifest_path"], workspace["project"] / filename)


def install_dav2(monkeypatch: pytest.MonkeyPatch, *, warmup: int = 2, failure: str | None = None) -> list:
    from src.models import depth_anything_v2_adapter as module
    instances = []

    class Adapter:
        def __init__(self, **kwargs):
            self.arguments, self.calls = kwargs, []
            self.metadata = {"network_input_hw": [14, 28], "fixture_adapter": "dav2"}
            instances.append(self)

        def infer(self, rgb):
            self.calls.append(rgb.copy())
            if failure == "raise":
                raise RuntimeError("fixture model failed")
            shape = rgb.shape[:2] if failure != "shape" else (2, 2)
            result = np.full(shape, len(self.calls) - warmup - 1, dtype=np.float32)
            if failure == "nan":
                result[0, 0] = np.nan
            return result

    monkeypatch.setattr(module, "DepthAnythingV2Adapter", Adapter)
    return instances


def install_vda(monkeypatch: pytest.MonkeyPatch, *, warmup: int = 2, reuse_buffer: bool = False) -> list:
    from src.models import video_depth_anything_adapter as module
    instances = []

    class Adapter:
        def __init__(self, **kwargs):
            self.arguments, self.calls = kwargs, []
            self.metadata = {"network_input_hw": [14, 28], "fixture_adapter": "vda", "inference_mode": "offline_noncausal"}
            self.buffer = None
            instances.append(self)

        def infer(self, rgb, fps):
            self.calls.append((rgb.copy(), fps))
            if self.buffer is None or not reuse_buffer:
                self.buffer = np.empty(rgb.shape[:3], dtype=np.float32)
            self.buffer.fill(len(self.calls) - warmup - 1)
            return self.buffer

    monkeypatch.setattr(module, "VideoDepthAnythingAdapter", Adapter)
    return instances


def test_freeze_binds_real_input_scientific_source_weights_and_refuses_overwrite(workspace: dict) -> None:
    path = freeze(workspace)
    protocol = read_json(path)
    assert protocol["manifest_sha256"] == sha256_file(workspace["manifest_path"])
    assert protocol["project_source_sha256"]["src/video/scientific.py"] == sha256_file(workspace["scientific"])
    assert base64.b64decode(protocol["project_source_archive_base64"]["src/video/scientific.py"], validate=True) == workspace["scientific"].read_bytes()
    assert protocol["resources"]["dav2"]["checkpoint_sha256"] == workspace["config"]["models"]["dav2"]["sha256"]
    assert protocol["scope"]["is_course_self_recorded"] is False
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        freeze(workspace)
    assert path.read_bytes() == before


@pytest.mark.parametrize("tamper", ["manifest", "rgb", "source", "checkpoint"])
def test_load_protocol_rejects_actual_frozen_dependencies_changed(workspace: dict, tamper: str) -> None:
    path = freeze(workspace)
    if tamper == "manifest":
        payload = read_json(workspace["manifest_path"])
        payload["split"] = "test"
        write_json(workspace["manifest_path"], payload)
    elif tamper == "rgb":
        image = workspace["manifest_path"].parent / "rgb/000000.png"
        image.write_bytes(b"not the frozen PNG")
    elif tamper == "source":
        workspace["scientific"].write_text("# scientific implementation changed\n")
    else:
        checkpoint = workspace["project"] / workspace["config"]["models"]["dav2"]["checkpoint"]
        checkpoint.write_bytes(b"different model")
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.load_protocol(path, verify_resources=True)


def test_load_protocol_rejects_configuration_edited_inside_frozen_json(workspace: dict) -> None:
    path = freeze(workspace)
    protocol = read_json(path)
    protocol["configuration"]["evaluation"]["percentile_high"] = 99.
    write_json(path, protocol)
    with pytest.raises(ValueError):
        runner.load_protocol(path, verify_resources=True)


def test_frozen_source_archive_cannot_disagree_with_scientific_hash(workspace: dict) -> None:
    path = freeze(workspace)
    protocol = read_json(path)
    protocol["project_source_archive_base64"]["src/video/scientific.py"] = base64.b64encode(b"wrong source bytes").decode("ascii")
    # Reseal to isolate the archive-to-code SHA check from the outer payload
    # integrity check; the wrong scientific bytes must still be rejected.
    protocol["protocol_content_sha256"] = runner._protocol_digest(protocol)
    write_json(path, protocol)
    with pytest.raises(ValueError):
        runner.load_protocol(path)


@pytest.mark.parametrize("bad", ["commit", "weights", "reference"])
def test_freeze_rejects_wrong_resource_identity_or_reference(workspace: dict, bad: str) -> None:
    config = deepcopy(workspace["config"])
    if bad == "commit":
        config["models"]["dav2"]["commit"] = "wrong-commit"
    elif bad == "weights":
        config["models"]["dav2"]["sha256"] = "0" * 64
    else:
        config["evaluation"]["reference_index"] = 3
    with pytest.raises(ValueError):
        runner.freeze_protocol(config, workspace["manifest_path"], workspace["project"] / "invalid.json")


def test_dav2_preserves_zero_and_first_measured_pass_with_warmup_and_timing(workspace: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    instances = install_dav2(monkeypatch)
    path = freeze(workspace)
    output = workspace["project"] / "dav2-run"
    result = runner.infer_run(path, "dav2", output)
    record = read_json(result)
    adapter = instances[0]
    assert len(adapter.calls) == 2 + 3 * 2
    assert adapter.arguments["device"] == "cpu"
    assert adapter.arguments["precision"] == "float32"
    for index, entry in enumerate(record["frames"]):
        raw = np.load(output / entry["raw_path"], allow_pickle=False)
        np.testing.assert_array_equal(raw, np.full((8, 12), index, dtype=np.float32))
        assert entry["source_index"] == index * 2
        assert entry["raw_sha256"] == sha256_file(output / entry["raw_path"])
    assert record["model"]["depth_type"] == "relative"
    assert record["model"]["unit"] == "arbitrary"
    timing = record["timing"]
    assert timing["warmup_count"] == 2
    assert len(timing["measured_sequence_seconds"]) == 2
    assert all(duration > 0 for duration in timing["measured_sequence_seconds"])
    assert [len(row) for row in timing["framewise_latency_seconds"]] == [3, 3]
    assert timing["offline_sequence_latency_seconds"] is None
    assert timing["output_frames"] == 3
    assert timing["peak_gpu_allocated_bytes"] is None
    assert timing["model_fps"] == pytest.approx(3 / np.mean(timing["measured_sequence_seconds"]))
    assert runner.verify_run(output)["status"] == "passed"
    with pytest.raises(FileExistsError):
        runner.infer_run(path, "dav2", output)


def test_vda_runs_identical_sequence_and_records_offline_latency(workspace: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    instances = install_vda(monkeypatch)
    path = freeze(workspace)
    output = workspace["project"] / "vda-run"
    record = read_json(runner.infer_run(path, "vda", output))
    assert len(instances[0].calls) == 4
    for rgb, fps in instances[0].calls:
        assert rgb.shape == (3, 8, 12, 3)
        np.testing.assert_array_equal(rgb[:, 0, 0, 0], [0, 20, 40])
        assert fps == 10.
    for entry in record["frames"]:
        np.testing.assert_array_equal(np.load(output / entry["raw_path"]), np.zeros((8, 12), dtype=np.float32))
    assert len(record["timing"]["offline_sequence_latency_seconds"]) == 2
    assert record["timing"]["framewise_latency_seconds"] is None
    assert "not online latency" in record["timing"]["latency_scope"]


def test_vda_first_measured_pass_does_not_alias_reused_model_output(workspace: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    install_vda(monkeypatch, reuse_buffer=True)
    output = workspace["project"] / "vda-reused-buffer"
    record = read_json(runner.infer_run(freeze(workspace), "vda", output))
    for entry in record["frames"]:
        np.testing.assert_array_equal(np.load(output / entry["raw_path"]), np.zeros((8, 12), dtype=np.float32))


def test_repeated_timing_does_not_select_the_temporally_smoother_prediction(workspace: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.models import video_depth_anything_adapter as module

    class Adapter:
        def __init__(self, **kwargs):
            self.calls = 0
            self.metadata = {"network_input_hw": [14, 28]}

        def infer(self, rgb, fps):
            self.calls += 1
            result = np.zeros(rgb.shape[:3], dtype=np.float32)
            if self.calls == 1:
                result[:, :, :] = np.arange(len(rgb), dtype=np.float32)[:, None, None] * 10.
            return result

    monkeypatch.setattr(module, "VideoDepthAnythingAdapter", Adapter)
    path = freeze(workspace, changes={"inference": {"warmup": 0}})
    output = workspace["project"] / "no-best-pass-selection"
    record = read_json(runner.infer_run(path, "vda", output))
    for index, entry in enumerate(record["frames"]):
        # First pass flickers strongly (0,10,20); second is all zero.  Only
        # throughput may use both passes, never choose the prettier result.
        np.testing.assert_array_equal(np.load(output / entry["raw_path"]), np.full((8, 12), index * 10., dtype=np.float32))


@pytest.mark.parametrize("failure", ["raise", "shape", "nan"])
def test_failed_prediction_keeps_truthful_metadata(workspace: dict, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    install_dav2(monkeypatch, failure=failure)
    path = freeze(workspace)
    output = workspace["project"] / "failed-run"
    with pytest.raises((ValueError, RuntimeError)):
        runner.infer_run(path, "dav2", output)
    record = read_json(output / "run.json")
    assert record["status"] == "failed"
    assert record["error"]["message"]
    assert record["finished_at_utc"]
    assert record["elapsed_seconds"] >= 0
    with pytest.raises(ValueError):
        runner.verify_run(output)


def test_cpu_fp16_refused_before_freezing_or_adapter_construction(workspace: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    instances = install_dav2(monkeypatch)
    with pytest.raises(ValueError):
        freeze(workspace, changes={"runtime": {"precision": "float16"}})
    assert instances == []
    assert not (workspace["project"] / "protocol.json").exists()


@pytest.mark.parametrize("tamper", ["raw", "order", "timestamp", "protocol", "escape", "nonfinite"])
def test_verify_run_rejects_prediction_or_binding_tampering(workspace: dict, monkeypatch: pytest.MonkeyPatch, tamper: str) -> None:
    install_dav2(monkeypatch)
    path = freeze(workspace)
    output = workspace["project"] / "verified-run"
    run_path = runner.infer_run(path, "dav2", output)
    record = read_json(run_path)
    if tamper == "raw":
        raw_path = output / record["frames"][0]["raw_path"]
        np.save(raw_path, np.ones((8, 12), dtype=np.float32))
    elif tamper == "order":
        record["frames"] = list(reversed(record["frames"]))
    elif tamper == "timestamp":
        record["frames"][0]["timestamp_seconds"] = 123.
    elif tamper == "protocol":
        protocol = read_json(path)
        protocol["scope"]["split"] = "test"
        write_json(path, protocol)
    elif tamper == "escape":
        outside = output.parent / "outside.npy"
        np.save(outside, np.zeros((8, 12), dtype=np.float32))
        record["frames"][0].update(raw_path="../outside.npy", raw_sha256=sha256_file(outside))
    else:
        raw_path = output / record["frames"][0]["raw_path"]
        np.save(raw_path, np.full((8, 12), np.nan, dtype=np.float32))
        record["frames"][0]["raw_sha256"] = sha256_file(raw_path)
    write_json(run_path, record)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.verify_run(output)


def test_flow_preserves_forward_backward_order_geometry_and_input_binding(workspace: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.video import flow
    calls = []

    class Estimator:
        def __init__(self, **kwargs):
            self.metadata = {"backend": "fixture", "arguments": kwargs}

        def estimate(self, previous, current):
            calls.append((previous.copy(), current.copy()))
            forward = np.zeros((*previous.shape[:2], 2), dtype=np.float32)
            forward[..., 0] = 1.
            backward = -forward
            return forward, backward

    monkeypatch.setattr(flow, "FlowEstimator", Estimator)
    path = freeze(workspace)
    output = workspace["project"] / "shared-flow"
    record = read_json(runner.cache_flow(path, output))
    assert record["status"] == "complete"
    assert record["manifest_sha256"] == sha256_file(workspace["manifest_path"])
    assert record["protocol_sha256"] == sha256_file(path)
    assert [(int(previous[0, 0, 0]), int(current[0, 0, 0])) for previous, current in calls] == [(0, 20), (20, 40)]
    assert len(record["pairs"]) == 2
    assert [(entry["previous_index"], entry["current_index"]) for entry in record["pairs"]] == [(0, 1), (1, 2)]
    for entry in record["pairs"]:
        forward = np.load(output / entry["forward_path"])
        backward = np.load(output / entry["backward_path"])
        assert forward.shape == backward.shape == (8, 12, 2)
        np.testing.assert_array_equal(forward[..., 0], 1.)
        np.testing.assert_array_equal(backward, -forward)
        assert entry["forward_sha256"] == sha256_file(output / entry["forward_path"])
    with pytest.raises(FileExistsError):
        runner.cache_flow(path, output)


def test_flow_failure_is_recorded_without_claiming_complete(workspace: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.video import flow

    class Estimator:
        def __init__(self, **kwargs):
            self.metadata = {"backend": "fixture"}

        def estimate(self, previous, current):
            return np.zeros((2, 2, 2), dtype=np.float32), np.zeros((2, 2, 2), dtype=np.float32)

    monkeypatch.setattr(flow, "FlowEstimator", Estimator)
    output = workspace["project"] / "bad-flow"
    with pytest.raises(ValueError):
        runner.cache_flow(freeze(workspace), output)
    record = read_json(output / "flow_manifest.json")
    assert record["status"] == "failed"
    assert record["error"]["message"]
