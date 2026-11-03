"""Command-line contracts and early failures; never invoke an upstream model."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from pipelines.video import cli
from src.video.common import read_json, sha256_file, write_json
from src.video.runner import load_config


PROJECT = Path(__file__).resolve().parents[1]


def test_real_parser_exposes_all_stages_and_uses_absolute_default_config() -> None:
    parser = cli.build_parser()
    for command in ("check", "prepare", "freeze", "infer", "flow", "evaluate", "verify", "summarize"):
        assert command in parser.format_help()
    check = parser.parse_args(["check"])
    assert check.config == PROJECT / "configs/video/relative_video.yaml"
    prepare = parser.parse_args(["prepare", "--public-clip", "tokyo_walk", "--output-dir", "/tmp/video-prepare"])
    assert prepare.source is None
    assert prepare.public_clip == "tokyo_walk"


def test_generic_protocol_alias_still_resolves(tmp_path):
    alias = tmp_path / "alias.yaml"
    alias.write_text("shared_protocol: configs/video/relative_video.yaml\n")
    assert load_config(alias) == load_config(PROJECT / "configs/video/relative_video.yaml")


def test_direct_cli_help_works_from_outside_project_directory(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, str(PROJECT / "pipelines/video/cli.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "Part 1B/2B" in result.stdout
    assert "summarize" in result.stdout


def test_shell_entry_changes_to_project_root_and_selects_the_project_conda_env(tmp_path: Path) -> None:
    binary = tmp_path / "bin"
    binary.mkdir()
    conda = binary / "conda"
    conda.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD"\nprintf "%s\\n" "$@"\n')
    conda.chmod(0o755)
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", str(PROJECT / "scripts/video/run.sh"), "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(PROJECT), "run", "--no-capture-output", "-n", "aiaa3201", "python", "pipelines/video/cli.py", "--help"]


@pytest.mark.parametrize("kind,extra", [("self_recorded", []), ("public", []), ("mandatory", ["--split", "test"])])
def test_custom_input_requires_real_provenance_before_any_decoder_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, extra: list[str]) -> None:
    from src.video import frames
    called = []
    monkeypatch.setattr(frames, "prepare_frames", lambda *args, **kwargs: called.append((args, kwargs)))
    with pytest.raises(ValueError):
        cli.main(["prepare", "--source", str(tmp_path / "source.mp4"), "--clip-id", "my-clip",
                  "--source-kind", kind, "--output-dir", str(tmp_path / "cache"), *extra])
    assert called == []


def test_self_recorded_session_is_passed_to_the_real_preparation_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.video import frames
    captured = []

    def prepare(source, output, **kwargs):
        captured.append((source, output, kwargs))
        return output / "manifest.json"

    monkeypatch.setattr(frames, "prepare_frames", prepare)
    assert cli.main(["prepare", "--source", str(tmp_path / "my-phone.mp4"), "--clip-id", "room-fixed",
                     "--source-kind", "self_recorded", "--session-id", "phone-session-1", "--camera-motion", "fixed",
                     "--split", "test", "--target-fps", "7", "--max-long-edge", "320", "--output-dir", str(tmp_path / "cache")]) == 0
    kwargs = captured[0][2]
    assert kwargs["provenance"]["recording_session"] == "phone-session-1"
    assert kwargs["source_kind"] == "self_recorded"
    assert kwargs["camera_motion"] == "fixed"
    assert kwargs["split"] == "test"
    assert kwargs["target_fps"] == 7.
    assert kwargs["max_long_edge"] == 320


def test_public_provenance_json_can_supply_the_source_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.video import frames
    provenance = tmp_path / "provenance.json"
    write_json(provenance, {"source_url": "https://example.test/clip.mp4", "license": "fixture"})
    captured = []
    monkeypatch.setattr(frames, "prepare_frames", lambda *args, **kwargs: captured.append(kwargs) or Path("manifest.json"))
    assert cli.main(["prepare", "--source", str(tmp_path / "public.mp4"), "--clip-id", "public-clip",
                     "--provenance-json", str(provenance), "--output-dir", str(tmp_path / "cache")]) == 0
    assert captured[0]["provenance"]["source_url"] == "https://example.test/clip.mp4"


def test_explicit_zero_fps_reaches_validation_instead_of_being_replaced_by_default(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"unchanged original")
    output = tmp_path / "zero-fps"
    with pytest.raises(ValueError, match="target_fps"):
        cli.main(["prepare", "--source", str(source), "--clip-id", "zero", "--source-url", "https://example.test/source",
                  "--target-fps", "0", "--output-dir", str(output)])
    assert source.read_bytes() == b"unchanged original"
    assert read_json(output / "manifest.json")["complete"] is False
    assert "target_fps" in read_json(output / "failure.json")["error"]["message"]


@pytest.mark.parametrize("overrides", [
    ["--device", "cpu", "--precision", "float16"],
    ["--warmup", "-1"], ["--timing-repeats", "0"],
    ["--torch-threads", "0"], ["--input-size", "13"],
])
def test_invalid_runtime_overrides_are_rejected_before_freezing_or_model_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: list[str]) -> None:
    calls = []
    monkeypatch.setattr(cli, "freeze_protocol", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(ValueError):
        cli.main(["freeze", "--manifest", str(tmp_path / "manifest.json"), "--output", str(tmp_path / "protocol.json"), *overrides])
    assert calls == []
    assert not (tmp_path / "protocol.json").exists()


@pytest.mark.parametrize("mismatched", ["dav2", "vda", "flow"])
def test_evaluation_refuses_a_different_protocol_before_loading_models_or_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatched: str) -> None:
    from src.video import evaluation
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, {"fixture": "frozen protocol"})
    digest = sha256_file(protocol_path)
    monkeypatch.setattr(cli, "load_protocol", lambda path: ({"configuration": {}}, {}, tmp_path / "manifest.json"))
    called = []
    monkeypatch.setattr(evaluation, "evaluate_pair", lambda *args, **kwargs: called.append((args, kwargs)))
    for name, filename in (("dav2", "run.json"), ("vda", "run.json"), ("flow", "flow_manifest.json")):
        write_json(tmp_path / name / filename, {"protocol_sha256": "0" * 64 if name == mismatched else digest})
    with pytest.raises(ValueError):
        cli.main(["evaluate", "--protocol", str(protocol_path), "--dav2-run", str(tmp_path / "dav2"),
                  "--vda-run", str(tmp_path / "vda"), "--flow-cache", str(tmp_path / "flow"),
                  "--output-dir", str(tmp_path / "untrusted-evaluation")])
    assert called == []
    assert not (tmp_path / "untrusted-evaluation").exists()


def test_check_returns_nonzero_if_frozen_gpu_device_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "check_resources", lambda config: {"fixture": "local resources verified"})
    monkeypatch.setattr(cli, "environment_info", lambda: {"cuda_available": False})
    output = tmp_path / "check.json"
    assert cli.main(["check", "--output", str(output)]) == 2
    assert read_json(output)["ready_for_requested_device"] is False
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        cli.main(["check", "--output", str(output)])
    assert output.read_bytes() == original
