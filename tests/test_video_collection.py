"""Collections preserve independent clips, provenance groups and missingness."""

import csv
from pathlib import Path

import pytest

from src.video import collection, evaluation
from src.video.common import read_json, sha256_file, write_json


@pytest.fixture
def verified(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls = []

    def verify(directory, *, verify_sources):
        assert verify_sources is True
        calls.append(Path(directory))
        return {"status": "passed"}

    monkeypatch.setattr(evaluation, "verify_evaluation", verify)
    return calls


def make_evaluation(tmp_path: Path, name: str, *, value: float | None,
                    frames: int = 3, source_kind: str = "public",
                    split: str = "development", motion: str = "moving",
                    source_sha: str | None = None, session: str | None = None) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    source = directory / "source.json"
    provenance = {"recording_session": session} if session else {}
    write_json(source, {"source": {"sha256": source_sha or name.ljust(64, "a")[:64], "provenance": provenance}})
    write_json(directory / "summary.json", {
        "clip": {"clip_id": name, "source_kind": source_kind, "split": split, "camera_motion": motion},
        "frames_count": frames,
        "sources": {"manifest_path": str(source), "manifest_sha256": sha256_file(source)},
        "aggregate": {
            "dav2": {"mean_pair_normalized_mae": value,
                     "mean_pair_normalized_rmse": None if value is None else value * 2},
            "vda": {"mean_pair_normalized_mae": None if value is None else value / 2,
                    "mean_pair_normalized_rmse": value},
        },
        "coverage": {"mean": .75, "empty_pairs": 1 if value is None else 0,
                     "low_coverage_pairs": 1 if value is None else 0},
    })
    return directory


def test_equal_clip_mean_does_not_weight_long_clips_or_pixels(tmp_path: Path, verified: list) -> None:
    short = make_evaluation(tmp_path, "short", value=1., frames=3)
    long = make_evaluation(tmp_path, "long", value=3., frames=1001)
    path = collection.summarize_evaluations([short, long], tmp_path / "collection")
    summary = read_json(path)
    groups = {group["model"]: group for group in summary["groups"]}
    assert groups["dav2"]["clip_equal_mean_normalized_mae"] == pytest.approx(2.)
    assert groups["vda"]["clip_equal_mean_normalized_mae"] == pytest.approx(1.)
    assert groups["dav2"]["clip_count"] == groups["dav2"]["evaluable_clips"] == 2
    assert verified == [short.resolve(), long.resolve()]
    assert summary["statistical_significance_claimed"] is False
    assert summary["clips_csv_sha256"] == sha256_file(path.parent / "clips.csv")
    with (path.parent / "clips.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {int(row["frames"]) for row in rows} == {3, 1001}


def test_source_split_motion_and_model_are_separate_groups(tmp_path: Path, verified: list) -> None:
    inputs = [
        make_evaluation(tmp_path, "public-dev-moving", value=1.),
        make_evaluation(tmp_path, "public-test-moving", value=2., split="test"),
        make_evaluation(tmp_path, "self-dev-fixed", value=3., source_kind="self_recorded", motion="fixed"),
        make_evaluation(tmp_path, "self-test-fixed", value=4., source_kind="self_recorded", split="test", motion="fixed"),
        make_evaluation(tmp_path, "self-test-moving", value=5., source_kind="self_recorded", split="test"),
        make_evaluation(tmp_path, "mandatory", value=6., source_kind="mandatory", split="mandatory"),
    ]
    summary = read_json(collection.summarize_evaluations(inputs, tmp_path / "groups"))
    keys = {(group["source_kind"], group["split"], group["camera_motion"], group["model"]) for group in summary["groups"]}
    expected = {(kind, split, motion, model) for kind, split, motion in (
        ("public", "development", "moving"), ("public", "test", "moving"),
        ("self_recorded", "development", "fixed"), ("self_recorded", "test", "fixed"),
        ("self_recorded", "test", "moving"), ("mandatory", "mandatory", "moving"),
    ) for model in ("dav2", "vda")}
    assert keys == expected
    assert all(group["clip_count"] == 1 for group in summary["groups"])


def test_missing_results_stay_null_and_do_not_lower_the_average(tmp_path: Path, verified: list) -> None:
    unavailable = make_evaluation(tmp_path, "unavailable", value=None)
    measured = make_evaluation(tmp_path, "measured", value=4.)
    summary = read_json(collection.summarize_evaluations([unavailable, measured], tmp_path / "mixed"))
    dav2 = next(group for group in summary["groups"] if group["model"] == "dav2")
    assert dav2["clip_count"] == 2
    assert dav2["evaluable_clips"] == 1
    assert dav2["clip_equal_mean_normalized_mae"] == 4.
    only_missing = read_json(collection.summarize_evaluations([unavailable], tmp_path / "missing"))
    assert all(group["evaluable_clips"] == 0 and group["clip_equal_mean_normalized_mae"] is None and group["clip_equal_mean_normalized_rmse"] is None for group in only_missing["groups"])


@pytest.mark.parametrize("different_kind", [False, True])
def test_same_original_recording_is_not_counted_twice_even_if_relabeled(tmp_path: Path, verified: list, different_kind: bool) -> None:
    sha = "0" * 64
    first = make_evaluation(tmp_path, "original", value=1., source_sha=sha)
    second = make_evaluation(tmp_path, "cut-or-relabeled", value=2., source_sha=sha,
                             source_kind="self_recorded" if different_kind else "public")
    with pytest.raises(ValueError):
        collection.summarize_evaluations([first, second], tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()


def test_recording_session_cannot_leak_between_development_and_test(tmp_path: Path, verified: list) -> None:
    first = make_evaluation(tmp_path, "session-dev", value=1., source_kind="self_recorded", session="same-session")
    second = make_evaluation(tmp_path, "session-test", value=2., source_kind="self_recorded", session="same-session", split="test")
    with pytest.raises(ValueError):
        collection.summarize_evaluations([first, second], tmp_path / "leak")
    assert not (tmp_path / "leak").exists()


def test_failed_verification_stops_before_creating_summary_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_evaluation(tmp_path, "not-verified", value=1.)

    def fail(directory, *, verify_sources):
        raise ValueError("raw dependency changed")

    monkeypatch.setattr(evaluation, "verify_evaluation", fail)
    with pytest.raises(ValueError):
        collection.summarize_evaluations([source], tmp_path / "untrusted")
    assert not (tmp_path / "untrusted").exists()


def test_duplicate_directory_empty_list_and_existing_output_are_rejected(tmp_path: Path, verified: list) -> None:
    source = make_evaluation(tmp_path, "valid", value=1.)
    with pytest.raises(ValueError):
        collection.summarize_evaluations([], tmp_path / "empty")
    with pytest.raises(ValueError):
        collection.summarize_evaluations([source, source], tmp_path / "repeated")
    output = tmp_path / "existing"
    output.mkdir()
    keep = output / "keep.txt"
    keep.write_text("user file")
    with pytest.raises(FileExistsError):
        collection.summarize_evaluations([source], output)
    assert keep.read_text() == "user file"
    assert list(output.iterdir()) == [keep]
