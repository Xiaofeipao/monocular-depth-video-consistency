"""Fixed-color diagnostics are checked independently of model predictions."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.video.evaluation import _sha, verify_evaluation
from src.video.report import colorize_relative, generate_report
from test_video_evaluation import grid, make_case, read_json, run_case, write_json


def test_colorization_keeps_zero_and_negative_finite_values_and_records_clipping():
    values = np.array([[-1.0, 0.0, 5.0, 10.0, 11.0, np.nan]])
    image, stats = colorize_relative(values, [0, 10])
    assert stats["valid_pixels"] == 5
    assert stats["invalid_pixels"] == 1
    assert stats["below_range_fraction"] == pytest.approx(0.2)
    assert stats["above_range_fraction"] == pytest.approx(0.2)
    assert image[0, 1].any()  # finite zero uses the low TURBO color, not invalid black
    assert not image[0, -1].any()
    np.testing.assert_array_equal(image[0, 0], image[0, 1])
    assert not np.array_equal(image[0, 1], image[0, 3])


@pytest.mark.parametrize("limits", [[1, 1], [2, 1], [0, np.inf], [np.nan, 1]])
def test_display_cannot_use_nonfinite_or_empty_range(limits):
    with pytest.raises(ValueError):
        colorize_relative(np.zeros((2, 2)), limits)


def test_color_of_the_same_value_does_not_change_with_other_pixels():
    first = np.array([[0.0, 10.0, 20.0]])
    second = np.array([[5.0, 10.0, 100.0]])
    image1, _ = colorize_relative(first, [0, 50])
    image2, _ = colorize_relative(second, [0, 50])
    np.testing.assert_array_equal(image1[0, 1], image2[0, 1])


def test_png_and_video_report_reload_binds_actual_sources_without_mutation(tmp_path):
    base = grid(8, 10)
    case = make_case(tmp_path, [base, base + 2, base - 1], report=True, video=True)
    before = _sha(case["dav2"] / "raw_000001.npy")
    summary = run_case(case)
    report = summary["report"]
    assert report["per_frame_rescaling"] is False
    assert report["raw_range"] == [0, 50]
    assert report["normalized_range"] == [0, 1]
    assert report["near_direction"] == "high"
    assert report["saturation"]["normalized"]["dav2"]["above_range_fraction"] > 0
    for role, expected in (("comparison_raw_video", 3), ("comparison_normalized_video", 3), ("flow_difference_video", 2)):
        item = next(item for item in report["assets"] if item["role"] == role)
        assert item["frames"] == item["decoded_frames_checked"] == expected
        assert item["fps"] == 8
        capture = cv2.VideoCapture(str(case["output"] / item["path"]))
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == expected
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) % 2 == 0
        capture.release()
    assert before == _sha(case["dav2"] / "raw_000001.npy")
    assert verify_evaluation(case["output"])["status"] == "verified"


def test_plots_and_contact_sheets_work_without_video_and_negative_medians(tmp_path):
    base = grid() - 20
    case = make_case(tmp_path, [base, base + 1], report=True, video=False)
    summary = run_case(case)
    assets = summary["report"]["assets"]
    assert any(item["role"] == "curves" for item in assets)
    assert any(item["role"] == "worst_pair_contact" for item in assets)
    assert not any(item["path"].endswith(".mp4") for item in assets)
    assert summary["report"]["saturation"]["raw"]["dav2"]["below_range_fraction"] == 1
    assert verify_evaluation(case["output"])["status"] == "verified"


def test_background_map_is_present_only_for_explicit_fixed_camera_roi(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base + 1], roi=np.ones(base.shape), motion="fixed", report=True)
    summary = run_case(case)
    assert any(item["role"] == "background_std_figure" for item in summary["report"]["assets"])


def test_zero_reference_span_can_still_show_raw_without_fabricating_normalized_values(tmp_path):
    base = np.zeros((6, 8), dtype=np.float32)
    case = make_case(tmp_path, [base, base + 1], report=True)
    summary = run_case(case)
    assert summary["report"]["saturation"]["raw"]["dav2"]["invalid_pixels"] == 0
    assert summary["report"]["saturation"]["normalized"]["dav2"]["valid_pixels"] == 0
    assert summary["report"]["saturation"]["normalized"]["dav2"]["invalid_pixels"] == base.size * 2


def test_report_refuses_to_relabel_changed_raw_as_the_same_evaluation(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base + 1], report=False)
    run_case(case)
    raw = case["dav2"] / "raw_000001.npy"
    np.save(raw, base + 2)
    run_path = case["dav2"] / "run.json"
    run = read_json(run_path)
    run["frames"][1]["raw_sha256"] = _sha(raw)
    write_json(run_path, run)
    with pytest.raises(ValueError, match="source bindings"):
        generate_report(case["output"] / "summary.json")


def test_report_cannot_overwrite_existing_images_or_movies(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base + 1], report=True)
    run_case(case)
    path = case["output"] / "report/contact_raw.png"
    checksum = _sha(path)
    with pytest.raises(FileExistsError):
        generate_report(case["output"] / "summary.json")
    assert _sha(path) == checksum


def test_report_asset_corruption_is_detected(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base + 1], report=True)
    run_case(case)
    (case["output"] / "report/contact_raw.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        verify_evaluation(case["output"])
