"""Numerical scenarios for the relative-video scientific protocol, CPU only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video.evaluation import _sha, evaluate_pair, verify_evaluation


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def make_case(tmp_path, dav2, vda=None, *, forward=None, backward=None, roi=None,
              motion="moving", minimum_coverage=0.05, report=False, video=False):
    dav2 = np.asarray(dav2, dtype=np.float32)
    vda = dav2.copy() if vda is None else np.asarray(vda, dtype=np.float32)
    count, height, width = dav2.shape
    shape = (height, width)
    config = {"evaluation": {"reference_index": 0, "percentile_low": 5.0, "percentile_high": 95.0,
                              "fb_threshold_px": 1.0, "minimum_coverage": minimum_coverage, "alignment": "none"},
              "flow": {"backend": "synthetic"},
              "visualization": {"generate_report": report, "generate_video": video, "raw_range": [0, 50],
                                "normalized_range": [0, 1], "colormap": "turbo"}}
    manifest_dir = tmp_path / "frames"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "manifest.json"
    manifest = {"schema_version": 1, "complete": True, "clip_id": "synthetic", "source_kind": "synthetic_test",
                "camera_motion": motion, "split": "development", "fps": 8.0, "height": height, "width": width,
                "roi_path": None, "frames": []}
    for i in range(count):
        path = manifest_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(path), np.full((*shape, 3), 20 + i, dtype=np.uint8))
        manifest["frames"].append({"index": i, "source_index": i * 2, "timestamp_seconds": i / 8,
                                   "rgb_path": path.name, "rgb_sha256": _sha(path)})
    if roi is not None:
        path = manifest_dir / "roi.png"
        cv2.imwrite(str(path), np.asarray(roi, dtype=np.uint8) * 255)
        manifest.update(roi_path=path.name, roi_sha256=_sha(path))
    write_json(manifest_path, manifest)
    directories = []
    for name, arrays in (("dav2", dav2), ("vda", vda)):
        directory = tmp_path / name
        directory.mkdir()
        run = {"schema_version": 1, "status": "complete", "manifest_path": str(manifest_path),
               "manifest_sha256": _sha(manifest_path), "config": config,
               "model": {"name": name, "depth_type": "relative", "representation": "inverse_depth_like",
                         "unit": "arbitrary", "alignment": "none", "precision": "float32", "network_input_hw": [14, 28],
                         "windowing": {"window_length": 32, "overlap": 10, "stride": 22, "window_starts": [0, 22]} if name == "vda" else {}},
               "timing": {"output_frames": count, "model_fps": 4.0, "model_seconds": count / 4}, "frames": []}
        for i, array in enumerate(arrays):
            path = directory / f"raw_{i:06d}.npy"
            np.save(path, array)
            run["frames"].append({"index": i, "raw_path": path.name, "raw_sha256": _sha(path),
                                  "shape": list(shape), "dtype": "float32"})
        write_json(directory / "run.json", run)
        directories.append(directory)
    flow_dir = tmp_path / "flow"
    flow_dir.mkdir()
    flow_manifest = {"schema_version": 1, "status": "complete", "manifest_sha256": _sha(manifest_path),
                     "frames_count": count, "shape": list(shape), "flow": {"backend": "synthetic"}, "pairs": []}
    for i in range(1, count):
        entry = {"previous_index": i - 1, "current_index": i}
        for name, values in (("forward", forward), ("backward", backward)):
            array = np.zeros((*shape, 2), dtype=np.float32) if values is None else np.asarray(values, dtype=np.float32)
            if array.ndim == 4:
                array = array[i - 1]
            path = flow_dir / f"{name}_{i:06d}.npy"
            np.save(path, array)
            entry.update({name + "_path": path.name, name + "_sha256": _sha(path)})
        flow_manifest["pairs"].append(entry)
    write_json(flow_dir / "flow_manifest.json", flow_manifest)
    return {"manifest": manifest_path, "dav2": directories[0], "vda": directories[1], "flow": flow_dir,
            "output": tmp_path / "evaluation", "config": config}


def run_case(case):
    path = evaluate_pair(case["manifest"], case["dav2"], case["vda"], case["flow"], case["output"], config=case["config"])
    return read_json(path)


def grid(height=5, width=6):
    yy, xx = np.indices((height, width), dtype=np.float32)
    return xx + 2 * yy


def rows(case, filename="pair_metrics.csv"):
    with (case["output"] / filename).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_known_translation_is_exactly_consistent_without_alignment(tmp_path):
    previous = grid()
    current = np.roll(previous, 1, axis=1)
    forward = np.zeros((*previous.shape, 2), dtype=np.float32)
    forward[..., 0] = 1
    backward = -forward
    case = make_case(tmp_path, [previous, current], forward=forward, backward=backward)
    summary = run_case(case)
    assert summary["aggregate"]["dav2"]["mean_pair_normalized_mae"] == 0
    assert summary["aggregate"]["dav2"]["included_pixels"] == previous.shape[0] * (previous.shape[1] - 1)
    assert summary["protocol"]["alignment"] == "none"
    assert verify_evaluation(case["output"])["numeric_rows_recomputed_from_raw"]


def test_positive_affine_units_do_not_change_normalized_warp_error(tmp_path):
    base = grid()
    sequence = np.stack([base, base + 1, base + 3])
    case = make_case(tmp_path, sequence, 4 * sequence + 13)
    summary = run_case(case)
    assert summary["aggregate"]["dav2"]["mean_pair_normalized_mae"] == pytest.approx(summary["aggregate"]["vda"]["mean_pair_normalized_mae"])
    assert summary["comparison"]["vda_mean_pair_mae_reduction_percent"] == pytest.approx(0)
    frame_rows = rows(case, "frame_metrics.csv")
    assert [float(row["dav2_normalized_median"]) for row in frame_rows] == pytest.approx([float(row["vda_normalized_median"]) for row in frame_rows])


def test_real_global_flicker_is_not_removed_by_per_frame_rescaling(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base + 5, base], [base, base, base])
    summary = run_case(case)
    assert summary["aggregate"]["dav2"]["mean_pair_normalized_mae"] > 0.3
    assert summary["aggregate"]["vda"]["mean_pair_normalized_mae"] == 0
    assert summary["comparison"]["vda_mean_pair_mae_reduction_percent"] == 100
    assert summary["median_summary"]["dav2"]["max_abs_log_median_ratio"] > 0
    assert not summary["protocol"]["per_frame_alignment"]


def test_zero_relative_values_are_valid_pixels(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base])
    summary = run_case(case)
    assert summary["aggregate"]["dav2"]["included_pixels"] == base.size
    assert np.load(case["output"] / "masks/pair_000000_000001.npy")[0, 0]


def test_shared_mask_removes_both_models_fractional_nonfinite_support(tmp_path):
    first = grid()
    sequence = np.stack([first, first + 1])
    other = sequence.copy()
    other[0, 1, 1] = np.nan
    forward = np.zeros((*first.shape, 2), dtype=np.float32)
    forward[..., 0], forward[..., 1] = 0.5, -0.5
    case = make_case(tmp_path, sequence, other, forward=forward, backward=-forward)
    run_case(case)
    actual = np.load(case["output"] / "masks/pair_000000_000001.npy")
    expected = np.ones(first.shape, dtype=bool)
    expected[:, 0] = False
    expected[-1] = False
    expected[0:2, 1:3] = False
    np.testing.assert_array_equal(actual, expected)
    assert int(rows(case)[0]["common_pixels"]) == int(expected.sum())


def test_current_nonfinite_in_either_model_is_removed_for_both(tmp_path):
    base = grid()
    other = np.stack([base, base])
    other[1, 2, 3] = np.inf
    case = make_case(tmp_path, [base, base], other)
    summary = run_case(case)
    assert summary["aggregate"]["dav2"]["included_pixels"] == base.size - 1
    assert summary["aggregate"]["vda"]["included_pixels"] == base.size - 1


def test_empty_masks_are_unavailable_not_perfect_scores(tmp_path):
    base = grid()
    backward = np.full((*base.shape, 2), 100, dtype=np.float32)
    case = make_case(tmp_path, [base, base], backward=backward)
    summary = run_case(case)
    assert summary["coverage"]["empty_pairs"] == 1
    assert summary["aggregate"]["dav2"]["mean_pair_normalized_mae"] is None
    assert rows(case)[0]["exclusion_reason"] == "empty_common_mask"
    assert rows(case)[0]["dav2_raw_mae"] == ""


def test_low_coverage_is_flagged_but_not_dropped_from_primary_mean(tmp_path):
    base = grid()
    backward = np.full((*base.shape, 2), 100, dtype=np.float32)
    backward[0, 0] = 0
    case = make_case(tmp_path, [base, base + 1], backward=backward, minimum_coverage=0.1)
    summary = run_case(case)
    assert summary["coverage"]["low_coverage_pairs"] == 1
    assert summary["aggregate"]["dav2"]["mean_pair_normalized_mae"] > 0
    row = rows(case)[0]
    assert row["quality_flag"] == "below_minimum_coverage"
    assert row["exclusion_reason"] == ""
    assert float(row["dav2_normalized_mae"]) > 0


def test_constant_reference_span_is_unavailable_even_if_raw_error_exists(tmp_path):
    case = make_case(tmp_path, np.stack([np.ones((4, 5)), np.ones((4, 5)) * 2]))
    summary = run_case(case)
    assert summary["reference"]["dav2"]["span"] is None
    assert summary["reference"]["dav2"]["unavailable_reason"] == "reference_percentile_span_is_zero"
    assert rows(case)[0]["dav2_raw_mae"] == "1.0"
    assert rows(case)[0]["dav2_normalized_mae"] == ""
    assert verify_evaluation(case["output"])["status"] == "verified"


def test_background_flicker_uses_predeclared_roi_and_fixed_units(tmp_path):
    base = grid()
    roi = np.zeros(base.shape, dtype=bool)
    roi[1:4, 1:5] = True
    case = make_case(tmp_path, [base, base + 1, base + 2], roi=roi, motion="fixed")
    summary = run_case(case)
    span = summary["reference"]["dav2"]["span"]
    item = summary["background_flicker"]["dav2"]
    assert summary["background_flicker"]["applicable"]
    assert summary["background_flicker"]["region_pixels"] == int(roi.sum())
    assert item["normalized_std_mean"] == pytest.approx(np.sqrt(2 / 3) / span)
    assert item["adjacent_normalized_mae_mean"] == pytest.approx(1 / span)
    std = np.load(case["output"] / "dav2_background_normalized_std.npy")
    assert np.isnan(std[~roi]).all()
    np.testing.assert_allclose(std[roi], np.sqrt(2 / 3) / span, rtol=1e-6)


@pytest.mark.parametrize("motion,with_roi", [("fixed", False), ("moving", True), ("unknown", True)])
def test_background_claim_requires_both_fixed_camera_and_explicit_roi(tmp_path, motion, with_roi):
    base = grid()
    case = make_case(tmp_path, [base, base], roi=np.ones(base.shape) if with_roi else None, motion=motion)
    assert not run_case(case)["background_flicker"]["applicable"]


def test_median_curve_uses_one_time_invariant_common_spatial_region(tmp_path):
    base = grid()
    sequence = np.stack([base, base])
    sequence[1, -1, -1] = np.nan
    case = make_case(tmp_path, sequence)
    summary = run_case(case)
    assert summary["fixed_common_region_pixels"] == base.size - 1
    values = rows(case, "frame_metrics.csv")
    assert values[0]["dav2_raw_median"] == values[1]["dav2_raw_median"]


@pytest.mark.parametrize("offset", [-20, -6.5])
def test_nonpositive_reference_median_does_not_invent_log_ratio(tmp_path, offset):
    base = grid() + offset
    case = make_case(tmp_path, [base, base + 1])
    summary = run_case(case)
    assert summary["median_summary"]["dav2"]["mean_abs_log_median_ratio"] is None
    assert summary["aggregate"]["dav2"]["mean_pair_normalized_mae"] > 0
    assert rows(case, "frame_metrics.csv")[0]["dav2_ratio_unavailable_reason"] == "nonpositive_raw_median_not_ratio_compatible"


def test_pair_mean_and_pixel_weighted_mean_are_separately_reported(tmp_path):
    base = grid()
    sequence = np.stack([base, base + 1, base + 4])
    sequence[0, 0, 0] = np.nan
    case = make_case(tmp_path, sequence)
    summary = run_case(case)
    result = summary["aggregate"]["dav2"]
    span = summary["reference"]["dav2"]["span"]
    assert result["mean_pair_normalized_mae"] == pytest.approx(2 / span)
    assert result["pixel_weighted_normalized_mae"] == pytest.approx(((base.size - 1) + 3 * base.size) / (2 * base.size - 1) / span)
    assert result["pixel_weighted_normalized_mae"] != result["mean_pair_normalized_mae"]


def test_vda_is_allowed_to_be_worse(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base + 1], [base, base + 3])
    summary = run_case(case)
    assert summary["comparison"]["vda_mean_pair_mae_reduction_percent"] == pytest.approx(-200)


@pytest.mark.parametrize("field,value", [("network_input_hw", [28, 28]), ("precision", "float16"), ("alignment", "oracle"), ("unit", "meter")])
def test_incompatible_runs_fail_and_record_actual_failure(tmp_path, field, value):
    base = grid()
    case = make_case(tmp_path, [base, base])
    path = case["vda"] / "run.json"
    payload = read_json(path)
    payload["model"][field] = value
    write_json(path, payload)
    with pytest.raises(ValueError):
        run_case(case)
    assert read_json(case["output"] / "evaluation_status.json")["status"] == "failed"


def test_evaluation_cannot_silently_change_frozen_reference_settings(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base])
    case["config"]["evaluation"]["percentile_low"] = 10
    with pytest.raises(ValueError, match="configuration differs"):
        run_case(case)


def test_second_evaluation_cannot_overwrite_any_output(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base])
    run_case(case)
    checksum = _sha(case["output"] / "summary.json")
    with pytest.raises(FileExistsError):
        run_case(case)
    assert _sha(case["output"] / "summary.json") == checksum


def test_reload_detects_changed_saved_prediction(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base])
    run_case(case)
    path = case["dav2"] / "raw_000001.npy"
    np.save(path, base + 1)
    with pytest.raises(ValueError, match="SHA256"):
        verify_evaluation(case["output"])
    assert verify_evaluation(case["output"], verify_sources=False)["source_verification"] is False


def test_reload_detects_changed_summary_before_trusting_aggregates(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base])
    run_case(case)
    path = case["output"] / "summary.json"
    payload = read_json(path)
    payload["aggregate"]["dav2"]["mean_pair_normalized_mae"] = 5
    write_json(path, payload)
    with pytest.raises(ValueError, match="SHA256"):
        verify_evaluation(case["output"])


def test_reload_recomputes_raw_metrics_not_just_summary_hashes(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base + 1])
    run_case(case)
    csv_path = case["output"] / "pair_metrics.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames, data = reader.fieldnames, list(reader)
    data[0]["dav2_raw_mae"] = "999"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    summary_path = case["output"] / "summary.json"
    summary = read_json(summary_path)
    next(item for item in summary["artifacts"] if item["role"] == "pair_metrics")["sha256"] = _sha(csv_path)
    write_json(summary_path, summary)
    status_path = case["output"] / "evaluation_status.json"
    status = read_json(status_path)
    status["summary_sha256"] = _sha(summary_path)
    write_json(status_path, status)
    assert verify_evaluation(case["output"], verify_sources=False)["status"] == "verified"
    with pytest.raises(ValueError, match="pair CSV from raw"):
        verify_evaluation(case["output"])


def test_window_context_and_actual_output_blending_are_reported_separately(tmp_path):
    base = grid()
    sequence = np.repeat(base[None], 33, axis=0)
    case = make_case(tmp_path, sequence)
    summary = run_case(case)
    boundary = summary["window_boundaries"]["boundaries"][0]
    assert boundary["window_start"] == 22
    assert boundary["interpolation_start"] == 24
    assert boundary["interpolation_stop_exclusive"] == 32
    assert boundary["new_output_start"] == 32
    assert 32 in boundary["nearby_pair_current_indices"]
    assert all(item["dav2_normalized_mae"] == 0 for item in boundary["pair_scores"])


def test_normalized_median_delta_has_a_fixed_reference_and_preserves_drift(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base + 2])
    summary = run_case(case)
    result = rows(case, "frame_metrics.csv")
    assert float(result[0]["dav2_normalized_median_delta"]) == 0
    assert float(result[1]["dav2_normalized_median_delta"]) == pytest.approx(2 / summary["reference"]["dav2"]["span"])


def test_reload_recomputes_background_std_map_content(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base + 1], motion="fixed", roi=np.ones(base.shape))
    run_case(case)
    path = case["output"] / "dav2_background_normalized_std.npy"
    np.save(path, np.zeros(base.shape, dtype=np.float32))
    summary_path = case["output"] / "summary.json"
    summary = read_json(summary_path)
    next(item for item in summary["artifacts"] if item["role"] == "dav2_background_std")["sha256"] = _sha(path)
    write_json(summary_path, summary)
    status_path = case["output"] / "evaluation_status.json"
    status = read_json(status_path)
    status["summary_sha256"] = _sha(summary_path)
    write_json(status_path, status)
    with pytest.raises(ValueError, match="recomputed shared masks and background std arrays"):
        verify_evaluation(case["output"])


def test_source_bound_fixed_roi_cannot_change_silently(tmp_path):
    base = grid()
    case = make_case(tmp_path, [base, base], roi=np.ones(base.shape), motion="fixed")
    run_case(case)
    roi = np.zeros(base.shape, dtype=np.uint8)
    roi[0, 0] = 255
    cv2.imwrite(str(case["manifest"].parent / "roi.png"), roi)
    with pytest.raises(ValueError, match="SHA256"):
        verify_evaluation(case["output"])
