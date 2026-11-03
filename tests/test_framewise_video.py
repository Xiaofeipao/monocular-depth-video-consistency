"""Part 1B numerical and reconstruction scenarios; no depth-model execution."""

from copy import deepcopy
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video.common import read_json, sha256_file, write_json
from src.video import framewise, runner
from src.video.framewise_roi import derive_background_roi


ROOT = Path(__file__).resolve().parents[1]


def grid():
    yy, xx = np.indices((8, 12), dtype=np.float32)
    return xx + yy * 2


@pytest.fixture
def make_case(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "check_resources", lambda _: {})
    counter = 0

    def make(raw=None, *, roi=None, motion="moving", forward=None, backward=None,
             report=False, video=False, source_kind="public", split="development", session=None, identity=None, input_size=518):
        nonlocal counter
        counter += 1
        root = tmp_path / f"case_{counter}"
        cache, run_dir, flow_dir = root / "frames", root / "dav2", root / "flow"
        for p in (cache, run_dir, flow_dir):
            p.mkdir(parents=True)
        sequence = np.asarray([grid(), grid() + 1, grid() + 3] if raw is None else raw, dtype=np.float32)
        count, height, width = sequence.shape
        config = deepcopy(runner.load_config(ROOT / "configs/video/relative_video.yaml"))
        config["models"] = {"dav2": config["models"]["dav2"]}  # 此夹具明确验证只有单模型的情况
        config["runtime"].update(device="cpu", precision="float32")
        config["inference"]["input_size"] = input_size
        config["flow"].update(backend="farneback", iterations=2)
        config["visualization"].update(generate_report=report, generate_video=video)
        manifest = {"schema_version": "video_frames_v1", "complete": True, "clip_id": f"synthetic_{counter}",
                    "source_kind": source_kind, "camera_motion": motion, "split": split,
                    "height": height, "width": width, "fps": 10.0, "roi_path": None, "roi_sha256": None,
                    "source": {"frame_count": count, "first_pts_seconds": 10.0,
                               "sha256": identity or hashlib.sha256(str(root).encode()).hexdigest(),
                               "provenance": {"recording_session": session}}, "frames": [],
                    "selection": {"target_fps": 10.0, "selected_source_indices": list(range(count)),
                                  "selected_timestamps_seconds": [i / 10 for i in range(count)]}}
        for i in range(count):
            name = f"rgb/{i:06d}.png"
            (cache / "rgb").mkdir(exist_ok=True)
            cv2.imwrite(str(cache / name), np.full((height, width, 3), i * 10 + 30, dtype=np.uint8))
            manifest["frames"].append({"index": i, "source_index": i, "timestamp_seconds": i / 10,
                                       "source_timestamp_seconds": 10 + i / 10,
                                       "rgb_path": name, "rgb_sha256": sha256_file(cache / name)})
        if roi is not None:
            path = cache / "roi.png"
            cv2.imwrite(str(path), np.asarray(roi, dtype=np.uint8) * 255)
            manifest.update(roi_path=path.name, roi_sha256=sha256_file(path))
        write_json(cache / "manifest.json", manifest)
        protocol = runner.freeze_protocol(config, cache / "manifest.json", root / "protocol.json")
        common = {"status": "complete", "manifest_path": str(cache / "manifest.json"),
                  "manifest_sha256": sha256_file(cache / "manifest.json"),
                  "protocol_path": str(protocol), "protocol_sha256": sha256_file(protocol)}
        run = {**common, "config": config,
               "model": {"name": "dav2", "depth_type": "relative", "representation": "inverse_depth_like",
                         "unit": "arbitrary", "alignment": "none", "gt_used": False,
                         "precision": "float32", "network_input_hw": [14, 28]},
               "timing": {"output_frames": count, "model_fps": 4.0, "warmup_count": 0, "timing_repeats": 1}, "frames": []}
        for frame, array in zip(manifest["frames"], sequence):
            name = f"raw_{frame['index']:06d}.npy"
            np.save(run_dir / name, array)
            run["frames"].append({"index": frame["index"], "source_index": frame["source_index"],
                                  "timestamp_seconds": frame["timestamp_seconds"], "raw_path": name,
                                  "raw_sha256": sha256_file(run_dir / name), "shape": [height, width], "dtype": "float32"})
        write_json(run_dir / "run.json", run)
        flow = {**common, "frames_count": count, "shape": [height, width],
                "flow": {"backend": "farneback", "iterations": 2}, "pairs": []}
        for i in range(1, count):
            entry = {"previous_index": i - 1, "current_index": i}
            for direction, supplied in (("forward", forward), ("backward", backward)):
                array = np.zeros((height, width, 2), dtype=np.float32) if supplied is None else np.asarray(supplied, dtype=np.float32)
                array = array[i - 1] if array.ndim == 4 else array
                name = f"{direction}_{i:06d}.npy"
                np.save(flow_dir / name, array)
                entry.update({direction + "_path": name, direction + "_sha256": sha256_file(flow_dir / name)})
            flow["pairs"].append(entry)
        write_json(flow_dir / "flow_manifest.json", flow)
        return {"root": root, "manifest": cache / "manifest.json", "run": run_dir, "flow": flow_dir,
                "output": root / "analysis", "protocol": protocol, "config": config}
    return make


def analyze(case):
    return read_json(framewise.analyze_framewise(case["run"], case["flow"], case["output"]))


def test_one_model_is_sufficient_and_affine_units_preserve_score(make_case):
    sequence = np.stack([grid(), grid() + 1, grid() + 3])
    a, b = make_case(sequence), make_case(sequence * 4 + 13)
    left, right = analyze(a), analyze(b)
    assert set(a["config"]["models"]) == {"dav2"}
    assert not (a["root"] / "vda").exists()
    assert left["aggregate"]["mean_pair_normalized_mae"] == pytest.approx(right["aggregate"]["mean_pair_normalized_mae"])
    assert left["aggregate"]["mean_pair_normalized_rmse"] == pytest.approx(right["aggregate"]["mean_pair_normalized_rmse"])
    assert framewise.verify_single_analysis(a["output"])["numeric_rows_recomputed_from_raw"]


def test_known_translation_warps_in_the_correct_direction(make_case):
    first = grid()
    flow = np.zeros((*first.shape, 2), dtype=np.float32)
    flow[..., 0] = 1
    case = make_case([first, np.roll(first, 1, axis=1)], forward=flow, backward=-flow)
    summary = analyze(case)
    assert summary["aggregate"]["mean_pair_normalized_mae"] == 0
    assert summary["aggregate"]["included_pixels"] == first.shape[0] * (first.shape[1] - 1)


def test_zero_values_remain_valid_and_flicker_is_not_aligned_away(make_case):
    case = make_case([grid(), grid() + 5, grid()])
    summary = analyze(case)
    assert summary["aggregate"]["mean_pair_normalized_mae"] > 0.1
    assert np.load(case["output"] / "masks/pair_000000_000001.npy")[0, 0]
    assert summary["median_summary"]["max_abs_log_ratio"] > 0
    assert not summary["scientific_protocol"]["per_frame_normalization"]


def test_static_background_std_matches_known_temporal_signal(make_case):
    roi = np.ones(grid().shape, dtype=bool)
    summary = analyze(make_case([grid(), grid() + 3, grid()], roi=roi, motion="fixed"))
    span = summary["reference"]["span"]
    assert summary["background_flicker"]["normalized_std_mean"] == pytest.approx(np.std([0, 3, 0]) / span)
    assert summary["background_flicker"]["adjacent_normalized_mae_mean"] == pytest.approx(3 / span)


def test_background_roi_does_not_make_full_frame_error_artificially_smaller(make_case):
    first = grid()
    second = first.copy()
    second[:, 6:] += 10
    roi = np.zeros(first.shape, dtype=bool)
    roi[:, :4] = True
    with_roi = analyze(make_case([first, second], roi=roi, motion="fixed"))
    without = analyze(make_case([first, second], motion="fixed"))
    assert with_roi["aggregate"] == without["aggregate"]
    assert with_roi["reference"] == without["reference"]
    assert with_roi["background_flicker"]["normalized_std_mean"] == 0
    assert not without["background_flicker"]["applicable"]


@pytest.mark.parametrize("motion,roi", [("moving", True), ("fixed", False), ("unknown", True)])
def test_background_flicker_requires_fixed_camera_and_roi(make_case, motion, roi):
    summary = analyze(make_case(roi=np.ones(grid().shape) if roi else None, motion=motion))
    assert not summary["background_flicker"]["applicable"]
    assert summary["background_flicker"]["normalized_std_mean"] is None


def test_empty_masks_and_zero_spans_are_not_zero_scores(make_case):
    f = np.ones((*grid().shape, 2), dtype=np.float32) * 1000
    summary = analyze(make_case(forward=f, backward=f))
    assert summary["coverage"]["empty_pairs"] == 2
    assert summary["aggregate"]["mean_pair_normalized_mae"] is None
    summary = analyze(make_case([np.ones(grid().shape), np.ones(grid().shape) * 2]))
    assert summary["reference"]["span"] is None
    assert summary["aggregate"]["mean_pair_normalized_mae"] is None


def test_fractional_nonfinite_interpolation_support_is_excluded(make_case):
    first = grid()
    first[3, 3] = np.nan
    f = np.zeros((*first.shape, 2), dtype=np.float32)
    f[..., 0] = 0.5
    case = make_case([first, grid()], forward=f, backward=-f)
    analyze(case)
    mask = np.load(case["output"] / "masks/pair_000000_000001.npy")
    assert not mask[3, 3] and not mask[3, 4]
    assert not mask[:, 0].any()


def test_nonpositive_median_ratio_is_unavailable(make_case):
    case = make_case([grid(), np.zeros_like(grid())])
    summary = analyze(case)
    rows = framewise._read_csv(case["output"] / "frame_metrics.csv")
    assert rows[1]["median_ratio"] is None
    assert rows[1]["raw_median"] == 0
    assert summary["median_summary"]["ratio_frames"] == 1


def test_low_coverage_is_a_flag_and_pair_mean_differs_from_pixel_weighting(make_case):
    first = grid()
    count, height, width = 3, *first.shape
    f = np.zeros((count - 1, height, width, 2), dtype=np.float32)
    f[1, ..., 0] = width - 1
    case = make_case([first, first + 1, np.ones_like(first) * 20], forward=f, backward=-f)
    sources = framewise.load_single_sources(case["run"], case["flow"])
    sources["run"]["config"]["evaluation"]["minimum_coverage"] = 0.5
    summary, pairs, _, _ = framewise.measure_single(sources)
    assert summary["aggregate"]["included_pairs"] == 2
    assert summary["coverage"]["low_coverage_pairs"] == 1
    assert pairs[1]["quality_flag"] == "below_minimum_coverage"
    assert summary["aggregate"]["mean_pair_normalized_mae"] == pytest.approx(np.mean([p["normalized_mae"] for p in pairs]))
    assert summary["aggregate"]["mean_pair_normalized_mae"] != pytest.approx(summary["aggregate"]["pixel_weighted_normalized_mae"])


def test_median_uses_a_fixed_spatial_region_not_flow_masks(make_case):
    first = grid()
    second = first.copy()
    second[0, 0] = np.nan
    case = make_case([first, second, first])
    summary = analyze(case)
    rows = framewise._read_csv(case["output"] / "frame_metrics.csv")
    assert len({r["raw_median"] for r in rows}) == 1
    assert summary["median_summary"]["fixed_region_pixels"] == first.size - 1


@pytest.mark.parametrize("field,value", [("name", "other"), ("unit", "meter"), ("alignment", "oracle"), ("gt_used", True)])
def test_wrong_model_semantics_are_refused_with_failed_metadata(make_case, field, value):
    case = make_case()
    path = case["run"] / "run.json"
    run = read_json(path)
    run["model"][field] = value
    write_json(path, run)
    with pytest.raises(ValueError):
        analyze(case)
    assert read_json(case["output"] / "analysis_status.json")["status"] == "failed"


@pytest.mark.parametrize("component", ["raw", "flow", "rgb", "protocol"])
def test_input_tampering_is_detected(make_case, component):
    case = make_case()
    if component == "raw":
        target = case["run"] / "raw_000000.npy"
    elif component == "flow":
        target = case["flow"] / "backward_000001.npy"
    elif component == "rgb":
        target = case["manifest"].parent / "rgb/000000.png"
    else:
        target = case["protocol"]
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises((ValueError, OSError)):
        analyze(case)


def test_recomputed_metrics_detect_resealed_wrong_csv(make_case):
    case = make_case()
    summary = analyze(case)
    path = case["output"] / "frame_metrics.csv"
    rows = framewise._read_csv(path)
    rows[1]["raw_median"] += 7
    framewise._write_csv(path, rows)
    next(a for a in summary["artifacts"] if a["path"] == path.name)["sha256"] = sha256_file(path)
    write_json(case["output"] / "summary.json", summary)
    status = read_json(case["output"] / "analysis_status.json")
    status["summary_sha256"] = sha256_file(case["output"] / "summary.json")
    write_json(case["output"] / "analysis_status.json", status)
    with pytest.raises(ValueError, match="recomputed from raw"):
        framewise.verify_single_analysis(case["output"])


def test_report_writes_standalone_png_pdf_and_decodable_videos(make_case):
    case = make_case(roi=np.ones(grid().shape), motion="fixed", report=True, video=True)
    summary = analyze(case)
    output = case["output"] / "report"
    for name in ("curves.png", "curves.pdf", "background_std.png", "analysis.md", "report.json", "worst_pairs.png"):
        assert (output / name).is_file()
    for name, count in (("depth_raw.mp4", 3), ("depth_fixed_reference.mp4", 3), ("temporal_difference.mp4", 2)):
        cap = cv2.VideoCapture(str(output / name))
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == count
        cap.release()
    assert "VDA" not in (output / "analysis.md").read_text()
    assert not summary["report"]["per_frame_rescaling"]
    assert framewise.verify_single_analysis(case["output"])["masks_and_background_recomputed"]


def test_diagnostics_and_mask_cannot_be_overwritten(make_case):
    case = make_case()
    analyze(case)
    snapshot = sha256_file(case["output"] / "summary.json")
    with pytest.raises(FileExistsError):
        analyze(case)
    assert sha256_file(case["output"] / "summary.json") == snapshot
    path = case["output"] / "masks/pair_000000_000001.npy"
    value = np.load(path)
    value[0, 0] = False
    np.save(path, value)
    with pytest.raises(ValueError):
        framewise.verify_single_analysis(case["output"])


def test_roi_derivation_preserves_original_cache_and_pixels(make_case):
    case = make_case(motion="fixed")
    original = sha256_file(case["manifest"])
    output = case["root"] / "with_roi"
    path = derive_background_roi(case["manifest"], output, rectangles=[[1, 1, 10, 7]], exclude_rectangles=[[3, 2, 5, 4]])
    result = read_json(path)
    region = cv2.imread(str(output / result["roi_path"]), 0) > 0
    assert region.sum() == 9 * 6 - 2 * 2
    assert sha256_file(case["manifest"]) == original
    for f in result["frames"]:
        assert sha256_file(output / f["rgb_path"]) == f["rgb_sha256"]
    assert (output / "roi_preview.png").is_file()
    with pytest.raises(FileExistsError):
        derive_background_roi(case["manifest"], output, rectangles=[[1, 1, 10, 7]])


@pytest.mark.parametrize("rect", [[0, 0, 99, 8], [3, 3, 2, 4], [-1, 0, 3, 3], [0, 0, 0, 8]])
def test_bad_roi_coordinates_fail_before_creating_cache(make_case, rect):
    case = make_case()
    output = case["root"] / "bad_roi"
    with pytest.raises(ValueError):
        derive_background_roi(case["manifest"], output, rectangles=[rect])
    assert not output.exists()


def test_collection_does_not_pool_sources_or_splits_and_deduplicates_recordings(make_case):
    a = make_case(source_kind="public")
    b = make_case(source_kind="self_recorded", split="test", motion="fixed", roi=np.ones(grid().shape))
    for case in (a, b):
        analyze(case)
    out = a["root"] / "collection"
    summary = read_json(framewise.summarize_single_analyses([a["output"], b["output"]], out))
    assert len(summary["groups"]) == 2
    assert all(group["clips"] == 1 for group in summary["groups"])
    duplicate = make_case(identity=read_json(a["manifest"])["source"]["sha256"], source_kind="self_recorded")
    analyze(duplicate)
    with pytest.raises(ValueError, match="Duplicate source"):
        framewise.summarize_single_analyses([a["output"], duplicate["output"]], a["root"] / "bad_summary")


def test_session_split_leakage_is_rejected(make_case):
    a = make_case(session="same_recording", split="development")
    b = make_case(session="same_recording", split="test")
    for case in (a, b):
        analyze(case)
    with pytest.raises(ValueError, match="session"):
        framewise.summarize_single_analyses([a["output"], b["output"]], a["root"] / "bad_summary")


def test_collection_rejects_mixed_input_protocols_in_one_group(make_case):
    a, b = make_case(), make_case(input_size=252)
    for case in (a, b):
        analyze(case)
    with pytest.raises(ValueError, match="scientific protocols"):
        framewise.summarize_single_analyses([a["output"], b["output"]], a["root"] / "bad_summary")


def test_collection_weights_recordings_equally_and_does_not_turn_na_into_zero(make_case):
    base = grid()
    a = make_case([base, base + 1])
    b = make_case([base, base + 4, base + 8, base + 12, base + 16])
    c = make_case([np.ones_like(base), np.ones_like(base)])
    for case in (a, b, c):
        analyze(case)
    out = a["root"] / "collection"
    result = read_json(framewise.summarize_single_analyses([a["output"], b["output"], c["output"]], out))
    group = result["groups"][0]
    assert group["clips"] == 3 and group["evaluable_clip_counts"]["normalized_mae"] == 2
    x = read_json(a["output"] / "summary.json")["aggregate"]["mean_pair_normalized_mae"]
    y = read_json(b["output"] / "summary.json")["aggregate"]["mean_pair_normalized_mae"]
    assert group["equal_clip_mean"]["normalized_mae"] == pytest.approx((x + y) / 2)


def test_preview_marks_exclusion_holes(make_case):
    case = make_case()
    path = derive_background_roi(case["manifest"], case["root"] / "roi_preview", rectangles=[[0, 0, 12, 8]], exclude_rectangles=[[4, 3, 8, 5]])
    preview = cv2.imread(str(path.parent / "roi_preview.png"))
    # The internal hole boundary must be visible as green, not only the outer
    # rectangle border. Pixel [2,5] lies on the contour above the excluded hole.
    assert preview[2, 5, 1] == 255 and preview[2, 5, 0] == 0


def test_header_is_checked_against_raw_after_summary_is_resealed(make_case):
    case = make_case()
    summary = analyze(case)
    summary["frames_count"] = 999
    path = case["output"] / "summary.json"
    write_json(path, summary)
    status = read_json(case["output"] / "analysis_status.json")
    status["summary_sha256"] = sha256_file(path)
    write_json(case["output"] / "analysis_status.json", status)
    with pytest.raises(ValueError, match="frame count"):
        framewise.verify_single_analysis(case["output"])
