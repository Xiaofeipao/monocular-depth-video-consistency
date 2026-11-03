import numpy as np
import pytest

from src.stereo.metrics import compute_disparity_metrics, compute_stereo_depth_metrics


def test_disparity_mae_excludes_invalid_but_fraction_keeps_denominator() -> None:
    target = np.ones((1, 4), dtype=np.float32)
    prediction = np.array([[1.0, 2.0, np.nan, 1.0]], dtype=np.float32)
    metrics = compute_disparity_metrics(
        prediction,
        target,
        evaluation_mask=np.ones((1, 4), dtype=bool),
    )
    assert metrics["disparity_mae_px"] == pytest.approx(1 / 3)
    assert metrics["valid_prediction_fraction"] == pytest.approx(0.75)
    assert metrics["bad1"] == .25
    assert metrics["bad2"] == .25
    assert metrics["bad4"] == .25


def test_stereo_depth_metrics_use_fixed_near_mid_far_bins() -> None:
    target = np.array([[1.0, 3.0, 5.0]], dtype=np.float32)
    prediction = target * 2.0
    metrics = compute_stereo_depth_metrics(
        prediction,
        target,
        evaluation_mask=np.ones(target.shape, dtype=bool),
        depth_bin_edges_m=(0.0, 2.0, 4.0, float("inf")),
    )
    assert metrics["depth_abs_rel"] == pytest.approx(1.0)
    assert metrics["depth_near_pixels"] == 1
    assert metrics["depth_mid_pixels"] == 1
    assert metrics["depth_far_pixels"] == 1


def test_course_metrics_use_all_finite_positive_gt_and_disclose_invalid():
    gt = np.array([[1., 2., 3., 4., 0., -1., np.nan, np.inf]])
    pred = np.array([[2., 4., np.nan, 0., 99., 99., 99., 99.]])
    result = compute_disparity_metrics(pred, gt)
    assert result == dict(disparity_mae_px=1.5, valid_prediction_fraction=.5,
                         evaluation_pixels=4, valid_prediction_pixels=2, invalid_prediction_pixels=2,
                         bad1=.75, bad2=.5, bad4=.5)
    valid = np.ones(gt.shape, bool); valid[0,0] = False
    result = compute_disparity_metrics(pred, gt, prediction_valid_mask=valid)
    assert result['evaluation_pixels'] == 4
    assert result['valid_prediction_fraction'] == .25
    assert result['disparity_mae_px'] == 2.


def test_no_valid_prediction_is_undefined_mae_with_zero_fraction():
    result = compute_disparity_metrics(np.full((2,3), np.nan), np.ones((2,3)))
    assert result['disparity_mae_px'] is None
    assert result['valid_prediction_fraction'] == 0
    assert all(result[k] == 1. for k in ('bad1','bad2','bad4'))
    with pytest.raises(ValueError): compute_disparity_metrics(np.ones(1), np.zeros(1))
    with pytest.raises(ValueError): compute_disparity_metrics(np.ones(2), np.ones(1))
    with pytest.raises(ValueError): compute_disparity_metrics(np.ones(2), np.ones(2), prediction_valid_mask=np.ones(1))


def test_bad_pixel_thresholds_are_strict_and_invalid_pixels_are_failures():
    gt = np.ones((1,6))
    pred = np.array([[2., 3., 5., 6., np.nan, -1.]])
    result = compute_disparity_metrics(pred, gt)
    assert result['bad1'] == pytest.approx(5/6)
    assert result['bad2'] == pytest.approx(4/6)
    assert result['bad4'] == pytest.approx(3/6)
    assert result['valid_prediction_fraction'] == pytest.approx(4/6)
