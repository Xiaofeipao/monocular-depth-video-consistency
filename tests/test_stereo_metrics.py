import numpy as np
import pytest

from src.stereo.metrics import compute_disparity_metrics, compute_stereo_depth_metrics


def test_disparity_metrics_count_invalid_predictions_as_bad() -> None:
    target = np.ones((1, 4), dtype=np.float32)
    prediction = np.array([[1.0, 2.0, np.nan, 1.0]], dtype=np.float32)
    metrics = compute_disparity_metrics(
        prediction,
        target,
        evaluation_mask=np.ones((1, 4), dtype=bool),
    )
    assert metrics["disparity_mae_px"] == pytest.approx(1 / 3)
    assert metrics["valid_coverage"] == pytest.approx(0.75)
    assert metrics["bad1"] == pytest.approx(0.25)
    assert metrics["bad2"] == pytest.approx(0.25)


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
