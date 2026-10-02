import numpy as np
import pytest

from src.metrics.depth import build_valid_mask, compute_depth_metrics


def test_identity_prediction_has_perfect_metrics() -> None:
    target = np.array([[0.5, 1.0], [2.0, 5.0]], dtype=np.float32)
    metrics = compute_depth_metrics(target.copy(), target, min_depth=0.1, max_depth=10.0)
    assert metrics["abs_rel"] == pytest.approx(0.0)
    assert metrics["sq_rel"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["rmse_log"] == pytest.approx(0.0)
    assert metrics["delta1"] == pytest.approx(1.0)
    assert metrics["delta2"] == pytest.approx(1.0)
    assert metrics["delta3"] == pytest.approx(1.0)
    assert metrics["valid_pixels"] == 4


def test_metrics_match_hand_calculation_without_clipping() -> None:
    target = np.array([1.0, 2.0], dtype=np.float32)
    prediction = np.array([2.0, 4.0], dtype=np.float32)
    metrics = compute_depth_metrics(
        prediction,
        target,
        min_depth=0.1,
        max_depth=10.0,
        clip_prediction=False,
    )
    assert metrics["abs_rel"] == pytest.approx(1.0)
    assert metrics["sq_rel"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["rmse_log"] == pytest.approx(np.log(2.0))
    assert metrics["delta1"] == pytest.approx(0.0)
    assert metrics["delta2"] == pytest.approx(0.0)
    assert metrics["delta3"] == pytest.approx(0.0)


def test_invalid_values_and_external_mask_are_excluded() -> None:
    target = np.array([0.0, 0.1, 1.0, 2.0, 10.0, 20.0], dtype=np.float32)
    prediction = np.array([1.0, 1.0, 1.0, np.nan, 1.0, 2.0], dtype=np.float32)
    external = np.array([True, True, True, True, True, False])
    valid = build_valid_mask(
        prediction,
        target,
        min_depth=0.1,
        max_depth=10.0,
        external_mask=external,
    )
    np.testing.assert_array_equal(valid, [False, False, True, False, False, False])


def test_no_valid_pixels_raises() -> None:
    with pytest.raises(ValueError, match="No valid pixels"):
        compute_depth_metrics(
            np.array([np.nan]),
            np.array([0.0]),
            min_depth=0.1,
            max_depth=10.0,
        )
