import numpy as np
import pytest

from src.metrics.temporal import (
    compute_median_scale_drift,
    compute_temporal_warp_metrics,
    forward_backward_consistency_mask,
    warp_with_backward_flow,
)


def test_zero_flow_identity_warp_and_metrics() -> None:
    depth = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    flow = np.zeros((3, 3, 2), dtype=np.float32)
    warped, valid = warp_with_backward_flow(depth, flow)
    np.testing.assert_array_equal(valid, True)
    np.testing.assert_allclose(warped, depth)
    metrics = compute_temporal_warp_metrics(depth, depth, flow)
    assert metrics["temporal_abs_rel"] == pytest.approx(0.0)
    assert metrics["temporal_rmse"] == pytest.approx(0.0)
    assert metrics["temporal_rmse_log"] == pytest.approx(0.0)


def test_backward_flow_direction_for_rightward_translation() -> None:
    previous = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    backward = np.zeros((3, 4, 2), dtype=np.float32)
    backward[..., 0] = -1.0
    warped, valid = warp_with_backward_flow(previous, backward)
    assert not valid[:, 0].any()
    assert valid[:, 1:].all()
    np.testing.assert_allclose(warped[:, 1:], previous[:, :-1])


def test_forward_backward_cycle_rejects_border_and_inconsistent_flow() -> None:
    forward = np.zeros((3, 4, 2), dtype=np.float32)
    backward = np.zeros((3, 4, 2), dtype=np.float32)
    forward[..., 0] = 1.0
    backward[..., 0] = -1.0
    consistent = forward_backward_consistency_mask(
        forward, backward, threshold_px=0.01
    )
    assert not consistent[:, 0].any()
    assert consistent[:, 1:].all()

    backward[1, 2, 0] = -3.0
    consistent = forward_backward_consistency_mask(
        forward, backward, threshold_px=0.01
    )
    assert not consistent[1, 2]


def test_temporal_metrics_honor_external_occlusion_mask() -> None:
    previous = np.ones((2, 2), dtype=np.float32)
    current = np.ones((2, 2), dtype=np.float32)
    current[0, 0] = 2.0
    flow = np.zeros((2, 2, 2), dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)
    mask[0, 0] = False
    metrics = compute_temporal_warp_metrics(
        previous, current, flow, valid_mask=mask
    )
    assert metrics["valid_pixels"] == 3
    assert metrics["temporal_abs_rel"] == pytest.approx(0.0)


def test_median_scale_drift_matches_known_frame_scaling() -> None:
    reference = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    sequence = np.stack([reference, reference * 2.0, reference * 0.5])
    result = compute_median_scale_drift(sequence)
    np.testing.assert_allclose(result["frame_medians"], [2.5, 5.0, 1.25])
    np.testing.assert_allclose(result["scale_ratios"], [1.0, 2.0, 0.5])
    assert result["mean_abs_log_drift"] == pytest.approx(2 * np.log(2) / 3)
    assert result["max_abs_log_drift"] == pytest.approx(np.log(2))
