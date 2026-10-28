import numpy as np
import pytest

from src.stereo.geometry import depth_sensitivity, disparity_to_depth


def test_disparity_to_depth_includes_middlebury_offset() -> None:
    disparity = np.array([10.0, 20.0], dtype=np.float32)
    depth, valid = disparity_to_depth(
        disparity,
        focal_length_px=100.0,
        baseline=0.2,
        disparity_offset_px=10.0,
    )
    np.testing.assert_allclose(depth, [1.0, 2.0 / 3.0])
    np.testing.assert_array_equal(valid, [True, True])


def test_nonpositive_denominator_is_invalid() -> None:
    depth, valid = disparity_to_depth(
        np.array([-2.0, 0.0, 2.0]),
        focal_length_px=100.0,
        baseline=0.1,
    )
    assert np.isnan(depth[0]) and np.isnan(depth[1])
    assert depth[2] == pytest.approx(5.0)
    np.testing.assert_array_equal(valid, [False, False, True])


def test_first_order_error_grows_quadratically_with_depth() -> None:
    errors = depth_sensitivity(
        np.array([1.0, 2.0]),
        focal_length_px=100.0,
        baseline=0.1,
        disparity_error_px=1.0,
    )
    assert errors[1] / errors[0] == pytest.approx(4.0)

