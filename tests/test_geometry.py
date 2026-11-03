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



def test_middlebury_mm_offset_and_opencv_units():
    from src.stereo.geometry import middlebury_disparity_to_depth
    from src.stereo.matching import decode_opencv_disparity
    raw = np.array([[160, 0, -16]], dtype=np.int16)
    px, _ = decode_opencv_disparity(raw, min_disparity=0)
    depth, valid = middlebury_disparity_to_depth(px, focal_length_px=1000., baseline_mm=200., doffs_px=10.)
    assert depth[0,0] == pytest.approx(10.)
    assert valid.tolist() == [[True,False,False]]
    assert np.isnan(depth[0,1:]).all()
    for invalid in (0., -1., np.inf, np.nan):
        with pytest.raises(ValueError):
            middlebury_disparity_to_depth(px, focal_length_px=1000., baseline_mm=invalid, doffs_px=0.)
    depth, valid = middlebury_disparity_to_depth(np.array([1.]), focal_length_px=1000., baseline_mm=200., doffs_px=-1.)
    assert not valid.any() and np.isnan(depth).all()
