import numpy as np

from src.stereo.matching import (
    decode_opencv_disparity,
    left_right_consistency_mask,
    round_num_disparities,
)


def test_num_disparities_rounds_up_to_multiple_of_sixteen() -> None:
    assert round_num_disparities(1) == 16
    assert round_num_disparities(64) == 64
    assert round_num_disparities(73) == 80


def test_opencv_fixed_point_decode_and_invalid_code() -> None:
    raw = np.array([[-16, 0, 16, 24]], dtype=np.int16)
    disparity, valid = decode_opencv_disparity(raw, min_disparity=0)
    np.testing.assert_array_equal(valid, [[False, True, True, True]])
    assert np.isnan(disparity[0, 0])
    np.testing.assert_allclose(disparity[0, 1:], [0.0, 1.0, 1.5])


def test_left_right_consistency_uses_right_image_coordinate() -> None:
    left = np.full((2, 5), 2.0, dtype=np.float32)
    right = np.full((2, 5), -2.0, dtype=np.float32)
    consistent = left_right_consistency_mask(left, right, threshold_px=0.01)
    expected = np.zeros((2, 5), dtype=bool)
    expected[:, 2:] = True
    np.testing.assert_array_equal(consistent, expected)
