import numpy as np
import pytest

from src.stereo.io import MiddleburyCalibration, MiddleburyScene
from src.stereo.sensitivity import analyze_prediction_error_by_distance


def test_empirical_sensitivity_counts_invalid_predictions_as_bad2() -> None:
    camera = np.array(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    calibration = MiddleburyCalibration(
        cam0=camera,
        cam1=camera,
        doffs_px=0.0,
        baseline_mm=1000.0,
        width=4,
        height=1,
        num_disparities=16,
        is_integer_disparity=False,
    )
    scene = MiddleburyScene(
        name="synthetic",
        left_rgb=np.zeros((1, 4, 3), dtype=np.uint8),
        right_rgb=np.zeros((1, 4, 3), dtype=np.uint8),
        gt_disparity_px=np.full((1, 4), 10.0, dtype=np.float32),
        non_occluded_mask=np.ones((1, 4), dtype=bool),
        occluded_mask=np.zeros((1, 4), dtype=bool),
        calibration=calibration,
    )
    prediction = np.array([[10.0, 13.0, np.nan, 10.0]], dtype=np.float32)
    rows = analyze_prediction_error_by_distance(
        [scene], {scene.name: prediction}, depth_bin_width_m=1.0
    )

    assert len(rows) == 1
    assert rows[0]["valid_coverage"] == pytest.approx(0.75)
    assert rows[0]["bad2"] == pytest.approx(0.5)
    assert rows[0]["mean_abs_disparity_error_px"] == pytest.approx(1.0)
