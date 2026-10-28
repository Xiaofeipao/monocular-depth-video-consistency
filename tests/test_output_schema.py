import numpy as np
import pytest

from src.data.output_schema import validate_depth_output


def test_native_metric_output_is_valid() -> None:
    validate_depth_output(
        np.ones((2, 3), dtype=np.float32),
        {"depth_type": "metric", "unit": "meter", "alignment": "none"},
    )


def test_metric_output_cannot_claim_alignment() -> None:
    with pytest.raises(ValueError, match="cannot declare"):
        validate_depth_output(
            np.ones((2, 3), dtype=np.float32),
            {"depth_type": "metric", "unit": "meter", "alignment": "median"},
        )


def test_relative_output_uses_arbitrary_units() -> None:
    validate_depth_output(
        np.ones((2, 3), dtype=np.float32),
        {"depth_type": "relative", "unit": "arbitrary", "alignment": "scale_shift"},
    )


def test_non_float_or_invalid_depth_is_rejected() -> None:
    metadata = {"depth_type": "relative", "unit": "arbitrary", "alignment": "none"}
    with pytest.raises(ValueError, match="float32"):
        validate_depth_output(np.ones((2, 2), dtype=np.float64), metadata)
    with pytest.raises(ValueError, match="strictly positive"):
        validate_depth_output(np.zeros((2, 2), dtype=np.float32), metadata)


def test_geometry_output_allows_nan_only_outside_boolean_valid_mask() -> None:
    metadata = {"depth_type": "metric", "unit": "meter", "alignment": "none"}
    depth = np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32)
    mask = np.array([[True, False], [True, False]], dtype=bool)
    validate_depth_output(depth, metadata, valid_mask=mask)
    with pytest.raises(ValueError, match="Valid raw depth contains"):
        validate_depth_output(depth, metadata, valid_mask=np.ones((2, 2), dtype=bool))
