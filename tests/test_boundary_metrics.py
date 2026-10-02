import numpy as np
import pytest

from src.metrics.boundary import (
    binary_boundary_f1,
    compute_depth_boundary_metrics,
    depth_edge_map,
)


def test_depth_edge_map_finds_step_discontinuity() -> None:
    depth = np.ones((3, 4), dtype=np.float32)
    depth[:, 2:] = 2.0
    edges = depth_edge_map(depth, log_threshold=0.1)
    expected = np.zeros_like(edges)
    expected[:, 1:3] = True
    np.testing.assert_array_equal(edges, expected)


def test_identical_depth_boundaries_are_perfect() -> None:
    depth = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]], dtype=np.float32)
    metrics = compute_depth_boundary_metrics(
        depth.copy(), depth, log_threshold=0.1, tolerance_px=0
    )
    assert metrics["boundary_precision"] == pytest.approx(1.0)
    assert metrics["boundary_recall"] == pytest.approx(1.0)
    assert metrics["boundary_f1"] == pytest.approx(1.0)


def test_boundary_tolerance_matches_one_pixel_shift() -> None:
    target = np.zeros((5, 5), dtype=bool)
    prediction = np.zeros((5, 5), dtype=bool)
    target[:, 2] = True
    prediction[:, 3] = True
    assert binary_boundary_f1(prediction, target, tolerance_px=0)["boundary_f1"] == 0.0
    assert binary_boundary_f1(prediction, target, tolerance_px=1)["boundary_f1"] == 1.0


def test_empty_boundary_maps_are_a_perfect_match() -> None:
    empty = np.zeros((3, 3), dtype=bool)
    assert binary_boundary_f1(empty, empty)["boundary_f1"] == 1.0
