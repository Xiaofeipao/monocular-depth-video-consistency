import numpy as np
import pytest

from src.metrics.relative_alignment import align_inverse_depth
from src.metrics.depth import compute_depth_metrics


def problem():
    raw = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
    target = 1 / (0.7 * raw.astype(np.float64) + 0.2)
    return raw, target


def test_known_inverse_affine_fit_recovers_scale_shift_and_depth():
    raw, gt = problem()
    before = raw.copy()
    depth, mask, info = align_inverse_depth(raw, gt, crop=[0, 3, 0, 4])
    assert info['scale'] == pytest.approx(0.7)
    assert info['shift'] == pytest.approx(0.2)
    np.testing.assert_allclose(depth, gt, atol=1e-12)
    np.testing.assert_array_equal(raw, before)
    assert mask.all() and raw[0, 0] == 0
    assert info['fallback_reason'] is None and info['gt_used']


def test_affine_change_of_raw_does_not_change_aligned_result():
    raw, gt = problem()
    a = align_inverse_depth(raw, gt, crop=[0, 3, 0, 4])[0]
    b = align_inverse_depth((raw * 2 + 3).astype(np.float32), gt, crop=[0, 3, 0, 4])[0]
    np.testing.assert_allclose(a, b, rtol=2e-6)


def test_fit_matches_independent_general_least_squares_solver():
    rng = np.random.default_rng(123)
    raw = rng.uniform(0, 4, (10, 11)).astype(np.float32)
    inverse_gt = 0.6 * raw + 0.2 + rng.normal(0, 0.01, raw.shape)
    _, _, info = align_inverse_depth(raw, 1 / inverse_gt, crop=[0, 10, 0, 11])
    expected = np.linalg.lstsq(np.column_stack([raw.ravel(), np.ones(raw.size)]),
                               inverse_gt.ravel(), rcond=None)[0]
    np.testing.assert_allclose([info['scale'], info['shift']], expected, atol=1e-12)


def test_crop_and_valid_gt_define_both_fit_and_score_mask():
    raw, gt = problem()
    gt[:, 0] = 9.9  # outside crop must not affect fit
    gt[0, 1] = np.nan
    gt[1, 1] = 10.0
    gt[2, 1] = 0.001
    depth, mask, info = align_inverse_depth(raw, gt, crop=[0, 3, 1, 4])
    assert info['scale'] == pytest.approx(0.7)
    assert info['shift'] == pytest.approx(0.2)
    assert not mask[:, :2].any() and info['valid_pixels'] == 6
    result = compute_depth_metrics(depth, gt, min_depth=0.001, max_depth=10, external_mask=mask)
    assert result['valid_pixels'] == 6 and result['abs_rel'] < 1e-12


@pytest.mark.parametrize('value', [0.0, 3.0])
def test_constant_prediction_is_reported_and_not_dropped(value):
    raw, gt = problem()
    depth, mask, info = align_inverse_depth(np.full_like(raw, value), gt, crop=[0, 3, 0, 4])
    assert info['fallback_reason'] == 'constant_prediction'
    assert info['scale'] == 0 and mask.sum() == raw.size
    np.testing.assert_allclose(depth, 1 / np.mean(1 / gt))


def test_negative_scale_is_constrained_and_reported_not_flipped():
    raw, gt = problem()
    depth, mask, info = align_inverse_depth(raw.max() - raw, gt, crop=[0, 3, 0, 4])
    assert info['fallback_reason'] == 'nonpositive_slope'
    assert info['unconstrained_scale'] < 0 and info['scale'] == 0
    assert np.isfinite(depth).all() and mask.all()


def test_negative_fitted_inverse_kept_in_scoring_at_max_depth():
    raw = np.arange(4, dtype=np.float32)[None, :]
    gt = 1 / np.array([[0.11, 0.12, 0.13, 2.0]])
    depth, mask, info = align_inverse_depth(raw, gt, crop=[0, 1, 0, 4])
    assert info['nonpositive_inverse_pixels'] > 0
    assert mask.sum() == 4 and depth[0, 0] == 10
    assert compute_depth_metrics(depth, gt, min_depth=0.001, max_depth=10, external_mask=mask)['valid_pixels'] == 4


@pytest.mark.parametrize('bad', [np.nan, np.inf, -1.0])
def test_invalid_raw_is_rejected_instead_of_hiding_pixels(bad):
    raw, gt = problem()
    raw[0, 0] = bad
    with pytest.raises(ValueError, match='finite and nonnegative'):
        align_inverse_depth(raw, gt, crop=[0, 3, 0, 4])


def test_empty_valid_gt_and_invalid_crop_fail():
    raw, gt = problem()
    with pytest.raises(ValueError, match='two valid'):
        align_inverse_depth(raw, np.zeros_like(gt), crop=[0, 3, 0, 4])
    with pytest.raises(ValueError, match='crop'):
        align_inverse_depth(raw, gt, crop=[0, 4, 0, 4])
