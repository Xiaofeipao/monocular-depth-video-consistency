"""Explicit GT-assisted affine alignment of inverse-depth-like predictions."""

from __future__ import annotations

import numpy as np


def align_inverse_depth(raw, target_m, *, crop, minimum=0.001, maximum=10.0):
    """Fit a>=0,b to a*raw+b ~= 1/GT on the evaluation crop only.

    This is test-GT-assisted relative evaluation, never native metric inference.
    Nonpositive fitted inverse values are clipped, never removed from scoring.
    Degenerate/negative-slope fits use the constrained constant optimum; the
    fallback is reported and the image remains in the benchmark denominator.
    """
    raw, target = np.asarray(raw), np.asarray(target_m, dtype=np.float64)
    if raw.ndim != 2 or raw.shape != target.shape or raw.dtype != np.float32:
        raise ValueError("Expected matching HxW float32 raw and GT")
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("Relative raw must be finite and nonnegative; zero is valid")
    if not 0 < minimum < maximum:
        raise ValueError("Invalid depth bounds")
    top, bottom, left, right = map(int, crop)
    if not (0 <= top < bottom <= raw.shape[0] and 0 <= left < right <= raw.shape[1]):
        raise ValueError("Evaluation crop outside image")
    mask = np.zeros(raw.shape, dtype=bool)
    mask[top:bottom, left:right] = True
    mask &= np.isfinite(target) & (target > minimum) & (target < maximum)
    if mask.sum() < 2:
        raise ValueError("Need at least two valid GT pixels")
    x, y = raw[mask].astype(np.float64), 1.0 / target[mask]
    xc, yc = x - x.mean(), y - y.mean()
    variance = float(xc @ xc)
    tolerance = np.finfo(np.float64).eps * max(float(x @ x), len(x))
    unconstrained = None if variance <= tolerance else float((xc @ yc) / variance)
    fallback = "constant_prediction" if unconstrained is None else (
        "nonpositive_slope" if unconstrained <= 0 else None)
    scale = 0.0 if fallback else unconstrained
    shift = float(y.mean() - scale * x.mean())
    fitted = scale * raw.astype(np.float64) + shift
    clipped_inverse = np.clip(fitted, 1.0 / maximum, 1.0 / minimum)
    depth = 1.0 / clipped_inverse
    diagnostics = {
        "method": "nonnegative_scale_shift_least_squares", "domain": "inverse_depth",
        "scope": "per_image_evaluation_crop_valid_gt", "gt_used": True,
        "scale": float(scale), "shift": shift, "unconstrained_scale": unconstrained,
        "fallback_reason": fallback, "valid_pixels": int(mask.sum()),
        "nonpositive_inverse_pixels": int(np.count_nonzero(fitted[mask] <= 0)),
        "inverse_below_range_pixels": int(np.count_nonzero(fitted[mask] < 1.0 / maximum)),
        "inverse_above_range_pixels": int(np.count_nonzero(fitted[mask] > 1.0 / minimum)),
        "inverse_fit_rmse": float(np.sqrt(np.mean((scale * x + shift - y) ** 2))),
    }
    return depth, mask, diagnostics
