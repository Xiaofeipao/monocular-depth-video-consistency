"""Validation for raw prediction arrays and their scale metadata."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

_DEPTH_TYPES = {"metric", "relative"}
_UNITS = {"meter", "arbitrary"}
_ALIGNMENTS = {"none", "median", "scale_shift"}


def validate_depth_output(
    depth: np.ndarray,
    metadata: Mapping[str, object],
    *,
    valid_mask: np.ndarray | None = None,
) -> None:
    """Raise ``ValueError`` when an output violates the shared contract.

    Dense model outputs may omit ``valid_mask`` and must then be finite and
    positive everywhere. Geometry pipelines may provide a boolean mask; only
    valid pixels must be finite and positive, while invalid pixels may be NaN.
    """

    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected an HxW depth array, got shape {depth.shape}")
    if depth.dtype != np.float32:
        raise ValueError(f"Raw depth must be float32, got {depth.dtype}")
    if valid_mask is None:
        valid = np.ones(depth.shape, dtype=bool)
        if not np.all(np.isfinite(depth)):
            raise ValueError("Raw depth contains NaN or infinite values")
    else:
        valid = np.asarray(valid_mask)
        if valid.dtype != np.bool_:
            raise ValueError(f"Valid mask must be bool, got {valid.dtype}")
        if valid.shape != depth.shape:
            raise ValueError(
                f"Valid mask shape differs: {valid.shape} != {depth.shape}"
            )
        if not valid.any():
            raise ValueError("Valid mask contains no valid pixels")
    if not np.all(np.isfinite(depth[valid])):
        raise ValueError("Valid raw depth contains NaN or infinite values")
    if np.any(depth[valid] <= 0):
        raise ValueError("Valid raw depth must be strictly positive")

    depth_type = metadata.get("depth_type")
    unit = metadata.get("unit")
    alignment = metadata.get("alignment")
    if depth_type not in _DEPTH_TYPES:
        raise ValueError(f"Unsupported depth_type: {depth_type!r}")
    if unit not in _UNITS:
        raise ValueError(f"Unsupported unit: {unit!r}")
    if alignment not in _ALIGNMENTS:
        raise ValueError(f"Unsupported alignment: {alignment!r}")
    if depth_type == "metric" and unit != "meter":
        raise ValueError("Metric depth must use unit='meter'")
    if depth_type == "metric" and alignment != "none":
        raise ValueError("Native metric output cannot declare test-time alignment")
    if depth_type == "relative" and unit != "arbitrary":
        raise ValueError("Relative depth must use unit='arbitrary'")
