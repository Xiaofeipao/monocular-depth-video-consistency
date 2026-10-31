"""Derive an immutable background-ROI frame cache without decoding again."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

from .common import create_output_directory, sha256_file, utc_now, write_json
from .frames import validate_manifest
from .evaluation import _path_inside


def derive_background_roi(manifest_path: Path, output_dir: Path, *, rectangles=None, exclude_rectangles=None, mask_path=None) -> Path:
    manifest_path = Path(manifest_path).resolve()
    manifest = validate_manifest(manifest_path)
    shape = (manifest["height"], manifest["width"])
    if bool(rectangles) == bool(mask_path):
        raise ValueError("Provide either include rectangles or an existing mask PNG")
    region = np.zeros(shape, dtype=bool)
    if mask_path:
        mask_path = Path(mask_path).resolve()
        value = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if value is None or value.shape != shape:
            raise ValueError("ROI mask must match the saved RGB canvas")
        region = value > 0

    def slices(rect):
        if len(rect) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in rect):
            raise ValueError("Rectangles use integer x0 y0 x1 y1 coordinates")
        x0, y0, x1, y1 = rect
        if not 0 <= x0 < x1 <= shape[1] or not 0 <= y0 < y1 <= shape[0]:
            raise ValueError("Rectangle is empty or outside the RGB canvas")
        return slice(y0, y1), slice(x0, x1)

    for rect in rectangles or []:
        region[slices(rect)] = True
    for rect in exclude_rectangles or []:
        region[slices(rect)] = False
    if not region.any():
        raise ValueError("Background ROI is empty")
    output = create_output_directory(output_dir)
    result = deepcopy(manifest)
    try:
        for frame in result["frames"]:
            # Existing cache paths have already passed contained-path validation.
            source = manifest_path.parent / frame["rgb_path"]
            target = _path_inside(output, frame["rgb_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        path = output / "background_roi.png"
        if not cv2.imwrite(str(path), region.astype(np.uint8) * 255):
            raise RuntimeError("Could not write ROI")
        preview = cv2.imread(str(output / result["frames"][0]["rgb_path"]))
        # Draw internal exclusion boundaries too; otherwise a hole could look
        # included even though it is excluded from scientific measurements.
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (0, 255, 0), 2)
        if not cv2.imwrite(str(output / "roi_preview.png"), preview):
            raise RuntimeError("Could not write ROI preview")
        result.update(roi_path=path.name, roi_sha256=sha256_file(path),
                      roi_derivation={"source_manifest": str(manifest_path), "source_manifest_sha256": sha256_file(manifest_path),
                                      "created_at_utc": utc_now(), "command": sys.argv,
                                      "include_rectangles_xyxy": rectangles, "exclude_rectangles_xyxy": exclude_rectangles,
                                      "mask_source_sha256": sha256_file(mask_path) if mask_path else None,
                                      "source_code_sha256": sha256_file(Path(__file__)),
                                      "preview_sha256": sha256_file(output / "roi_preview.png"),
                                      "operation": "copy unchanged RGB and add explicit ROI; original cache unchanged"})
        write_json(output / "manifest.json", result)
        validate_manifest(output / "manifest.json")
        return output / "manifest.json"
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "error": str(exc), "command": sys.argv})
        raise
