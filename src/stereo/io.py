"""Middlebury Stereo v3 input, PFM, and calibration readers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MiddleburyCalibration:
    cam0: np.ndarray
    cam1: np.ndarray
    doffs_px: float
    baseline_mm: float
    width: int
    height: int
    num_disparities: int
    is_integer_disparity: bool
    vmin_px: float | None = None
    vmax_px: float | None = None
    dyavg_px: float | None = None
    dymax_px: float | None = None

    @property
    def focal_length_px(self) -> float:
        return float(self.cam0[0, 0])

    @property
    def baseline_m(self) -> float:
        return self.baseline_mm / 1000.0


@dataclass(frozen=True)
class MiddleburyScene:
    name: str
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    gt_disparity_px: np.ndarray
    non_occluded_mask: np.ndarray
    occluded_mask: np.ndarray
    calibration: MiddleburyCalibration


def _read_noncomment_ascii_line(handle) -> str:
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("Unexpected end of PFM header")
        decoded = line.decode("ascii").strip()
        if decoded and not decoded.startswith("#"):
            return decoded


def read_pfm_header(path: Path) -> dict[str, object]:
    """Read PFM dimensions, scale, channels, and byte order."""

    with Path(path).open("rb") as handle:
        magic = _read_noncomment_ascii_line(handle)
        if magic not in {"PF", "Pf"}:
            raise ValueError(f"Invalid PFM magic in {path}: {magic!r}")
        dimensions = _read_noncomment_ascii_line(handle)
        match = re.fullmatch(r"(\d+)\s+(\d+)", dimensions)
        if match is None:
            raise ValueError(f"Invalid PFM dimensions in {path}: {dimensions!r}")
        width, height = (int(value) for value in match.groups())
        signed_scale = float(_read_noncomment_ascii_line(handle))
    if width <= 0 or height <= 0 or signed_scale == 0:
        raise ValueError(f"Invalid PFM metadata in {path}")
    return {
        "width": width,
        "height": height,
        "channels": 3 if magic == "PF" else 1,
        "scale": abs(signed_scale),
        "endianness": "little" if signed_scale < 0 else "big",
    }


def read_pfm(path: Path) -> np.ndarray:
    """Load a PFM image with the Middlebury SDK convention.

    The scale sign selects byte order. Middlebury disparity payloads already
    contain pixel-valued floats, so the scale magnitude is metadata and must
    not be multiplied into the samples. Rows are stored bottom-to-top.
    """

    path = Path(path)
    with path.open("rb") as handle:
        magic = _read_noncomment_ascii_line(handle)
        if magic not in {"PF", "Pf"}:
            raise ValueError(f"Invalid PFM magic in {path}: {magic!r}")
        dimensions = _read_noncomment_ascii_line(handle)
        match = re.fullmatch(r"(\d+)\s+(\d+)", dimensions)
        if match is None:
            raise ValueError(f"Invalid PFM dimensions in {path}: {dimensions!r}")
        width, height = (int(value) for value in match.groups())
        signed_scale = float(_read_noncomment_ascii_line(handle))
        if signed_scale == 0:
            raise ValueError(f"PFM scale cannot be zero in {path}")
        channels = 3 if magic == "PF" else 1
        dtype = np.dtype("<f4" if signed_scale < 0 else ">f4")
        data = np.fromfile(handle, dtype=dtype)

    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"PFM payload size differs in {path}: {data.size} != {expected}")
    shape = (height, width, channels) if channels == 3 else (height, width)
    image = np.flipud(data.reshape(shape))
    return np.asarray(image, dtype=np.float32)


def _parse_calibration_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if "=" in line:
            key, value = line.split("=", maxsplit=1)
            entries[key.strip()] = value.strip()
    return entries


def _parse_camera_matrix(value: str, path: Path, name: str) -> np.ndarray:
    values = [float(item) for item in re.findall(r"[-+0-9.eE]+", value)]
    if len(values) != 9:
        raise ValueError(f"Expected a 3x3 {name} matrix in {path}")
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def load_middlebury_calibration(path: Path) -> MiddleburyCalibration:
    """Parse the complete calibration needed for matching and metric depth."""

    path = Path(path)
    entries = _parse_calibration_entries(path)
    required = {"cam0", "cam1", "doffs", "baseline", "width", "height", "ndisp"}
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"Missing calibration keys in {path}: {sorted(missing)}")
    calibration = MiddleburyCalibration(
        cam0=_parse_camera_matrix(entries["cam0"], path, "cam0"),
        cam1=_parse_camera_matrix(entries["cam1"], path, "cam1"),
        doffs_px=float(entries["doffs"]),
        baseline_mm=float(entries["baseline"]),
        width=int(entries["width"]),
        height=int(entries["height"]),
        num_disparities=int(entries["ndisp"]),
        is_integer_disparity=bool(int(entries.get("isint", "0"))),
        vmin_px=float(entries["vmin"]) if "vmin" in entries else None,
        vmax_px=float(entries["vmax"]) if "vmax" in entries else None,
        dyavg_px=float(entries["dyavg"]) if "dyavg" in entries else None,
        dymax_px=float(entries["dymax"]) if "dymax" in entries else None,
    )
    if calibration.focal_length_px <= 0 or calibration.baseline_mm <= 0:
        raise ValueError(f"Non-positive focal length or baseline in {path}")
    return calibration


def parse_middlebury_calibration(path: Path) -> dict[str, float | int]:
    """Return JSON-friendly calibration scalars for dataset validation."""

    calibration = load_middlebury_calibration(path)
    return {
        "focal_length_px": calibration.focal_length_px,
        "doffs_px": calibration.doffs_px,
        "baseline_mm": calibration.baseline_mm,
        "width": calibration.width,
        "height": calibration.height,
        "num_disparities": calibration.num_disparities,
    }


def load_middlebury_scene(scene_root: Path) -> MiddleburyScene:
    """Load one training scene and validate all array dimensions."""

    scene_root = Path(scene_root)
    calibration = load_middlebury_calibration(scene_root / "calib.txt")
    left_bgr = cv2.imread(str(scene_root / "im0.png"), cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(str(scene_root / "im1.png"), cv2.IMREAD_COLOR)
    mask_values = cv2.imread(str(scene_root / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE)
    if left_bgr is None or right_bgr is None or mask_values is None:
        raise FileNotFoundError(f"Incomplete Middlebury scene: {scene_root}")
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    gt_disparity = read_pfm(scene_root / "disp0GT.pfm")
    expected_shape = (calibration.height, calibration.width)
    shapes = {
        "left": left_rgb.shape[:2],
        "right": right_rgb.shape[:2],
        "ground_truth": gt_disparity.shape,
        "mask": mask_values.shape,
    }
    if any(shape != expected_shape for shape in shapes.values()):
        raise ValueError(
            f"Shape mismatch in {scene_root.name}: expected={expected_shape}, got={shapes}"
        )
    known_mask_values = set(np.unique(mask_values).tolist())
    if not known_mask_values <= {0, 128, 255}:
        raise ValueError(f"Unexpected mask values in {scene_root}: {known_mask_values}")
    return MiddleburyScene(
        name=scene_root.name,
        left_rgb=left_rgb,
        right_rgb=right_rgb,
        gt_disparity_px=gt_disparity,
        non_occluded_mask=mask_values == 255,
        occluded_mask=mask_values == 128,
        calibration=calibration,
    )
