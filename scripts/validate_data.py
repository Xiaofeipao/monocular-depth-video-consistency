#!/usr/bin/env python3
"""Validate Step 0 dataset structure without loading full datasets into memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat


EXPECTED_MIDDLEBURY_SCENES = {
    "Adirondack",
    "ArtL",
    "Jadeplant",
    "Motorcycle",
    "MotorcycleE",
    "Piano",
    "PianoL",
    "Pipes",
    "Playroom",
    "Playtable",
    "PlaytableP",
    "Recycle",
    "Shelves",
    "Teddy",
    "Vintage",
}


def read_pfm_header(path: Path) -> dict[str, object]:
    """Read only PFM metadata, avoiding a full disparity allocation."""

    with path.open("rb") as handle:
        magic = handle.readline().decode("ascii").strip()
        if magic not in {"PF", "Pf"}:
            raise ValueError(f"Invalid PFM magic in {path}: {magic!r}")
        dimensions = handle.readline().decode("ascii").strip()
        while dimensions.startswith("#"):
            dimensions = handle.readline().decode("ascii").strip()
        match = re.fullmatch(r"(\d+)\s+(\d+)", dimensions)
        if match is None:
            raise ValueError(f"Invalid PFM dimensions in {path}: {dimensions!r}")
        width, height = (int(value) for value in match.groups())
        scale = float(handle.readline().decode("ascii").strip())
    if width <= 0 or height <= 0 or scale == 0:
        raise ValueError(f"Invalid PFM metadata in {path}")
    return {
        "width": width,
        "height": height,
        "channels": 3 if magic == "PF" else 1,
        "scale": abs(scale),
        "endianness": "little" if scale < 0 else "big",
    }


def parse_middlebury_calibration(path: Path) -> dict[str, float | int]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" in line:
            key, value = line.split("=", maxsplit=1)
            entries[key.strip()] = value.strip()
    required = {"cam0", "cam1", "doffs", "baseline", "width", "height", "ndisp"}
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"Missing calibration keys in {path}: {sorted(missing)}")
    camera_values = {
        name: [float(value) for value in re.findall(r"[-+0-9.eE]+", entries[name])]
        for name in ("cam0", "cam1")
    }
    if any(len(values) != 9 for values in camera_values.values()):
        raise ValueError(f"Expected 3x3 camera matrices in {path}")
    result: dict[str, float | int] = {
        "focal_length_px": camera_values["cam0"][0],
        "doffs_px": float(entries["doffs"]),
        "baseline_mm": float(entries["baseline"]),
        "width": int(entries["width"]),
        "height": int(entries["height"]),
        "num_disparities": int(entries["ndisp"]),
    }
    if result["focal_length_px"] <= 0 or result["baseline_mm"] <= 0:
        raise ValueError(f"Non-positive focal length/baseline in {path}")
    return result


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_middlebury(data_root: Path) -> dict[str, object]:
    training_root = data_root / "middlebury/MiddEval3/trainingQ"
    if not training_root.is_dir():
        raise FileNotFoundError(training_root)
    scenes = {path.name for path in training_root.iterdir() if path.is_dir()}
    if scenes != EXPECTED_MIDDLEBURY_SCENES:
        raise ValueError(
            f"Unexpected Middlebury scenes. Missing={sorted(EXPECTED_MIDDLEBURY_SCENES-scenes)}, "
            f"extra={sorted(scenes-EXPECTED_MIDDLEBURY_SCENES)}"
        )
    required = ["im0.png", "im1.png", "calib.txt", "disp0GT.pfm", "mask0nocc.png"]
    missing: list[str] = []
    scene_geometry: dict[str, dict[str, object]] = {}
    for scene in sorted(scenes):
        for filename in required:
            path = training_root / scene / filename
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing Middlebury files: " + ", ".join(missing))

    for scene in sorted(scenes):
        scene_root = training_root / scene
        with Image.open(scene_root / "im0.png") as image0:
            image0_size = image0.size
        with Image.open(scene_root / "im1.png") as image1:
            image1_size = image1.size
        with Image.open(scene_root / "mask0nocc.png") as mask:
            mask_size = mask.size
        pfm = read_pfm_header(scene_root / "disp0GT.pfm")
        calibration = parse_middlebury_calibration(scene_root / "calib.txt")
        expected_size = (pfm["width"], pfm["height"])
        if not image0_size == image1_size == mask_size == expected_size:
            raise ValueError(
                f"Middlebury shape mismatch in {scene}: im0={image0_size}, "
                f"im1={image1_size}, mask={mask_size}, pfm={expected_size}"
            )
        if (calibration["width"], calibration["height"]) != image0_size:
            raise ValueError(
                f"Calibration/image shape mismatch in {scene}: "
                f"calib={(calibration['width'], calibration['height'])}, "
                f"image={image0_size}"
            )
        scene_geometry[scene] = {"pfm": pfm, "calibration": calibration}
    return {
        "root": str(training_root.relative_to(data_root.parent)),
        "scene_count": len(scenes),
        "scenes": sorted(scenes),
        "required_files_per_scene": required,
        "shape_and_geometry_validation": scene_geometry,
    }


def validate_nyuv2(data_root: Path) -> dict[str, object]:
    nyu_root = data_root / "nyuv2"
    split_path = nyu_root / "splits.mat"
    labeled_path = nyu_root / "nyu_depth_v2_labeled.mat"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if not labeled_path.is_file():
        raise FileNotFoundError(labeled_path)

    splits = loadmat(split_path)
    train = np.asarray(splits["trainNdxs"]).reshape(-1).astype(np.int64)
    test = np.asarray(splits["testNdxs"]).reshape(-1).astype(np.int64)
    if len(train) != 795 or len(test) != 654:
        raise ValueError(f"Unexpected NYUv2 split sizes: train={len(train)}, test={len(test)}")
    if np.intersect1d(train, test).size:
        raise ValueError("NYUv2 train/test splits overlap")
    combined = np.sort(np.concatenate([train, test]))
    np.testing.assert_array_equal(combined, np.arange(1, 1450))

    with h5py.File(labeled_path, "r") as handle:
        keys = sorted(handle.keys())
        if "images" not in handle or "depths" not in handle:
            raise KeyError(f"NYUv2 labeled MAT is missing images/depths; keys={keys}")
        image_shape = list(handle["images"].shape)
        depth_shape = list(handle["depths"].shape)
        if 1449 not in image_shape or 1449 not in depth_shape:
            raise ValueError(
                f"NYUv2 labeled arrays do not contain 1449 samples: "
                f"images={image_shape}, depths={depth_shape}"
            )
        image_dtype = str(handle["images"].dtype)
        depth_dtype = str(handle["depths"].dtype)

    return {
        "labeled_mat": str(labeled_path.relative_to(data_root.parent)),
        "labeled_mat_bytes": labeled_path.stat().st_size,
        "labeled_mat_sha256": sha256(labeled_path),
        "split_file": str(split_path.relative_to(data_root.parent)),
        "split_sha256": sha256(split_path),
        "train_count": len(train),
        "test_count": len(test),
        "mat_keys": keys,
        "images_shape_on_disk": image_shape,
        "depths_shape_on_disk": depth_shape,
        "images_dtype": image_dtype,
        "depths_dtype": depth_dtype,
        "indexing": "MATLAB one-based",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data/manifests/validated_datasets.json",
    )
    args = parser.parse_args()
    result = {
        "middlebury": validate_middlebury(args.data_root),
        "nyuv2": validate_nyuv2(args.data_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
