"""Streaming access to the NYU Depth V2 labeled MATLAB/HDF5 file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class NYUv2Sample:
    """One RGB/depth pair with stable split and source identifiers."""

    test_position: int
    matlab_index_1based: int
    rgb: np.ndarray
    depth_m: np.ndarray
    scene: str
    scene_type: str


def load_nyuv2_test_indices(
    split_file: Path, *, variable: str = "testNdxs"
) -> np.ndarray:
    """Load MATLAB one-based test indices as a validated 1-D int array."""

    contents = loadmat(Path(split_file))
    if variable not in contents:
        raise KeyError(f"Missing split variable {variable!r} in {split_file}")
    indices = np.asarray(contents[variable]).reshape(-1).astype(np.int64)
    if indices.size == 0 or np.any(indices <= 0) or len(np.unique(indices)) != len(indices):
        raise ValueError("NYUv2 test split must contain unique positive MATLAB indices")
    return indices


class NYUv2LabeledDataset:
    """Read selected samples without loading the 2.8 GB dataset into memory."""

    def __init__(
        self,
        labeled_mat: Path,
        split_file: Path,
        *,
        split_variable: str = "testNdxs",
    ) -> None:
        self.labeled_mat = Path(labeled_mat)
        self.test_indices_1based = load_nyuv2_test_indices(
            split_file, variable=split_variable
        )
        self._file: h5py.File | None = None

    def __len__(self) -> int:
        return int(self.test_indices_1based.size)

    def __enter__(self) -> "NYUv2LabeledDataset":
        self._file = h5py.File(self.labeled_mat, "r")
        if self._file["images"].shape[0] < int(self.test_indices_1based.max()):
            raise ValueError("NYUv2 split index exceeds the labeled dataset length")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError("Use NYUv2LabeledDataset as a context manager")
        return self._file

    def _decode_reference_string(self, field: str, zero_based_index: int) -> str:
        reference = self.file[field][0, zero_based_index]
        codepoints = np.asarray(self.file[reference]).reshape(-1)
        return "".join(chr(int(value)) for value in codepoints)

    def get_test_sample(self, test_position: int) -> NYUv2Sample:
        """Return an RGB HxWx3 uint8 image and aligned HxW float32 depth."""

        if not 0 <= test_position < len(self):
            raise IndexError(f"Test position out of range: {test_position}")
        matlab_index = int(self.test_indices_1based[test_position])
        index = matlab_index - 1
        # HDF5 exposes reversed MATLAB dimensions:
        # RGB 3x640x480 -> 480x640x3; depth 640x480 -> 480x640.
        rgb = np.asarray(self.file["images"][index]).transpose(2, 1, 0).copy()
        depth = np.asarray(self.file["depths"][index]).T.astype(np.float32, copy=True)
        if rgb.dtype != np.uint8 or rgb.shape[:2] != depth.shape or rgb.shape[2] != 3:
            raise ValueError(
                f"Invalid NYUv2 sample shapes/dtypes: {rgb.shape}/{rgb.dtype}, "
                f"{depth.shape}/{depth.dtype}"
            )
        return NYUv2Sample(
            test_position=test_position,
            matlab_index_1based=matlab_index,
            rgb=rgb,
            depth_m=depth,
            scene=self._decode_reference_string("scenes", index),
            scene_type=self._decode_reference_string("sceneTypes", index),
        )
