from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from src.data.nyuv2 import NYUv2LabeledDataset, load_nyuv2_test_indices


def _string_reference(file: h5py.File, name: str, value: str):
    dataset = file.create_dataset(
        name,
        data=np.asarray([ord(character) for character in value], dtype=np.uint16)[:, None],
    )
    return dataset.ref


def test_load_nyuv2_test_indices_preserves_matlab_one_based_values(tmp_path: Path) -> None:
    split = tmp_path / "splits.mat"
    savemat(split, {"testNdxs": np.array([[1], [3]], dtype=np.int32)})
    np.testing.assert_array_equal(load_nyuv2_test_indices(split), [1, 3])


def test_nyuv2_loader_restores_matlab_image_orientation(tmp_path: Path) -> None:
    labeled = tmp_path / "labeled.mat"
    split = tmp_path / "splits.mat"
    savemat(split, {"testNdxs": np.array([[1]], dtype=np.int32)})
    expected_rgb = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    expected_depth = np.arange(8, dtype=np.float32).reshape(2, 4) + 1.0
    with h5py.File(labeled, "w") as file:
        file.create_dataset("images", data=expected_rgb.transpose(2, 1, 0)[None])
        file.create_dataset("depths", data=expected_depth.T[None])
        scenes = file.create_dataset("scenes", (1, 1), dtype=h5py.ref_dtype)
        scene_types = file.create_dataset("sceneTypes", (1, 1), dtype=h5py.ref_dtype)
        scenes[0, 0] = _string_reference(file, "scene_string", "kitchen_0001")
        scene_types[0, 0] = _string_reference(file, "type_string", "kitchen")

    with NYUv2LabeledDataset(labeled, split) as dataset:
        sample = dataset.get_test_sample(0)
    np.testing.assert_array_equal(sample.rgb, expected_rgb)
    np.testing.assert_allclose(sample.depth_m, expected_depth)
    assert sample.matlab_index_1based == 1
    assert sample.scene == "kitchen_0001"
    assert sample.scene_type == "kitchen"
