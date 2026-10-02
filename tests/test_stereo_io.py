from pathlib import Path

import numpy as np

from src.stereo.io import load_middlebury_calibration, read_pfm


def test_read_pfm_uses_scale_sign_and_vertical_flip_without_rescaling(tmp_path: Path) -> None:
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    payload = np.flipud(expected).astype("<f4").tobytes()
    path = tmp_path / "disparity.pfm"
    path.write_bytes(b"Pf\n2 2\n-0.5\n" + payload)
    np.testing.assert_allclose(read_pfm(path), expected)


def test_load_middlebury_calibration_converts_baseline_to_meters(tmp_path: Path) -> None:
    path = tmp_path / "calib.txt"
    path.write_text(
        "cam0=[1000 0 320; 0 1000 240; 0 0 1]\n"
        "cam1=[1000 0 300; 0 1000 240; 0 0 1]\n"
        "doffs=20\n"
        "baseline=193\n"
        "width=640\nheight=480\nndisp=70\nisint=0\n"
        "vmin=4\nvmax=60\ndyavg=0.1\ndymax=0.3\n",
        encoding="ascii",
    )
    calibration = load_middlebury_calibration(path)
    assert calibration.focal_length_px == 1000.0
    assert calibration.baseline_m == 0.193
    assert calibration.doffs_px == 20.0
    assert calibration.num_disparities == 70
