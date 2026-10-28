from pathlib import Path

from scripts.validate_data import parse_middlebury_calibration, read_pfm_header


def test_read_pfm_header_reports_shape_scale_and_endianness(tmp_path: Path) -> None:
    path = tmp_path / "depth.pfm"
    path.write_bytes(b"Pf\n# comment\n4 3\n-2.0\n")
    assert read_pfm_header(path) == {
        "width": 4,
        "height": 3,
        "channels": 1,
        "scale": 2.0,
        "endianness": "little",
    }


def test_parse_middlebury_calibration_extracts_metric_geometry(tmp_path: Path) -> None:
    path = tmp_path / "calib.txt"
    path.write_text(
        "cam0=[1000 0 320; 0 1000 240; 0 0 1]\n"
        "cam1=[1000 0 300; 0 1000 240; 0 0 1]\n"
        "doffs=20\n"
        "baseline=193.0\n"
        "width=640\n"
        "height=480\n"
        "ndisp=128\n",
        encoding="ascii",
    )
    calibration = parse_middlebury_calibration(path)
    assert calibration["focal_length_px"] == 1000.0
    assert calibration["doffs_px"] == 20.0
    assert calibration["baseline_mm"] == 193.0
    assert calibration["width"] == 640
    assert calibration["height"] == 480
