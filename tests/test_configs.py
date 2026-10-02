from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative_path: str) -> dict:
    with (PROJECT_ROOT / relative_path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_all_project_yaml_files_parse() -> None:
    paths = sorted((PROJECT_ROOT / "configs").rglob("*.yaml"))
    assert paths
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            assert yaml.safe_load(handle) is not None, path


def test_nyuv2_candidate_protocol_prevents_alignment() -> None:
    config = load_yaml("configs/part2/depth_pro_nyuv2.yaml")
    assert config["dataset"]["expected_test_images"] == 654
    assert config["evaluation"]["test_time_alignment"] == "none"
    assert config["evaluation"]["native_scale"] == "metric"


def test_video_tracks_are_not_mixed() -> None:
    config = load_yaml("configs/part2/video_depth_anything.yaml")
    assert config["relative_track"]["unit"] == "arbitrary"
    assert config["metric_track"]["unit"] == "meter"
    assert config["metric_track"]["alignment"] == "none"


def test_shared_candidate_protocol_preserves_native_metric_scale() -> None:
    config = load_yaml("configs/evaluation_protocol.yaml")
    nyuv2 = config["nyuv2"]
    assert nyuv2["expected_test_images"] == 654
    assert nyuv2["native_metric_track"]["test_time_alignment"] == "none"
    assert nyuv2["valid_depth_m"] == {
        "minimum": 0.001,
        "maximum": 10.0,
        "bounds": "exclusive",
    }
    assert config["video"]["flow_convention"] == "backward_target_to_source_dx_dy"


def test_middlebury_protocol_has_four_frozen_experiments_and_disjoint_split() -> None:
    config = load_yaml("configs/part1/middlebury_sgbm.yaml")
    development = config["split"]["development_scenes"]
    held_out = config["split"]["evaluation_scenes"]
    assert config["protocol_status"] == "frozen_v2"
    assert len(development) == 5
    assert len(held_out) == 10
    assert not set(development) & set(held_out)
    assert len(set(development + held_out)) == 15
    assert set(config["experiments"]) == {
        "bm_fixed",
        "bm_tuned",
        "sgbm_fixed",
        "sgbm_tuned",
    }
    assert config["experiments"]["bm_fixed"]["algorithm"] == "bm"
    assert config["experiments"]["bm_tuned"]["algorithm"] == "bm"
    assert config["experiments"]["sgbm_fixed"]["algorithm"] == "sgbm"
    assert config["experiments"]["sgbm_tuned"]["algorithm"] == "sgbm"
    for experiment in config["experiments"].values():
        assert experiment["parameters"]["lr_threshold_px"] == 1.0
    assert set(config["tuning"]) >= {"bm", "sgbm", "lr_threshold_px"}
    assert config["tuning"]["lr_threshold_px"] == 1.0
    for method in ("bm", "sgbm"):
        selection = config["tuning"][method]["selection"]
        parameters = config["experiments"][selection["experiment"]]["parameters"]
        for name in config["tuning"][method]["search"]:
            assert parameters[name] == selection[name]
