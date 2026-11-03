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
    visualization = config["visualization"]
    assert "failure_case_count" not in visualization
    assert set(visualization["failure_thresholds"]) == {
        "abs_rel",
        "delta1",
        "si_boundary_f1",
    }


def test_unidepth_v2_uses_same_nyuv2_protocol_without_camera_input() -> None:
    depth_pro = load_yaml("configs/part2/depth_pro_nyuv2.yaml")
    unidepth = load_yaml("configs/part2/unidepth_v2_nyuv2.yaml")
    assert unidepth["dataset"] == depth_pro["dataset"]
    assert unidepth["evaluation"] == depth_pro["evaluation"]
    assert unidepth["metrics"] == depth_pro["metrics"]
    assert unidepth["statistics"] == depth_pro["statistics"]
    assert unidepth["visualization"]["failure_thresholds"] == depth_pro[
        "visualization"
    ]["failure_thresholds"]
    assert unidepth["model"]["camera_output"] == "intrinsics_px"
    assert unidepth["model"]["provide_external_intrinsics"] is False


def test_part2_key_analysis_preserves_primary_protocol_and_scope() -> None:
    config = load_yaml("configs/part2/key_analysis.yaml")
    assert config["dataset"]["expected_test_images"] == 654
    assert config["evaluation"]["test_time_alignment"] == "none"
    assert config["evaluation"]["valid_depth_m"] == {
        "minimum": 0.001,
        "maximum": 10.0,
        "bounds": "exclusive",
    }
    assert config["boundary_analysis"]["tolerance_pixels"] == [1, 2, 3]
    assert config["absolute_scale_analysis"]["oracle_alignment"][
        "forbidden_as_primary_result"
    ]
    assert set(config["baseline_outputs"]) == {"depth_pro", "unidepth_v2"}
    assert not {"camera_analysis", "cross_domain_analysis", "unidepth_model"} & config.keys()


def test_video_tracks_are_not_mixed() -> None:
    from src.evaluation.temporal_clip import load_config
    config = load_config(PROJECT_ROOT / "configs/part2/vda_video.yaml")
    baseline = load_yaml("configs/part1/dav2_video.yaml")
    assert config["protocol"] == "course_part2b_offline_v1"
    assert config['inference']['mode'] == 'offline'
    assert config['inference']['metric'] is False
    assert config['input']['max_long_edge'] == 640
    assert config['input']['requested_frames'] == 100
    assert config['input']['original_fps'] is True
    assert config['source'] == baseline['source']
    assert config['diagnostics'] == baseline['diagnostics']
    assert config['visualization'] == baseline['visualization']
    assert config['runtime']['precision'] == baseline['runtime']['precision'] == 'float16'


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
    assert config["protocol_status"] == "middlebury_course_v1"
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
    assert config['evaluation']['mask_name'] == 'finite_positive_gt'
    assert config['evaluation']['primary_metrics'] == ['disparity_mae_px','valid_prediction_fraction']
    assert config['tuning']['objective'] == 'minimum_mae_with_fixed_fraction_floor'
    for method in ('bm', 'sgbm'):
        parameters = config['experiments'][method+'_fixed']['parameters']
        assert parameters['lr_threshold_px'] == 1.0
        assert config['experiments'][method+'_tuned']['parameters'] is None
        for name, values in config['tuning'][method]['search'].items():
            assert parameters[name] in values
    assert config['outputs']['evaluation_run'] == 'evaluation'
    from src.stereo.protocol import load_config
    assert load_config(PROJECT_ROOT/'configs/part1/middlebury_sgbm.yaml') == config
