from pipelines.part2.run_depth_pro_nyuv2 import select_failure_cases


THRESHOLDS = {
    "abs_rel": {"operator": "greater_than", "value": 0.25},
    "delta1": {"operator": "less_than", "value": 0.60},
    "si_boundary_f1": {"operator": "less_than", "value": 0.01},
}


def make_row(position: int, *, abs_rel: float, delta1: float, boundary: float) -> dict:
    return {
        "test_position": position,
        "matlab_index_1based": position + 1,
        "scene": f"scene-{position}",
        "scene_type": "test",
        "abs_rel": abs_rel,
        "delta1": delta1,
        "si_boundary_f1": boundary,
    }


def test_failure_selection_uses_thresholds_without_fixed_count() -> None:
    rows = [
        make_row(0, abs_rel=0.10, delta1=0.90, boundary=0.20),
        make_row(1, abs_rel=0.30, delta1=0.90, boundary=0.20),
        make_row(2, abs_rel=0.10, delta1=0.50, boundary=0.005),
        make_row(3, abs_rel=0.25, delta1=0.60, boundary=0.01),
    ]

    selected = select_failure_cases(rows, THRESHOLDS)

    assert [item["test_position"] for item in selected] == [2, 1]
    assert selected[0]["triggered_metrics"] == ["delta1", "si_boundary_f1"]
    assert selected[1]["triggered_metrics"] == ["abs_rel"]
