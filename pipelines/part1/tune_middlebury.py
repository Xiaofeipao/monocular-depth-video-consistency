#!/usr/bin/env python3
"""Tune BM and SGBM once on the declared Middlebury development scenes."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stereo.io import load_middlebury_scene
from src.stereo.matching import StereoMatcherConfig, match_stereo
from src.stereo.metrics import compute_disparity_metrics


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["protocol_status"] not in {"tuning_v2", "frozen_v2"}:
        raise ValueError(f"Unexpected protocol status: {config['protocol_status']}")
    return config


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tune_method(
    *, method: str, config: dict, scenes: list, output_dir: Path
) -> dict[str, object]:
    tuning = config["tuning"]
    method_tuning = tuning[method]
    search = method_tuning["search"]
    fixed = method_tuning["fixed"]
    parameter_names = list(search)
    combinations = list(itertools.product(*(search[name] for name in parameter_names)))
    rows: list[dict[str, float | int]] = []
    method_output_dir = output_dir / method
    method_output_dir.mkdir(parents=True, exist_ok=True)

    for index, combination in enumerate(combinations, start=1):
        candidate = dict(zip(parameter_names, combination))
        matcher_config = StereoMatcherConfig(method=method, **fixed, **candidate)
        scene_metrics = []
        for scene in scenes:
            result = match_stereo(
                scene.left_rgb,
                scene.right_rgb,
                calibration_num_disparities=scene.calibration.num_disparities,
                config=matcher_config,
                lr_threshold_px=float(tuning["lr_threshold_px"]),
            )
            scene_metrics.append(
                compute_disparity_metrics(
                    result.left_disparity_px,
                    scene.gt_disparity_px,
                    evaluation_mask=scene.non_occluded_mask,
                )
            )
        row: dict[str, float | int] = {
            **candidate,
            "bad2": float(np.mean([metrics["bad2"] for metrics in scene_metrics])),
            "disparity_mae_px": float(
                np.mean([metrics["disparity_mae_px"] for metrics in scene_metrics])
            ),
            "valid_coverage": float(
                np.mean([metrics["valid_coverage"] for metrics in scene_metrics])
            ),
        }
        rows.append(row)
        candidate_text = " ".join(f"{key}={value}" for key, value in candidate.items())
        print(
            f"[{method} {index:03d}/{len(combinations)}] {candidate_text}: "
            f"bad2={row['bad2']:.4f}, mae={row['disparity_mae_px']:.3f}, "
            f"coverage={row['valid_coverage']:.4f}",
            flush=True,
        )

    rows.sort(
        key=lambda row: (
            float(row["bad2"]),
            float(row["disparity_mae_px"]),
            -float(row["valid_coverage"]),
        )
    )
    write_csv(method_output_dir / "candidates.csv", rows)
    best: dict[str, object] = {
        "method": method,
        "objective": tuning["objective"],
        "tie_breakers": tuning["tie_breakers"],
        "development_scenes": config["split"]["development_scenes"],
        "candidate_count": len(rows),
        "lr_threshold_px": float(tuning["lr_threshold_px"]),
        "best": rows[0],
        "fixed": fixed,
    }
    (method_output_dir / "best.json").write_text(
        json.dumps(best, indent=2) + "\n", encoding="utf-8"
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/part1/middlebury_sgbm.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/part1/middlebury/tuning_v2",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=("bm", "sgbm"), default=("bm", "sgbm")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_root = PROJECT_ROOT / config["dataset"]["root"]
    development_names = config["split"]["development_scenes"]
    scenes = [load_middlebury_scene(dataset_root / name) for name in development_names]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        method: tune_method(
            method=method,
            config=config,
            scenes=scenes,
            output_dir=args.output_dir,
        )
        for method in args.methods
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
