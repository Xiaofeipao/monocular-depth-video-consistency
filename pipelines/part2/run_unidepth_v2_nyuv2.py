#!/usr/bin/env python3
"""Zero-shot UniDepthV2-S evaluation on the frozen NYUv2 test split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.part2.run_depth_pro_nyuv2 import (
    evaluate_saved_predictions,
    prediction_stem,
    resize_prediction,
    resolve_and_verify,
    selected_positions,
    sha256,
)
from src.data.nyuv2 import NYUv2LabeledDataset
from src.data.output_schema import validate_depth_output
from src.models.unidepth_v2_adapter import UniDepthV2Adapter


def load_config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["protocol_status"] != "frozen_v1":
        raise ValueError(f"Unsupported protocol status: {config['protocol_status']}")
    if config["evaluation"]["test_time_alignment"] != "none":
        raise ValueError("Native metric evaluation must not use test-time alignment")
    if config["model"]["provide_external_intrinsics"]:
        raise ValueError("Primary UniDepthV2 track must predict its own intrinsics")
    if config["model"]["camera_output"] != "intrinsics_px":
        raise ValueError("UniDepthV2 must save a full predicted intrinsics matrix")
    return config


def infer_predictions(
    *,
    config: dict,
    dataset: NYUv2LabeledDataset,
    positions: list[int],
    output_dir: Path,
    device: str,
    resume: bool,
    checkpoint_dir: Path,
) -> dict[str, object]:
    predictions_dir = output_dir / "predictions"
    metadata_dir = output_dir / "metadata"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    adapter = UniDepthV2Adapter(
        repository=PROJECT_ROOT / config["model"]["repository"],
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    environment = {
        **adapter.environment(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "model_load_seconds": adapter.load_seconds,
    }
    (output_dir / "inference_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )

    warmup_iterations = int(config["efficiency"]["warmup_iterations"])
    if warmup_iterations and positions:
        warmup_sample = dataset.get_test_sample(positions[0])
        for _ in range(warmup_iterations):
            adapter.infer(warmup_sample.rgb)

    inferred = skipped = 0
    start = time.perf_counter()
    for sequence_index, position in enumerate(positions, start=1):
        sample = dataset.get_test_sample(position)
        stem = prediction_stem(position, sample.matlab_index_1based)
        prediction_path = predictions_dir / f"{stem}.npz"
        metadata_path = metadata_dir / f"{stem}.json"
        if prediction_path.exists() or metadata_path.exists():
            if resume and prediction_path.exists() and metadata_path.exists():
                skipped += 1
                print(f"[{sequence_index:03d}/{len(positions)}] resume {stem}", flush=True)
                continue
            raise FileExistsError(
                f"Refusing to overwrite incomplete/existing output for {stem}; use --resume"
            )
        depth, intrinsics, runtime, peak_memory = adapter.infer(sample.rgb)
        depth = resize_prediction(depth, sample.depth_m.shape)
        if (
            intrinsics.shape != (3, 3)
            or not np.all(np.isfinite(intrinsics))
            or intrinsics[0, 0] <= 0
            or intrinsics[1, 1] <= 0
        ):
            raise ValueError(f"Invalid predicted intrinsics for {stem}: {intrinsics}")
        metadata = {
            "sample_id": stem,
            "test_position": position,
            "matlab_index_1based": sample.matlab_index_1based,
            "scene": sample.scene,
            "scene_type": sample.scene_type,
            "model": config["model"]["name"],
            "depth_type": "metric",
            "unit": "meter",
            "alignment": "none",
            "intrinsics_source": "model_prediction",
            "intrinsics_px": intrinsics.tolist(),
            "external_intrinsics_provided": False,
            "input_shape": list(sample.rgb.shape[:2]),
            "output_shape": list(depth.shape),
            "inference_seconds": runtime,
            "peak_gpu_memory_bytes": peak_memory,
        }
        validate_depth_output(depth, metadata)
        np.savez_compressed(
            prediction_path,
            depth_m=depth,
            intrinsics_px=intrinsics.astype(np.float32),
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        inferred += 1
        print(
            f"[{sequence_index:03d}/{len(positions)}] {stem}: "
            f"fx={intrinsics[0, 0]:.1f}px, fy={intrinsics[1, 1]:.1f}px, "
            f"{runtime:.3f}s, peak={peak_memory / 2**30:.2f}GiB",
            flush=True,
        )
    wall_seconds = time.perf_counter() - start
    return {
        "positions_requested": len(positions),
        "inferred": inferred,
        "resumed": skipped,
        "wall_seconds": wall_seconds,
        "end_to_end_fps_including_io_and_save": (
            inferred / wall_seconds if inferred else None
        ),
        "environment": environment,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/part2/unidepth_v2_nyuv2.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/part2/unidepth_v2_nyuv2/frozen_v1",
    )
    parser.add_argument(
        "--mode", choices=("infer-evaluate", "infer", "evaluate"), default="infer-evaluate"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--positions", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = config["dataset"]
    labeled_mat = resolve_and_verify(
        PROJECT_ROOT,
        dataset_config["labeled_mat"],
        dataset_config["labeled_mat_sha256"],
    )
    split_file = resolve_and_verify(
        PROJECT_ROOT,
        dataset_config["split_file"],
        dataset_config["split_file_sha256"],
    )
    model_config = resolve_and_verify(
        PROJECT_ROOT,
        config["model"]["config"],
        config["model"]["config_sha256"],
    )
    checkpoint = resolve_and_verify(
        PROJECT_ROOT,
        config["model"]["checkpoint"],
        config["model"]["checkpoint_sha256"],
    )
    if model_config.parent != checkpoint.parent:
        raise ValueError("UniDepthV2 config and checkpoint must share one local directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with NYUv2LabeledDataset(
        labeled_mat,
        split_file,
        split_variable=dataset_config["split_variable"],
    ) as dataset:
        if len(dataset) != int(dataset_config["expected_test_images"]):
            raise ValueError(f"Unexpected NYUv2 test count: {len(dataset)}")
        positions = selected_positions(len(dataset), args.max_samples, args.positions)
        inference = None
        if args.mode in {"infer-evaluate", "infer"}:
            inference = infer_predictions(
                config=config,
                dataset=dataset,
                positions=positions,
                output_dir=args.output_dir,
                device=args.device,
                resume=args.resume,
                checkpoint_dir=checkpoint.parent,
            )
            (args.output_dir / "inference_summary.json").write_text(
                json.dumps(inference, indent=2) + "\n", encoding="utf-8"
            )
        evaluation = None
        if args.mode in {"infer-evaluate", "evaluate"}:
            evaluation = evaluate_saved_predictions(
                config=config,
                dataset=dataset,
                positions=positions,
                output_dir=args.output_dir,
            )

    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "protocol_status": config["protocol_status"],
        "mode": args.mode,
        "sample_count": len(positions),
        "full_test_split": len(positions) == int(dataset_config["expected_test_images"]),
        "formal_training_run": False,
        "test_time_alignment": "none",
        "external_intrinsics_provided": False,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"inference": inference, "evaluation": evaluation}, indent=2))


if __name__ == "__main__":
    main()
