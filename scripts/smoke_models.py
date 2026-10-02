#!/usr/bin/env python3
"""Run inference-only smoke tests for the mandatory pretrained model path."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def summarize(array: np.ndarray) -> dict[str, object]:
    array = np.asarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_fraction": float(np.isfinite(array).mean()),
        "minimum": float(np.nanmin(array)),
        "maximum": float(np.nanmax(array)),
    }


def release(model: torch.nn.Module) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_depth_anything(project_root: Path, image_bgr: np.ndarray) -> dict[str, object]:
    sys.path.insert(0, str(project_root / "Depth-Anything-V2"))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
    )
    checkpoint = project_root / "checkpoints/depth_anything_v2/depth_anything_v2_vits.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model = model.cuda().eval()
    depth = model.infer_image(image_bgr, input_size=252)
    result = summarize(depth)
    result["depth_type"] = "relative"
    release(model)
    return result


def run_depth_pro(project_root: Path, image_path: Path) -> dict[str, object]:
    sys.path.insert(0, str(project_root / "ml-depth-pro/src"))
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    config = dataclasses.replace(
        DEFAULT_MONODEPTH_CONFIG_DICT,
        checkpoint_uri=str(project_root / "checkpoints/depth_pro/depth_pro.pt"),
    )
    model, transform = depth_pro.create_model_and_transforms(
        config=config,
        device=torch.device("cuda"),
        precision=torch.float16,
    )
    model.eval()
    image = Image.open(image_path).convert("RGB")
    prediction = model.infer(transform(image))
    depth = prediction["depth"].detach().float().cpu().numpy()
    result = summarize(depth)
    result["depth_type"] = "metric"
    result["unit"] = "meter"
    result["focal_length_px"] = float(
        prediction["focallength_px"].detach().float().cpu().item()
    )
    release(model)
    return result


def run_video_depth(
    project_root: Path,
    image_bgr: np.ndarray,
    *,
    metric: bool,
) -> dict[str, object]:
    sys.path.insert(0, str(project_root / "Video-Depth-Anything"))
    from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        metric=metric,
    )
    filename = (
        "metric_video_depth_anything_vits.pth"
        if metric
        else "video_depth_anything_vits.pth"
    )
    checkpoint = project_root / "checkpoints/video_depth_anything" / filename
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model = model.cuda().eval()
    frame_count = 8
    frames = np.repeat(image_bgr[None, ...], repeats=frame_count, axis=0)
    depths, returned_fps = model.infer_video_depth(
        frames,
        target_fps=1,
        input_size=252,
        device="cuda",
        fp32=False,
    )
    result = summarize(depths)
    result["depth_type"] = "metric" if metric else "relative"
    result["unit"] = "meter" if metric else "arbitrary"
    result["input_frame_count"] = frame_count
    result["returned_fps"] = returned_fps
    release(model)
    return result


def run_raft_small(project_root: Path, image_bgr: np.ndarray) -> dict[str, object]:
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=None)
    checkpoint = (
        project_root
        / "checkpoints/torchvision/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth"
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model = model.cuda().eval()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image = torch.nn.functional.interpolate(
        image, size=(128, 128), mode="bilinear", align_corners=False
    )
    image1, image2 = weights.transforms()(image, image.clone())
    with torch.inference_mode():
        flow = model(image1.cuda(), image2.cuda())[-1].float().cpu().numpy()
    result = summarize(flow)
    result["mean_flow_magnitude_px"] = float(
        np.linalg.norm(flow[0], axis=0).mean()
    )
    release(model)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/step0/model_smoke.json"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else project_root / args.output

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script inside the allocated GPU job")

    image_path = project_root / "ml-depth-pro/data/example.jpg"
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)

    results: dict[str, object] = {
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
        },
        "depth_anything_v2_small": run_depth_anything(project_root, image_bgr),
        "depth_pro": run_depth_pro(project_root, image_path),
        "video_depth_anything_small_relative": run_video_depth(
            project_root, image_bgr, metric=False
        ),
        "video_depth_anything_small_metric": run_video_depth(
            project_root, image_bgr, metric=True
        ),
        "raft_small": run_raft_small(project_root, image_bgr),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
