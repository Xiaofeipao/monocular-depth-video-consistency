"""Inference-only adapter for the official Depth Pro implementation."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch


class DepthProAdapter:
    """Load one official checkpoint and emit native metric depth in meters."""

    def __init__(
        self,
        *,
        repository: Path,
        checkpoint: Path,
        device: str = "cuda:0",
        precision: str = "float16",
    ) -> None:
        repository = Path(repository).resolve()
        source = repository / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

        if precision != "float16":
            raise ValueError("The frozen Depth Pro protocol currently requires float16")
        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("CUDA is unavailable for Depth Pro inference")
        self.device = torch.device(device)
        self.precision = torch.float16
        config = dataclasses.replace(
            DEFAULT_MONODEPTH_CONFIG_DICT,
            checkpoint_uri=str(Path(checkpoint).resolve()),
        )
        start = time.perf_counter()
        self.model, self.transform = depth_pro.create_model_and_transforms(
            config=config,
            device=self.device,
            precision=self.precision,
        )
        self.model.eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.load_seconds = time.perf_counter() - start

    def infer(self, rgb: np.ndarray) -> tuple[np.ndarray, float, float, int]:
        """Return native depth, predicted focal length, time and peak CUDA bytes."""

        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 HxWx3 RGB, got {rgb.shape}/{rgb.dtype}")
        image = Image.fromarray(rgb)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            transformed = self.transform(image)
            prediction = self.model.infer(transformed)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        runtime = time.perf_counter() - start
        depth = prediction["depth"].detach().float().cpu().numpy().astype(np.float32)
        focal_length_px = float(
            prediction["focallength_px"].detach().float().cpu().item()
        )
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        return depth, focal_length_px, runtime, peak_memory

    def environment(self) -> dict[str, object]:
        result: dict[str, object] = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(self.device),
            "precision": "float16",
        }
        if self.device.type == "cuda":
            result.update(
                {
                    "gpu_name": torch.cuda.get_device_name(self.device),
                    "gpu_count_visible": torch.cuda.device_count(),
                    "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                        self.device
                    ).total_memory,
                }
            )
        return result
