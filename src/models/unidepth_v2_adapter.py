"""Inference-only adapter for the official UniDepthV2 implementation."""

from __future__ import annotations

from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch


class UniDepthV2Adapter:
    """Load a local UniDepthV2 checkpoint and emit native metric depth."""

    def __init__(
        self,
        *,
        repository: Path,
        checkpoint_dir: Path,
        device: str = "cuda:0",
    ) -> None:
        repository = Path(repository).resolve()
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        from unidepth.models import UniDepthV2

        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("CUDA is unavailable for UniDepthV2 inference")
        self.device = torch.device(device)
        start = time.perf_counter()
        self.model = UniDepthV2.from_pretrained(str(Path(checkpoint_dir).resolve()))
        self.model = self.model.to(self.device).eval()
        self.model.interpolation_mode = "bilinear"
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.load_seconds = time.perf_counter() - start

    def infer(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Return native depth, predicted intrinsics, time and peak CUDA bytes."""

        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 HxWx3 RGB, got {rgb.shape}/{rgb.dtype}")
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="!! self.resolution_level not set.*"
            )
            prediction = self.model.infer(image)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        runtime = time.perf_counter() - start
        depth = (
            prediction["depth"]
            .squeeze(0)
            .squeeze(0)
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        intrinsics = (
            prediction["intrinsics"].squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        )
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        return depth, intrinsics, runtime, peak_memory

    def environment(self) -> dict[str, object]:
        result: dict[str, object] = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(self.device),
            "precision": "official_float16_autocast",
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
