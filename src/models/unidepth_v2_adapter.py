"""Inference-only adapter for the official UniDepthV2 implementation."""

from __future__ import annotations

from pathlib import Path
import sys
import time
import warnings

import numpy as np


class UniDepthV2Adapter:
    """Load a local UniDepthV2 checkpoint and emit native metric depth."""

    def __init__(
        self,
        *,
        repository: Path,
        checkpoint_dir: Path,
        device: str = "cuda:0",
    ) -> None:
        import torch
        self.torch = torch
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

    def infer(
        self,
        rgb: np.ndarray,
        *,
        intrinsics_px: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Return metric depth, effective intrinsics, runtime, and peak CUDA bytes.

        ``intrinsics_px=None`` preserves the frozen self-predicted-camera track.
        Supplying a 3x3 pinhole matrix is reserved for a separately labelled
        extra-camera-information ablation.
        """

        torch = self.torch
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 HxWx3 RGB, got {rgb.shape}/{rgb.dtype}")
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
        camera = None
        if intrinsics_px is not None:
            intrinsics_px = np.asarray(intrinsics_px, dtype=np.float32)
            if (
                intrinsics_px.shape != (3, 3)
                or not np.all(np.isfinite(intrinsics_px))
                or intrinsics_px[0, 0] <= 0
                or intrinsics_px[1, 1] <= 0
            ):
                raise ValueError(f"Invalid 3x3 intrinsics: {intrinsics_px}")
            camera = torch.from_numpy(intrinsics_px.copy())
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="!! self.resolution_level not set.*"
            )
            prediction = self.model.infer(image, camera=camera)
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
        if intrinsics_px is None:
            intrinsics = (
                prediction["intrinsics"]
                .squeeze(0)
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        else:
            # UniDepth still exposes its camera-head prediction even when rays
            # are built from a supplied camera. For ablation metadata we return
            # the effective input camera, not that unused auxiliary prediction.
            intrinsics = intrinsics_px.astype(np.float32, copy=True)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        return depth, intrinsics, runtime, peak_memory

    def environment(self) -> dict[str, object]:
        torch = self.torch
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
