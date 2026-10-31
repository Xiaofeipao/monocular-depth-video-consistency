"""Bidirectional optical flow for evaluation, never for depth modification."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.models.depth_anything_v2_adapter import _runtime, _sha256, _validate_rgb


class FlowEstimator:
    """Return (previous→current, current→previous), in original-canvas pixels.

    RAFT uses local weights exclusively. Farneback is an explicit CPU backend,
    never an automatic fallback after a failed RAFT or CUDA request.
    """

    def __init__(self, *, backend="raft_small", device="cuda:0", checkpoint=None,
                 max_long_edge=None, iterations=12):
        if backend not in {"raft_small", "farneback"}:
            raise ValueError("backend must be raft_small or farneback")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if max_long_edge is not None and (not isinstance(max_long_edge, int) or isinstance(max_long_edge, bool) or max_long_edge < 1):
            raise ValueError("max_long_edge must be a positive integer or None")
        self.backend, self.max_long_edge, self.iterations = backend, max_long_edge, iterations
        self.metadata = {
            "backend": backend, "iterations": iterations,
            "max_long_edge": max_long_edge, "unit": "evaluation_canvas_pixel",
            "directions": ["previous_to_current", "current_to_previous"],
            "used_for": "evaluation_only", "precision": "float32",
        }
        if backend == "farneback":
            if str(device) != "cpu":
                raise ValueError("Farneback is an explicit CPU backend; request device='cpu'")
            if checkpoint is not None:
                raise ValueError("Farneback does not use a checkpoint")
            self.device = "cpu"
            self.metadata.update(device="cpu", checkpoint_sha256=None,
                                 parameters=0, checkpoint_bytes=0,
                                 implementation_version=cv2.__version__)
            return
        self.torch, self.device = _runtime(device, "float32")
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError("RAFT-Small requires an existing local checkpoint; downloads are disabled")
        import torchvision
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        self.model = raft_small(weights=None, progress=False)
        self.model.load_state_dict(self.torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.to(self.device).eval()
        self.transform = Raft_Small_Weights.DEFAULT.transforms()
        self.metadata.update(
            device=str(self.device), checkpoint=str(Path(checkpoint).resolve()),
            checkpoint_sha256=_sha256(Path(checkpoint)),
            checkpoint_bytes=Path(checkpoint).stat().st_size,
            parameters=sum(p.numel() for p in self.model.parameters()),
            implementation_version=torchvision.__version__,
            preprocessing="official torchvision OpticalFlow uint8 -> float32 [-1,1]",
            padding="replicate right/bottom; multiple of 8, minimum 128",
        )

    def _raft(self, first, second):
        torch = self.torch
        a = torch.from_numpy(first.copy()).permute(2, 0, 1).unsqueeze(0)
        b = torch.from_numpy(second.copy()).permute(2, 0, 1).unsqueeze(0)
        a, b = self.transform(a, b)
        height, width = first.shape[:2]
        padded_height = max(128, ((height + 7) // 8) * 8)
        padded_width = max(128, ((width + 7) // 8) * 8)
        pad = (0, padded_width - width, 0, padded_height - height)
        a = torch.nn.functional.pad(a, pad, mode="replicate").to(self.device)
        b = torch.nn.functional.pad(b, pad, mode="replicate").to(self.device)
        with torch.inference_mode():
            prediction = self.model(a, b, num_flow_updates=self.iterations)[-1]
        flow = prediction[0, :, :height, :width].permute(1, 2, 0).float().cpu().numpy()
        self.metadata["network_input_shape"] = [padded_height, padded_width]
        self.metadata["padding_right_bottom"] = [pad[1], pad[3]]
        return flow

    def _one(self, first, second):
        if self.backend == "raft_small":
            return self._raft(first, second)
        return cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(first, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(second, cv2.COLOR_RGB2GRAY), None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=self.iterations,
            poly_n=5, poly_sigma=1.2, flags=0,
        )

    def estimate(self, previous_rgb: np.ndarray, current_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _validate_rgb(previous_rgb)
        _validate_rgb(current_rgb)
        if previous_rgb.shape != current_rgb.shape:
            raise ValueError("Both RGB frames must share an evaluation canvas")
        height, width = previous_rgb.shape[:2]
        scale = min(1.0, self.max_long_edge / max(height, width)) if self.max_long_edge else 1.0
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if size != (width, height):
            previous_rgb = cv2.resize(previous_rgb, size, interpolation=cv2.INTER_AREA)
            current_rgb = cv2.resize(current_rgb, size, interpolation=cv2.INTER_AREA)
        results = []
        for first, second in ((previous_rgb, current_rgb), (current_rgb, previous_rgb)):
            flow = np.asarray(self._one(first, second), dtype=np.float32)
            if flow.shape != (size[1], size[0], 2) or not np.isfinite(flow).all():
                raise ValueError("Optical flow must be finite HxWx2")
            if size != (width, height):
                flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
                flow[..., 0] *= width / size[0]
                flow[..., 1] *= height / size[1]
            results.append(np.ascontiguousarray(flow, dtype=np.float32))
        self.metadata["evaluation_shape"] = [height, width]
        self.metadata["flow_estimation_shape"] = [size[1], size[0]]
        self.metadata["component_rescaling_xy"] = [width / size[0], height / size[1]]
        return results[0], results[1]
