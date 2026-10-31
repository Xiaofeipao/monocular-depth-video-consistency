"""Offline relative VDA-S adapter; preserves official window stitching."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .depth_anything_v2_adapter import (
    SMALL_CONFIG, _base_metadata, _configure_cpu_attention, _import_official, _prediction, _runtime,
    effective_input_size, network_input_shape,
)


def window_metadata(frame_count: int) -> dict:
    if frame_count < 1:
        raise ValueError("A video must contain at least one frame")
    window, overlap, interpolation = 32, 10, 8
    stride = window - overlap
    starts = list(range(0, frame_count, stride))
    padded = (stride - (frame_count % stride)) % stride + overlap
    return {
        "window_length": window, "overlap": overlap, "stride": stride,
        "keyframes": [0, 12, 24, 25, 26, 27, 28, 29, 30, 31],
        "interpolation_length": interpolation,
        "window_count": len(starts), "window_starts": starts,
        "original_frame_count": frame_count, "padded_frame_count": padded,
        "padding_method": "repeat_last_rgb_frame",
        "processed_window_frames": len(starts) * window,
        "keyframe_reused_inputs": max(0, len(starts) - 1) * overlap,
        "interpolated_overlap_frames": max(0, len(starts) - 1) * interpolation,
        "stitch_output_frames_before_trim": frame_count + padded,
        "trimmed_frame_count": padded,
        "window_alignment": "official prediction-to-prediction scale_and_shift; no GT",
        "alignment_reference_keyframes": [0, 12],
    }


class VideoDepthAnythingAdapter:
    """VDA-S relative offline sequence inference, not causal streaming."""

    def __init__(self, *, repository, checkpoint, device="cuda:0", precision="float16", input_size=518):
        torch, actual = _runtime(device, precision)
        repository, checkpoint = Path(repository).resolve(), Path(checkpoint).resolve()
        if not repository.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError("Provide an existing official repository and local checkpoint")
        effective_input_size(1, 1, input_size)
        module = _import_official(repository, "video_depth_anything.video_depth", ("video_depth_anything", "utils"))
        constants = (module.INFER_LEN, module.OVERLAP, list(module.KEYFRAMES), module.INTERP_LEN)
        if constants != (32, 10, [0, 12, 24, 25, 26, 27, 28, 29, 30, 31], 8):
            raise RuntimeError(f"Official VDA window constants changed: {constants}")
        self.model = module.VideoDepthAnything(**SMALL_CONFIG, metric=False)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.to(actual).eval()
        attention = _configure_cpu_attention(self.model, actual)
        self.torch, self.device, self.precision, self.input_size = torch, actual, precision, input_size
        self.metadata = _base_metadata(repository, checkpoint, self.model, actual, precision, input_size)
        self.metadata["attention"] = attention
        self.metadata.update(
            model="video_depth_anything_small", temporal_context=True,
            inference_mode="offline_noncausal",
        )

    def infer(self, frames_rgb: np.ndarray, fps: float) -> np.ndarray:
        if not isinstance(frames_rgb, np.ndarray) or frames_rgb.dtype != np.uint8 or frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3 or min(frames_rgb.shape[:3]) <= 0:
            raise ValueError("Expected a nonempty uint8 TxHxWx3 RGB array")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        expected = network_input_shape(*frames_rgb.shape[1:3], self.input_size)
        observed = []

        def capture_shape(_module, inputs):
            # Official temporal head receives features, patch_h, patch_w, T.
            actual = (int(inputs[1]) * 14, int(inputs[2]) * 14)
            if actual != expected:
                raise RuntimeError(f"Actual VDA input grid {actual} differs from shared protocol {expected}")
            observed.append(actual)

        handle = self.model.head.register_forward_pre_hook(capture_shape)
        # The upstream implementation passes this string to autocast, which
        # requires 'cuda', not 'cuda:0'.  Current-device context preserves the
        # explicitly requested GPU for its .to('cuda') tensors.
        context = self.torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext()
        try:
            with self.torch.inference_mode(), context:
                output, output_fps = self.model.infer_video_depth(
                    frames_rgb, fps, input_size=self.input_size,
                    device=self.device.type, fp32=self.precision == "float32",
                )
        finally:
            handle.remove()
        windows = window_metadata(len(frames_rgb))
        if len(observed) != windows["window_count"]:
            raise RuntimeError("VDA actual head-call count differs from the recorded window protocol")
        if not np.isclose(output_fps, fps, rtol=0, atol=1e-9):
            raise RuntimeError("Official VDA changed the requested FPS")
        self.metadata.update(windows)
        self.metadata["windowing"] = windows
        self.metadata["network_input_shape"] = list(observed[0])
        self.metadata["network_input_hw"] = list(observed[0])
        self.metadata["observed_window_input_shapes"] = [list(s) for s in observed]
        return _prediction(output, frames_rgb.shape[:3])
