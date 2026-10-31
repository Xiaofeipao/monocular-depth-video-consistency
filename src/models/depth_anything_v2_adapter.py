"""Local-checkpoint DA-V2-S adapter with auditable, shared video preprocessing.

The official DA image helper selects its own device.  This adapter instead
constructs the official transform and explicitly places the tensor on the
requested device.  RGB conversion follows VDA's float32 path for both models.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
from types import MethodType

import cv2
import numpy as np


SMALL_CONFIG = dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_identity(repository: Path) -> dict:
    def git(*args):
        completed = subprocess.run(
            ["git", "-C", str(repository), *args], capture_output=True, text=True,
            check=False, timeout=15,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    sources = {
        str(path.relative_to(repository)): _sha256(path)
        for path in sorted(repository.rglob("*.py"))
        if ".git" not in path.parts and "__pycache__" not in path.parts
    }
    return {
        "repository": str(repository), "upstream_git_sha": revision,
        "repository_dirty": None if status is None else bool(status),
        "source_files_sha256": sources,
    }


def _import_official(repository: Path, module: str, namespaces: tuple[str, ...]):
    """Reject modules already cached from a different copy of an upstream repo."""
    def check_loaded():
        for name, loaded in tuple(sys.modules.items()):
            if not any(name == prefix or name.startswith(prefix + ".") for prefix in namespaces):
                continue
            locations = []
            if getattr(loaded, "__file__", None):
                locations.append(Path(loaded.__file__).resolve())
            locations.extend(Path(p).resolve() for p in getattr(loaded, "__path__", []))
            if not locations or any(not p.is_relative_to(repository) for p in locations):
                raise RuntimeError(
                    f"Imported namespace {name!r} does not originate in {repository}; "
                    "start a clean Python process instead of mixing source copies"
                )

    check_loaded()
    location = str(repository)
    if location in sys.path:
        sys.path.remove(location)
    sys.path.insert(0, location)
    importlib.invalidate_caches()
    loaded = importlib.import_module(module)
    check_loaded()
    return loaded


def _runtime(device: str, precision: str):
    import torch

    if precision not in {"float16", "float32"}:
        raise ValueError("precision must be float16 or float32")
    actual = torch.device(device)
    if actual.type not in {"cuda", "cpu"}:
        raise ValueError("Video adapters support an explicit CPU or CUDA device")
    if actual.type == "cpu" and precision != "float32":
        raise ValueError("CPU inference requires precision=float32")
    if actual.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; request CPU/float32 explicitly")
        index = torch.cuda.current_device() if actual.index is None else actual.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        actual = torch.device("cuda", index)
    return torch, actual


def _configure_cpu_attention(model, device) -> dict:
    """Select official PyTorch reference attention on CPU, per instance only.

    An installed xFormers package can import successfully without providing CPU
    kernels. The upstream DINO base class already supplies the exact reference
    attention computation, and VDA motion attention has its own ordinary-bmm
    branch. No third-party file, global class, weight or CUDA instance changes.
    """
    is_cpu = str(device).split(":", 1)[0] == "cpu"
    info = {
        "backend": "official_pytorch_reference" if is_cpu else "official_upstream",
        "cpu_reference_enabled": is_cpu,
        "dino_reference_instance_count": 0,
        "motion_xformers_disabled_instance_count": 0,
        "dino_reference_instance_paths": [],
        "motion_xformers_disabled_instance_paths": [],
    }
    if not is_cpu:
        return info
    dino_modules = {
        "depth_anything_v2.dinov2_layers.attention",
        "video_depth_anything.dinov2_layers.attention",
    }
    for name, layer in model.named_modules():
        hierarchy = type(layer).__mro__
        optimized = next((base for base in hierarchy if base.__name__ == "MemEffAttention" and base.__module__ in dino_modules), None)
        reference = next((base for base in hierarchy if base.__name__ == "Attention" and base.__module__ in dino_modules), None)
        if optimized is not None and reference is not None and getattr(layer.forward, "__func__", None) is not reference.forward:
            layer.forward = MethodType(reference.forward, layer)
            info["dino_reference_instance_paths"].append(name)
        motion = any(base.__module__.startswith("video_depth_anything.motion_module.") for base in hierarchy)
        if motion and getattr(layer, "_use_memory_efficient_attention_xformers", False):
            layer._use_memory_efficient_attention_xformers = False
            info["motion_xformers_disabled_instance_paths"].append(name)
    info["dino_reference_instance_count"] = len(info["dino_reference_instance_paths"])
    info["motion_xformers_disabled_instance_count"] = len(info["motion_xformers_disabled_instance_paths"])
    return info


def effective_input_size(height: int, width: int, input_size: int = 518) -> int:
    """Official VDA panoramic adjustment, also applied to framewise DA-V2."""
    if min(height, width) <= 0 or input_size < 14:
        raise ValueError("Positive image dimensions and input_size >=14 are required")
    ratio = max(height, width) / min(height, width)
    size = input_size
    if ratio > 1.78:
        size = round(int(input_size * 1.777 / ratio) / 14) * 14
    if size < 14:
        raise ValueError("Aspect ratio is too extreme for the chosen input_size")
    return int(size)


def network_input_shape(height: int, width: int, input_size: int = 518) -> tuple[int, int]:
    """Pure shape equivalent of official lower-bound, multiple-of-14 Resize."""
    size = effective_input_size(height, width, input_size)
    scale = max(size / height, size / width)

    def constrain(value):
        rounded = int(np.round(value / 14) * 14)
        return int(np.ceil(value / 14) * 14) if rounded < size else rounded

    return constrain(scale * height), constrain(scale * width)


def _transform(module, rgb: np.ndarray, input_size: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    size = effective_input_size(height, width, input_size)
    sample = {"image": rgb.astype(np.float32) / 255.0}
    sample = module.Resize(
        width=size, height=size, resize_target=False, keep_aspect_ratio=True,
        ensure_multiple_of=14, resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_CUBIC,
    )(sample)
    sample = module.NormalizeImage(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )(sample)
    return module.PrepareForNet()(sample)["image"]


def _validate_rgb(rgb: np.ndarray):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) <= 0:
        raise ValueError("Expected a nonempty uint8 HxWx3 RGB array")


def _prediction(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Nonfinite or incorrectly shaped relative prediction: {value.shape}, expected {shape}")
    if np.any(value < 0):
        raise ValueError("The official relative branch must return nonnegative values")
    return value


def _base_metadata(repository: Path, checkpoint: Path, model, device, precision: str, input_size: int) -> dict:
    return {
        **_repository_identity(repository), "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "encoder": "vits", "depth_type": "relative", "unit": "arbitrary",
        "representation": "inverse_depth_like", "near_is_large": True,
        "alignment": "none", "gt_used": False, "device": str(device),
        "precision": precision, "input_size": input_size,
        "preprocessing": {
            "color": "RGB", "input_range": "float32 RGB / 255",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "resize": "official lower_bound, keep_aspect_ratio, multiple_of_14",
            "panorama": "official VDA ratio>1.78 adjustment shared by both models",
            "resize_interpolation": "OpenCV INTER_CUBIC",
            "output_interpolation": "torch bilinear align_corners=True",
            "common_float_path": "official VDA float32 conversion applied to both models",
        },
    }


class DepthAnythingV2Adapter:
    """Independent, framewise DA-V2-S; zero is retained as a relative value."""

    def __init__(self, *, repository, checkpoint, device="cuda:0", precision="float16", input_size=518):
        torch, actual = _runtime(device, precision)
        repository, checkpoint = Path(repository).resolve(), Path(checkpoint).resolve()
        if not repository.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError("Provide an existing official repository and local checkpoint")
        effective_input_size(1, 1, input_size)
        module = _import_official(repository, "depth_anything_v2.dpt", ("depth_anything_v2",))
        self.transforms = _import_official(repository, "depth_anything_v2.util.transform", ("depth_anything_v2",))
        self.model = module.DepthAnythingV2(**SMALL_CONFIG)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.to(actual).eval()
        attention = _configure_cpu_attention(self.model, actual)
        self.torch, self.device, self.precision, self.input_size = torch, actual, precision, input_size
        self.metadata = _base_metadata(repository, checkpoint, self.model, actual, precision, input_size)
        self.metadata["attention"] = attention
        self.metadata.update(model="depth_anything_v2_small", temporal_context=False)

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        _validate_rgb(rgb)
        torch = self.torch
        tensor = torch.from_numpy(_transform(self.transforms, rgb, self.input_size)).unsqueeze(0).to(self.device)
        shape = tuple(tensor.shape[-2:])
        expected = network_input_shape(*rgb.shape[:2], self.input_size)
        if shape != expected:
            raise RuntimeError(f"Actual DA preprocessing shape {shape} differs from common protocol {expected}")
        self.metadata["network_input_shape"] = list(shape)
        self.metadata["network_input_hw"] = list(shape)
        context = torch.autocast("cuda", dtype=torch.float16) if self.precision == "float16" else nullcontext()
        with torch.inference_mode(), context:
            output = self.model(tensor)
            output = torch.nn.functional.interpolate(
                output[:, None], rgb.shape[:2], mode="bilinear", align_corners=True,
            )[0, 0]
        return _prediction(output.float().cpu().numpy(), rgb.shape[:2])
