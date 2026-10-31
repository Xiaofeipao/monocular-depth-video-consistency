"""Small, model-independent helpers for auditable video runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse non-standard NaN/Infinity; unavailable measurements use null + reason.
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                         default=_json_default) + "\n"
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def create_output_directory(path: str | Path) -> Path:
    path = Path(path).resolve()
    # Even an empty existing directory can belong to a different attempt.
    path.mkdir(parents=True, exist_ok=False)
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_info() -> dict:
    import cv2
    import torch
    import torchvision
    result = {
        "recorded_at_utc": utc_now(), "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "numpy": np.__version__, "opencv": cv2.__version__,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_num_threads": torch.get_num_threads(),
    }
    if result["cuda_available"]:
        result["visible_gpus"] = [
            {"index": i, "name": torch.cuda.get_device_name(i),
             "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())
        ]
    return result


def source_fingerprints() -> dict[str, str]:
    paths = sorted((PROJECT_ROOT / "src/video").glob("*.py"))
    paths += [PROJECT_ROOT / "src/metrics/temporal.py"]
    paths += [PROJECT_ROOT / "src/models/depth_anything_v2_adapter.py",
              PROJECT_ROOT / "src/models/video_depth_anything_adapter.py"]
    paths += sorted((PROJECT_ROOT / "pipelines/video").glob("*.py"))
    return {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in paths if p.is_file()}


def verify_source_fingerprints(expected: dict[str, str]) -> None:
    for relative, digest in expected.items():
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or sha256_file(path) != digest:
            raise ValueError(f"Scientific source changed: {relative}; use a new run")


def repository_identity(repository: Path) -> dict:
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repository), *args], text=True, stderr=subprocess.PIPE
        ).strip()
    return {"path": str(repository.resolve()), "commit": git("rev-parse", "HEAD"),
            "has_substantive_changes": bool(git("diff", "--ignore-space-at-eol", "--stat")),
            "staged_changes": bool(git("diff", "--cached", "--stat"))}


def inside_path(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Output path escapes its directory: {relative}")
    return path
