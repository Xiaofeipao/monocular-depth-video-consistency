"""Thin adapters around official implementations; weights are always local."""
from contextlib import nullcontext
from dataclasses import replace
import importlib
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from .io import sha256


CONFIGS = {"vits":dict(encoder="vits", features=64, out_channels=[48,96,192,384]),
           "vitb":dict(encoder="vitb", features=128, out_channels=[96,192,384,768])}


class DepthModel:
    def __init__(self, name, repo, checkpoint, device="cpu", encoder="vits", metric=False,
                 input_size=518, fp16=False):
        import torch
        if name not in ("dav2","depth-pro","vda"):
            raise ValueError("Model must be dav2, depth-pro or vda")
        if name == "dav2" and metric:
            raise ValueError("dav2 adapter is relative; use Depth Pro for metric depth")
        if encoder not in CONFIGS or input_size < 14:
            raise ValueError("Use Small/vits or Base/vitb, input_size >=14")
        if fp16 and device != "cuda":
            raise ValueError("FP16 adapter requires CUDA; use FP32 on CPU")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable: use --device cpu")
        repo,checkpoint = Path(repo).resolve(),Path(checkpoint).resolve()
        if not repo.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError("Provide the official source directory and local checkpoint; see checkpoints/README.md")
        self.torch,self.name,self.device = torch,name,device
        self.fp16,self.input_size = fp16,input_size
        self.metric = name == "depth-pro" or metric
        sys.path.insert(0,str(repo/"src" if name == "depth-pro" else repo))
        importlib.invalidate_caches()
        self.transform = None
        if name == "depth-pro":
            import depth_pro
            from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
            config = replace(DEFAULT_MONODEPTH_CONFIG_DICT,checkpoint_uri=str(checkpoint))
            self.model,self.transform = depth_pro.create_model_and_transforms(config,
                device=torch.device(device),precision=torch.float16 if fp16 else torch.float32)
        elif name == "dav2":
            from depth_anything_v2.dpt import DepthAnythingV2
            self.model = DepthAnythingV2(**CONFIGS[encoder])
        else:
            from video_depth_anything.video_depth import VideoDepthAnything
            self.model = VideoDepthAnything(**CONFIGS[encoder],metric=metric)
        if name != "depth-pro":
            self.model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
            self.model.to(device)
        self.model.eval()
        try:
            revision = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            revision = None
        self.metadata = {"model":name,"encoder":encoder if name != "depth-pro" else "official",
                         "depth_type":"metric" if self.metric else "relative",
                         "unit":"meter" if self.metric else "arbitrary",
                         "representation":"depth" if self.metric else "inverse_depth",
                         "alignment":"none","checkpoint_sha256":sha256(checkpoint),
                         "checkpoint_bytes":checkpoint.stat().st_size,"upstream_git_sha":revision,
                         "parameters":sum(p.numel() for p in self.model.parameters()),
                         "device":device,"precision":"fp16" if fp16 else "fp32",
                         "input_size":1536 if name == "depth-pro" else input_size}

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def image(self, bgr):
        if self.name == "vda":
            raise ValueError("VDA must use sequence inference")
        torch = self.torch
        context = torch.autocast("cuda",dtype=torch.float16) if self.fp16 and self.name != "depth-pro" else nullcontext()
        with torch.inference_mode(),context:
            if self.name == "depth-pro":
                from PIL import Image
                transformed = self.transform(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
                # No GT/EXIF focal length in native zero-shot metric evaluation.
                result = self.model.infer(transformed)
                depth = result["depth"].float().cpu().numpy()
                extras = {"focal_px":float(result["focallength_px"].item()),"intrinsics_source":"predicted_focal_center_assumed"}
            else:
                # Mirror official VDA's panoramic input-size adjustment so the
                # framewise comparison uses the same network resolution.
                ratio = max(bgr.shape[:2])/min(bgr.shape[:2])
                effective_size = self.input_size
                if ratio > 1.78:
                    effective_size = round(int(self.input_size*1.777/ratio)/14)*14
                if effective_size < 14:
                    raise ValueError("Aspect ratio is too extreme for the requested input size")
                depth = self.model.infer_image(bgr,effective_size)
                extras = {"effective_input_size":effective_size}
        return np.asarray(depth,np.float32),extras

    def video(self, frames, fps):
        if self.name != "vda":
            raise ValueError("Sequence inference requires VDA")
        rgb = np.stack([cv2.cvtColor(f,cv2.COLOR_BGR2RGB) for f in frames])
        with self.torch.inference_mode():
            depths,output_fps = self.model.infer_video_depth(rgb,fps,input_size=self.input_size,
                device=self.device,fp32=not self.fp16)
        if len(depths) != len(frames) or not np.isclose(output_fps,fps):
            raise ValueError("Official video output changed frame count or FPS")
        return np.asarray(depths,np.float32)
