"""Self-describing predictions; no dependence on coloured depth images."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess

import cv2
import numpy as np


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def read_bgr(path):
    # imdecode handles Windows non-ASCII paths.
    image = cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path, image):
    ok, data = cv2.imencode(Path(path).suffix,image)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    data.tofile(path)


def load_depth(path, scale=1.):
    path = Path(path)
    if path.suffix == ".npy":
        depth = np.load(path,allow_pickle=False)
    else:
        depth = cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Cannot read depth: {path}")
    if depth.ndim != 2 or scale <= 0:
        raise ValueError("Depth must be HxW and scale positive")
    return depth.astype(np.float32)*scale


def read_frames(path, target_fps=None, max_frames=None, max_side=None):
    path = Path(path)
    frames, indices = [], []
    if path.is_dir():
        paths = sorted(p for p in path.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg"))
        if not paths:
            raise ValueError("No image frames")
        if target_fps is None or target_fps <= 0:
            raise ValueError("Image directories require --fps")
        fps = float(target_fps)
        source = ((i,read_bgr(p)) for i,p in enumerate(paths))
        cap = None
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            cap.release()
            raise ValueError("Video has invalid FPS")
        def generate():
            i = 0
            while True:
                ok,frame = cap.read()
                if not ok:
                    break
                yield i,frame
                i += 1
        source = generate()
    output_fps = fps if target_fps is None else min(fps,float(target_fps))
    if output_fps <= 0 or (max_frames is not None and max_frames <= 0):
        raise ValueError("FPS/max_frames must be positive")
    next_sample = 0
    try:
        for i,frame in source:
            if i+1e-7 < next_sample:
                continue
            next_sample += fps/output_fps
            if max_side and max(frame.shape[:2]) > max_side:
                h,w = frame.shape[:2]
                ratio = max_side/max(h,w)
                frame = cv2.resize(frame,(round(w*ratio),round(h*ratio)),interpolation=cv2.INTER_AREA)
            if frames and frame.shape != frames[0].shape:
                raise ValueError("All frames must have the same dimensions")
            frames.append(frame)
            indices.append(i)
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        if cap is not None:
            cap.release()
    if not frames:
        raise ValueError("No frames decoded")
    return frames,indices,output_fps


def environment():
    versions = {}
    for name in ("numpy","opencv-python","torch","torchvision","timm","einops","Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        revision = None
    return {"python":platform.python_version(),"platform":platform.platform(),"packages":versions,"git_sha":revision}


class RunWriter:
    def __init__(self, path, metadata, config=None):
        self.path = Path(path)
        # Frozen baselines must never be overwritten accidentally.
        self.path.mkdir(parents=True,exist_ok=False)
        for name in ("depth","valid","rgb","visualization"):
            (self.path/name).mkdir()
        if metadata["depth_type"] not in ("metric","relative"):
            raise ValueError("Invalid depth type")
        if metadata["depth_type"] == "metric" and (metadata["unit"] != "meter" or metadata["representation"] != "depth"):
            raise ValueError("Metric runs must contain depth in meters")
        self.metadata = {"schema_version":1,"alignment":"none",**metadata}
        self.records = []
        write_json(self.path/"metadata.json",self.metadata)
        write_json(self.path/"config_resolved.json",config or {})
        write_json(self.path/"environment.json",environment())

    def add(self, depth, rgb, source, source_index=None, extras=None):
        depth = np.asarray(depth,np.float32)
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError("Depth and RGB dimensions must match")
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise ValueError("Prediction has no valid positive values")
        key = f"{len(self.records):06d}"
        np.save(self.path/"depth"/f"{key}.npy",depth)
        write_image(self.path/"valid"/f"{key}.png",valid.astype(np.uint8)*255)
        write_image(self.path/"rgb"/f"{key}.png",rgb)
        self.records.append({"id":key,"source":str(source),"source_index":source_index,
                             "rgb_sha256":sha256(self.path/"rgb"/f"{key}.png"),
                             "depth_sha256":sha256(self.path/"depth"/f"{key}.npy"),
                             "shape":list(depth.shape),**(extras or {})})

    def finish(self, timing):
        write_json(self.path/"manifest.json",self.records)
        write_json(self.path/"timing.json",timing)


def load_run(path):
    path = Path(path)
    metadata = json.loads((path/"metadata.json").read_text(encoding="utf-8"))
    records = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    if not records:
        raise ValueError("Empty run")
    ids = [record['id'] for record in records]
    if len(set(ids)) != len(ids) or any(not key.isdigit() for key in ids):
        raise ValueError("Invalid or duplicate saved frame IDs")
    return metadata,records


def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty result list")
    with open(path,"w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
