"""Rectified stereo and pinhole geometry. Distances are in metres."""
from pathlib import Path
import re

import cv2
import numpy as np


def disparity_to_depth(disparity, focal_px, baseline_m, doffs=0.0):
    if focal_px <= 0 or baseline_m <= 0:
        raise ValueError("Focal length and baseline must be positive")
    d = np.asarray(disparity, dtype=np.float32)
    denominator = d + doffs
    valid = np.isfinite(d) & (d > 0) & (denominator > 0)
    z = np.full(d.shape, np.nan, np.float32)
    np.divide(focal_px * baseline_m, denominator, out=z, where=valid)
    return z


def read_middlebury_calibration(path):
    entries = dict(line.split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line)
    camera = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", entries["cam0"])]
    return {"focal_px": camera[0], "baseline_m": float(entries["baseline"]) / 1000,
            "doffs": float(entries.get("doffs", 0)), "ndisp": int(entries.get("ndisp", 128))}


def read_pfm(path):
    with open(path, "rb") as f:
        kind = f.readline().strip()
        if kind not in (b"Pf", b"PF"):
            raise ValueError("Invalid PFM header")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        scale = float(f.readline())
        values = np.fromfile(f, "<f4" if scale < 0 else ">f4")
    shape = (h, w, 3) if kind == b"PF" else (h, w)
    return np.flipud(values.reshape(shape)).astype(np.float32) * abs(scale)


def stereo_depth(left, right, focal_px, baseline_m, doffs=0, method="sgbm", num_disparities=128, block_size=5, lr_threshold=1.0):
    if left.shape != right.shape:
        raise ValueError("Rectified images must have the same shape")
    if num_disparities <= 0 or num_disparities % 16:
        raise ValueError("num_disparities must be a positive multiple of 16")
    if block_size % 2 != 1 or block_size < (5 if method == "bm" else 3):
        raise ValueError("block_size must be odd; BM needs >=5, SGBM >=3")
    gray = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    def matcher(minimum):
        if method == "bm":
            m = cv2.StereoBM_create(num_disparities, block_size)
            m.setMinDisparity(minimum)
            return m
        if method != "sgbm":
            raise ValueError("method must be bm or sgbm")
        return cv2.StereoSGBM_create(minDisparity=minimum, numDisparities=num_disparities,
            blockSize=block_size, P1=8*block_size**2, P2=32*block_size**2,
            uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
            disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    a, b = gray(left), gray(right)
    dl = matcher(0).compute(a, b).astype(np.float32) / 16
    minimum_right = -num_disparities + 1
    dr = matcher(minimum_right).compute(b, a).astype(np.float32) / 16
    y, x = np.indices(dl.shape, dtype=np.float32)
    xr = x - dl
    sampled = cv2.remap(dr, xr, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=float(minimum_right - 1))
    valid = (dl > 0) & (xr >= 0) & (xr <= dl.shape[1]-1)
    valid &= (sampled >= minimum_right) & (sampled <= 0) & (np.abs(dl + sampled) <= lr_threshold)
    depth = disparity_to_depth(dl, focal_px, baseline_m, doffs)
    depth[~valid] = np.nan
    return dl, depth, valid


def backproject(depth, fx, fy, cx, cy):
    if fx <= 0 or fy <= 0:
        raise ValueError("Focal lengths must be positive")
    y, x = np.indices(depth.shape)
    return np.stack(((x-cx)*depth/fx, (y-cy)*depth/fy, depth), axis=-1)


def write_ply(path, depth, rgb, intrinsics, stride=2):
    if stride < 1 or depth.shape != rgb.shape[:2]:
        raise ValueError("Invalid stride or mismatched RGB/depth")
    xyz = backproject(depth, **intrinsics)[::stride, ::stride]
    color = rgb[::stride, ::stride]
    valid = np.isfinite(xyz).all(-1) & (xyz[..., 2] > 0)
    points, colors = xyz[valid], color[valid]
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(points))
        f.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, colors):
            f.write("%.6f %.6f %.6f %d %d %d\n" % (*p, *c))


def measure_point(depth, x, y, intrinsics, radius=2):
    h, w = depth.shape
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("Click is outside the image")
    patch = depth[max(0,y-radius):y+radius+1, max(0,x-radius):x+radius+1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        raise ValueError("No valid depth near the click")
    z = float(np.median(valid))
    return np.array([(x-intrinsics['cx'])*z/intrinsics['fx'],
                     (y-intrinsics['cy'])*z/intrinsics['fy'], z])
