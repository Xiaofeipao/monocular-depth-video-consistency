"""固定参数的C组件移除；完整C仍调用移植的同公式CPU/CUDA实现。"""
import numpy as np
from .edge_aware import refine_edges, CUDAEdgeRefiner

METHODS = ('C', 'C_no_RGB', 'C_no_depth_gate')


def cpu_ablation(depth, rgb, *, use_rgb=True, use_depth=True):
    if use_rgb and use_depth:
        return refine_edges(depth, rgb)
    depth = np.asarray(depth, np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    safe = np.where(valid, depth, 1)
    inverse = np.where(valid, 1 / safe, 0)
    logs = np.log(safe)
    colors = rgb.astype(np.float32) / 255
    h, w = depth.shape
    pad = ((2, 2), (2, 2))
    qp, lp, vp = (np.pad(x, pad, mode='edge') for x in (inverse, logs, valid))
    cp = np.pad(colors, (*pad, (0, 0)), mode='edge')
    total, weights = np.zeros_like(depth), np.zeros_like(depth)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            region = (slice(2 + dy, 2 + dy + h), slice(2 + dx, 2 + dx + w))
            rgb_error = np.mean((cp[region] - colors) ** 2, axis=-1)
            weight = np.exp(-(dx * dx + dy * dy) / (2 * 1.5 ** 2)
                            - (rgb_error / (2 * .10 ** 2) if use_rgb else 0)
                            - ((lp[region] - logs) ** 2 / (2 * .18 ** 2) if use_depth else 0)) * vp[region] * valid
            total += weight * qp[region]
            weights += weight
    output = np.full_like(depth, np.nan)
    np.divide(weights, total, out=output, where=valid & (total > 0))
    return output


def make_refiner(method, backend):
    if method not in METHODS or backend not in ('cpu', 'cuda'):
        raise ValueError('Unknown C method/backend')
    rgb, depth = method != 'C_no_RGB', method != 'C_no_depth_gate'
    if backend == 'cuda':
        return CUDAEdgeRefiner(use_rgb_guidance=rgb, use_depth_guidance=depth)
    return lambda z, image: cpu_ablation(z, image, use_rgb=rgb, use_depth=depth)
