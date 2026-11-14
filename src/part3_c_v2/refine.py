"""Bounded, RGB-supported directional morphological shock in log depth.

Inspired by Osher/Rudin (1990), not a reproduction of their PDE solver.
All support comes from the input prediction and RGB; never ground truth.
"""
from dataclasses import asdict, dataclass
import cv2
import numpy as np
from src.refinement.edge_aware import refine_edges


@dataclass(frozen=True)
class Parameters:
    denoise_blend: float = .5
    iterations: int = 3
    step: float = .15
    derivative_sigma: float = .8
    support_radius: int = 6
    minimum_depth_ratio: float = 1.10
    maximum_plateau_fraction: float = .15
    rgb_gradient_scale: float = .03
    max_log_change: float = .04

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError('Parameters must be finite')
        if not 0 <= self.denoise_blend <= 1 or not 1 <= self.iterations <= 8:
            raise ValueError('Invalid blend/iteration count')
        if not isinstance(self.iterations, int) or not isinstance(self.support_radius, int):
            raise ValueError('Integer iterations/support required')
        if (not 0 < self.step <= 1 or self.derivative_sigma <= 0 or self.support_radius < 1
                or self.minimum_depth_ratio <= 1 or not 0 < self.maximum_plateau_fraction < .5
                or self.rgb_gradient_scale <= 0
                or self.max_log_change <= 0):
            raise ValueError('Invalid shock support/bounds')


def _inputs(depth, rgb):
    z = np.asarray(depth, np.float32)
    if z.ndim != 2 or rgb.shape != (*z.shape, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected HxW depth and matching uint8 RGB')
    if not np.isfinite(z).all() or (z <= 0).any():
        raise ValueError('Dense positive depth required; no hole filling')
    return z


def denoise(depth, rgb):
    return refine_edges(_inputs(depth, rgb), rgb)


def refine(depth, rgb, parameters=Parameters(), *, method='C_v2', smoothed=None,
           return_diagnostics=False):
    """Return float32 metres, without alignment or range clipping for scoring.

    A log-space convex blend of RAW and v1 C is followed by bounded 1-pixel
    directional erosions/dilations. Curvature determines the side; magnitude
    is gated by predicted depth contrast and parallel RGB gradients. Constant
    depth cannot acquire RGB texture. Seed-fixed signs/gates limit feedback.
    Bounds constrain the entire result relative to RAW and its local envelope.
    """
    if method not in ('C_v2', 'C_v2_no_sharpen', 'C_v2_no_RGB_support'):
        raise ValueError('Unknown v2 method')
    p = parameters
    z = _inputs(depth, rgb)
    smooth = denoise(z, rgb) if smoothed is None else _inputs(smoothed, rgb)
    raw = np.log(z.astype(np.float64))
    seed = (1-p.denoise_blend)*raw+p.denoise_blend*np.log(smooth.astype(np.float64))
    guide = cv2.GaussianBlur(seed, (0, 0), p.derivative_sigma, borderType=cv2.BORDER_REPLICATE)
    def sobel(value, dx, dy):
        return cv2.Sobel(value, cv2.CV_64F, dx, dy, ksize=3,
                         borderType=cv2.BORDER_REPLICATE) / (8 if dx+dy == 1 else 4)
    gx, gy = sobel(guide, 1, 0), sobel(guide, 0, 1)
    norm = np.hypot(gx, gy)
    nx, ny = gx/np.maximum(norm, 1e-12), gy/np.maximum(norm, 1e-12)
    curvature = nx*nx*sobel(guide, 2, 0)+2*nx*ny*sobel(guide, 1, 1)+ny*ny*sobel(guide, 0, 2)
    # A soft sign avoids binary curvature switches creating fresh pixel edges.
    sign = np.tanh(curvature/np.maximum(.25*norm, 1e-6))
    sign = np.where(np.abs(curvature) > 1e-7, sign, 0.)
    kernel = np.ones((2*p.support_radius+1,)*2, np.uint8)
    lower, upper = cv2.erode(raw, kernel), cv2.dilate(raw, kernel)
    y, x = np.indices(z.shape, dtype=np.float32)
    def along(value, offset):
        return cv2.remap(value, np.ascontiguousarray(x+offset*nx, dtype=np.float32),
                         np.ascontiguousarray(y+offset*ny, dtype=np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    r = p.support_radius
    left, right = along(guide, -r), along(guide, r)
    contrast = right-left  # normals point towards increasing predicted depth
    depth_gate = np.clip((contrast-np.log(p.minimum_depth_ratio))/np.log(p.minimum_depth_ratio), 0, 1)
    instability = np.maximum(np.abs(along(guide, -r-2)-left), np.abs(along(guide, r+2)-right))
    stable = np.clip(1-instability/(p.maximum_plateau_fraction*np.maximum(contrast, 1e-12)), 0, 1)
    position = (seed-left)/np.maximum(contrast, 1e-12)
    # Modify only the interior of a two-sided transition; preserve the plateaus.
    interior = np.clip((position-.10)/.15, 0, 1)*np.clip((.90-position)/.15, 0, 1)
    depth_gate *= stable*interior
    color = rgb.astype(np.float64)/255
    # Squared projection across all channels handles chromatic edges as well.
    cx, cy = sobel(color, 1, 0), sobel(color, 0, 1)
    projection = np.sqrt(np.sum((cx*nx[..., None]+cy*ny[..., None])**2, axis=-1)/3)
    rgb_gate = np.clip(cv2.dilate(projection, np.ones((5, 5), np.uint8))/p.rgb_gradient_scale, 0, 1)
    color_contrast = np.sqrt(np.mean((along(color, r)-along(color, -r))**2, axis=-1))
    rgb_gate *= np.clip(color_contrast/.10, 0, 1)
    gate = depth_gate*(1 if method == 'C_v2_no_RGB_support' else rgb_gate)
    gate *= norm > 1e-7
    # Do not modify the image rim where the normal or side support is incomplete.
    r += 2
    gate[:r] = gate[-r:] = 0
    gate[:, :r] = gate[:, -r:] = 0
    maps = [(np.ascontiguousarray(x+k*nx, dtype=np.float32),
             np.ascontiguousarray(y+k*ny, dtype=np.float32)) for k in (-1, 1)]
    lo = np.maximum(lower, raw-p.max_log_change)
    hi = np.minimum(upper, raw+p.max_log_change)
    value = np.clip(seed, lo, hi)
    if method != 'C_v2_no_sharpen':
        for _ in range(p.iterations):
            neighbours = [cv2.remap(value, xx, yy, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE) for xx, yy in maps]
            low = np.minimum(value, np.minimum(*neighbours))
            high = np.maximum(value, np.maximum(*neighbours))
            target = np.where(sign > 0, low, np.where(sign < 0, high, value))
            value = np.clip(value+p.step*gate*np.abs(sign)*(target-value), lo, hi)
    result = np.exp(value).astype(np.float32)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise FloatingPointError('Invalid refinement output')
    if return_diagnostics:
        return result, dict(depth_gate=depth_gate.astype(np.float32),
                            rgb_gate=rgb_gate.astype(np.float32), gate=gate.astype(np.float32),
                            log_change=(value-raw).astype(np.float32))
    return result
