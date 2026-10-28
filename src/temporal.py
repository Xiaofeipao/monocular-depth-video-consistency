"""A0-A6: frozen-prediction temporal fusion ablations."""
from dataclasses import dataclass
import cv2
import numpy as np
from .flow import warp, consistency


@dataclass
class FusionConfig:
    ablation: int = 6
    alpha: float = .4
    fb_threshold: float = 1.0
    rgb_threshold: float = .12
    depth_threshold: float = .2
    edge_threshold: float = .08
    scale_shrink: float = 1.0
    max_log_scale: float = .1
    minimum_reliable: float = .02
    scene_cut_threshold: float = .35
    domain: str = "inverse"

    def __post_init__(self):
        if not 0 <= self.ablation <= 6 or not 0 <= self.alpha <= 1:
            raise ValueError("ablation must be A0-A6 and alpha in [0,1]")
        if min(self.fb_threshold,self.rgb_threshold,self.depth_threshold,self.edge_threshold) <= 0:
            raise ValueError("Gate thresholds must be positive")
        if self.scale_shrink < 0 or self.domain not in ("depth", "inverse"):
            raise ValueError("Invalid scale shrink or fusion domain")


class TemporalFusion:
    def __init__(self, config=None):
        self.config = config or FusionConfig()
        self.previous = self.rgb = None
        self.resets = 0

    def step(self, depth, rgb, forward=None, backward=None):
        c = self.config
        depth = np.asarray(depth, np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        debug = {"weight": np.zeros_like(depth), "scale": 1., "reset": False}
        def finish(result):
            result = np.asarray(result, np.float32)
            result[~valid] = np.nan
            self.previous, self.rgb = result.copy(), rgb.copy()
            return result, debug
        if self.previous is None or c.ablation == 0:
            return finish(depth.copy())
        if depth.shape != self.previous.shape or rgb.shape != self.rgb.shape:
            raise ValueError("Reset the fusion before changing frame dimensions")
        # Scene-cut/low-coverage reset belongs to the full method, keeping weak
        # baselines faithful to their definition.
        rgb_error_unwarped = np.mean(np.abs(rgb.astype(np.float32)-self.rgb.astype(np.float32)))/255
        if c.ablation == 6 and rgb_error_unwarped > c.scene_cut_threshold:
            self.resets += 1
            debug["reset"] = True
            return finish(depth.copy())
        old_valid = np.isfinite(self.previous) & (self.previous > 0)
        old = np.where(old_valid, self.previous, 0)
        if c.ablation == 1:
            historical, usable = old, old_valid & valid
            fb_error = np.zeros_like(depth)
            rgb_error = np.zeros_like(depth)
        else:
            if forward is None or backward is None:
                raise ValueError("A2-A6 require forward and backward flow")
            if c.domain == "inverse":
                old_inverse = np.zeros_like(old)
                np.divide(1,old,out=old_inverse,where=old_valid)
                historical_inverse,inside = warp(old_inverse,backward)
                historical = np.zeros_like(old)
                np.divide(1,historical_inverse,out=historical,where=historical_inverse>0)
            else:
                historical, inside = warp(old, backward)
            warped_valid, _ = warp(old_valid.astype(np.float32), backward)
            usable = inside & (warped_valid > .999) & valid & (historical > 0)
            fb_valid, fb_error = consistency(forward, backward, c.fb_threshold)
            if c.ablation >= 3:
                usable &= fb_valid
            old_rgb, _ = warp(self.rgb, backward)
            rgb_error = np.mean(np.abs(rgb.astype(np.float32)-old_rgb), axis=-1)/255
        log_delta = np.zeros_like(depth)
        np.subtract(np.log(np.maximum(depth,1e-6)), np.log(np.maximum(historical,1e-6)),
                    out=log_delta, where=usable)
        if c.ablation >= 4:
            usable &= (rgb_error < c.rgb_threshold) & (np.abs(log_delta) < c.depth_threshold)
        edge = np.zeros_like(depth, bool)
        if c.ablation >= 5:
            # Neighbour differences rather than texture edges alone; dilate to
            # avoid bilinear interpolation across a depth discontinuity.
            logd = np.log(np.where(valid, depth, 1))
            edge[:,1:] |= np.abs(np.diff(logd,axis=1)) > c.edge_threshold
            edge[1:,:] |= np.abs(np.diff(logd,axis=0)) > c.edge_threshold
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
            edge[:,1:] |= np.abs(np.diff(gray,axis=1)) > c.edge_threshold
            edge[1:,:] |= np.abs(np.diff(gray,axis=0)) > c.edge_threshold
            edge = cv2.dilate(edge.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            usable &= ~edge
        if c.ablation == 6:
            if usable.mean() < c.minimum_reliable:
                self.resets += 1
                debug["reset"] = True
                return finish(depth.copy())
            log_scale = np.clip(np.median(log_delta[usable])/(1+c.scale_shrink),
                                -c.max_log_scale, c.max_log_scale)
            scale = float(np.exp(log_scale))
            historical = historical * scale
            debug["scale"] = scale
        weight = usable.astype(np.float32)*c.alpha
        if c.ablation >= 4:
            exponent = -fb_error/c.fb_threshold-rgb_error/c.rgb_threshold-np.abs(log_delta)/c.depth_threshold
            confidence = np.zeros_like(depth)
            np.exp(exponent,out=confidence,where=usable)
            weight *= confidence
        current_safe = np.where(valid, depth, 1)
        historical = np.where(usable, historical, current_safe)
        if c.domain == "inverse":
            result = 1/((1-weight)/current_safe + weight/np.maximum(historical,1e-6))
        else:
            result = (1-weight)*current_safe + weight*historical
        debug.update(weight=weight, reliable=usable, fb_error=fb_error, rgb_error=rgb_error, edge=edge)
        return finish(result)
