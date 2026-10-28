"""Direction E: reduced/adaptive flow resolution, with full-resolution depths.

This accelerates postprocessing; it is not an ONNX or backbone-speed claim.
"""
import cv2
import numpy as np
from ..flow import FarnebackFlow


class ReducedFlow:
    def __init__(self, scale=.5, adaptive=False, motion_threshold=.06, local_motion=False):
        if not 0<scale<=1:
            raise ValueError('Flow scale must be in (0,1]')
        self.scale,self.adaptive,self.motion_threshold = scale,adaptive,motion_threshold
        self.local_motion = local_motion
        self.backend = FarnebackFlow()
        self.name = 'adaptive-farneback' if adaptive else f'farneback-{scale:g}'
        self.used_scales = []

    def __call__(self, previous, current):
        scale = self.scale
        if self.adaptive:
            difference = np.mean(np.abs(previous.astype(np.float32)-current.astype(np.float32)),axis=-1)/255
            score = float(difference.mean())
            if self.local_motion:
                # An 8x8 grid keeps a small moving object from disappearing in
                # the full-image average. This is a change trigger, not flow.
                tiles = np.array_split(difference,min(8,difference.shape[0]),axis=0)
                score = max(float(t.mean()) for row in tiles for t in np.array_split(row,min(8,row.shape[1]),axis=1))
            if score > self.motion_threshold:
                scale = 1.
        h,w = previous.shape[:2]
        if scale == 1:
            self.used_scales.append(1.)
            return self.backend(previous,current)
        hh,ww = max(8,round(h*scale)),max(8,round(w*scale))
        frames = [cv2.resize(f,(ww,hh),interpolation=cv2.INTER_AREA) for f in (previous,current)]
        fields = self.backend(*frames)
        self.used_scales.append(scale)
        return tuple(cv2.resize(field,(w,h),interpolation=cv2.INTER_LINEAR)*np.array([w/ww,h/hh],np.float32) for field in fields)
