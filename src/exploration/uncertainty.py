"""Direction D: flow-aligned fusion weighted by explicit heuristic uncertainty.

Local log-depth residual is a proxy, NOT a calibrated model uncertainty output.
If supplied, model uncertainty must be a positive log-depth variance map.
"""
import cv2
import numpy as np
from ..flow import warp,consistency


def local_variance(depth):
    valid = np.isfinite(depth)&(depth>0)
    logd = np.log(np.where(valid,depth,1)).astype(np.float32)
    center = cv2.medianBlur(logd,3)
    variance = cv2.GaussianBlur((logd-center)**2,(3,3),0)+1e-4
    variance[~valid] = np.inf
    return variance


class UncertaintyFusion:
    def __init__(self, alpha_max=.7, max_log_update=None, edge_threshold=None):
        if not 0<=alpha_max<=1:
            raise ValueError('alpha_max must be in [0,1]')
        self.alpha_max = alpha_max
        if max_log_update is not None and (not np.isfinite(max_log_update) or max_log_update <= 0):
            raise ValueError('max_log_update must be positive and finite')
        if edge_threshold is not None and (not np.isfinite(edge_threshold) or edge_threshold <= 0):
            raise ValueError('edge_threshold must be positive and finite')
        self.max_log_update, self.edge_threshold = max_log_update, edge_threshold
        self.depth = self.rgb = self.variance = None

    def step(self, depth, rgb, forward=None, backward=None, variance=None):
        depth = np.asarray(depth,np.float32)
        current = local_variance(depth) if variance is None else np.asarray(variance,np.float32)
        valid = np.isfinite(depth)&(depth>0)
        if current.shape != depth.shape or not np.isfinite(current[valid]).all() or (current[valid]<=0).any():
            raise ValueError('Uncertainty must be positive finite log-depth variance on valid pixels')
        output = depth.copy()
        weight = np.zeros_like(depth)
        if self.depth is not None:
            if forward is None or backward is None:
                raise ValueError('Uncertainty fusion requires bidirectional flow')
            if np.mean(np.abs(rgb.astype(float)-self.rgb.astype(float)))/255 < .35:
                previous_valid = np.isfinite(self.depth)&(self.depth>0)&np.isfinite(self.variance)
                inverse = np.zeros_like(depth)
                np.divide(1,self.depth,out=inverse,where=previous_valid)
                old_inverse,inside = warp(inverse,backward)
                old_var,_ = warp(np.where(previous_valid,self.variance,0),backward)
                old_mask,_ = warp(previous_valid.astype(np.float32),backward)
                old_rgb,_ = warp(self.rgb,backward)
                mask,fb_error = consistency(forward,backward,1.)
                photo = np.mean(np.abs(old_rgb-rgb.astype(np.float32)),axis=-1)/255
                mask &= inside&(old_mask>.999)&valid&(old_inverse>0)&(photo<.12)
                old_depth = np.ones_like(depth)
                np.divide(1,old_inverse,out=old_depth,where=old_inverse>0)
                innovation = np.abs(np.log(np.where(valid,depth,1))-np.log(old_depth))
                mask &= innovation < .35
                if self.edge_threshold is not None:
                    # Protect both sides of a discontinuity, including a one-pixel
                    # neighbourhood that bilinear flow sampling can mix.
                    logd = np.log(np.where(valid,depth,1))
                    edge = np.zeros_like(valid)
                    dx = abs(np.diff(logd,axis=1)) > self.edge_threshold
                    dy = abs(np.diff(logd,axis=0)) > self.edge_threshold
                    edge[:,1:] |= dx
                    edge[:,:-1] |= dx
                    edge[1:,:] |= dy
                    edge[:-1,:] |= dy
                    edge |= ~valid
                    mask &= ~cv2.dilate(edge.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
                # Estimate which observation is less reliable. Process variance
                # prevents historical certainty from growing indefinitely.
                ratio = np.zeros_like(depth)
                np.divide(current,current+old_var+2e-4,out=ratio,where=mask)
                weight[mask] = np.minimum(self.alpha_max,ratio[mask])*np.exp(-photo[mask]/.12-fb_error[mask])
                safe = np.where(valid,depth,1)
                output = 1/((1-weight)/safe+weight*np.where(mask,old_inverse,1/safe))
                if self.max_log_update is not None:
                    # Reduce the actual inverse-depth blend weight rather than
                    # clipping the result without updating its variance estimate.
                    bounded = safe*np.exp(np.clip(np.log(output/safe),-self.max_log_update,self.max_log_update))
                    difference = old_inverse-1/safe
                    actual = np.zeros_like(weight)
                    np.divide(1/bounded-1/safe,difference,out=actual,where=mask&(abs(difference)>1e-8))
                    weight = np.minimum(weight,np.maximum(actual,0))
                    output = 1/((1-weight)/safe+weight*np.where(mask,old_inverse,1/safe))
                current = (1-weight)**2*np.where(valid,current,0)+weight**2*old_var+1e-4
        output[~valid] = np.nan
        current[~valid] = np.inf
        self.depth,self.rgb,self.variance = output.copy(),rgb.copy(),current.copy()
        return output,{'uncertainty':current,'weight':weight}
