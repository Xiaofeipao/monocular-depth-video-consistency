"""Direction C: joint RGB/inverse-depth bilateral refinement with hole masks."""
import numpy as np


def refine_edges(depth, rgb, radius=2, sigma_rgb=.10, sigma_log=.18, sigma_space=1.5):
    if radius < 1 or min(sigma_rgb,sigma_log,sigma_space) <= 0:
        raise ValueError('Invalid joint bilateral settings')
    depth = np.asarray(depth,np.float32)
    if rgb.shape[:2] != depth.shape:
        raise ValueError('RGB/depth dimensions differ')
    valid = np.isfinite(depth)&(depth>0)
    safe = np.where(valid,depth,1)
    q = np.where(valid,1/safe,0)
    logd = np.log(safe)
    color = rgb.astype(np.float32)/255
    h,w = depth.shape
    pad = ((radius,radius),(radius,radius))
    qp,lp,vp = (np.pad(x,pad,mode='edge') for x in (q,logd,valid))
    cp = np.pad(color,(*pad,(0,0)),mode='edge')
    total,weights = np.zeros_like(q),np.zeros_like(q)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            sl = (slice(radius+dy,radius+dy+h),slice(radius+dx,radius+dx+w))
            rgb_error = np.mean((cp[sl]-color)**2,axis=-1)
            weight = np.exp(-(dx*dx+dy*dy)/(2*sigma_space**2)-rgb_error/(2*sigma_rgb**2)
                            -(lp[sl]-logd)**2/(2*sigma_log**2))*vp[sl]*valid
            total += weight*qp[sl]
            weights += weight
    out = np.full_like(depth,np.nan)
    np.divide(weights,total,out=out,where=valid&(total>0))
    return out


class CUDAEdgeRefiner:
    """Same 5x5 joint bilateral formula, vectorized on CUDA in FP32.

    Host transfers and the synchronizing output copy belong to processing time.
    No learned parameters, lower-resolution approximation or half precision.
    """
    def __init__(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('C_cuda requires a CUDA-capable PyTorch environment')
        self.torch = torch

    def __call__(self, depth, rgb):
        torch = self.torch
        import torch.nn.functional as functional
        depth = np.asarray(depth,np.float32)
        if depth.ndim != 2 or rgb.shape != (*depth.shape,3):
            raise ValueError('Expected HxW depth and HxWx3 RGB')
        with torch.inference_mode():
            z = torch.as_tensor(depth,device='cuda')
            valid = torch.isfinite(z)&(z>0)
            safe = torch.where(valid,z,1.)
            inverse = torch.where(valid,1/safe,0.)
            logd = torch.log(safe)
            color = torch.as_tensor(np.ascontiguousarray(rgb),device='cuda',dtype=torch.float32).permute(2,0,1)/255
            h,w = depth.shape
            def neighbours(value):
                padded = functional.pad(value[None],(2,2,2,2),mode='replicate')
                return functional.unfold(padded,kernel_size=5).view(value.shape[0],25,h,w)
            colors = neighbours(color)
            rgb_error = ((colors-color[:,None])**2).mean(dim=0)
            del colors
            logs = neighbours(logd[None])[0]
            offsets = torch.arange(-2,3,device='cuda',dtype=torch.float32)
            spatial = (offsets[:,None]**2+offsets[None,:]**2).reshape(25,1,1)/(2*1.5**2)
            weight = torch.exp(-spatial-rgb_error/(2*.10**2)-(logs-logd)**2/(2*.18**2))
            weight *= neighbours(valid.float()[None])[0]*valid
            total = (weight*neighbours(inverse[None])[0]).sum(dim=0)
            denominator = weight.sum(dim=0)
            result = torch.where(valid&(total>0),denominator/total,float('nan'))
            return result.cpu().numpy()
