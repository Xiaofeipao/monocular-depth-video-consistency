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
