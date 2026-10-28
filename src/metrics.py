"""Explicit masks/protocols, image macro averages and flow-aligned proxies."""
import cv2
import numpy as np
from .flow import warp, consistency


def evaluation_mask(gt, minimum=.1, maximum=10., mask=None, crop=None):
    valid = np.isfinite(gt) & (gt >= minimum) & (gt <= maximum) & (gt > 0)
    if mask is not None:
        if mask.shape != gt.shape:
            raise ValueError("Mask and GT dimensions differ")
        valid &= mask.astype(bool)
    if crop is not None:
        top, bottom, left, right = crop
        if not (0 <= top < bottom <= gt.shape[0] and 0 <= left < right <= gt.shape[1]):
            raise ValueError("Crop must be [top,bottom,left,right] in GT pixels")
        region = np.zeros(gt.shape, bool)
        region[top:bottom,left:right] = True
        valid &= region
    if not valid.any():
        raise ValueError("No valid GT pixels after protocol mask")
    return valid


def align_prediction(pred, gt, valid, alignment="none", representation="depth"):
    p = np.asarray(pred, np.float64).copy()
    if p.shape != gt.shape or not np.isfinite(p[valid]).all():
        raise ValueError("Prediction must have GT dimensions and finite values on the evaluation mask")
    if alignment == "none":
        if representation == "inverse_depth":
            if (p[valid] <= 0).any():
                raise ValueError("Inverse depth must be positive")
            p = 1/np.maximum(p,1e-12)
        return p
    target = 1/np.maximum(gt,1e-12) if representation == "inverse_depth" else gt
    if alignment == "median":
        denominator = np.median(p[valid])
        if denominator <= 0:
            raise ValueError("Median alignment requires positive median")
        p *= np.median(target[valid])/denominator
    elif alignment == "scale_shift":
        a = np.column_stack((p[valid], np.ones(valid.sum())))
        if np.linalg.matrix_rank(a) < 2:
            raise ValueError("Scale/shift alignment is underdetermined for constant predictions")
        scale, shift = np.linalg.lstsq(a,target[valid],rcond=None)[0]
        p = p*scale+shift
    else:
        raise ValueError("Unknown alignment")
    if representation == "inverse_depth":
        p = 1/np.maximum(p,1e-12)
    return p


def depth_metrics(pred, gt, minimum=.1, maximum=10., mask=None, crop=None,
                  alignment="none", depth_type="metric", representation="depth", clip=False):
    if depth_type == "metric" and (alignment != "none" or representation != "depth"):
        raise ValueError("Native metric evaluation forbids GT alignment and inverse representation")
    valid = evaluation_mask(gt,minimum,maximum,mask,crop)
    p = align_prediction(pred,gt,valid,alignment,representation)
    # Never silently drop failed predictions using a method-dependent mask.
    if not np.isfinite(p[valid]).all() or (p[valid] <= 0).any():
        raise ValueError("Invalid/nonpositive prediction on valid GT; refusing a biased score")
    if clip:
        p = np.clip(p,minimum,maximum)
    a,b = p[valid],np.asarray(gt,np.float64)[valid]
    error = a-b
    ratio = np.maximum(a/b,b/a)
    return {"AbsRel":float(np.mean(np.abs(error)/b)), "SqRel":float(np.mean(error**2/b)),
            "RMSE":float(np.sqrt(np.mean(error**2))),
            "RMSE_log":float(np.sqrt(np.mean((np.log(a)-np.log(b))**2))),
            **{f"delta{i}":float(np.mean(ratio < 1.25**i)) for i in (1,2,3)},
            "valid_pixels":int(valid.sum())}, p, valid


def boundary_metrics(pred, gt, valid, threshold=.08, tolerance=2):
    """Log-depth jump edges; fixed thresholds. This is NOT Depth Pro SI_boundary_F1."""
    if threshold <= 0 or tolerance < 0:
        raise ValueError("Invalid boundary settings")
    def edges(d):
        safe = np.log(np.where(valid & np.isfinite(d) & (d>0), d, 1))
        out = np.zeros(d.shape,bool)
        out[:,1:] |= (np.abs(np.diff(safe,axis=1)) > threshold) & valid[:,1:] & valid[:,:-1]
        out[1:,:] |= (np.abs(np.diff(safe,axis=0)) > threshold) & valid[1:,:] & valid[:-1,:]
        return out
    pe,ge = edges(pred),edges(gt)
    k = np.ones((2*tolerance+1,2*tolerance+1),np.uint8)
    near = lambda e: cv2.dilate(e.astype(np.uint8),k).astype(bool)
    precision = float((pe & near(ge)).sum()/pe.sum()) if pe.any() else float(not ge.any())
    recall = float((ge & near(pe)).sum()/ge.sum()) if ge.any() else float(not pe.any())
    return {"edge_precision":precision, "edge_recall":recall,
            "edge_F1":2*precision*recall/(precision+recall) if precision+recall else 0.}


def temporal_metrics(previous, current, forward, backward, threshold=1.0):
    old_valid = np.isfinite(previous) & (previous > 0)
    old, inside = warp(np.where(old_valid,previous,0),backward)
    sampled_valid,_ = warp(old_valid.astype(np.float32),backward)
    mask,_ = consistency(forward,backward,threshold)
    mask &= inside & (sampled_valid > .999) & np.isfinite(current) & (current > 0) & (old > 0)
    if not mask.any():
        return {"temporal_log_error":None,"local_flicker":None,"log_scale_ratio":None,"coverage":0.}
    residual = np.log(current[mask])-np.log(old[mask])
    center = float(np.median(residual))
    return {"temporal_log_error":float(np.mean(np.abs(residual))),
            "local_flicker":float(np.mean(np.abs(residual-center))),
            "log_scale_ratio":center,"coverage":float(mask.mean())}


def summarize(rows, bootstrap=1000, seed=0):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    rng = np.random.default_rng(seed)
    result = {}
    for key in rows[0]:
        values = [r[key] for r in rows if isinstance(r.get(key),(int,float)) and np.isfinite(r[key])]
        if not values:
            continue
        a = np.asarray(values)
        if bootstrap == 0:
            result[key] = {"mean":float(a.mean()),"n":len(a)}
            continue
        means = np.array([rng.choice(a,len(a),replace=True).mean() for _ in range(bootstrap)])
        result[key] = {"mean":float(a.mean()),"ci95":np.quantile(means,[.025,.975]).tolist(),"n":len(a)}
    return result


def paired_bootstrap(baseline, improved, seed=0, samples=2000):
    a,b = np.asarray(baseline),np.asarray(improved)
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a-b).all():
        raise ValueError("Paired observations must match and be finite")
    delta = b-a
    rng = np.random.default_rng(seed)
    means = [rng.choice(delta,len(delta),replace=True).mean() for _ in range(samples)]
    return {"mean_difference":float(delta.mean()),"ci95":np.quantile(means,[.025,.975]).tolist()}
