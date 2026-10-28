"""Direction B: freeze a robust scale using independent measured landmarks."""
import numpy as np


def fit_scale(predicted_m, measured_m, minimum_points=3, bounds=(.25,4.)):
    predicted, measured = np.asarray(predicted_m,float),np.asarray(measured_m,float)
    if predicted.shape != measured.shape or predicted.ndim != 1:
        raise ValueError('Calibration observations must be paired vectors')
    valid = np.isfinite(predicted)&np.isfinite(measured)&(predicted>0)&(measured>0)
    if valid.sum() < minimum_points:
        raise ValueError('At least three valid independent calibration observations required')
    ratios = np.log(measured[valid]/predicted[valid])
    center = np.median(ratios)
    mad = np.median(np.abs(ratios-center))
    keep = np.abs(ratios-center) <= max(3*1.4826*mad,.03)
    if keep.sum() < minimum_points:
        raise ValueError('Insufficient inliers after robust calibration')
    scale = float(np.exp(np.median(ratios[keep])))
    if not bounds[0] <= scale <= bounds[1]:
        raise ValueError('Measured calibration scale outside configured bounds')
    return {'scale':scale,'observations':int(valid.sum()),'inliers':int(keep.sum()),
            'median_log_residual':float(np.median(np.abs(ratios[keep]-np.log(scale)))),
            'source':'independent measured landmarks; never test-GT alignment'}


def apply_scale(depth, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Scale must be positive and finite')
    return np.asarray(depth,np.float32)*scale
