"""仅由开发图拟合一个冻结全局尺度，不按测试图对齐。"""
import numpy as np


def image_log_ratio(prediction, target):
    prediction = np.asarray(prediction, np.float32)
    target = np.asarray(target, np.float32)
    if prediction.shape != target.shape:
        raise ValueError('Prediction/GT grids differ')
    valid = np.isfinite(target) & (target > .001) & (target < 10.) & np.isfinite(prediction) & (prediction > 0)
    if not valid.any():
        raise ValueError('No valid development pixels')
    return float(np.median(np.log(target[valid] / prediction[valid])))


def fit_global_scale(log_ratios, *, expected_images=795):
    values = np.asarray(log_ratios, np.float64)
    if values.ndim != 1 or len(values) != expected_images or not np.isfinite(values).all():
        raise ValueError('Require the complete finite development split')
    scale = float(np.exp(np.median(values)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid frozen scale')
    return scale


def apply_global_scale(prediction, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid frozen scale')
    return np.asarray(prediction, np.float32) * scale
