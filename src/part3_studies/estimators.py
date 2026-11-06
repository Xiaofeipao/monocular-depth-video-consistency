"""三个开发集全局尺度：图像等权，绝不读取测试GT拟合。"""
import numpy as np

METHODS = ('B_median_log', 'B_mean_log', 'B_absrel')


def calibration_values(prediction, target):
    # 与冻结协议一致：范围比较保留原GT dtype，浮点运算改用float64。
    original = np.asarray(target)
    valid = np.isfinite(original) & (original > np.asarray(.001, dtype=original.dtype)) & (original < 10)
    p, g = np.asarray(prediction, np.float64)[valid], np.asarray(target, np.float64)[valid]
    if not p.size or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError('Empty GT support or invalid development prediction')
    return p, g


def weighted_median(values, weights):
    """最小化sum(w*abs(s-value))，平局选较小端点；无需测试或候选搜索。"""
    values, weights = np.asarray(values, np.float64), np.asarray(weights, np.float64)
    if (values.ndim != 1 or values.shape != weights.shape or not values.size
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(values <= 0) or np.any(weights <= 0)):
        raise ValueError('Positive finite nonempty values/weights required')
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2, side='left'))
    return float(values[order[index]])


def fit_rules(image_stats, ratios, weights, expected_images=795):
    if len(image_stats) != expected_images or expected_images < 1:
        raise ValueError('All official development images are required')
    median_logs = np.array([float(r['median_log_ratio']) for r in image_stats])
    mean_logs = np.array([float(r['mean_log_ratio']) for r in image_stats])
    if not np.isfinite(median_logs).all() or not np.isfinite(mean_logs).all():
        raise ValueError('Nonfinite image log ratios')
    result = dict(zip(METHODS, (float(np.exp(np.median(median_logs))),
                                float(np.exp(np.mean(mean_logs))), weighted_median(ratios, weights))))
    if not all(np.isfinite(s) and s > 0 for s in result.values()):
        raise ValueError('Invalid fitted scale')
    return result
