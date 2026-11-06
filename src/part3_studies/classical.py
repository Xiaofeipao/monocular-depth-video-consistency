"""固定经典对照：log-depth Gaussian与He等的彩色RGB Guided Filter。"""
import cv2
import numpy as np

METHODS = ('Gaussian_log', 'RGB_guided_log')


def guided_log(depth, rgb, radius=2, epsilon=.01):
    """彩色局部线性模型；RGB[0,1]，精确box均值，无降采样或ximgproc依赖。

    参考He/Sun/Tang ECCV2010算法1及彩色引导式；引导目标为log-depth，
    指数变换保证正深度。输入GT、语义mask及另一模型均不参与处理。
    """
    depth = np.asarray(depth, np.float64)
    if radius < 1 or epsilon <= 0 or rgb.shape != (*depth.shape, 3):
        raise ValueError('Invalid guided-filter settings or RGB shape')
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError('Dense positive depth required; no hole filling')
    image = rgb.astype(np.float64) / 255
    p = np.log(depth)
    k = (2 * radius + 1,) * 2
    def mean(x): return cv2.boxFilter(x, -1, k, normalize=True, borderType=cv2.BORDER_REPLICATE)
    mi, mp = mean(image), mean(p)
    cov = np.empty((*depth.shape, 3, 3), np.float64)
    for i in range(3):
        for j in range(3):
            cov[..., i, j] = mean(image[..., i] * image[..., j]) - mi[..., i] * mi[..., j]
    cov += epsilon * np.eye(3)
    cross = mean(image * p[..., None]) - mi * mp[..., None]
    a = np.linalg.solve(cov, cross[..., None])[..., 0]
    b = mp - np.sum(a * mi, axis=-1)
    return np.exp(np.sum(mean(a) * image, axis=-1) + mean(b)).astype(np.float32)


def apply(method, depth, rgb, config):
    if method not in METHODS: raise ValueError('Unknown classical method')
    if method == 'RGB_guided_log':
        return guided_log(depth, rgb, **config['guided'])
    depth = np.asarray(depth, np.float64)
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError('Dense positive depth required')
    radius, sigma = config['gaussian']['radius'], config['gaussian']['sigma']
    if radius < 1 or sigma <= 0: raise ValueError('Invalid Gaussian parameters')
    return np.exp(cv2.GaussianBlur(np.log(depth), (2*radius+1,)*2, sigmaX=sigma,
                                 sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)).astype(np.float32)
