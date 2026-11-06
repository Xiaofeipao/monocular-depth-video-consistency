"""经典双目的开发集误差—覆盖率选择与分区域统计。"""
from __future__ import annotations

import cv2
import numpy as np

from src.stereo.metrics import compute_disparity_metrics

LABELS = ('fixed', 'matched_coverage', 'matched_error')
REGIONS = ('all_GT', 'non_occluded', 'occluded', 'low_texture', 'depth_boundary')


def select_candidates(rows, tolerance=0.01):
    """约束只针对开发集；候选顺序固定，用于最终平局。"""
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError('Invalid coverage tolerance')
    ids = [r['candidate'] for r in rows]
    if len(ids) != len(set(ids)) or 0 not in ids:
        raise ValueError('Expected unique candidates and fixed reference 0')
    for r in rows:
        if (not np.isfinite(r['coverage']) or not 0 <= r['coverage'] <= 1
                or (r['mae'] is not None and (not np.isfinite(r['mae']) or r['mae'] < 0))):
            raise ValueError('Non-finite or invalid development score')
    base = next(r for r in rows if r['candidate'] == 0)
    if base['mae'] is None:
        raise ValueError('Fixed reference is undefined in a development scene')
    defined = [r for r in rows if r['mae'] is not None]
    near = [r for r in defined if abs(r['coverage'] - base['coverage']) <= tolerance]
    accurate = [r for r in defined if r['mae'] <= base['mae']]
    return dict(fixed=base,
                matched_coverage=min(near, key=lambda r: (r['mae'], -r['coverage'], r['candidate'])),
                matched_error=min(accurate, key=lambda r: (-r['coverage'], r['mae'], r['candidate'])))


def pareto_frontier(rows):
    defined = [r for r in rows if r['mae'] is not None]
    return [r for r in defined if not any(
        q['mae'] <= r['mae'] and q['coverage'] >= r['coverage']
        and (q['mae'] < r['mae'] or q['coverage'] > r['coverage']) for q in defined)]


def regional_masks(scene, settings):
    gray = cv2.cvtColor(scene.left_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    window = settings['texture_window']
    variance = cv2.boxFilter(gray * gray, -1, (window, window)) - cv2.boxFilter(gray, -1, (window, window)) ** 2
    gt = scene.gt_disparity_px
    safe = np.where(np.isfinite(gt) & (gt > 0), gt, 0)
    strength = np.abs(cv2.Sobel(safe, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(safe, cv2.CV_32F, 0, 1))
    return dict(all_GT=np.ones(gt.shape, bool), non_occluded=scene.non_occluded_mask,
                occluded=scene.occluded_mask, low_texture=variance < settings['texture_variance_threshold'],
                depth_boundary=strength > settings['boundary_sobel_threshold_px'])


def regional_metrics(prediction, target, mask, valid):
    if not (np.asarray(mask, bool) & np.isfinite(target) & (target > 0)).any():
        return dict(mae=None, coverage=None, evaluation_pixels=0, valid_prediction_pixels=0,
                    invalid_prediction_pixels=0, bad1=None, bad2=None, bad4=None)
    metrics = dict(compute_disparity_metrics(prediction, target, evaluation_mask=mask,
                                           prediction_valid_mask=valid))
    metrics['mae'] = metrics.pop('disparity_mae_px')
    metrics['coverage'] = metrics.pop('valid_prediction_fraction')
    return metrics


def summarize(rows):
    result = []
    for method in ('bm', 'sgbm'):
        for label in LABELS:
            for region in REGIONS:
                group = [r for r in rows if (r['method'], r['selection'], r['region']) == (method, label, region)]
                supported = [r for r in group if r['evaluation_pixels'] > 0]
                record = dict(method=method, selection=label, region=region, scenes=len(group),
                              supported_scenes=len(supported))
                for key in ('mae', 'coverage', 'bad1', 'bad2', 'bad4'):
                    values = [r[key] for r in supported if r[key] is not None]
                    record[key] = float(np.mean(values)) if values and len(values) == len(supported) else None
                    record[key + '_defined_scenes'] = len(values)
                for key in ('evaluation_pixels', 'valid_prediction_pixels', 'invalid_prediction_pixels'):
                    record[key] = sum(r[key] for r in group)
                result.append(record)
    return result
