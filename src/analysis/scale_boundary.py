"""Native metric scale, depth/object edges and GT-supported normal profiles.

No routine changes a prediction. Oracle/own-plateau normalizations are diagnostic.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_cdt, distance_transform_edt
from src.metrics.boundary import depth_edge_map

METRICS = ('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'delta1', 'delta2', 'delta3')


def error_statistics(prediction, target, mask):
    """N/A for empty support; units are explicit, squared sums enable accounting."""
    p, g = np.asarray(prediction, np.float64)[mask], np.asarray(target, np.float64)[mask]
    if not p.size:
        return dict(pixel_count=0, **{k: None for k in (
            'abs_rel', 'mae_m', 'rmse_m', 'bias_m', 'relative_bias',
            'large_relative_error_fraction', 'squared_error_sum', 'absolute_relative_error_sum')})
    d = p-g
    return dict(pixel_count=int(p.size), abs_rel=float(np.mean(np.abs(d)/g)),
                mae_m=float(np.mean(np.abs(d))), rmse_m=float(np.sqrt(np.mean(d*d))),
                bias_m=float(np.mean(d)), relative_bias=float(np.mean(d/g)),
                large_relative_error_fraction=float(np.mean(np.abs(d)/g > .25)),
                squared_error_sum=float(np.sum(d*d)),
                absolute_relative_error_sum=float(np.sum(np.abs(d)/g)))


def scale_statistics(prediction, target, valid, minimum=.001, maximum=10.):
    p, g = np.asarray(prediction, np.float64)[valid], np.asarray(target, np.float64)[valid]
    if not p.size or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError('Native predictions must be finite positive values on the fixed GT domain')
    clipped = np.clip(p, minimum, maximum)
    ratio = p/g
    log_error = np.log(ratio)
    b = float(np.mean(log_error))
    mse = float(np.mean(log_error**2))
    residual = float(np.mean((log_error-b)**2))
    raw_scale = float(np.median(ratio))
    result = dict(valid_pixels=int(p.size), raw_scale_ratio=raw_scale,
                  clipped_scale_ratio=float(np.median(clipped/g)),
                  raw_abs_log_scale=float(abs(np.log(raw_scale))),
                  raw_ratio_p05=float(np.quantile(ratio, .05)),
                  raw_ratio_p95=float(np.quantile(ratio, .95)),
                  raw_min_m=float(p.min()), raw_max_m=float(p.max()),
                  raw_below_minimum_fraction=float(np.mean(p < minimum)),
                  raw_above_maximum_fraction=float(np.mean(p > maximum)),
                  raw_out_of_range_fraction=float(np.mean((p < minimum) | (p > maximum))),
                  log_bias=b, log_bias_squared=b*b, log_mse=mse,
                  centered_log_mse=residual, log_decomposition_residual=float(mse-b*b-residual))
    tracks = {'native_clipped': np.clip(prediction, minimum, maximum),
              'native_raw': np.asarray(prediction, np.float64),
              'oracle_raw_median_clipped': np.clip(np.asarray(prediction, np.float64)/raw_scale, minimum, maximum)}
    for name, depth in tracks.items():
        result.update({f'{name}_{k}': v for k, v in error_statistics(depth, target, valid).items()})
    return result, tracks


def annotation_edges(labels, instances, valid):
    """Combine class AND instance IDs; only label=0 means unlabelled.

    Instance numbers may repeat across classes, and zero is not discarded solely
    because it is an instance number. Adjacent labelled pixels must both be valid.
    """
    labels, instances = np.asarray(labels), np.asarray(instances)
    if labels.shape != instances.shape or labels.shape != valid.shape:
        raise ValueError('Annotation shapes must match the depth crop')
    labelled = valid & (labels > 0)
    edges = np.zeros(valid.shape, bool)
    for axis in (0, 1):
        a, b = [slice(None)]*2, [slice(None)]*2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        different = (labels[a] != labels[b]) | (instances[a] != instances[b])
        hit = labelled[a] & labelled[b] & different
        edges[a] |= hit
        edges[b] |= hit
    return edges


def chebyshev_distance(edges):
    if not np.any(edges):
        return np.full(edges.shape, np.inf)
    return distance_transform_cdt(~edges, metric='chessboard').astype(np.float64)


def edge_comparison(pred_edges, gt_edges, valid, object_edges, tolerances, distance_cap):
    """Match with square/Chebyshev tolerance; distance diagnostics use Euclidean px."""
    p = np.asarray(pred_edges, bool) & valid
    g = np.asarray(gt_edges, bool) & valid
    obj = np.asarray(object_edges, bool) & g
    npred, ngt, nobj = int(p.sum()), int(g.sum()), int(obj.sum())
    pd, gd = chebyshev_distance(p), chebyshev_distance(g)
    rows = []
    for tolerance in tolerances:
        pmatch = int(np.count_nonzero(p & (gd <= tolerance)))
        gmatch = int(np.count_nonzero(g & (pd <= tolerance)))
        if npred == 0 and ngt == 0:
            precision = recall = f1 = 1.
        elif npred == 0 or ngt == 0:
            precision = recall = f1 = 0.
        else:
            precision, recall = pmatch/npred, gmatch/ngt
            f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
        rows.append(dict(tolerance_px=tolerance, precision=precision, recall=recall, f1=f1,
                         prediction_edge_pixels=npred, gt_edge_pixels=ngt,
                         matched_prediction_pixels=pmatch, matched_gt_pixels=gmatch,
                         object_edge_pixels=nobj,
                         object_matched_pixels=int(np.count_nonzero(obj & (pd <= tolerance))),
                         object_edge_recall=float(np.mean(pd[obj] <= tolerance)) if nobj else None))
    # No-edge cases are retained as capped misses, never silently dropped.
    p_dist = distance_transform_edt(~g)[p] if ngt else np.full(npred, distance_cap)
    g_dist = distance_transform_edt(~p)[g] if npred else np.full(ngt, distance_cap)
    distances = dict(
        prediction_edge_distance_capped_px=float(np.mean(np.minimum(p_dist, distance_cap))) if npred else None,
        gt_edge_distance_capped_px=float(np.mean(np.minimum(g_dist, distance_cap))) if ngt else None,
        gt_edge_distance_p95_capped_px=float(np.quantile(np.minimum(g_dist, distance_cap), .95)) if ngt else None,
        gt_edge_miss_beyond_3px_fraction=float(np.mean(pd[g] > 3)) if ngt else None,
        prediction_edge_spurious_beyond_3px_fraction=float(np.mean(gd[p] > 3)) if npred else None)
    return rows, distances


def region_masks(gt_edges, object_edges, valid):
    distance = chebyshev_distance(gt_edges)
    obj_distance = chebyshev_distance(object_edges)
    return {
        'edge_0_1px': valid & (distance <= 1),
        'edge_1_3px': valid & (distance > 1) & (distance <= 3),
        'edge_3_5px': valid & (distance > 3) & (distance <= 5),
        'interior_gt5px': valid & (distance > 5),
        'edge_le3px': valid & (distance <= 3),
        'object_edge_le3px': valid & (obj_distance <= 3),
    }


def crossings(values, level):
    values = np.asarray(values, np.float64)
    indices = np.where((values[:-1] < level) & (values[1:] >= level))[0]
    return [float(i + (level-values[i])/(values[i+1]-values[i])) for i in indices]


def profile_shape(values, plateau_samples, minimum_ratio):
    """Own-side plateau normalization is a shape diagnostic, not metric alignment.

    Unique rising 10/50/90 crossings are required; oscillation/weak contrast
    gives an explicit failure and N/A geometry rather than an optimistic width.
    """
    values = np.asarray(values, np.float64)
    left, right = float(np.median(values[:plateau_samples])), float(np.median(values[-plateau_samples:]))
    contrast = abs(np.log(right/left))
    out = dict(left_m=left, right_m=right, log_contrast=contrast,
               midpoint_index=None, width_10_90_px=None, status='weak_contrast',
               crossing_50_count=0, overshoot_fraction=None)
    if contrast < np.log(minimum_ratio):
        return out
    normalized = (values-left)/(right-left)
    xs = {level: crossings(normalized, level) for level in (.1, .5, .9)}
    out['crossing_50_count'] = len(xs[.5])
    out['overshoot_fraction'] = float(np.mean((normalized < -.1) | (normalized > 1.1)))
    if any(len(v) != 1 for v in xs.values()):
        out['status'] = 'ambiguous_crossings'
        return out
    if not xs[.1][0] <= xs[.5][0] <= xs[.9][0]:
        out['status'] = 'unordered_crossings'
        return out
    out.update(status='measurable', midpoint_index=xs[.5][0], width_10_90_px=xs[.9][0]-xs[.1][0])
    return out


def choose_profiles(target, valid, object_support, config):
    """Choose only from GT, with fixed contrast, side stability and separation.

    The sampling lines are horizontal/vertical cross-sections, not a claim of
    exact continuous geometric normals on arbitrary curved boundaries.
    """
    radius, side = config['radius_px'], config['plateau_samples']
    log_gt = np.log(np.maximum(target, .001)).astype(np.float64)
    candidates = []
    for axis in (1, 0):
        # Broad GT jump proposes a cross-section; exact endpoint stability is checked below.
        jump = np.abs(np.diff(log_gt, axis=axis))
        ys, xs = np.where(jump >= np.log(config['minimum_gt_ratio']))
        scores = jump[ys, xs]
        order = np.argsort(-scores, kind='stable')
        kept = 0
        for index in order:
            y, x = int(ys[index]), int(xs[index])
            if not object_support[y, x]:
                continue
            coordinate = x if axis == 1 else y
            length = target.shape[axis]
            if coordinate-radius < 0 or coordinate+radius >= length:
                continue
            sl = (y, slice(x-radius, x+radius+1)) if axis == 1 else (slice(y-radius, y+radius+1), x)
            if not valid[sl].all():
                continue
            line = log_gt[sl]
            if max(float(np.std(line[:side])), float(np.std(line[-side:]))) > config['maximum_gt_log_std']:
                continue
            shape = profile_shape(target[sl], side, config['minimum_gt_ratio'])
            if shape['status'] != 'measurable' or shape['width_10_90_px'] > config['maximum_gt_width_px']:
                continue
            # Reject oscillatory GT cross-sections even when the net contrast is large.
            v = (line-line[0])/(line[-1]-line[0]) if line[-1] != line[0] else np.zeros_like(line)
            if float(np.sum(np.maximum(-np.diff(v), 0))) > config['maximum_gt_backtracking']:
                continue
            if any(np.hypot(y-p['y'], x-p['x']) < config['separation_px'] for p in candidates):
                continue
            candidates.append(dict(y=y, x=x, axis='x' if axis == 1 else 'y', radius_px=radius,
                                   gt_midpoint_index=shape['midpoint_index'],
                                   gt_width_10_90_px=shape['width_10_90_px'],
                                   gt_log_contrast=shape['log_contrast']))
            kept += 1
            if kept >= config['maximum_per_image']:
                break
    # Rank GT support deterministically, independently of either model.
    candidates.sort(key=lambda p: (-p['gt_log_contrast'], p['y'], p['x'], p['axis']))
    return candidates[:config['maximum_per_image']]


def extract_profile(array, profile):
    y, x, r = profile['y'], profile['x'], profile['radius_px']
    return np.asarray(array[y, x-r:x+r+1] if profile['axis']=='x' else array[y-r:y+r+1, x], np.float64)


def measure_profile(prediction, target, profile, config):
    pred = extract_profile(prediction, profile)
    gt = extract_profile(target, profile)
    result = profile_shape(pred, config['plateau_samples'], config['minimum_prediction_ratio'])
    left, right = np.median(gt[:config['plateau_samples']]), np.median(gt[-config['plateau_samples']:])
    result['correct_depth_order'] = bool((right-left)*(result['right_m']-result['left_m']) > 0)
    if not result['correct_depth_order'] and result['log_contrast'] >= np.log(config['minimum_prediction_ratio']):
        result.update(status='wrong_depth_order', midpoint_index=None, width_10_90_px=None)
    result['signed_midpoint_error_px'] = (result['midpoint_index']-profile['gt_midpoint_index']) if result['midpoint_index'] is not None else None
    result['absolute_midpoint_error_px'] = abs(result['signed_midpoint_error_px']) if result['signed_midpoint_error_px'] is not None else None
    result['width_excess_px'] = result['width_10_90_px']-profile['gt_width_10_90_px'] if result['width_10_90_px'] is not None else None
    result['contrast_ratio_to_gt'] = result['log_contrast']/profile['gt_log_contrast']
    result['profile_mae_m'] = float(np.mean(np.abs(pred-gt)))
    return result
