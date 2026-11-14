"""BM/SGBM开发候选的公共评分；具体选择规则由coverage研究管理。"""
import numpy as np
from src.stereo.matching import StereoMatcherConfig, match_stereo
from src.stereo.geometry import middlebury_disparity_to_depth
from src.stereo.metrics import compute_disparity_metrics


def measure(parameters,method,scenes):
    values=dict(parameters);lr=values.pop('lr_threshold_px',1.)
    matcher=StereoMatcherConfig(method=method,**values);rows=[]
    for scene in scenes:
        result=match_stereo(scene.left_rgb,scene.right_rgb,
                            calibration_num_disparities=scene.calibration.num_disparities,
                            config=matcher,lr_threshold_px=lr)
        _, geometry_valid=middlebury_disparity_to_depth(result.left_disparity_px,
            focal_length_px=scene.calibration.focal_length_px,
            baseline_mm=scene.calibration.baseline_mm,doffs_px=scene.calibration.doffs_px)
        rows.append({'scene':scene.name,**compute_disparity_metrics(result.left_disparity_px,
            scene.gt_disparity_px,prediction_valid_mask=result.valid_mask & geometry_valid)})
    means={key:float(np.mean([r[key] for r in rows])) if all(r[key] is not None for r in rows) else None
           for key in ('disparity_mae_px','valid_prediction_fraction')}
    return means,rows

