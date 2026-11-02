"""Rescore saved stereo predictions on all finite positive GT, without matching."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.stereo.io import load_middlebury_scene
from src.stereo.metrics import compute_disparity_metrics
from src.io import write_csv,write_json,sha256


def main():
    source=ROOT/'outputs/full_benchmark_20261029/middlebury'
    out=ROOT/'outputs/course_submission/stereo';out.mkdir(parents=True,exist_ok=True)
    rows=[];hashes=[]
    for path in sorted((ROOT/'data/middlebury/MiddEval3/trainingQ').iterdir()):
        if not (path/'disp0GT.pfm').exists(): continue
        scene=load_middlebury_scene(path)
        for method in ('bm_fixed','bm_tuned','sgbm_fixed','sgbm_tuned'):
            file=source/path.name/method/'prediction.npz'
            with np.load(file) as z: pred=np.where(z['valid_mask'],z['disparity_px'],np.nan)
            for label,mask in [('all_positive_GT',np.ones(pred.shape,bool)),('non_occluded',scene.non_occluded_mask)]:
                scores=compute_disparity_metrics(pred,scene.gt_disparity_px,evaluation_mask=mask)
                rows.append(dict(scene=path.name,method=method,mask=label,**scores))
            hashes.append(dict(path=file.relative_to(ROOT).as_posix(),sha256=sha256(file),GT_sha256=sha256(path/'disp0GT.pfm')))
    if len(rows)!=120: raise ValueError('Need 15 scenes x 4 methods x 2 masks')
    means={}
    for label in ('all_positive_GT','non_occluded'):
        means[label]={}
        for method in ('bm_fixed','bm_tuned','sgbm_fixed','sgbm_tuned'):
            r=[x for x in rows if x['method']==method and x['mask']==label]
            means[label][method]=dict(scenes=len(r),macro_disparity_mae_px=float(np.mean([x['disparity_mae_px'] for x in r])),
                macro_valid_coverage=float(np.mean([x['valid_coverage'] for x in r])),
                pooled_disparity_mae_px=sum(x['disparity_mae_px']*x['valid_prediction_pixels'] for x in r)/sum(x['valid_prediction_pixels'] for x in r),
                pooled_valid_coverage=sum(x['valid_prediction_pixels'] for x in r)/sum(x['evaluation_pixels'] for x in r))
    write_csv(out/'per_scene.csv',rows);write_csv(out/'source_checksums.csv',hashes)
    write_json(out/'summary.json',dict(protocol='Course full finite positive GT; original prediction validity incl. LR consistency; MAE on valid predictions, coverage denominator all GT; non-occluded retained separately',means=means,matching_rerun=False))
    print(json.dumps(means['all_positive_GT']),flush=True)


if __name__=='__main__':main()
