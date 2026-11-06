"""B补充：三种预先声明的全局尺度全部冻结、并列测试，不按测试结果选法。"""
from pathlib import Path
import argparse
import hashlib
import sys
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pipelines.part3 import common as c
from pipelines.part3_studies import common as s
from src.refinement.global_scale import image_log_ratio, apply_global_scale
from src.part3_studies.estimators import calibration_values, fit_rules, METHODS


def fit_values(development,mc):
    saved=c.read_csv(development/'predictions.csv')
    if len(saved)!=795 or len({r['sample_id'] for r in saved})!=795:
        raise ValueError('795 unique official development images required')
    if {p.stem for p in (development/'predictions').glob('*.npz')}!={r['sample_id'] for r in saved}:
        raise ValueError('Development array coverage differs')
    rows=[];ratios=[];weights=[]
    with c.dataset(mc,development=True) as data:
        if len(data)!=795:raise ValueError('Wrong development split')
        for i,row in enumerate(saved):
            sample=data.get_test_sample(i);key=c.development_stem(sample)
            path=development/'predictions'/(key+'.npz')
            if (row['sample_id']!=key or int(row['position'])!=i
                or int(row['matlab_index'])!=sample.matlab_index_1based
                or row['rgb_sha256']!=hashlib.sha256(sample.rgb.tobytes()).hexdigest()
                or row['prediction_sha256']!=c.sha(path)):
                raise ValueError('Development identity/SHA differs')
            z=s.read_depth(path);gt=sample.depth_m[c.CROP]
            p,g=calibration_values(z[c.CROP],gt)
            ratio=g/p
            ratios.append(ratio);weights.append((p/g)/p.size)
            rows.append(dict(sample_id=key,scene=sample.scene,matlab_index=sample.matlab_index_1based,
                             prediction_sha256=c.sha(path),valid_pixels=int(p.size),
                             median_log_ratio=image_log_ratio(z[c.CROP],gt),
                             mean_log_ratio=float(np.mean(np.log(ratio)))))
            if i%100==0:print(f'开发图比例 {i+1}/795',flush=True)
    print('求全开发像素加权中位数（内存需求较高）',flush=True)
    scales=fit_rules(rows,np.concatenate(ratios),np.concatenate(weights))
    return scales,rows


def fit(output,cfg,base,mc,model):
    development=ROOT/cfg['development_root']/model/'development_v1'
    reference=ROOT/cfg['development_root']/model/'frozen_scale_v1'
    parent=c.checked(development,'b_development');ref=c.checked(reference,'b_frozen_scale')
    if (parent['model']!=model or parent['config']!=base or ref['model']!=model or ref['config']!=base
        or Path(ref['parents'][0]).resolve()!=development.resolve()):
        raise ValueError('Development/reference model or config differs')
    inputs=c.resources(base,model,mc)
    record=s.begin(output,'study_b_fit',cfg,model,inputs,[reference],depth_type='metric',unit='meter',
                   alignment='none',native_metric=False,development_images=795,test_images_used_for_fitting=0)
    try:
        scales,rows=fit_values(development,mc)
        c.assert_close(scales['B_median_log'],c.read_json(reference/'frozen_scale.json')['scale'],'Original B scale')
        c.write_csv(output/'development_statistics.csv',rows)
        c.write_json(output/'frozen_scales.json',dict(scales=scales,development_images=795,
            test_images_used_for_fitting=0,frozen_at_utc=datetime.now(timezone.utc).isoformat(),
            objectives=dict(B_median_log='exp(median_images(median_valid_pixels(log(GT/pred))))',
                            B_mean_log='exp(mean_images(mean_valid_pixels(log(GT/pred))))',
                            B_absrel='weighted_median(GT/pred, weights=(pred/GT)/valid_pixels_per_image)'),
            fitting_predictions_clipped=False,testing_predictions_clipped=True,
            note='All rules retained; B_absrel minimizes UNCLIPPED image-macro AbsRel, not clipped test objective'))
        s.seal(output,record,True)
    except BaseException as exc:c.failed(output,exc);raise


def evaluate_rows(output,source,mc,scales,baseline,limit=None,verify=False):
    rows=[];keys=[]
    with c.dataset(mc) as data:
        for i in c.positions(len(data),limit):
            sample=data.get_test_sample(i);raw,digest=c.native(source,sample);key=c.stem(sample);keys.append(key)
            for method in ('RAW',*METHODS):
                z=raw if method=='RAW' else apply_global_scale(raw,scales[method])
                if method!='RAW':
                    path=output/'predictions'/method/(key+'.npz')
                    if verify:c.assert_close(s.read_depth(path),z,'Calibrated saved array/'+method)
                    else:
                        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,depth_m=z)
                metrics=c.score(z,sample.depth_m,boundary=False)
                if method=='RAW':c.raw_matches(metrics,baseline[key])
                rows.append(dict(sample_id=key,scene=sample.scene,method=method,
                                 source_prediction_sha256=digest,**metrics))
            if i%50==0:print(f'B三估计规则 {i+1}/654',flush=True)
    for method in METHODS:
        if {p.stem for p in (output/'predictions'/method).glob('*.npz')}!=set(keys):
            raise ValueError('Prediction coverage differs')
    return rows


def summary(rows,cfg):
    return dict(samples=len(rows)//4,means=c.means(rows,c.METRICS[:-1]),
                paired_scene_deltas=s.paired(rows,cfg['methods'],['abs_rel','rmse','delta1'],
                                            cfg['bootstrap_resamples'],cfg['bootstrap_seed']),
                comparison='All predeclared rules, not a test-selected winner; calibrated, not untouched native metric')


def evaluate(output,frozen,cfg,base,mc,model,limit):
    parent=s.checked(frozen,'study_b_fit')
    if parent['config']!=cfg or parent['model']!=model:raise ValueError('Frozen rules differ')
    scales=c.read_json(frozen/'frozen_scales.json')['scales']
    source,baseline,inputs=c.baseline_files(base,model,mc);inputs.update(c.resources(base,model,mc))
    record=s.begin(output,'study_b_evaluation',cfg,model,inputs,[frozen],limit=limit,scales=scales,
                   depth_type='metric',unit='meter',alignment='none',native_metric=False)
    try:
        rows=evaluate_rows(output,source,mc,scales,baseline,limit)
        c.write_csv(output/'per_image.csv',rows);c.write_json(output/'summary.json',summary(rows,cfg))
        s.seal(output,record,limit is None and len(rows)==2616)
    except BaseException as exc:c.failed(output,exc);raise


def verify(output,cfg,base,mc,model):
    record=s.checked(output,full=False)
    if record['config']!=cfg or record['model']!=model:raise ValueError('Run model/config differs')
    if record['kind']=='study_b_fit':
        development=ROOT/cfg['development_root']/model/'development_v1'
        scales,rows=fit_values(development,mc)
        frozen=c.read_json(output/'frozen_scales.json')
        if frozen['development_images']!=795 or frozen['test_images_used_for_fitting']!=0:
            raise ValueError('Invalid development/test disclosure')
        for method,value in scales.items():c.assert_close(value,frozen['scales'][method],'Refit scale/'+method)
        s.table_equal(rows,c.read_csv(output/'development_statistics.csv'))
        return dict(passed=True,development_images=795,all_three_scales_refitted=True,formal_complete=True)
    if record['kind']!='study_b_evaluation':raise ValueError('Not a B study')
    source=ROOT/base['models'][model]['baseline']
    baseline={r['sample_id']:r for r in c.read_csv(source/'metrics_per_image.csv')}
    scales=c.read_json(Path(record['parents'][0])/'frozen_scales.json')['scales']
    if scales!=record['scales']:raise ValueError('Recorded scales differ from frozen parent')
    rows=evaluate_rows(output,source,mc,scales,baseline,record['limit'],verify=True)
    s.table_equal(rows,c.read_csv(output/'per_image.csv'))
    if summary(rows,cfg)!=c.read_json(output/'summary.json'):raise ValueError('Summary differs')
    return dict(passed=True,samples=len(rows)//4,predictions_recomputed=3*len(rows)//4,
                formal_complete=record['formal_complete'],scope='Test arrays/GT/tables/statistics; fit refit has its own verify')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','fit','evaluate','verify'])
    p.add_argument('--model',required=True,choices=['depth_pro','unidepth_v2'])
    p.add_argument('--config',type=Path);p.add_argument('--output-dir',type=Path)
    p.add_argument('--frozen-dir',type=Path);p.add_argument('--run',type=Path)
    p.add_argument('--limit',type=int);p.add_argument('--report',type=Path)
    a=p.parse_args();cfg,base,mc=s.config('b',a.model,a.config)
    folder=ROOT/cfg['development_root']/a.model
    if a.action=='check':
        c.resources(base,a.model,mc);c.baseline_files(base,a.model,mc)
        c.checked(folder/'development_v1','b_development');c.checked(folder/'frozen_scale_v1','b_frozen_scale')
        print('795开发/原尺度/654基线与来源核对通过；无需推理');return
    if a.action=='verify':
        if not a.run:p.error('--run is required')
        c.report(a.run,verify(a.run.resolve(),cfg,base,mc,a.model),a.report);return
    if a.action=='fit':
        if a.limit is not None:p.error('795 development images required; --limit is evaluation-only')
        fit((a.output_dir or folder/cfg['fit_name']).resolve(),cfg,base,mc,a.model)
    else:evaluate((a.output_dir or folder/cfg['evaluation_name']).resolve(),
                  (a.frozen_dir or folder/cfg['fit_name']).resolve(),cfg,base,mc,a.model,a.limit)


if __name__=='__main__':main()
