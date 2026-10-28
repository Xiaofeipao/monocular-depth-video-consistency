"""Metric and GT-pose-compensated temporal evaluation for TUM RGB-D manifests."""
import argparse
import json
from pathlib import Path

from .evaluate import evaluate_run
from .io import load_run,load_depth,write_csv,write_json
from .metrics.pose import pose_temporal_metrics
from .metrics import evaluation_mask


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--manifest',required=True)
    a=p.parse_args(argv)
    run,manifest=Path(a.run),Path(a.manifest)
    data=json.loads(manifest.read_text(encoding='utf-8'))
    meta,records=load_run(run)
    if meta['depth_type']!='metric': p.error('GT-pose metric evaluation requires native metric depth')
    # Also checks exact input RGB, sample IDs and frozen depth hashes.
    evaluate_run(run,manifest,bootstrap=0)
    samples={s['id']:s for s in data['samples']}
    intrinsics=json.loads((manifest.parent/data['intrinsics']).read_text(encoding='utf-8'))
    rows=[]
    def ground_truth(sample):
        gt=load_depth(manifest.parent/sample['depth'],sample['depth_scale'])
        mask=load_depth(manifest.parent/sample['mask'])>0 if sample.get('mask') else None
        protocol=data['protocol']
        valid=evaluation_mask(gt,protocol['min_depth'],protocol['max_depth'],mask,protocol.get('crop'),protocol.get('depth_bounds','inclusive'))
        import numpy as np
        return np.where(valid,gt,np.nan)
    for before,after in zip(records,records[1:]):
        s0,s1=samples[before['sample_id']],samples[after['sample_id']]
        if s1['timestamp']<=s0['timestamp']: raise ValueError('Video timestamps are not strictly increasing')
        result=pose_temporal_metrics(load_depth(run/'depth'/f"{before['id']}.npy"),load_depth(run/'depth'/f"{after['id']}.npy"),
            ground_truth(s0),ground_truth(s1),
            s0['pose_world_camera'],s1['pose_world_camera'],intrinsics)
        rows.append({'sample_id':s1['id'],'delta_seconds':s1['timestamp']-s0['timestamp'],**result})
    directory=run/'evaluation'/'rgbd_pose'
    directory.mkdir(parents=True,exist_ok=False)
    write_csv(directory/'per_pair.csv',rows)
    write_json(directory/'summary.json',{'pose_convention':data['pose_convention'],
        'visibility':'GT-defined rigid reprojection, z-buffer, 3cm+3% depth tolerance; not optical-flow proxy',
        'pairs':len(rows),'metric_means':{key:sum(r[key] for r in rows if r[key] is not None)/sum(r[key] is not None for r in rows)
            for key in ('pose_temporal_log_error','pose_temporal_MAE_m','gt_reprojection_MAE_m','coverage') if any(r[key] is not None for r in rows)}})


if __name__=='__main__': main()
