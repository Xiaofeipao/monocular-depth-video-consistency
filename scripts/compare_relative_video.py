"""Reproduce fixed-reference relative warp scores from saved video predictions."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.flow import FarnebackFlow,warp,consistency
from src.io import load_run,load_depth,read_bgr,sha256,write_json,write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dav2',required=True)
    p.add_argument('--vda',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    roots=[Path(a.dav2),Path(a.vda)]
    runs=[load_run(root) for root in roots]
    records=[run[1] for run in runs]
    if len(records[0])!=len(records[1]) or len(records[0])<2:
        raise ValueError('Paired runs required')
    frames=[]
    depths=[[],[]]
    for i,(r0,r1) in enumerate(zip(*records)):
        if (r0['source_index'],r0['id'])!=(r1['source_index'],r1['id']):
            raise ValueError('Frame identity mismatch')
        for j,r in enumerate((r0,r1)):
            for kind,ext,key in [('rgb','png','rgb_sha256'),('depth','npy','depth_sha256')]:
                if sha256(roots[j]/kind/f"{r['id']}.{ext}")!=r[key]:
                    raise ValueError('Saved prediction/RGB hash mismatch')
            depths[j].append(load_depth(roots[j]/'depth'/f"{r['id']}.npy"))
        rgb=[read_bgr(roots[j]/'rgb'/f"{r['id']}.png") for j,r in enumerate((r0,r1))]
        if not np.array_equal(*rgb): raise ValueError('RGB differs between methods')
        frames.append(rgb[0])
    spans=[float(np.percentile(d[0],95)-np.percentile(d[0],5)) for d in depths]
    if min(spans)<=0: raise ValueError('Degenerate reference frame')
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'flow_masks').mkdir()
    cv2.setNumThreads(1)
    rows=[]
    flow=FarnebackFlow()
    for i in range(1,len(frames)):
        forward,backward=flow(frames[i-1],frames[i])
        mask,error=consistency(forward,backward,1.)
        warped=[]
        for d in depths:
            old,inside=warp(d[i-1],backward)
            valid,_=warp(np.isfinite(d[i-1]).astype(np.float32),backward)
            mask &= inside & (valid>.999) & np.isfinite(old) & np.isfinite(d[i])
            warped.append(old)
        if not mask.any(): raise ValueError('Empty common mask')
        row=dict(previous_id=records[0][i-1]['id'],current_id=records[0][i]['id'],
                 previous_source_index=records[0][i-1]['source_index'],current_source_index=records[0][i]['source_index'],
                 valid_pixels=int(mask.sum()),coverage=float(mask.mean()))
        for name,d,old,span in zip(('DAV2','VDA'),depths,warped,spans):
            row[name]=float(np.mean(abs(old[mask]-d[i][mask]))/span)
        rows.append(row)
        np.savez_compressed(out/'flow_masks'/f'{i-1:06d}_{i:06d}.npz',forward=forward,backward=backward,common_mask=mask,fb_error=error)
    report=dict(dataset='relative_video_demo, 32 frames only, no depth GT',
        metric='raw relative warp absolute error divided by fixed frame0 P95-P5; common FB mask, zero allowed; no per-frame alignment',
        scores={name:dict(reference_span=span,mean=float(np.mean([r[name] for r in rows]))) for name,span in zip(('DAV2','VDA'),spans)},
        not_metric_accuracy=True)
    write_json(out/'relative_comparison.json',report)
    write_csv(out/'per_pair.csv',rows)
    write_json(out/'settings.json',dict(reference_frame_id=records[0][0]['id'],reference_source_index=records[0][0]['source_index'],
        percentiles=[5,95],percentile_method='numpy default linear; all frame-0 pixels',
        flow='OpenCV Farneback on saved BGR converted to grayscale',
        farneback=dict(pyr_scale=.5,levels=3,winsize=21,iterations=3,poly_n=5,poly_sigma=1.2,flags=0),
        warp='cv2.remap INTER_LINEAR, BORDER_CONSTANT=0; current-to-previous backward flow',
        mask='common forward/backward Euclidean error <=1px, inside source bounds, finite current and fully finite interpolated previous for both models; zero allowed',
        aggregation='equal-weight arithmetic mean of 31 per-pair mean absolute errors, divided by model-specific fixed reference span',
        alignment='none; no inversion, no scale/shift fitting, no GT',opencv=cv2.__version__,numpy=np.__version__))
    print(report)


if __name__=='__main__': main()
