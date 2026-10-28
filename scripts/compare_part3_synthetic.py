"""CPU-only controlled synthetic diagnostics for preserved Part 3 directions.

No learned model is run. This cannot establish a best direction on real videos.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.exploration.compare import compare
from src.io import RunWriter,write_json,write_csv


def make_case(root, name, seed):
    rng=np.random.default_rng(seed)
    root.mkdir(parents=True,exist_ok=False)
    h,w=64,96
    y,x=np.indices((h,w))
    background=(3.+x/w*.7).astype(np.float32)
    texture=rng.integers(60,170,(h,w,3),dtype=np.uint8)
    texture=cv2.GaussianBlur(texture,(5,5),0)
    bias=1.25 if name=='scale_bias' else 1.
    metadata=dict(model='SYNTHETIC_NO_BACKBONE',synthetic=True,depth_type='metric',unit='meter',
                  representation='depth',fps=10.,visualization_range=[.5,5.])
    raw=RunWriter(root/'baseline',metadata,{'seed':seed,'scenario':name,'bias':bias})
    # Separate calibration frames: external exact simulated tape measurements,
    # never points on the test frames. Noise and sample IDs are independent.
    calibration=RunWriter(root/'calibration',metadata)
    points=[]
    for i,z in enumerate((1.,2.,3.)):
        pred=np.full((h,w),z*bias,np.float32)*np.exp(rng.normal(0,.015,(h,w))).astype(np.float32)
        calibration.add(pred,texture,'synthetic_calibration',i,{'sample_id':f'{name}_dev{i}'})
        points.append(dict(sample_id=f'{name}_dev{i}',x=12,y=12,depth_m=z))
    calibration.finish({'synthetic':True})
    write_json(root/'calibration.json',{'source':'independent_measured_landmarks','run':'calibration','points':points})
    samples=[]
    foreground_distances=[1.70,1.60,1.52,1.48,1.45,1.40,1.43,1.49,1.53,1.58,1.70,1.80]
    for i in range(12):
        gt=background.copy()
        rgb=texture.copy()
        left=24+(i*2 if name=='moving_occlusion' else 0)
        distance=foreground_distances[i] if name=='threshold_jitter' else 1.2
        gt[18:48,left:left+24]=distance
        rgb[18:48,left:left+24]=(45,170,70)
        if name=='scene_cut' and i>=6:
            rgb=255-rgb
            gt=gt+1.5
        sigma=.08 if name in ('static_noise','scale_bias') else .045
        correlated=rng.normal(0,.035) if name=='threshold_jitter' else rng.normal(0,.008)
        prediction=gt*bias*np.exp(rng.normal(0,sigma,gt.shape)+correlated).astype(np.float32)
        key=f'{name}_test{i}'
        raw.add(prediction,rgb,'synthetic_test',i,{'sample_id':key,'focal_px':90.})
        np.save(root/f'gt_{i}.npy',gt)
        samples.append(dict(id=key,rgb=f'baseline/rgb/{i:06d}.png',depth=f'gt_{i}.npy'))
    raw.finish({'synthetic':True,'frames':12})
    write_json(root/'test_manifest.json',{'dataset':f'SYNTHETIC_{name}; no real model/data',
        'protocol':{'min_depth':.1,'max_depth':10.,'edge_threshold':.08,'edge_tolerance':2},'samples':samples})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/part3/directions_synthetic')
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args(argv)
    root=Path(a.output)
    root.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    reports,rows={},[]
    cases=('static_noise','scale_bias','moving_occlusion','threshold_jitter','scene_cut')
    for i,name in enumerate(cases):
        case=root/name
        make_case(case,name,20261021+i)
        report=compare(case/'baseline',case/'comparison',case/'test_manifest.json',case/'calibration.json',
                       repeats=a.repeats,roi=[8,56,8,88])
        reports[name]=report['winners']
        rows.extend({'scenario':name,**row} for row in report['methods'])
        print(f'Completed synthetic scenario: {name}',flush=True)
    write_csv(root/'all_scenarios.csv',rows)
    write_json(root/'summary.json',{'synthetic':True,'seed_base':20261021,'cases':reports,
        'claim':'Software/algorithm diagnostics only; real Part 2 predictions unavailable.',
        'comparison':'Same test frames, masks and full-resolution evaluation flow within each case.',
        'timing':'CPU postprocessing + warning compute; single OpenCV thread; one warmup and repeated full passes.',
        'calibration':'B uses separate generated calibration frames and simulated measured distances.',
        'not_run':['real pretrained Part 2 inference: weights/sources/data absent','GPU inference: deferred by user']})
    print(f'Results: {root.resolve()}',flush=True)


if __name__=='__main__':
    main()
