"""Native metric model resource audit and development-only global calibration."""
import argparse,hashlib,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from src.data.nyuv2 import NYUv2LabeledDataset
from src.io import write_csv,write_json,sha256
from src.metrics.depth import compute_depth_metrics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['depth_pro','unidepth_v2'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    print(a.model,'loading official model',flush=True)
    if a.model=='depth_pro':
        from src.models.depth_pro_adapter import DepthProAdapter
        model=DepthProAdapter(repository=Path('third_party/ml-depth-pro'),checkpoint=Path('checkpoints/depth_pro/depth_pro.pt'))
        checkpoint=Path('checkpoints/depth_pro/depth_pro.pt')
    else:
        from src.models.unidepth_v2_adapter import UniDepthV2Adapter
        model=UniDepthV2Adapter(repository=Path('third_party/UniDepth'),checkpoint_dir=Path('checkpoints/unidepth_v2_vits14'))
        checkpoint=Path('checkpoints/unidepth_v2_vits14/model.safetensors')
    mat=Path('data/nyuv2/nyu_depth_v2_labeled.mat');split=Path('data/nyuv2/splits.mat')
    provenance=dict(model=a.model,checkpoint_sha256=sha256(checkpoint),split_sha256=sha256(split),mat_sha256=sha256(mat),source_sha256=sha256(__file__),precision='adapter official defaults',camera='RGB only, predicted camera',device='GPU-A')
    write_json(out/'protocol.json',dict(**provenance,calibration='median over training-image median log(GT/pred); one frozen scale; no test GT fit',resource='16 deterministic equally spaced test positions, external RGB scales .5/.75/1; one warmup, two measured repeats; restore depth to 480x640; same crop and metrics; not a full-test resource benchmark; external resizing does not imply changing fixed internal network size'))
    train=[]
    with NYUv2LabeledDataset(mat,split,split_variable='trainNdxs') as ds:
        assert len(ds)==795
        for i in range(len(ds)):
            s=ds.get_test_sample(i);z,_,seconds,mem=model.infer(s.rgb)
            z=cv2.resize(z,(640,480),interpolation=cv2.INTER_LINEAR)
            g=s.depth_m[45:471,41:601];d=z[45:471,41:601];valid=np.isfinite(g)&(g>.001)&(g<10)&np.isfinite(d)&(d>0)
            if not valid.any():raise ValueError('Empty training mask')
            train.append(dict(train_position=i,matlab_index=s.matlab_index_1based,log_ratio=float(np.median(np.log(g[valid]/d[valid]))),prediction_sha256=hashlib.sha256(z.tobytes()).hexdigest(),seconds=seconds,peak_bytes=mem))
            if i%50==0: print(a.model,'development',i,'/795',flush=True)
    scale=float(np.exp(np.median([r['log_ratio'] for r in train])))
    write_csv(out/'calibration_development.csv',train)
    write_json(out/'frozen_scale.json',dict(scale=scale,development_images=795,test_images_used_for_fitting=0,development_csv_sha256=sha256(out/'calibration_development.csv')))
    rows=[];resource=[]
    with NYUv2LabeledDataset(mat,split) as ds:
        positions=set(np.linspace(0,653,16,dtype=int).tolist())
        for i in range(len(ds)):
            s=ds.get_test_sample(i);key=f'test_{i:04d}_mat_{s.matlab_index_1based:04d}'
            file=Path('outputs/full_benchmark_20261029')/(a.model+'_nyuv2')/'predictions'/(key+'.npz')
            with np.load(file,allow_pickle=False) as f:raw=f['depth_m']
            if raw.shape!=(480,640):raw=cv2.resize(raw,(640,480),interpolation=cv2.INTER_LINEAR)
            for method,z in [('RAW',raw),('B_global',raw*scale)]:
                rows.append(dict(sample_id=key,method=method,source_sha256=sha256(file),**compute_depth_metrics(z[45:471,41:601],s.depth_m[45:471,41:601],min_depth=.001,max_depth=10.)))
            if i not in positions:continue
            for factor in (.5,.75,1.):
                rgb=cv2.resize(s.rgb,(int(640*factor),int(480*factor)),interpolation=cv2.INTER_AREA) if factor!=1 else s.rgb
                model.infer(rgb) # warmup for this input shape, excluded from timing
                for repeat in range(2):
                    z,_,seconds,mem=model.infer(rgb);z=cv2.resize(z,(640,480),interpolation=cv2.INTER_LINEAR)
                    resource.append(dict(sample_id=key,input_scale=factor,repeat=repeat,inference_seconds=seconds,peak_gpu_memory_bytes=mem,**compute_depth_metrics(z[45:471,41:601],s.depth_m[45:471,41:601],min_depth=.001,max_depth=10.)))
    write_csv(out/'calibration_test.csv',rows);write_csv(out/'resource.csv',resource)
    metrics=('abs_rel','rmse','delta1')
    cal={m:{k:float(np.mean([r[k] for r in rows if r['method']==m])) for k in metrics} for m in ('RAW','B_global')}
    res={str(f):{k:float(np.mean([r[k] for r in resource if r['input_scale']==f])) for k in (*metrics,'inference_seconds','peak_gpu_memory_bytes')} for f in (.5,.75,1.)}
    write_json(out/'summary.json',dict(status='complete',scale=scale,calibration=cal,resource=res,test_images=654,resource_images=16,limitations='B is a Part 3 improvement, not untouched Part 2; source resizing may leave model internal resolution fixed; timing excludes dataset IO/scoring and includes adapter preprocessing and synchronization; no claim of universally optimal scale or resolution'))
    print(a.model,'complete',flush=True)

if __name__=='__main__':main()
