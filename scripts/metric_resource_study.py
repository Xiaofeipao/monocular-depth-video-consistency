"""Resource-only 16-image audit; no scale fitting or test-set parameter selection."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from src.data.nyuv2 import NYUv2LabeledDataset
from src.metrics.depth import compute_depth_metrics
from src.io import write_csv,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['depth_pro','unidepth_v2'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    if a.model=='depth_pro':
        from src.models.depth_pro_adapter import DepthProAdapter
        model=DepthProAdapter(repository=Path('third_party/ml-depth-pro'),checkpoint=Path('checkpoints/depth_pro/depth_pro.pt'))
    else:
        from src.models.unidepth_v2_adapter import UniDepthV2Adapter
        model=UniDepthV2Adapter(repository=Path('third_party/UniDepth'),checkpoint_dir=Path('checkpoints/unidepth_v2_vits14'))
    rows=[]
    with NYUv2LabeledDataset(Path('data/nyuv2/nyu_depth_v2_labeled.mat'),Path('data/nyuv2/splits.mat')) as ds:
        for i in np.linspace(0,653,16,dtype=int):
            s=ds.get_test_sample(int(i));key=f'test_{i:04d}_mat_{s.matlab_index_1based:04d}'
            for factor in (.5,.75,1.):
                rgb=cv2.resize(s.rgb,(int(640*factor),int(480*factor)),interpolation=cv2.INTER_AREA) if factor!=1 else s.rgb
                model.infer(rgb)
                for repeat in range(2):
                    z,_,seconds,mem=model.infer(rgb);z=cv2.resize(z,(640,480),interpolation=cv2.INTER_LINEAR)
                    rows.append(dict(sample_id=key,input_scale=factor,repeat=repeat,inference_seconds=seconds,peak_gpu_memory_bytes=mem,**compute_depth_metrics(z[45:471,41:601],s.depth_m[45:471,41:601],min_depth=.001,max_depth=10.)))
    write_csv(out/'resource.csv',rows)
    write_json(out/'summary.json',{str(f):{k:float(np.mean([r[k] for r in rows if r['input_scale']==f])) for k in ('abs_rel','rmse','delta1','inference_seconds','peak_gpu_memory_bytes')} for f in (.5,.75,1.)})

if __name__=='__main__':main()
