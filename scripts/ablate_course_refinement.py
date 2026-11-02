"""Fixed C component removal on saved 654-image NYU predictions, no tuning."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'third_party/ml-depth-pro/src'))
from src.data.nyuv2 import NYUv2LabeledDataset
from src.exploration.edge_aware import CUDAEdgeRefiner
from src.metrics.depth import compute_depth_metrics
from src.metrics.boundary import compute_depth_pro_si_boundary_f1
from src.io import write_csv,write_json,sha256


def main():
    import torch;torch.set_num_threads(4)
    out=ROOT/'outputs/course_submission/C_ablation';out.mkdir(parents=True,exist_ok=True)
    methods={'C_no_RGB':CUDAEdgeRefiner(use_rgb_guidance=False),'C_no_depth_gate':CUDAEdgeRefiner(use_depth_guidance=False)}
    rows=[]
    with NYUv2LabeledDataset(ROOT/'data/nyuv2/nyu_depth_v2_labeled.mat',ROOT/'data/nyuv2/splits.mat') as data:
        for i in range(len(data)):
            s=data.get_test_sample(i);key=f'test_{i:04d}_mat_{s.matlab_index_1based:04d}'
            path=ROOT/'outputs/full_benchmark_20261029/depth_pro_nyuv2/predictions'/f'{key}.npz'
            with np.load(path) as f: raw=f['depth_m']
            for name,refine in methods.items():
                z=refine(raw,s.rgb)[45:471,41:601];gt=s.depth_m[45:471,41:601]
                m=dict(compute_depth_metrics(z,gt,min_depth=.001,max_depth=10.))
                m['si_boundary_f1']=compute_depth_pro_si_boundary_f1(np.clip(z,.001,10),gt)
                rows.append(dict(sample_id=key,method=name,source_sha256=sha256(path),**m))
            if i%50==0: print(f'C ablation {i+1}/654',flush=True)
    write_csv(out/'per_image.csv',rows)
    means={name:{k:float(np.mean([r[k] for r in rows if r['method']==name])) for k in ('abs_rel','rmse','delta1','si_boundary_f1')} for name in methods}
    write_json(out/'summary.json',dict(samples=654,means=means,selection='Fixed component removal, no test-time parameter search; primary C defaults unchanged',protocol='Filled NYU depth; native metric; crop [45:471,41:601]; .001<GT<10; clip prediction; image macro mean'))
    print(json.dumps(means),flush=True)


if __name__=='__main__':main()
