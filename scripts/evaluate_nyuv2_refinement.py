"""Frozen CPU/CUDA C refinement on the full NYUv2 test split, without test tuning."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.data.nyuv2 import NYUv2LabeledDataset
from src.exploration.edge_aware import CUDAEdgeRefiner, refine_edges
from src.metrics.depth import compute_depth_metrics
from src.metrics.boundary import compute_depth_pro_si_boundary_f1
from src.io import write_json,write_csv,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--mat',default='data/nyuv2/nyu_depth_v2_labeled.mat')
    p.add_argument('--splits',default='data/nyuv2/splits.mat')
    p.add_argument('--backend',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--check-only',action='store_true',help='Validate the full input split and prediction paths without inference or output writes')
    p.add_argument('--save-predictions',action='store_true',help='Also export refined metric depth NPZ files')
    a=p.parse_args()
    run,out=Path(a.run),Path(a.output)
    # Check all inputs before reserving a new immutable output directory.
    with NYUv2LabeledDataset(Path(a.mat),Path(a.splits)) as dataset:
        if len(dataset)!=654:
            raise ValueError('Course evaluation requires exactly 654 test images')
        for i in range(len(dataset)):
            index=int(dataset.test_indices_1based[i])
            file=run/'predictions'/f'test_{i:04d}_mat_{index:04d}.npz'
            if not file.is_file():
                raise FileNotFoundError(file)
    if a.check_only:
        print('Inputs available: 654 prediction paths and the NYUv2 test split; no model executed or outputs written')
        return
    refine=CUDAEdgeRefiner() if a.backend=='cuda' else refine_edges
    method='C_cuda' if a.backend=='cuda' else 'C'
    out.mkdir(parents=True,exist_ok=False)
    if a.save_predictions: (out/'predictions').mkdir()
    sys.path.insert(0,str(Path('third_party/ml-depth-pro/src').resolve()))
    cv2.setNumThreads(1)
    rows=[]
    with NYUv2LabeledDataset(Path(a.mat),Path(a.splits)) as dataset:
        for i in range(len(dataset)):
            sample=dataset.get_test_sample(i)
            stem=f'test_{i:04d}_mat_{sample.matlab_index_1based:04d}'
            file=run/'predictions'/f'{stem}.npz'
            with np.load(file,allow_pickle=False) as archive:
                raw=archive['depth_m'].astype(np.float32)
            if raw.shape!=sample.depth_m.shape:
                raw=cv2.resize(raw,(sample.depth_m.shape[1],sample.depth_m.shape[0]),interpolation=cv2.INTER_LINEAR)
            improved=refine(raw,sample.rgb)
            if a.save_predictions:
                np.savez_compressed(out/'predictions'/f'{stem}.npz',depth_m=improved)
            crop=np.s_[45:471,41:601]
            target=sample.depth_m[crop]
            for method_name,depth in [('RAW',raw),(method,improved)]:
                scores=dict(compute_depth_metrics(depth[crop],target,min_depth=.001,max_depth=10.))
                scores['si_boundary_f1']=compute_depth_pro_si_boundary_f1(np.clip(depth[crop],.001,10.),target)
                rows.append(dict(sample_id=stem,method=method_name,source_prediction_sha256=sha256(file),**scores))
            if i%20==0: print(f'Refinement evaluated {i+1}/{len(dataset)}',flush=True)
    write_csv(out/'per_image.csv',rows)
    metrics=('abs_rel','sq_rel','rmse','rmse_log','delta1','delta2','delta3','si_boundary_f1')
    means={m:{k:float(np.mean([r[k] for r in rows if r['method']==m])) for k in metrics} for m in ('RAW',method)}
    base,improved=means['RAW'],means[method]
    guard=improved['abs_rel']<=base['abs_rel']+.01 and improved['delta1']>=base['delta1']-.005 and improved['si_boundary_f1']>=base['si_boundary_f1']-.01
    write_json(out/'summary.json',dict(samples=len(rows)//2,means=means,accuracy_boundary_guard_passed=guard,
        abs_rel_improved=improved['abs_rel']<base['abs_rel'],parameters='Frozen 5x5 FP32 joint bilateral defaults; no test-set tuning',
        protocol='Full RGB; crop [45,471,41,601]; .001<GT<10; clip predictions; native metric; no alignment',
        limitation='Image-only refinement; no temporal claim; SI boundary guard is separately labelled from log-edge F1'))


if __name__=='__main__': main()
