"""Serial, fixed-thread C versus C_cuda benchmark with numerical equivalence checks."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from src.exploration.compare import compare
from src.exploration.edge_aware import refine_edges,CUDAEdgeRefiner
from src.io import load_run,load_depth,write_json


def verify_equivalence():
    rng=np.random.default_rng(20261028)
    refiner=CUDAEdgeRefiner()
    rows=[]
    for name,shape in [('odd_grid',(19,27)),('tiny',(2,3)),('single',(1,1)),('thin_structure',(24,32))]:
        depth=np.exp(rng.uniform(-2,2,shape)).astype(np.float32)
        rgb=rng.integers(0,256,(*shape,3),dtype=np.uint8)
        if name=='thin_structure':
            depth[:]=4
            depth[:,15]=1
            rgb[:]=100
        if depth.size>1:
            depth.flat[0]=np.nan
            depth.flat[-1]=0
        expected=refine_edges(depth,rgb)
        actual=refiner(depth,rgb)
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6,equal_nan=True)
        valid=np.isfinite(expected)
        rows.append(dict(case=name,max_relative_difference=float(np.max(abs(actual[valid]/expected[valid]-1))) if valid.any() else 0.))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--manifest')
    p.add_argument('--output',required=True)
    p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checks=verify_equivalence()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    report=compare(a.run,a.output,a.manifest,methods=['RAW','C','C_cuda'],repeats=a.repeats)
    _,records=load_run(a.run)
    max_relative=0.
    for record in records:
        key=record['id']
        reference=load_depth(Path(a.output)/'C'/'depth'/f'{key}.npy')
        actual=load_depth(Path(a.output)/'C_cuda'/'depth'/f'{key}.npy')
        np.testing.assert_allclose(actual,reference,rtol=2e-6,atol=2e-6,equal_nan=True)
        valid=np.isfinite(reference)
        if valid.any():
            max_relative=max(max_relative,float(np.max(abs(actual[valid]/reference[valid]-1))))
    timings={}
    for name in ('C','C_cuda'):
        import json
        timings[name]=json.loads((Path(a.output)/name/'timing.json').read_text())
    result=dict(numerical_checks=checks,real_max_relative_difference=max_relative,
        cpu_threads=cv2.getNumThreads(),torch_threads=torch.get_num_threads(),serial_methods=True,
        python='Python 3.x',torch='CUDA-enabled PyTorch',cuda='available',
        gpu='CUDA_GPU_A',environment_details='Machine identity and exact environment versions omitted in public reports',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        timings=timings,
        speedup_vs_C=timings['C']['mean_seconds']/timings['C_cuda']['mean_seconds'],
        scope='Same FP32 5x5 formula; processing plus warning, includes CPU/GPU transfers; excludes model, decode, I/O and scoring',
        limitations=['One GPU and fixed CPU thread setting; no cross-machine performance claim.',
                     'Later TUM segment is disjoint in time, but belongs to the same scene and sequence.'])
    by_method={row['method']:row for row in report['methods']}
    tolerances={'AbsRel':1e-5,'RMSE':1e-5,'delta1':1e-5,'edge_F1':1e-4,
                'temporal_log_error':1e-5,'pose_temporal_log_error':1e-5}
    differences={key:abs(by_method['C_cuda'][key]-by_method['C'][key])
                 for key in tolerances if by_method['C'][key] is not None}
    result['metric_absolute_differences']=differences
    result['metric_tolerances']=tolerances
    result['acceptance_passed']=result['speedup_vs_C']>1.1 and all(value<=tolerances[key] for key,value in differences.items())
    write_json(Path(a.output)/'cuda_validation.json',result)
    print(result)


if __name__=='__main__': main()
