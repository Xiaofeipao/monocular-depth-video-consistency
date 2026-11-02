"""Warmed offline RGB-file -> metric-depth-file timing, not streaming latency."""
import gc
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.io import load_run,read_bgr,write_json,write_csv,sha256,environment
from src.models.foundation import DepthModel
from src.flow import FarnebackFlow
from src.exploration.efficient import ReducedFlow
from src.exploration.uncertainty import UncertaintyFusion
from src.temporal import TemporalFusion,FusionConfig


def main():
    import torch
    torch.set_num_threads(4); cv2.setNumThreads(1)
    out=ROOT/'outputs/d_research/pipeline_timing'; out.mkdir(parents=True,exist_ok=True)
    selection=json.loads((out.parent/'selection.json').read_text(encoding='utf-8'))
    if not (out.parent/'summary.json').exists(): raise RuntimeError('Finish study before timing')
    begin=time.perf_counter()
    model=DepthModel('vda',ROOT/'third_party/Video-Depth-Anything',ROOT/'checkpoints/metric_video_depth_anything_vits.pth',
        device='cuda',metric=True,input_size=518)
    load_seconds=time.perf_counter()-begin
    rows=[]; cases=[]
    for name in ('freiburg3_walking_xyz','moving'):
        # Different aspect ratios require different attention workspaces.
        # Release inactive allocations before changing shape, outside timing.
        gc.collect(); torch.cuda.empty_cache()
        baseline=(out.parent/name/'baseline') if name.startswith('freiburg') else ROOT/'outputs/extended_video_validation_gpu518'/name/'baseline'
        meta,records=load_run(baseline)
        reference=np.stack([np.load(baseline/'depth'/f'{r["id"]}.npy') for r in records])
        methods=['RAW','A','D','E_half']+([] if selection['fallback_RAW'] else ['D_tuned'])
        for method in methods:
            gc.collect(); torch.cuda.empty_cache()
            for repeat in range(4):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                reserved_before=torch.cuda.memory_reserved()
                start=time.perf_counter()
                frames=[read_bgr(baseline/'rgb'/f'{r["id"]}.png') for r in records]
                read_done=time.perf_counter()
                raw=model.video(frames,5); torch.cuda.synchronize()
                infer_done=time.perf_counter()
                processor=(TemporalFusion(FusionConfig()) if method in ('A','E_half') else
                    UncertaintyFusion(**(selection['params'] if method=='D_tuned' else {})) if method.startswith('D') else None)
                backend=ReducedFlow(.5) if method=='E_half' else FarnebackFlow()
                result=[]
                for i,(z,rgb) in enumerate(zip(raw,frames)):
                    if processor is None: value=z.copy()
                    else:
                        f,b=backend(frames[i-1],rgb) if i else (None,None)
                        value,_=processor.step(z,rgb,f,b)
                    result.append(value)
                post_done=time.perf_counter()
                # Timing outputs are scratch replays; full method predictions
                # are already frozen in the study. Keep one replay per clip.
                np.savez_compressed(out/f'{name}_replay.npz',depth=np.stack(result))
                end=time.perf_counter()
                if not np.allclose(raw,reference,rtol=1e-4,atol=1e-5): raise ValueError('Timing model predictions differ from frozen baseline')
                if repeat:
                    rows.append(dict(clip=name,method=method,repeat=repeat,frames=len(frames),
                        read_seconds=read_done-start,model_seconds=infer_done-read_done,
                        flow_fusion_seconds=post_done-infer_done,write_seconds=end-post_done,total_seconds=end-start,
                        fps=len(frames)/(end-start),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                        reserved_before_bytes=reserved_before,reserved_after_bytes=torch.cuda.memory_reserved(),
                        model_max_abs_difference=float(np.max(abs(raw-reference)))))
                print(f'{name} {method} repeat {repeat}: {len(frames)/(end-start):.2f} offline file-pipeline FPS',flush=True)
            times=[r for r in rows if r['clip']==name and r['method']==method]
            cases.append(dict(clip=name,method=method,frames=len(frames),
                median_total_seconds=float(np.median([r['total_seconds'] for r in times])),
                median_fps=float(np.median([r['fps'] for r in times]))))
            write_csv(out/'repeats.csv',rows)
            write_json(out/'summary.json',dict(model_load_seconds=load_seconds,methods=cases,
                protocol='1 warmup + 3 repeats; inactive CUDA cache released at clip/method boundaries outside timing; warmed model; saved RGB PNG read + offline 518 FP32 VDA + freshly computed CPU flow/fusion + compressed depth write; excludes initial model load, source video extraction, display, application alerts and evaluation',
                selection_sha256=sha256(out.parent/'selection.json')))
    dest=ROOT/'docs/evidence/d_research/pipeline_timing'; dest.mkdir(parents=True,exist_ok=True)
    for name in ('repeats.csv','summary.json'): (dest/name).write_bytes((out/name).read_bytes())
    env=environment(); write_json(dest/'software_environment.json',{k:env[k] for k in ('python','packages')})
    # Replays were written to include real output IO in timing. The study
    # already retains all method predictions, so these are expendable scratch.
    for path in out.glob('*_replay.npz'):
        if not path.resolve().is_relative_to(out.resolve()): raise ValueError('Unsafe replay path')
        path.unlink()


if __name__=='__main__': main()
