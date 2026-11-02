"""Course relative-video baseline on all genuine frames, capped at 100."""
import argparse,json,sys,time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.io import read_frames,RunWriter,sha256,write_json,write_csv
from src.models.foundation import DepthModel
from src.visualization import render_run


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--model',choices=['dav2','vda'],required=True)
    a=p.parse_args(); import torch
    torch.set_num_threads(4);cv2.setNumThreads(1)
    out=ROOT/'outputs/course_submission'/('rollercoaster_'+a.model)
    if (out/'timing.json').exists(): print('Completed run already exists');return
    source=ROOT/'third_party/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4'
    start=time.perf_counter();frames,indices,fps=read_frames(source,max_frames=100,max_side=640)
    if indices!=list(range(len(frames))): raise ValueError('Original consecutive frames required')
    checkpoint=ROOT/('checkpoints/depth_anything_v2/depth_anything_v2_vits.pth' if a.model=='dav2' else 'checkpoints/video_depth_anything/video_depth_anything_vits.pth')
    repo=ROOT/'third_party'/('Depth-Anything-V2' if a.model=='dav2' else 'Video-Depth-Anything')
    model=DepthModel(a.model,repo,checkpoint,device='cuda',input_size=518,fp16=True)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[]
    if a.model=='vda':
        model.synchronize();begin=time.perf_counter();depths=model.video(frames,fps);model.synchronize()
        seconds=time.perf_counter()-begin;times=[seconds/len(frames)]*len(frames)
    else:
        depths=[]
        for rgb in frames:
            model.synchronize();begin=time.perf_counter();z,_=model.image(rgb);model.synchronize()
            times.append(time.perf_counter()-begin);depths.append(z)
        depths=np.stack(depths);seconds=sum(times)
    if not np.isfinite(depths).all() or (depths<0).any(): raise ValueError('Invalid raw output')
    lo,hi=np.percentile(depths,[2,98]).tolist()
    metadata={**model.metadata,'fps':fps,'frame_indices':indices,'source_sha256':sha256(source),
        'hardware':dict(alias='GPU-A',CUDA_compute_capability=list(torch.cuda.get_device_capability()),CUDA_visible_memory_bytes=torch.cuda.get_device_properties(0).total_memory),
        'requested_frames':100,'actual_frames':len(frames),'short_source_note':'Official source contains 70 genuine frames; no repeats, padding or FPS conversion in saved outputs',
        'visualization_range':[lo,hi],'visualization_normalization':'Fixed whole-clip raw P2-P98 per method, same TURBO palette; not a shared metric scale',
        'inference_mode':'offline' if a.model=='vda' else 'framewise'}
    writer=RunWriter(out,metadata,dict(source=source.relative_to(ROOT).as_posix(),encoder='vits',input_size=518,max_len=100,max_res=640,precision='fp16',metric=False))
    for i,z in enumerate(depths): writer.add(z,frames[i],source.relative_to(ROOT).as_posix(),indices[i],dict(sample_id=f'{indices[i]:06d}',timestamp=i/fps))
    render_run(out,writer.records,lo,hi,'relative (not meters)',fps,a.model)
    writer.finish(dict(model_seconds=seconds,model_fps=len(frames)/seconds,model_latency_ms=1000*seconds/len(frames),
        end_to_end_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),frames=len(frames),fps=fps,warmup=0,includes_model_load=True))
    write_csv(out/'per_frame_timing.csv',[dict(frame=i,seconds=t,kind='offline_amortized' if a.model=='vda' else 'per_frame') for i,t in enumerate(times)])
    print(json.dumps(dict(model=a.model,frames=len(frames),fps=fps,normalization=[lo,hi])),flush=True)


if __name__=='__main__':main()
