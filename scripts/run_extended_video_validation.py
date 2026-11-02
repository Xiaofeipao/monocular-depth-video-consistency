"""CPU/GPU validation of fixed directions on longer self-recorded and RGB-D clips.

Uses 64 central frames per phone clip and two disjoint TUM excerpts. Parameters
are shared across both tracks. RGB-D evaluation uses native metric depth and GT
camera motion, never fits depth to the test GT.
"""
import argparse
from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.review_capture import inspect
from src.io import read_bgr,load_depth,sha256,write_json,write_csv,load_run
from src.data.tum import prepare_sequence
from src.exploration.compare import compare
from src.flow import FarnebackFlow,RAFTFlow,consistency,warp

METHODS=['RAW','A','C','D','E_half']


def resized_tum(source,destination):
    data=json.loads((source/'manifest.json').read_text())
    if destination.exists():
        cached=json.loads((destination/'manifest.json').read_text())
        if cached['source_manifest_sha256']!=sha256(source/'manifest.json'):
            raise ValueError('Resized sample source changed')
        return destination/'manifest.json'
    destination.mkdir(parents=True)
    for name in ('rgb','depth','valid'): (destination/name).mkdir()
    samples=[]
    intrinsics=json.loads((source/data['intrinsics']).read_text())
    for s in data['samples']:
        rgb=read_bgr(source/s['rgb']); h,w=rgb.shape[:2]
        scale=448/max(h,w); ww,hh=round(w*scale),round(h*scale)
        color=cv2.resize(rgb,(ww,hh),interpolation=cv2.INTER_AREA)
        z=cv2.resize(load_depth(source/s['depth'],s.get('depth_scale',1)),(ww,hh),interpolation=cv2.INTER_NEAREST)
        valid=cv2.resize(load_depth(source/s['mask']),(ww,hh),interpolation=cv2.INTER_NEAREST)>0
        key=f'{len(samples):06d}'
        cv2.imwrite(str(destination/'rgb'/f'{key}.png'),color)
        np.save(destination/'depth'/f'{key}.npy',z)
        cv2.imwrite(str(destination/'valid'/f'{key}.png'),valid.astype(np.uint8)*255)
        samples.append({**s,'rgb':f'rgb/{key}.png','depth':f'depth/{key}.npy','mask':f'valid/{key}.png','depth_scale':1.,'rgb_sha256':sha256(destination/'rgb'/f'{key}.png'),'depth_sha256':sha256(destination/'depth'/f'{key}.npy')})
    k={**intrinsics,'fx':intrinsics['fx']*ww/w,'fy':intrinsics['fy']*hh/h,'cx':(intrinsics['cx']+.5)*ww/w-.5,'cy':(intrinsics['cy']+.5)*hh/h-.5}
    write_json(destination/'intrinsics.json',k)
    write_json(destination/'manifest.json',{**data,'samples':samples,'source_manifest_sha256':sha256(source/'manifest.json'),'resize':dict(RGB='INTER_AREA',depth_and_mask='INTER_NEAREST',intrinsics='pixel-center-aware scale',output=[hh,ww])})
    return destination/'manifest.json'


def score_common(folder,backend,name):
    runs={m:load_run(folder/'directions'/m) for m in METHODS}
    records=runs['RAW'][1]
    frames=[read_bgr(folder/'baseline/rgb'/f'{r["id"]}.png') for r in records]
    depths={m:[np.load(folder/'directions'/m/'depth'/f'{r["id"]}.npy') for r in records] for m in METHODS}
    target=folder/f'flow_{name}'; target.mkdir(exist_ok=True)
    rows=[]
    for i in range(1,len(records)):
        path=target/f'{i-1:06d}_{i:06d}.npz'
        if path.exists():
            f=np.load(path); forward,backward=f['forward'],f['backward']
        else: forward,backward=backend(frames[i-1],frames[i])
        mask,error=consistency(forward,backward,1.)
        old={}
        for m,ds in depths.items():
            old[m],inside=warp(ds[i-1],backward)
            valid,_=warp((np.isfinite(ds[i-1])&(ds[i-1]>0)).astype(np.float32),backward)
            mask &= inside&(valid>.999)&np.isfinite(old[m])&(old[m]>0)&np.isfinite(ds[i])&(ds[i]>0)
        if not mask.any(): raise ValueError('Empty common evaluation mask')
        np.savez_compressed(path,forward=forward,backward=backward,common_mask=mask,fb_error=error)
        for m,ds in depths.items():
            delta=np.log(ds[i][mask])-np.log(old[m][mask])
            rows.append(dict(method=m,previous_id=records[i-1]['id'],current_id=records[i]['id'],log_warp_proxy=float(np.mean(abs(delta))),coverage=float(mask.mean()),valid_pixels=int(mask.sum())))
    write_csv(folder/f'per_pair_{name}.csv',rows)
    return {m:float(np.mean([r['log_warp_proxy'] for r in rows if r['method']==m])) for m in METHODS}


def render(folder):
    records=load_run(folder/'baseline')[1]; writer=None
    try:
        for i,r in enumerate(records):
            rgb=read_bgr(folder/'baseline/rgb'/f'{r["id"]}.png')
            from src.visualization.video import colorize
            panels=[('RGB',rgb)]+[(m,colorize(np.load(folder/'directions'/m/'depth'/f'{r["id"]}.npy'),.1,10)) for m in METHODS]
            tiles=[]
            for label,img in panels:
                h,w=img.shape[:2]; scale=288/max(h,w)
                img=cv2.resize(img,(round(w*scale),round(h*scale)))
                tile=np.zeros((img.shape[0]+30,img.shape[1],3),np.uint8); tile[30:]=img
                cv2.putText(tile,label,(6,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1); tiles.append(tile)
            image=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])])
            footer=np.zeros((36,image.shape[1],3),np.uint8)
            cv2.putText(footer,f'{i/5:.1f}s | fixed 0.1-10 m',(6,24),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
            image=np.vstack([image,footer]); image=cv2.copyMakeBorder(image,0,image.shape[0]%2,0,image.shape[1]%2,cv2.BORDER_CONSTANT)
            if writer is None:
                writer=cv2.VideoWriter(str(folder/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5,(image.shape[1],image.shape[0]))
                if not writer.isOpened(): raise OSError('Video writer unavailable')
            writer.write(image)
            if i in (0,21,22,43,44,len(records)-1): cv2.imwrite(str(folder/f'frame_{i:06d}.jpg'),image)
    finally:
        if writer: writer.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/extended_video_validation_cpu')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--input-size',type=int,default=252)
    a=p.parse_args(); out=ROOT/a.output; out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(4); cv2.setNumThreads(1)
    env=os.environ.copy(); env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH']=str(ROOT/'.runtime')+os.pathsep+str(ROOT)
    settings=dict(device=a.device,precision='fp32',requested_frames=64,sample_may_be_shorter=True,minimum_sample_frames=32,fps=5,input_size=a.input_size,max_side=448,methods=METHODS,display_range=[.1,10],repeats=3,warmup=1,
        limitations=['Same four phone clips are one physical scene, not four independent environments.','Warp proxy includes real z changes for moving cameras/objects; RAFT is not pose compensation.','TUM GT-pose reprojection separately compensates rigid camera geometry.','No test GT fitting; shared frozen processing parameters; no test-set search.','E accelerates A postprocessing, not the Part 2 backbone.'])
    if (out/'settings.json').exists():
        old_settings=json.loads((out/'settings.json').read_text())
        if 'frames' in old_settings:
            old_settings['requested_frames']=old_settings.pop('frames')
            old_settings.update(sample_may_be_shorter=True,minimum_sample_frames=32)
        if old_settings!=settings: raise ValueError('Settings differ; use a new output directory')
    write_json(out/'settings.json',settings)
    captures=out/'self_inputs'; captures.mkdir(exist_ok=True)
    jobs=[]
    for video in sorted((ROOT/'data/self-recorded').glob('*.mp4')):
        slug=video.stem.replace(' ','_'); saved=captures/slug/'input/samples.json'
        if saved.exists():
            cached=json.loads(saved.read_text())
            if cached['source_sha256']!=sha256(video) or len(cached['samples'])!=64: raise ValueError('Phone source changed')
        else: inspect(video,captures,Namespace(frames=64,sample_fps=5,max_side=448))
        jobs.append(dict(name=slug,track='self-recorded',input=saved,gt=None))
    for name,start in [('tum_xyz_early',0),('tum_xyz_late',15)]:
        source=out/'sample_inputs'/f'{name}_original'
        if not (source/'manifest.json').exists():
            prepare_sequence(ROOT/'data/downloads/rgbd_dataset_freiburg1_xyz.tgz',source,max_frames=64,target_fps=5,start_seconds=start)
        manifest=resized_tum(source,out/'sample_inputs'/name)
        if len(json.loads(manifest.read_text())['samples'])<32: raise ValueError('Need >=32 synchronized sample frames')
        jobs.append(dict(name=name,track='TUM GT sample',input=manifest,gt=manifest))
    raft=RAFTFlow(a.device,ROOT/'checkpoints/raft_small_C_T_V2-01064c6d.pth')
    summaries=[]; commands=[]
    for job in jobs:
        folder=out/job['name']; folder.mkdir(exist_ok=True)
        source=job['input']; data=json.loads(source.read_text())
        baseline=folder/'baseline'
        args=['-m','src.infer','--input',str(source.relative_to(ROOT)),'--output',str(baseline.relative_to(ROOT)),'--model','vda','--metric','--repo','third_party/Video-Depth-Anything','--checkpoint','checkpoints/metric_video_depth_anything_vits.pth','--device',a.device,'--input-size',str(a.input_size),'--vis-min','.1','--vis-max','10']
        commands.append(args); write_json(out/'commands.json',commands)
        if not (baseline/'timing.json').exists():
            print(job['name']+f': infer {len(data["samples"])} metric VDA frames',flush=True)
            with (folder/'inference.log').open('w') as log:
                subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        meta,records=load_run(baseline)
        if meta['input_manifest_sha256']!=sha256(source) or len(records)!=len(data['samples']) or meta['depth_type']!='metric': raise ValueError('Frozen baseline mismatch')
        if meta['checkpoint_sha256']!=sha256(ROOT/'checkpoints/metric_video_depth_anything_vits.pth'): raise ValueError('Checkpoint changed')
        direction=folder/'directions'
        if not (direction/'comparison.json').exists():
            print(job['name']+': compare RAW/A/C/D/E with'+(' GT' if job['gt'] else 'out GT'),flush=True)
            compare(baseline,direction,manifest=job['gt'],methods=METHODS,repeats=3)
        elif json.loads((direction/'comparison.json').read_text())['source_manifest_sha256']!=sha256(baseline/'manifest.json'):
            raise ValueError('Direction parent changed')
        print(job['name']+': common-mask/RAFT score and render',flush=True)
        scores={name:score_common(folder,flow,name) for name,flow in [('farneback',FarnebackFlow()),('raft',raft)]}
        render(folder)
        report=json.loads((direction/'comparison.json').read_text())
        result=dict(name=job['name'],track=job['track'],input_manifest=str(source.relative_to(ROOT)),input_manifest_sha256=sha256(source),first_timestamp=data['samples'][0].get('timestamp'),last_timestamp=data['samples'][-1].get('timestamp'),frames=len(records),flow_scores=scores,methods=report['methods'],guard=report['winners'],source_sha256=data.get('source_sha256',data.get('archive_sha256')))
        summaries.append(result); write_json(out/'summary.json',dict(settings=settings,clips=summaries))
        print(json.dumps(result),flush=True)


if __name__=='__main__': main()
