"""CPU capture review. Run: python scripts/review_capture.py --models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.io import sha256,write_json,read_bgr


def inspect(path,out,a):
    cap=cv2.VideoCapture(str(path))
    fps=cap.get(cv2.CAP_PROP_FPS)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps<=0 or total<a.frames:
        raise ValueError(f'Invalid video: {path}')
    width,height=(int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT))
    slug=path.stem.replace(' ','_')
    folder=out/slug/'input'
    folder.mkdir(parents=True,exist_ok=True)
    start=max(0,(total/fps-(a.frames-1)/a.sample_fps)/2)
    ids=[round((start+i/a.sample_fps)*fps) for i in range(a.frames)]
    source_hash=sha256(path)
    if (folder/'samples.json').exists():
        previous_manifest=json.loads((folder/'samples.json').read_text(encoding='utf-8'))
        scale=min(1,a.max_side/max(height,width))
        saved_shape=read_bgr(folder/previous_manifest['samples'][0]['rgb']).shape[:2]
        if (previous_manifest['source_sha256']!=source_hash or
            [r['source_index'] for r in previous_manifest['samples']]!=ids or
            previous_manifest['video']['playback_fps']!=a.sample_fps or
            saved_shape!=(round(height*scale),round(width*scale))):
            raise ValueError('Source or sampling changed; choose a new --output directory')
    contact_ids=set(np.linspace(0,total-1,12).astype(int).tolist())
    tiles,samples,brightness,motion=[],[],[],[]
    previous=None
    n=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        ratio=480/max(height,width)
        small=cv2.resize(frame,(round(width*ratio),round(height*ratio)),interpolation=cv2.INTER_AREA)
        g=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        brightness.append(float(g.mean()))
        if n%max(1,round(fps/5))==0:
            if previous is not None:
                pts=cv2.goodFeaturesToTrack(previous,200,.01,10)
                if pts is not None:
                    nxt,status,_=cv2.calcOpticalFlowPyrLK(previous,g,pts,None)
                    if nxt is not None and status.any():
                        d=(nxt-pts)[status.ravel()==1].reshape(-1,2)
                        motion.append(float(np.median(np.linalg.norm(d,axis=1))))
            previous=g
        if n in contact_ids:
            tile=np.zeros((small.shape[0]+30,480,3),np.uint8)
            h,w=small.shape[:2]
            tile[30:30+h,(480-w)//2:(480-w)//2+w]=small
            cv2.putText(tile,f'{n/fps:.2f}s',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
            tiles.append(tile)
        if n in ids:
            scale=min(1,a.max_side/max(height,width))
            resized=cv2.resize(frame,(round(width*scale),round(height*scale)),interpolation=cv2.INTER_AREA)
            name=f'{len(samples):06d}.png'
            if (folder/name).exists():
                if not np.array_equal(read_bgr(folder/name),resized):
                    raise ValueError('Saved RGB differs; choose a new --output directory')
            else:
                cv2.imwrite(str(folder/name),resized)
            samples.append(dict(id=name[:-4],rgb=name,source_index=n,timestamp=n/fps))
        n+=1
    cap.release()
    if len(samples)!=a.frames: raise ValueError('Incomplete sampling')
    while len(tiles)<12: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out/f'{slug}_contact.jpg'),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,12,3)]))
    if not (folder/'samples.json').exists():
        write_json(folder/'samples.json',dict(video=dict(playback_fps=a.sample_fps),max_side=a.max_side,source=str(path.relative_to(ROOT)),source_sha256=source_hash,samples=samples))
    row=dict(file=str(path.relative_to(ROOT)),slug=slug,bytes=path.stat().st_size,sha256=source_hash,width=width,height=height,fps=fps,reported_frames=total,decoded_frames=n,complete_decode=n==total,duration_s=n/fps,estimated_bitrate_mbps=path.stat().st_size*8/(n/fps)/1e6,brightness_range=[min(brightness),max(brightness)],median_tracked_motion_px_per_0_2s=float(np.median(motion)) if motion else None,motion_analysis_long_side_px=480,sample_start_s=samples[0]['timestamp'],sample_end_s=samples[-1]['timestamp'],sampled_frames=len(samples))
    print(json.dumps(row),flush=True)
    return row


def command(args,log,env):
    with log.open('w',encoding='utf-8') as f:
        proc=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if proc.returncode: raise RuntimeError(f'Command failed; inspect {log}')


def render(folder,fps):
    depths=[[np.load(p) for p in sorted((folder/model/'depth').glob('*.npy'))] for model in ('dav2','vda')]
    refs=[(float(np.percentile(d[0],5)),float(np.percentile(d[0],95))) for d in depths]
    if any(hi<=lo for lo,hi in refs): raise ValueError('Degenerate display range')
    frames=sorted((folder/'dav2/rgb').glob('*.png'))
    writer=None
    try:
        for i,p in enumerate(frames):
            rgb=read_bgr(p)
            panels=[rgb]
            for d,(lo,hi) in zip(depths,refs):
                v=np.clip((d[i]-lo)/(hi-lo),0,1)
                panels.append(cv2.applyColorMap(np.uint8(v*255),cv2.COLORMAP_TURBO))
            image=np.hstack(panels)
            header=np.zeros((44,image.shape[1],3),np.uint8)
            for j,label in enumerate(('RGB','DAV2 relative','VDA relative')):
                cv2.putText(header,label,(j*rgb.shape[1]+4,27),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            image=np.vstack([header,image])
            image=cv2.copyMakeBorder(image,0,image.shape[0]%2,0,image.shape[1]%2,cv2.BORDER_CONSTANT)
            if writer is None:
                writer=cv2.VideoWriter(str(folder/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(image.shape[1],image.shape[0]))
                if not writer.isOpened(): raise OSError('Video encoder unavailable')
            writer.write(image)
            if i==len(frames)//2: cv2.imwrite(str(folder/'comparison.jpg'),image)
    finally:
        if writer: writer.release()
    write_json(folder/'display_settings.json',dict(representation='relative inverse depth; warmer is nearer',reference='model-specific frame-0 P5/P95 fixed for whole excerpt; common normalized display 0..1; raw arrays unchanged',ranges=dict(zip(('DAV2','VDA'),refs)),not_metric_depth=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/self-recorded')
    p.add_argument('--output',default='outputs/self_recorded_review')
    p.add_argument('--models',action='store_true')
    p.add_argument('--frames',type=int,default=16)
    p.add_argument('--sample-fps',type=float,default=5)
    p.add_argument('--input-size',type=int,default=252)
    p.add_argument('--max-side',type=int,default=448)
    a=p.parse_args()
    if a.frames<2 or a.sample_fps<=0: p.error('Need >=2 frames and positive FPS')
    out=ROOT/a.output
    out.mkdir(parents=True,exist_ok=True)
    paths=sorted((ROOT/a.input).glob('*.mp4'))
    if not paths: p.error('No MP4 files')
    rows=[inspect(path,out,a) for path in paths]
    write_json(out/'capture_checks.json',rows)
    if not a.models: return
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH']=str(ROOT/'.runtime')+os.pathsep+str(ROOT)
    commands,results=[],[]
    for row in rows:
        folder=out/row['slug']
        for model,repo,weight in [('dav2','Depth-Anything-V2','depth_anything_v2_vits.pth'),('vda','Video-Depth-Anything','video_depth_anything_vits.pth')]:
            dest=folder/model
            args=['-m','src.infer','--input',str((folder/'input/samples.json').relative_to(ROOT)),'--output',str(dest.relative_to(ROOT)),'--model',model,'--repo',f'third_party/{repo}','--checkpoint',f'checkpoints/{weight}','--device','cpu','--input-size',str(a.input_size),'--vis-min','0','--vis-max','50']
            commands.append(args)
            write_json(out/'commands.json',commands)
            if (dest/'timing.json').exists():
                config=json.loads((dest/'config_resolved.json').read_text(encoding='utf-8'))
                if config['input_size']!=a.input_size or config['device']!='cpu':
                    raise ValueError('Inference settings changed; choose a new --output directory')
                metadata=json.loads((dest/'metadata.json').read_text(encoding='utf-8'))
                if metadata['input_manifest_sha256']!=sha256(folder/'input/samples.json'):
                    raise ValueError('Input manifest changed; choose a new --output directory')
                if metadata['checkpoint_sha256']!=sha256(ROOT/f'checkpoints/{weight}'):
                    raise ValueError('Checkpoint changed; choose a new --output directory')
            if not (dest/'timing.json').exists():
                print(f'Running {row["slug"]} {model}',flush=True)
                command(args,folder/f'{model}.log',env)
        score=folder/'scores'
        if not (score/'relative_comparison.json').exists():
            command(['scripts/compare_relative_video.py','--dav2',str((folder/'dav2').relative_to(ROOT)),'--vda',str((folder/'vda').relative_to(ROOT)),'--output',str(score.relative_to(ROOT))],folder/'score.log',env)
        render(folder,a.sample_fps)
        result=dict(file=row['file'],sample_start_s=row['sample_start_s'],sample_end_s=row['sample_end_s'],**json.loads((score/'relative_comparison.json').read_text()))
        results.append(result)
        write_json(out/'model_checks.json',dict(protocol=dict(device='cpu',precision='fp32',input_size=a.input_size,max_side=a.max_side,frames=a.frames,sample_fps=a.sample_fps,scope='central short excerpts only; no GT, not metric accuracy; dynamic objects included in common flow-valid mask'),clips=results))
        print(json.dumps(result),flush=True)
    lines=['# Self-recorded capture review','',
           'CPU FP32 Small models; identical saved RGB frames and network input size.',
           f'Each clip: {a.frames} central frames at {a.sample_fps:g} FPS; input size {a.input_size}; saved RGB max side {a.max_side}.','',
           '## Source integrity','']
    for row in rows:
        lines.append(f'- {row["file"]}: {row["width"]} x {row["height"]}, {row["duration_s"]:.2f} s, {row["estimated_bitrate_mbps"]:.2f} Mbps; decoded {row["decoded_frames"]}/{row["reported_frames"]} frames.')
    lines+=['','## Temporal comparison','',
            'Lower is better: mean flow-warp absolute error / fixed frame-0 P95-P5 span. Common forward/backward-consistent mask <=1 px; no per-frame scale fitting.','']
    for row,result in zip(rows,results):
        d,v=(result['scores'][m]['mean'] for m in ('DAV2','VDA'))
        lines.append(f'- {row["slug"]}, {row["sample_start_s"]:.2f}-{row["sample_end_s"]:.2f} s: DAV2 {d:.8f}; VDA {v:.8f}; VDA reduction {100*(1-v/d):.1f}%. [Preview]({row["slug"]}/comparison.mp4).')
    lines+=['','## Interpretation','',
            'These are relative inverse-depth stability proxies, not metric depth accuracy. Dynamic objects are not segmented; their changes and flow errors affect the score. Smoothing can reduce this error without improving thin structures.',
            'The 16-frame default uses repeated-frame padding in the official 32-frame VDA window. It does not test long-video window transitions or whole-clip scale drift.',
            'Comparison videos use model-specific frame-0 P5/P95 display calibration, fixed across the excerpt, with common normalized display 0..1. Raw NPY predictions are unchanged; colors do not represent meters.',
            'Saved evidence: capture_checks.json, model_checks.json, commands.json, input/samples.json, per-model config/metadata/environment/timing/RGB/NPY/valid, scores/per_pair.csv, flow_masks and settings.json.','']
    if (out.parent/'self_recorded_part3_cpu/summary.json').exists():
        lines+=['## Part 3 CPU comparison','',
                'A/C/D/E were additionally evaluated on metric VDA predictions of the same RGB excerpts. These metric log-error scores differ from the relative inverse-depth scores above.',
                '[Part 3 results](../self_recorded_part3_cpu/review.md).','']
    (out/'review.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': main()
