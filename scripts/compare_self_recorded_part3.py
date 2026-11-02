"""CPU-only metric VDA + frozen Part 3 directions on existing capture excerpts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.io import load_run,read_bgr,sha256,write_json,write_csv
from src.flow import FarnebackFlow,RAFTFlow,warp,consistency
from src.exploration.compare import compare
from src.visualization.video import colorize
from src.temporal import FusionConfig
from dataclasses import asdict

METHODS=['RAW','A','A_conservative','C','D','D_guarded','E_half','E_local']


def score(folder,backend,name):
    runs={m:load_run(folder/'directions'/m) for m in METHODS}
    rgb=[read_bgr(folder/'metric_vda/rgb'/f'{r["id"]}.png') for r in runs['RAW'][1]]
    depths={m:[np.load(folder/'directions'/m/'depth'/f'{r["id"]}.npy') for r in runs[m][1]] for m in METHODS}
    for m in METHODS:
        if len(runs[m][1])!=len(rgb): raise ValueError('Frame count mismatch')
        for record,d in zip(runs[m][1],depths[m]):
            run=folder/'directions'/m
            if sha256(run/'depth'/f'{record["id"]}.npy')!=record['depth_sha256'] or sha256(run/'rgb'/f'{record["id"]}.png')!=record['rgb_sha256']:
                raise ValueError('Frozen derived output hash mismatch')
            if record['rgb_sha256']!=runs['RAW'][1][int(record['id'])]['rgb_sha256']:
                raise ValueError('RGB frames differ between methods')
    flowdir=folder/f'flow_{name}'
    flowdir.mkdir(exist_ok=True)
    rows=[]
    for i in range(1,len(rgb)):
        cache=flowdir/f'{i-1:06d}_{i:06d}.npz'
        if cache.exists():
            data=np.load(cache)
            forward,backward=data['forward'],data['backward']
        else:
            forward,backward=backend(rgb[i-1],rgb[i])
        mask,error=consistency(forward,backward,1.)
        warped={}
        for m,ds in depths.items():
            prev,inside=warp(ds[i-1],backward)
            valid,_=warp((np.isfinite(ds[i-1])&(ds[i-1]>0)).astype(np.float32),backward)
            mask &= inside&(valid>.999)&np.isfinite(prev)&(prev>0)&np.isfinite(ds[i])&(ds[i]>0)
            warped[m]=prev
        if not mask.any(): raise ValueError('Empty common evaluation mask')
        np.savez_compressed(cache,forward=forward,backward=backward,common_mask=mask,fb_error=error)
        for m,ds in depths.items():
            delta=np.log(ds[i][mask])-np.log(warped[m][mask])
            rows.append(dict(method=m,previous_id=f'{i-1:06d}',current_id=f'{i:06d}',coverage=float(mask.mean()),valid_pixels=int(mask.sum()),temporal_log_error=float(np.mean(abs(delta))),warp_MAE_m=float(np.mean(abs(ds[i][mask]-warped[m][mask])))))
    write_csv(folder/f'per_pair_{name}.csv',rows)
    means={m:{k:float(np.mean([r[k] for r in rows if r['method']==m])) for k in ('temporal_log_error','warp_MAE_m','coverage')} for m in METHODS}
    for m in METHODS:
        means[m]['reduction_vs_RAW_pct']=100*(1-means[m]['temporal_log_error']/means['RAW']['temporal_log_error'])
    write_json(folder/f'scores_{name}.json',means)
    return means


def diagnostics(folder):
    baseline=sorted((folder/'directions/RAW/depth').glob('*.npy'))
    results={}
    for method in METHODS:
        deviations,retentions,counts=[],[],[]
        for p in baseline:
            raw=np.load(p)
            d=np.load(folder/'directions'/method/'depth'/p.name)
            valid=np.isfinite(raw)&(raw>0)&np.isfinite(d)&(d>0)
            deviations.extend(abs(np.log(d[valid]/raw[valid]))[::16].tolist())
            def edge(z):
                log=np.log(np.maximum(z,1e-6)); e=np.zeros(z.shape,bool)
                e[:,1:] |= abs(np.diff(log,axis=1))>.08
                e[1:,:] |= abs(np.diff(log,axis=0))>.08
                return e&valid
            a,b=edge(raw),edge(d)
            near=cv2.dilate(b.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            retentions.append(float((a&near).sum()/max(1,a.sum())))
            counts.append(float(b.sum()/max(1,a.sum())))
        timing=json.loads((folder/'directions'/method/'timing.json').read_text())
        results[method]=dict(p95_abs_log_change_vs_RAW=float(np.percentile(deviations,95)),RAW_edge_retention_within_1px=float(np.mean(retentions)),edge_count_ratio_vs_RAW=float(np.mean(counts)),postprocess_fps=timing['postprocess_fps'],repeat_seconds=timing['repeat_seconds'])
    write_json(folder/'diagnostics.json',dict(metrics=results,caveat='RAW edges are a reference, NOT ground truth. Edge retention does not measure boundary accuracy. P95 samples every 16th valid pixel per frame. Timing excludes backbone, decode, I/O, evaluation; includes flow and warning compute; 1 OpenCV thread.'))
    return results


def render(folder):
    records=load_run(folder/'metric_vda')[1]
    writer=None
    try:
        for i,r in enumerate(records):
            rgb=read_bgr(folder/'metric_vda/rgb'/f'{r["id"]}.png')
            images=[('RGB',rgb)]+[(m,colorize(np.load(folder/'directions'/m/'depth'/f'{r["id"]}.npy'),.1,10)) for m in METHODS]
            tiles=[]
            for label,image in images:
                h,w=image.shape[:2]; scale=288/max(h,w)
                small=cv2.resize(image,(round(w*scale),round(h*scale)))
                hh,ww=small.shape[:2]
                tile=np.zeros((hh+30,ww,3),np.uint8)
                tile[30:30+hh]=small
                cv2.putText(tile,label,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                tiles.append(tile)
            image=np.vstack([np.hstack(tiles[j:j+3]) for j in (0,3,6)])
            footer=np.zeros((36,image.shape[1],3),np.uint8)
            cv2.putText(footer,'Fixed 0.1-10 m | blue=near, red=far',(6,24),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
            image=np.vstack([image,footer])
            image=cv2.copyMakeBorder(image,0,image.shape[0]%2,0,image.shape[1]%2,cv2.BORDER_CONSTANT)
            if writer is None:
                writer=cv2.VideoWriter(str(folder/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5,(image.shape[1],image.shape[0]))
                if not writer.isOpened(): raise OSError('Cannot open video writer')
            writer.write(image)
            if i in (0,8,15): cv2.imwrite(str(folder/f'comparison_{i:06d}.jpg'),image)
    finally:
        if writer: writer.release()


def write_review(out,summary):
    lines=['# 自录片段 Part 3 CPU 对照','',
           '基线为 metric VDA-Small，使用已有相对 VDA 检查完全相同的 RGB/时间段。',
           '4 段各 16 帧、5 FPS、input-size 252、RGB 长边 448；全部 CPU FP32。','',
           '以下是相对于 RAW 的时序 log-error 降幅，正数更好；不是深度准确率。',
           'Farneback 用于融合，RAFT 为独立光流评分；所有方法在同一共同有效区域评分。','']
    for clip in summary:
        lines += [f'## {clip["clip"]}：{clip["start_s"]:.2f}–{clip["end_s"]:.2f} 秒','']
        for m in METHODS:
            f,r=(clip['scores'][name][m] for name in ('farneback','raft'))
            lines.append(f'- {m}: Farneback {f["temporal_log_error"]:.8f}（降低 {f["reduction_vs_RAW_pct"]:.2f}%）；RAFT {r["temporal_log_error"]:.8f}（降低 {r["reduction_vs_RAW_pct"]:.2f}%）。')
        lines += ['',f'[同范围对比视频]({clip["clip"]}/comparison.mp4)','']
    lines += ['## 限制与证据','',
              '无深度/位姿 GT，不能证明 AbsRel、边界准确率、米制尺度漂移或是否通过既有精度约束。',
              '边界诊断只比较 RAW 边缘的保留情况，RAW 不是 GT；平滑可能降低时序分数但丢失结构。',
              '官方 VDA 对短序列使用末帧填充，尚未检验完整视频或跨窗口衔接。动态对象未分割。',
              '固定显示范围 0.1–10 m，蓝色近、红色远；方法之间与帧之间都不独立归一化。',
              'settings.json / method_parameters.json / commands.json 保存流程和参数；summary.json 保存汇总。',
              '各片段目录保存基线及派生 NPY/RGB/valid、配置/环境/计时、逐对 CSV 和完整 flow/common-mask。','']
    (out/'review.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures',default='outputs/self_recorded_review')
    p.add_argument('--output',default='outputs/self_recorded_part3_cpu')
    a=p.parse_args()
    out=ROOT/a.output; out.mkdir(parents=True,exist_ok=True)
    capture_root=ROOT/a.captures
    captures=json.loads((capture_root/'capture_checks.json').read_text())
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH']=str(ROOT/'.runtime')+os.pathsep+str(ROOT)
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    raft=RAFTFlow('cpu',ROOT/'checkpoints/raft_small_C_T_V2-01064c6d.pth')
    summary=[]; commands=[]
    settings=dict(device='cpu',precision='fp32',methods=METHODS,input_size=252,RGB_max_side=448,frames=16,fps=5,representation='metric depth, meters; native metric VDA Small',display_range_m=[.1,10],repeats=3,warmup_passes=1,
                  evaluation='same common FB-consistent mask <=1 px for all methods, finite positive depth, bilinear warp; equal-weight mean of 15 transition log errors; no GT or scale fitting',
                  farneback=dict(pyr_scale=.5,levels=3,winsize=21,iterations=3,poly_n=5,poly_sigma=1.2,flags=0),
                  independent_crosscheck='torchvision RAFT-Small CPU local checkpoint, default 12 updates, [-1,1] RGB, min 128 / next-multiple-of-8 resize, flow rescaled to saved RGB',raft_checkpoint_sha256=sha256(ROOT/'checkpoints/raft_small_C_T_V2-01064c6d.pth'),
                  limits=['No depth/pose GT: no AbsRel, delta1, true boundary F1 or metric scale-drift conclusion.','Farneback scoring shares fusion flow; RAFT provides independent-flow crosscheck, not GT.','Short central excerpts; official VDA pads 16 to 32, no long-video window transition test.','Dynamic objects are not segmented. Smoothing can improve temporal score but harm detail.'])
    write_json(out/'settings.json',settings)
    write_json(out/'method_parameters.json',dict(A=asdict(FusionConfig()),A_conservative=asdict(FusionConfig(alpha=.2)),C=dict(radius=2,sigma_rgb=.10,sigma_log=.18,sigma_space=1.5),D=dict(alpha_max=.7,max_log_update=None,edge_threshold=None),D_guarded=dict(alpha_max=.25,max_log_update=.01,edge_threshold=.08),E_half=dict(fusion=asdict(FusionConfig()),flow_scale=.5,adaptive=False),E_local=dict(fusion=asdict(FusionConfig()),flow_scale=.5,adaptive=True,local_motion=True,motion_threshold=.06),source_sha256={p:sha256(ROOT/p) for p in ['scripts/compare_self_recorded_part3.py','src/exploration/compare.py','src/exploration/edge_aware.py','src/exploration/uncertainty.py','src/exploration/efficient.py','src/temporal.py','src/flow.py']}))
    for capture in captures:
        slug=capture['slug']; folder=out/slug; folder.mkdir(exist_ok=True)
        source=capture_root/slug/'input/samples.json'
        manifest=json.loads(source.read_text())
        if sha256(ROOT/manifest['source'])!=manifest['source_sha256']: raise ValueError('Source video changed')
        baseline=folder/'metric_vda'
        args=['-m','src.infer','--model','vda','--metric','--input',str(source.relative_to(ROOT)),'--output',str(baseline.relative_to(ROOT)),'--repo','third_party/Video-Depth-Anything','--checkpoint','checkpoints/metric_video_depth_anything_vits.pth','--device','cpu','--input-size','252','--vis-min','.1','--vis-max','10']
        commands.append(args); write_json(out/'commands.json',commands)
        if not (baseline/'timing.json').exists():
            print(f'{slug}: metric VDA inference',flush=True)
            with (folder/'inference.log').open('w') as f:
                subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        meta,records=load_run(baseline)
        if meta['input_manifest_sha256']!=sha256(source) or meta['input_size']!=252 or len(records)!=16 or meta['depth_type']!='metric' or meta['checkpoint_sha256']!=sha256(ROOT/'checkpoints/metric_video_depth_anything_vits.pth'): raise ValueError('Baseline settings mismatch')
        write_json(folder/'source_sampling.json',manifest)
        if not (folder/'directions/comparison.json').exists():
            print(f'{slug}: directions',flush=True)
            compare(baseline,folder/'directions',methods=METHODS,repeats=3)
        else:
            previous=json.loads((folder/'directions/comparison.json').read_text())
            if previous['source_manifest_sha256']!=sha256(baseline/'manifest.json'):
                raise ValueError('Part 3 parent prediction changed; choose a new output directory')
        print(f'{slug}: common-mask Farneback and independent RAFT scoring',flush=True)
        scores={name:score(folder,backend,name) for name,backend in [('farneback',FarnebackFlow()),('raft',raft)]}
        diag=diagnostics(folder)
        render(folder)
        result=dict(clip=slug,source_sha256=manifest['source_sha256'],start_s=manifest['samples'][0]['timestamp'],end_s=manifest['samples'][-1]['timestamp'],scores=scores,diagnostics=diag)
        summary.append(result)
        write_json(out/'summary.json',dict(settings=settings,clips=summary))
        write_review(out,summary)
        print(json.dumps({name:{m:round(s['reduction_vs_RAW_pct'],2) for m,s in score.items()} for name,score in scores.items()}),flush=True)


if __name__=='__main__': main()
