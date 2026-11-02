"""Frozen development/heldout D ablations with lossless compact predictions."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.io import sha256,write_json,write_csv,load_run,read_bgr,load_depth
from src.flow import FarnebackFlow,RAFTFlow,warp,consistency
from src.temporal import TemporalFusion,FusionConfig
from src.exploration.uncertainty import UncertaintyFusion
from src.exploration.efficient import ReducedFlow
from src.exploration.edge_aware import refine_edges
from src.metrics.video_project import depth_metrics,boundary_metrics
from src.metrics.pose import pose_correspondences,score_pose_correspondences

OLD=ROOT/'outputs/extended_video_validation_gpu518'


def fuse(method,params,depths,frames,fields):
    processor=(TemporalFusion(FusionConfig()) if method in ('A','E_half') else
               UncertaintyFusion(**params) if method.startswith('D') else None)
    result=[]; diagnostics=[]
    for i,(z,rgb) in enumerate(zip(depths,frames)):
        if method in ('RAW','RAW_selected'): value=z.copy(); debug={}
        elif method=='C': value=refine_edges(z,rgb); debug={}
        else:
            f,b=fields[i-1] if i else (None,None)
            value,debug=processor.step(z,rgb,f,b)
        if not np.isfinite(value).all() or (value<=0).any(): raise ValueError('Invalid output')
        result.append(value)
        diagnostics.append(dict(frame=i,reset=bool(debug.get('reset',False)),
            mean_weight=float(np.mean(debug.get('weight',0))),coverage=debug.get('coverage')))
    return np.stack(result),diagnostics


def recipes(selection=None):
    result={'RAW':{},'A':{},'D_temporal':dict(use_confidence=False,enable_reset=False),
            'D_confidence':dict(enable_reset=False),'D':{},'E_half':{},'C':{}}
    if selection is not None: result['D_tuned']=selection
    return result


def eligible(row,raw,guard):
    return (row['AbsRel']<=raw['AbsRel']+guard['AbsRel_max_increase'] and
            row['delta1']>=raw['delta1']-guard['delta1_max_drop'] and
            row['edge_F1']>=raw['edge_F1']-guard['edge_F1_max_drop'])


class Study:
    def __init__(self,out,protocol):
        self.out=out; self.protocol=protocol; self.raft=None
        self.evidence=ROOT/'docs/evidence/d_research'; self.evidence.mkdir(parents=True,exist_ok=True)

    def baseline(self,name,kind):
        if kind in ('phone','reference'): return OLD/name/'baseline'
        source=ROOT/'data/public_samples/d_research'/name/'manifest.json'
        target=self.out/name/'baseline'; target.parent.mkdir(parents=True,exist_ok=True)
        if not (target/'timing.json').exists():
            args=['-m','src.infer','--input',str(source.relative_to(ROOT)),'--output',str(target.relative_to(ROOT)),
                '--model','vda','--metric','--repo','third_party/Video-Depth-Anything',
                '--checkpoint','checkpoints/metric_video_depth_anything_vits.pth','--device','cuda','--input-size','518',
                '--vis-min','.1','--vis-max','10']
            write_json(target.parent/'inference_command.json',args)
            with (target.parent/'inference.log').open('w') as log:
                subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        meta,_=load_run(target)
        if meta['input_manifest_sha256']!=sha256(source) or meta['device']!='cuda' or meta['input_size']!=518:
            raise ValueError('Cached inference config mismatch')
        return target

    def fields(self,folder,baseline,frames,kind,name):
        cached=OLD/name/'flow_farneback' if kind in ('phone','reference') else folder/'flow_farneback'
        cached.mkdir(parents=True,exist_ok=True)
        fields=[]
        for i in range(1,len(frames)):
            path=cached/f'{i-1:06d}_{i:06d}.npz'
            if path.exists():
                with np.load(path) as d: f,b=d['forward'],d['backward']
            else:
                f,b=FarnebackFlow()(frames[i-1],frames[i]); mask,_=consistency(f,b)
                np.savez_compressed(path,forward=f,backward=b,common_mask=mask)
            fields.append((f,b))
        return fields

    def evaluate(self,name,kind,methods):
        folder=self.out/name; folder.mkdir(parents=True,exist_ok=True)
        final=folder/'summary.json'
        if final.exists():
            old=json.loads(final.read_text(encoding='utf-8'))
            if old['recipes']!=methods: raise ValueError('Recipes differ; use a new study output')
            if sha256(folder/'predictions.npz')!=old['predictions_sha256']: raise ValueError('Predictions changed')
            print(name+': validated cached results',flush=True); return old
        baseline=self.baseline(name,kind); meta,records=load_run(baseline)
        if meta['checkpoint_sha256']!=sha256(ROOT/'checkpoints/metric_video_depth_anything_vits.pth'):
            raise ValueError('Checkpoint changed')
        frames=[]; depths=[]
        for r in records:
            for sub,ext,key in [('rgb','png','rgb_sha256'),('depth','npy','depth_sha256')]:
                if sha256(baseline/sub/f'{r["id"]}.{ext}')!=r[key]: raise ValueError('Frozen baseline changed')
            frames.append(read_bgr(baseline/'rgb'/f'{r["id"]}.png'))
            depths.append(np.load(baseline/'depth'/f'{r["id"]}.npy'))
        print(name+': fusion ablations',flush=True)
        fields=self.fields(folder,baseline,frames,kind,name)
        half=ReducedFlow(.5); half_fields=[half(a,b) for a,b in zip(frames,frames[1:])]
        results={}; timings={}; diagnostics={}
        for method,params in methods.items():
            elapsed=[]
            for repeat in range(4):
                start=time.perf_counter()
                z,debug=fuse(method,params,depths,frames,half_fields if method=='E_half' else fields)
                if repeat: elapsed.append(time.perf_counter()-start)
            results[method]=z; diagnostics[method]=debug
            timings[method]=dict(fusion_seconds=float(np.median(elapsed)),fusion_repeat_seconds=elapsed)
        # Flow generation is timed with the same warm-up/repeat policy. It is
        # counted once per method, separate from fusion-only and evaluation.
        flow_times={}
        for label,backend in [('full',FarnebackFlow()),('half',ReducedFlow(.5))]:
            times=[]
            for repeat in range(4):
                start=time.perf_counter()
                for a,b in zip(frames,frames[1:]): backend(a,b)
                if repeat: times.append(time.perf_counter()-start)
            flow_times[label]=times
        for method in methods:
            seconds=0 if method in ('RAW','RAW_selected','C') else float(np.median(flow_times['half' if method=='E_half' else 'full']))
            timings[method].update(flow_seconds=seconds,postprocess_seconds=seconds+timings[method]['fusion_seconds'],
                postprocess_fps=len(frames)/(seconds+timings[method]['fusion_seconds']))
        del half_fields
        manifest=None; data=None; gt=None; intrinsics=None
        if kind!='phone':
            manifest=(OLD/'sample_inputs'/name/'manifest.json') if kind=='reference' else ROOT/'data/public_samples/d_research'/name/'manifest.json'
            data=json.loads(manifest.read_text(encoding='utf-8'))
            if [r['sample_id'] for r in records]!=[s['id'] for s in data['samples']]: raise ValueError('GT ordering differs')
            gt=[]
            for s in data['samples']:
                z=load_depth(manifest.parent/s['depth'],s.get('depth_scale',1))
                valid=load_depth(manifest.parent/s['mask'])>0
                valid &= np.isfinite(z)&(z>=data['protocol']['min_depth'])&(z<=data['protocol']['max_depth'])
                gt.append(np.where(valid,z,np.nan))
            intrinsics=json.loads((manifest.parent/data['intrinsics']).read_text(encoding='utf-8'))
        image_rows=[]; pairs=[]
        for i in range(len(frames)):
            for method,zs in results.items():
                row=dict(method=method,frame=i)
                if gt is not None:
                    scores,_,valid=depth_metrics(zs[i],gt[i],minimum=data['protocol']['min_depth'],maximum=data['protocol']['max_depth'])
                    row.update(scores); row.update(boundary_metrics(zs[i],gt[i],valid))
                # Diagnostic against RAW, explicitly not GT boundary accuracy.
                raw_boundary=boundary_metrics(zs[i],depths[i],np.ones(depths[i].shape,bool))
                row['RAW_reference_edge_F1']=raw_boundary['edge_F1']
                row['RAW_log_change']=float(np.mean(abs(np.log(zs[i])-np.log(depths[i]))))
                image_rows.append(row)
        print(name+': common-mask flow and GT-pose evaluation',flush=True)
        for i in range(1,len(frames)):
            pose=None
            if gt is not None:
                pose=pose_correspondences(gt[i-1],gt[i],data['samples'][i-1]['pose_world_camera'],data['samples'][i]['pose_world_camera'],intrinsics)
            for backend in ('farneback','raft'):
                if backend=='farneback': f,b=fields[i-1]
                else:
                    cache=OLD/name/'flow_raft' if kind in ('phone','reference') else folder/'flow_raft'
                    cache.mkdir(exist_ok=True)
                    path=cache/f'{i-1:06d}_{i:06d}.npz'
                    if path.exists():
                        with np.load(path) as d: f,b=d['forward'],d['backward']
                    else:
                        if self.raft is None: self.raft=RAFTFlow('cuda',ROOT/'checkpoints/raft_small_C_T_V2-01064c6d.pth')
                        f,b=self.raft(frames[i-1],frames[i]); mask,_=consistency(f,b)
                        np.savez_compressed(path,forward=f,backward=b,common_mask=mask)
                mask,_=consistency(f,b)
                if not mask.any(): raise ValueError('Empty RGB-only common mask')
                for method,zs in results.items():
                    old,inside=warp(zs[i-1],b)
                    if not inside[mask].all() or (old[mask]<=0).any(): raise ValueError('Invalid on common mask')
                    row=dict(method=method,previous=i-1,current=i,backend=backend,
                        warp_proxy=float(np.mean(abs(np.log(zs[i][mask])-np.log(old[mask])))),coverage=float(mask.mean()),valid_pixels=int(mask.sum()))
                    if backend=='farneback' and pose is not None: row.update(score_pose_correspondences(zs[i-1],zs[i],pose))
                    pairs.append(row)
        rows=[]
        for method in methods:
            images=[r for r in image_rows if r['method']==method]
            row=dict(method=method,**timings[method],resets=sum(d['reset'] for d in diagnostics[method]))
            for key in ('AbsRel','RMSE','delta1','edge_F1','RAW_reference_edge_F1','RAW_log_change'):
                row[key]=float(np.mean([r[key] for r in images])) if key in images[0] else None
            for backend in ('farneback','raft'):
                row[backend+'_warp_proxy']=float(np.mean([r['warp_proxy'] for r in pairs if r['method']==method and r['backend']==backend]))
            scores=[r['pose_temporal_log_error'] for r in pairs if r['method']==method and r.get('pose_temporal_log_error') is not None]
            row['pose_temporal_log_error']=float(np.mean(scores)) if scores else None
            rows.append(row)
        raw=next(r for r in rows if r['method']=='RAW')
        for row in rows:
            row['guard_passed']=eligible(row,raw,self.protocol['guard']) if gt is not None else None
        # Reuse verified immutable arrays instead of copying old experiments.
        # Equal predictions (e.g. reset ablation on a clip without cuts) share
        # one logical source; metrics still retain every method name.
        stored={}; locations={}; seen={}
        for method,zs in results.items():
            digest=hashlib.sha256(zs.tobytes()).hexdigest()
            if digest in seen:
                locations[method]=locations[seen[digest]].copy(); continue
            seen[digest]=method
            if method in ('RAW','RAW_selected'):
                locations[method]=dict(run=str(baseline.relative_to(ROOT)),manifest_sha256=sha256(baseline/'manifest.json'))
            elif kind in ('phone','reference') and method in ('A','C','D','E_half'):
                run=OLD/name/'directions'/method
                old_meta,old_records=load_run(run)
                same=all(np.array_equal(z,np.load(run/'depth'/f'{r["id"]}.npy')) and sha256(run/'depth'/f'{r["id"]}.npy')==r['depth_sha256'] for z,r in zip(zs,old_records))
                if same: locations[method]=dict(run=str(run.relative_to(ROOT)),manifest_sha256=sha256(run/'manifest.json'))
                else: stored[method]=zs; locations[method]=dict(npz_key=method)
            else:
                stored[method]=zs; locations[method]=dict(npz_key=method)
        np.savez_compressed(folder/'predictions.npz',**stored)
        write_csv(folder/'per_image.csv',image_rows); write_csv(folder/'per_pair.csv',pairs)
        write_json(folder/'diagnostics.json',diagnostics)
        result=dict(name=name,kind=kind,frames=len(frames),recipes=methods,methods=rows,
            baseline=str(baseline.relative_to(ROOT)),baseline_manifest_sha256=sha256(baseline/'manifest.json'),
            input_manifest_sha256=sha256(manifest) if manifest else meta['input_manifest_sha256'],
            predictions_sha256=sha256(folder/'predictions.npz'),prediction_locations=locations,flow_repeat_seconds=flow_times)
        write_json(final,result)
        self.render(folder,frames,results)
        dest=self.evidence/name; dest.mkdir(exist_ok=True)
        for file in ('summary.json','per_image.csv','per_pair.csv','diagnostics.json'):
            (dest/file).write_bytes((folder/file).read_bytes())
        print(name+': complete',flush=True)
        return result

    @staticmethod
    def render(folder,frames,results):
        from src.visualization.video import colorize
        methods=['RAW','D_temporal','D','D_tuned','E_half']
        methods=[m for m in methods if m in results]
        if len(methods)<5: methods.append('A')
        writer=None
        try:
            for i,rgb in enumerate(frames):
                panels=[('RGB',rgb)]+[(m,colorize(results[m][i],.1,10)) for m in methods[:5]]
                tiles=[]
                for label,img in panels:
                    h,w=img.shape[:2]; factor=240/max(h,w)
                    img=cv2.resize(img,(round(w*factor),round(h*factor)))
                    tile=np.zeros((img.shape[0]+24,img.shape[1],3),np.uint8); tile[24:]=img
                    cv2.putText(tile,label,(4,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1); tiles.append(tile)
                image=np.vstack((np.hstack(tiles[:3]),np.hstack(tiles[3:])))
                image=cv2.copyMakeBorder(image,0,image.shape[0]%2,0,image.shape[1]%2,cv2.BORDER_CONSTANT)
                if writer is None:
                    writer=cv2.VideoWriter(str(folder/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5,(image.shape[1],image.shape[0]))
                    if not writer.isOpened(): raise RuntimeError('No video writer')
                writer.write(image)
                if i in (8,24,40,56): cv2.imwrite(str(folder/f'review_{i:06d}.jpg'),image)
        finally:
            if writer: writer.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/d_research')
    a=p.parse_args()
    import torch
    torch.set_num_threads(4); cv2.setNumThreads(1)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for the frozen 518 study')
    protocol=json.loads((ROOT/'configs/part3/d_research.json').read_text(encoding='utf-8'))
    out=ROOT/a.output; out.mkdir(parents=True,exist_ok=True)
    registered=out/'protocol.json'
    if registered.exists() and json.loads(registered.read_text(encoding='utf-8'))!=protocol: raise ValueError('Protocol changed')
    write_json(registered,protocol)
    study=Study(out,protocol)
    development=protocol['development'][0]
    candidates=recipes()
    for i,params in enumerate(protocol['candidates']): candidates[f'D_candidate_{i}']=params
    selection_path=out/'selection.json'
    if not selection_path.exists():
        # Heldout model inference and metrics start only AFTER this file exists.
        dev=study.evaluate(development,'development',candidates)
        possible=[r for r in dev['methods'] if r['method'].startswith('D_candidate_') and r['guard_passed']]
        best=min(possible,key=lambda r:r['pose_temporal_log_error']) if possible else None
        winner=best['method'] if best else 'RAW'
        selection=dict(development=development,selected=winner,params=candidates[winner],fallback_RAW=best is None,
            protocol_sha256=sha256(registered),development_summary_sha256=sha256(out/development/'summary.json'),
            reason='Development only; minimum pose error among guarded candidates; no heldout input to selection')
        write_json(selection_path,selection)
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    if selection['protocol_sha256']!=sha256(registered): raise ValueError('Selection protocol mismatch')
    methods=recipes(selection['params'] if not selection['fallback_RAW'] else None)
    if selection['fallback_RAW']: methods['RAW_selected']={}
    results=[]
    for kind,key in [('heldout','heldout'),('phone','phone_demonstration'),('reference','reference_only')]:
        for name in protocol[key]:
            results.append(study.evaluate(name,kind,methods))
            write_json(out/'summary.json',dict(protocol=protocol,selection=selection,cases=results))
    for file in ('protocol.json','selection.json','summary.json'):
        (study.evidence/file).write_bytes((out/file).read_bytes())
    print('Completed heldout, self-recorded and reference validation',flush=True)


if __name__=='__main__': main()
