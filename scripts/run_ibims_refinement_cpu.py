"""Cross-dataset CPU/GPU RAW/C audit on MoGe's processed 100-image iBims mirror.

This mirror and the explicitly masked boundary metric are NOT a reproduction of
the original unmasked paper table. Native depth is never fitted to test GT.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time
import warnings
from contextlib import nullcontext
import zipfile
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.io import sha256,write_json,write_csv,environment
from src.exploration.edge_aware import refine_edges
from src.metrics.video_project import depth_metrics
from src.metrics.boundary import _official_si_boundary


def prepare():
    archive=ROOT/'data/downloads/ibims1_moge_processed.zip'
    out=ROOT/'data/public_samples/ibims_moge_processed'
    if (out/'manifest.json').exists(): return out
    for name in ('rgb','depth','valid'): (out/name).mkdir(parents=True,exist_ok=True)
    samples=[]
    with zipfile.ZipFile(archive) as z:
        paths=sorted(n for n in z.namelist() if n.endswith('/depth.png'))
        if len(paths)!=100: raise ValueError('Expected complete 100-image mirror')
        for name in paths:
            prefix=name.rsplit('/',1)[0]; key=prefix.rsplit('/',1)[-1]
            image=Image.open(io.BytesIO(z.read(name))); code=np.array(image)
            near,far=float(image.info['near']),float(image.info['far'])
            normalized=(code.astype(np.float32)-1)/65533
            depth=near**(1-normalized)*far**normalized
            if 'unit' in image.info: depth*=float(image.info['unit'])
            depth[code==0]=np.nan; depth[code==65535]=np.inf
            valid=np.isfinite(depth)&(depth>=.1)&(depth<=10.)
            (out/'rgb'/f'{key}.jpg').write_bytes(z.read(prefix+'/image.jpg'))
            np.save(out/'depth'/f'{key}.npy',depth)
            cv2.imwrite(str(out/'valid'/f'{key}.png'),valid.astype(np.uint8)*255)
            samples.append(dict(id=key,rgb=f'rgb/{key}.jpg',depth=f'depth/{key}.npy',mask=f'valid/{key}.png',depth_sha256=sha256(out/'depth'/f'{key}.npy'),rgb_sha256=sha256(out/'rgb'/f'{key}.jpg')))
    write_json(out/'manifest.json',dict(dataset='iBims-1 MoGe processed mirror, 100 images',source='https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation',archive_sha256=sha256(archive),samples=samples,protocol=dict(min_depth=.1,max_depth=10.,crop=None,alignment='none',clip_metric_predictions=True),caveat='JPEG RGB / log-quantized PNG depth; original masks/edges are not all present. Not the original MAT or exact paper protocol.'))
    return out


def masked_si(pred,gt,valid):
    # Same ratio thresholds, orientation averaging and weighting as official
    # Depth Pro, with both endpoints required valid. No invented GT at holes.
    prediction=1/np.maximum(pred.astype(np.float64),1e-6)
    target=1/np.maximum(np.where(valid,gt,1).astype(np.float64),1e-6)
    h=valid[:,1:]&valid[:,:-1]; v=valid[1:]&valid[:-1]
    masks=(h,v,h,v)
    thresholds=np.linspace(1.05,1.25,10); scores=[]
    edges=_official_si_boundary().__globals__['fgbg_depth']
    for t in thresholds:
        p=[x&m for x,m in zip(edges(prediction,t),masks)]
        g=[x&m for x,m in zip(edges(target,t),masks)]
        precision=np.mean([np.count_nonzero(a&b)/max(1,np.count_nonzero(a)) for a,b in zip(p,g)])
        recall=np.mean([np.count_nonzero(a&b)/max(1,np.count_nonzero(b)) for a,b in zip(p,g)])
        scores.append(2*precision*recall/(precision+recall) if precision+recall else 0.)
    return float(np.dot(scores,thresholds/thresholds.sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['depth-pro','unidepth-v2'],required=True)
    p.add_argument('--limit',type=int,default=100)
    p.add_argument('--output',default='outputs/ibims_refinement_cpu')
    p.add_argument('--precision',choices=['auto','fp32','bf16'],default='auto')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args(); import torch
    if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if a.device=='cuda' and a.precision=='bf16': raise ValueError('GPU audit requires FP32')
    torch.set_num_threads(4); cv2.setNumThreads(1)
    data_root=prepare(); data=json.loads((data_root/'manifest.json').read_text())
    out=ROOT/a.output/a.model
    for name in ('RAW','C','visualization'): (out/name).mkdir(parents=True,exist_ok=True)
    precision='fp32' if a.precision=='auto' else a.precision
    settings=dict(model=a.model,device=a.device,precision='cpu_bf16_autocast_FP32_weights' if precision=='bf16' else 'fp32',limit=a.limit,input_manifest_sha256=sha256(data_root/'manifest.json'),native_metric=True,provide_camera=False,alignment='none',metric_valid_range_m=[.1,10],metric_prediction_clipping=[.1,10],boundary='masked_official_ratio_SI_F1: both adjacent GT pixels valid; NOT unmasked paper F1',C=dict(radius=2,sigma_rgb=.10,sigma_log=.18,sigma_space=1.5),mirror_caveat=data['caveat'])
    if (out/'settings.json').exists():
        old=json.loads((out/'settings.json').read_text())
        if old!=settings: raise ValueError('Output settings differ; choose a new --output directory')
    write_json(out/'settings.json',settings); write_json(out/'environment.json',environment())
    model=None; rows=[]
    import csv
    times=list(csv.DictReader((out/'inference_times.csv').open())) if (out/'inference_times.csv').exists() else []
    for i,s in enumerate(data['samples'][:a.limit]):
        rgb=cv2.imread(str(data_root/s['rgb'])); gt=np.load(data_root/s['depth']); valid=cv2.imread(str(data_root/s['mask']),0)>0
        raw_path=out/'RAW'/f'{s["id"]}.npy'
        if raw_path.exists(): raw=np.load(raw_path)
        else:
            if model is None:
                if a.model=='depth-pro':
                    from src.models.foundation import DepthModel
                    model=DepthModel('depth-pro',ROOT/'third_party/ml-depth-pro',ROOT/'checkpoints/depth_pro/depth_pro.pt',device=a.device)
                    settings_weights=dict(checkpoint_sha256=sha256(ROOT/'checkpoints/depth_pro/depth_pro.pt'),input_resolution=1536)
                else:
                    from src.models.unidepth_v2_adapter import UniDepthV2Adapter
                    model=UniDepthV2Adapter(repository=ROOT/'third_party/UniDepth',checkpoint_dir=ROOT/'checkpoints/unidepth_v2_vits14',device=a.device)
                    settings_weights=dict(checkpoint_sha256=sha256(ROOT/'checkpoints/unidepth_v2_vits14/model.safetensors'),input_resolution='official resolution policy')
                write_json(out/'weights.json',settings_weights)
                if next(model.model.parameters()).dtype!=torch.float32: raise ValueError('Expected FP32 weights')
            if a.device=='cuda': torch.cuda.synchronize()
            begin=time.perf_counter()
            if a.model=='depth-pro':
                with torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False) if precision=='bf16' else nullcontext():
                    raw,_=model.image(rgb)
            else:
                with torch.inference_mode(),warnings.catch_warnings():
                    warnings.simplefilter('ignore'); raw,_,_,_=model.infer(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
            if a.device=='cuda': torch.cuda.synchronize()
            times.append(dict(sample_id=s['id'],seconds=time.perf_counter()-begin))
            write_csv(out/'inference_times.csv',times)
            np.save(raw_path,raw)
        if raw.shape!=gt.shape or not np.isfinite(raw).all() or (raw<=0).any(): raise ValueError('Invalid native prediction')
        refined=refine_edges(raw,rgb); np.save(out/'C'/raw_path.name,refined)
        for method,z in [('RAW',raw),('C',refined)]:
            scores,_,_=depth_metrics(z,gt,minimum=.1,maximum=10.,mask=valid,depth_type='metric',clip=True)
            scores['masked_ratio_SI_boundary_F1']=masked_si(z,gt,valid)
            rows.append(dict(sample_id=s['id'],method=method,prediction_sha256=sha256(out/method/raw_path.name),**scores))
        write_csv(out/'per_image.csv',rows)
        if i%10==0: print(f'{a.model}: {i+1}/{min(a.limit,100)}',flush=True)
    means={m:{k:float(np.mean([r[k] for r in rows if r['method']==m])) for k in ('AbsRel','RMSE','delta1','masked_ratio_SI_boundary_F1')} for m in ('RAW','C')}
    write_json(out/'summary.json',dict(samples=len(rows)//2,means=means,settings=settings,not_exact_paper_protocol=True))
    print(json.dumps(means),flush=True)


if __name__=='__main__': main()
