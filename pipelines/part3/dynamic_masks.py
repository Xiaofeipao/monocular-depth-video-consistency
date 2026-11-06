"""Frozen full-sequence inference and mask-stratified diagnostics, without depth GT."""
import argparse,csv,gc,hashlib,importlib.metadata,json,sys,time
from pathlib import Path
import cv2,numpy as np
from src.video.dynamic_compat import DepthModel
from src.refinement.edge_aware import CUDAEdgeRefiner
from src.video.dynamic_compat import FarnebackFlow,RAFTFlow,warp,consistency

METHODS=('dav2','vda','depth_pro','C','C_no_RGB','C_no_depth_gate')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def write_csv(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def average(rows,key):
    values=[r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else None

def infer(args):
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(1)
    root=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    sources={};frames={}
    for name,count in [('bmx-trees',80),('tennis',70)]:
        files=sorted((root/name).glob('*'));masks={p.stem:p for p in (root/(name+'_mask')).glob('*.png')}
        assert len(files)==count and {p.stem for p in files}==set(masks)
        frames[name]=[cv2.imread(str(p)) for p in files]
        assert all(im is not None and im.shape==(240,432,3) for im in frames[name])
        sources[name]=[dict(frame=i,rgb=p.name,mask=masks[p.stem].name,rgb_sha256=digest(p),mask_sha256=digest(masks[p.stem])) for i,p in enumerate(files)]
        (out/name).mkdir()
    write_json(out/'inputs.json',sources)
    write_json(out/'protocol.json',dict(source_fps=None,frame_selection='all consecutive frames, no sampling',display_fps=10,
        vda_api_fps_placeholder=1,placeholder_scope='upstream infer_video_depth returns target_fps unchanged; no resampling in this API',
        input_size=518,effective_small_short_side=504,external_hw=[240,432],precision='FP16 official Small relative; Depth Pro FP16; C FP32',
        calibration='none',depth_gt=False,device='GPU-A',torch=torch.__version__,cuda=torch.version.cuda,
        timing='one warmup; sequence inference excludes load/IO; VDA offline amortized, not streaming latency',
        source_sha256=digest(__file__),region_rule='current foreground/background also agrees with warped previous segmentation; boundary=current mask +/-2 px; FB threshold 1px; >=8 common pixels per region',
        scoring='Farneback and independent RAFT-Small; mean absolute warp difference divided by fixed whole-clip P99-P1; C variants use unchanged Depth Pro span; not depth accuracy',
        edge_proxy='fraction of mask boundary within 2px of normalized depth Sobel edges >.02; segmentation silhouette is not depth-edge GT'))
    model_specs=[('dav2','dav2','Depth-Anything-V2','depth_anything_v2/depth_anything_v2_vits.pth'),
                 ('vda','vda','Video-Depth-Anything','video_depth_anything/video_depth_anything_vits.pth'),
                 ('depth_pro','depth-pro','ml-depth-pro','depth_pro/depth_pro.pt')]
    timings=[];identities={}
    for key,adapter,repo,ckpt in model_specs:
        model=DepthModel(adapter,Path(args.assets_root)/'third_party'/repo,Path(args.assets_root)/'checkpoints'/ckpt,device='cuda',input_size=518,fp16=True)
        identities[key]=model.metadata
        for name,images in frames.items():
            folder=out/name/key;folder.mkdir()
            if key=='vda':model.video(images[:2],1.)
            else:model.image(images[0])
            torch.cuda.reset_peak_memory_stats();model.synchronize();start=time.perf_counter()
            if key=='vda':z=model.video(images,1.)
            else:z=np.stack([model.image(im)[0] for im in images])
            model.synchronize();seconds=time.perf_counter()-start
            assert z.shape==(len(images),240,432) and np.isfinite(z).all()
            for i,d in enumerate(z):np.save(folder/f'{i:05d}.npy',d.astype(np.float32))
            timings.append(dict(sequence=name,method=key,frames=len(images),seconds=seconds,ms_per_frame=1000*seconds/len(images),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
            print(key,name,'complete',round(seconds,3),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(out/'models.json',identities)
    for method,kw in [('C',{}),('C_no_RGB',dict(use_rgb_guidance=False)),('C_no_depth_gate',dict(use_depth_guidance=False))]:
        refine=CUDAEdgeRefiner(**kw)
        for name,images in frames.items():
            folder=out/name/method;folder.mkdir();total=0.
            refine(np.load(out/name/'depth_pro/00000.npy'),cv2.cvtColor(images[0],cv2.COLOR_BGR2RGB))
            torch.cuda.reset_peak_memory_stats()
            for i,image in enumerate(images):
                raw=np.load(out/name/'depth_pro'/f'{i:05d}.npy');rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                start=time.perf_counter();d=refine(raw,rgb);total+=time.perf_counter()-start
                np.save(folder/f'{i:05d}.npy',d)
            timings.append(dict(sequence=name,method=method,frames=len(images),seconds=total,ms_per_frame=1000*total/len(images),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
    write_csv(out/'timing.csv',timings)
    write_json(out/'inference_complete.json',dict(status='complete',arrays=900,depth_gt=False))

def analyze(args):
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(1)
    out=Path(args.output);root=Path(args.data);rows=[];edges=[];summary={}
    assert (out/'inference_complete.json').exists()
    if (out/'summary.json').exists():raise FileExistsError('Completed analysis is immutable; use verify or a new run')
    backends=[('farneback',FarnebackFlow()),('raft',RAFTFlow(device='cuda',checkpoint=Path(args.assets_root)/'checkpoints/raft_small_C_T_V2-01064c6d.pth'))]
    for name in ('bmx-trees','tennis'):
        rgb=[cv2.imread(str(p)) for p in sorted((root/name).glob('*'))]
        masks=[cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)>0 for p in sorted((root/(name+'_mask')).glob('*.png'))]
        depths={m:np.stack([np.load(p) for p in sorted((out/name/m).glob('*.npy'))]) for m in METHODS}
        spans={m:float(np.percentile(z,99)-np.percentile(z,1)) for m,z in depths.items()}
        for m in METHODS[3:]:spans[m]=spans['depth_pro']
        assert all(v>0 for v in spans.values())
        write_json(out/name/'normalization.json',dict(spans=spans,note='fixed P99-P1 for scoring; shared RAW depth_pro span for its variants'))
        for backend,flow in backends:
            cache=out/name/('flow_'+backend);cache.mkdir(exist_ok=True)
            for i in range(1,len(rgb)):
                forward,backward=flow(rgb[i-1],rgb[i]);common,fb=consistency(forward,backward,1.)
                previous_mask,inside=warp(masks[i-1].astype(np.float32),backward)
                common &= inside
                warped={}
                for m,z in depths.items():
                    prev,inside=warp(z[i-1],backward);valid,_=warp(np.isfinite(z[i-1]).astype(np.float32),backward)
                    common &= inside&(valid>.999)&np.isfinite(prev)&np.isfinite(z[i]);warped[m]=prev
                b=masks[i].astype(np.uint8);kernel=np.ones((5,5),np.uint8)
                band=cv2.dilate(b,kernel)!=cv2.erode(b,kernel)
                regions={'all':np.ones(b.shape,bool),'foreground':masks[i]&(previous_mask>.999),
                         'background':~masks[i]&(previous_mask<.001),'boundary':band}
                np.savez_compressed(cache/f'{i:05d}.npz',forward=forward,backward=backward,common_mask=common,fb_error=fb)
                for region,mask in regions.items():
                    valid=mask&common;n=int(valid.sum())
                    for m,z in depths.items():rows.append(dict(sequence=name,backend=backend,previous=i-1,current=i,region=region,method=m,valid_pixels=n,
                        region_pixels=int(mask.sum()),coverage=float(n/max(1,int(mask.sum()))),normalized_warp_mae=float(np.mean(abs(z[i][valid]-warped[m][valid]))/spans[m]) if n>=8 else None))
            print('scored',name,backend,flush=True)
        for i,mask in enumerate(masks):
            b=mask.astype(np.uint8);band=(cv2.dilate(b,np.ones((3,3),np.uint8))!=cv2.erode(b,np.ones((3,3),np.uint8)))
            for m,z in depths.items():
                value=z[i]/spans[m];smooth=cv2.GaussianBlur(value,(0,0),.8)
                grad=(abs(cv2.Sobel(smooth,cv2.CV_32F,1,0))+abs(cv2.Sobel(smooth,cv2.CV_32F,0,1)))/8
                near=cv2.dilate((grad>.02).astype(np.uint8),np.ones((5,5),np.uint8))>0
                edges.append(dict(sequence=name,frame=i,method=m,silhouette_support=float(near[band].mean()),prediction_edge_fraction=float((grad>.02).mean())))
        # Fixed per-method display; C shares unchanged Depth Pro range. Playback FPS is illustrative only.
        ranges={m:np.percentile(z,[2,98]).tolist() for m,z in depths.items()}
        for m in METHODS[3:]:ranges[m]=ranges['depth_pro']
        write_json(out/name/'display.json',dict(ranges=ranges,fps=10,original_fps=None))
        writer=cv2.VideoWriter(str(out/name/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(432*3,264*2))
        assert writer.isOpened()
        for i,image in enumerate(rgb):
            tiles=[]
            for m in METHODS:
                lo,hi=ranges[m];z=depths[m][i];colored=cv2.applyColorMap((np.clip((z-lo)/(hi-lo),0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
                tile=np.zeros((264,432,3),np.uint8);tile[:240]=colored
                cv2.putText(tile,m+' | display 10 FPS',(5,257),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);tiles.append(tile)
            canvas=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]);writer.write(canvas)
            if i in (0,len(rgb)//2,len(rgb)-1):
                review=np.hstack([image,tiles[2][:240],tiles[3][:240]])
                cv2.imwrite(str(out/name/f'case_{i:05d}.jpg'),review)
        writer.release()
        summary[name]={}
        for backend,_ in backends:
            summary[name][backend]={}
            for region in ('all','foreground','background','boundary'):
                summary[name][backend][region]={m:dict(normalized_warp_mae=average([r for r in rows if r['sequence']==name and r['backend']==backend and r['region']==region and r['method']==m],'normalized_warp_mae'),evaluable_pairs=sum(r['normalized_warp_mae'] is not None for r in rows if r['sequence']==name and r['backend']==backend and r['region']==region and r['method']==m)) for m in METHODS}
    write_csv(out/'per_pair.csv',rows);write_csv(out/'silhouette.csv',edges)
    write_json(out/'summary.json',dict(status='complete',scores=summary,limitations='No depth GT; no real FPS; mask foreground is not per-pixel motion GT. Flow/occlusion errors and true depth motion remain. Mask support is not depth boundary accuracy. Independent backend is a sensitivity check, not temporal truth.',raft_checkpoint_sha256=digest(Path(args.assets_root)/'checkpoints/raft_small_C_T_V2-01064c6d.pth')))
    files={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file() and p.name!='FILES.sha256.json'}
    write_json(out/'FILES.sha256.json',files)

def check(args):
    root=Path(args.data)
    for name,count in [('bmx-trees',80),('tennis',70)]:
        images={p.stem:p for p in (root/name).glob('*') if p.is_file()}
        masks={p.stem:p for p in (root/(name+'_mask')).glob('*.png')}
        if len(images)!=count or set(images)!=set(masks) or set(images)!={f'{i:05d}' for i in range(count)}:raise ValueError('Missing or mismatched input frames')
        for stem,p in images.items():
            im=cv2.imread(str(p));mask=cv2.imread(str(masks[stem]),0)
            if im is None or mask is None or im.shape!=(240,432,3) or mask.shape!=(240,432) or not set(np.unique(mask)).issubset({0,255}):raise ValueError('Invalid RGB/mask')
    for rel in ['depth_anything_v2/depth_anything_v2_vits.pth','video_depth_anything/video_depth_anything_vits.pth','depth_pro/depth_pro.pt','raft_small_C_T_V2-01064c6d.pth']:
        if not (Path(args.assets_root)/'checkpoints'/rel).is_file():raise FileNotFoundError(rel)
    print('150 RGB/mask pairs and four local checkpoints present; no model executed, no output writes')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['check','predict','run','analyze','verify'])
    p.add_argument('--config',default='configs/part3/dynamic_masks.yaml')
    p.add_argument('--data')
    p.add_argument('--assets-root')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    import yaml
    config=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    expected=dict(protocol='dynamic_mask_regions_v1',sequences={'bmx-trees':80,'tennis':70},saved_hw=[240,432],input_size=518,source_fps=None,vda_api_fps_placeholder=1,display_fps=10,farneback=dict(pyr_scale=.5,levels=3,winsize=21,iterations=3,poly_n=5,poly_sigma=1.2),raft=dict(variant='small',iterations=12),forward_backward_threshold_px=1,minimum_region_pixels=8)
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen dynamic study protocol changed; use a separately named study')
    a.data=a.data or config['data'];a.assets_root=a.assets_root or config['assets_root']
    if a.stage=='check':check(a);return
    if a.stage=='verify':
        from scripts.verify_dynamic_masks import main as verify
        verify(['--run',a.output,'--data',a.data]);return
    if a.stage in ('run','predict'):infer(a)
    if a.stage in ('run','analyze'):analyze(a)

if __name__=='__main__':main()
