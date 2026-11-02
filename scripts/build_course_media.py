"""Build course comparison, provided-image application, failures and phone extras."""
import csv,json,sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.io import load_run,read_bgr,write_json,write_csv,sha256
from src.data.nyuv2 import NYUv2LabeledDataset
from src.exploration.edge_aware import CUDAEdgeRefiner
from src.metrics.depth import compute_depth_metrics
from src.metrics.boundary import depth_edge_map,compute_depth_pro_si_boundary_f1
from src.geometry import write_ply

OUT=ROOT/'outputs/course_submission'


def color(z,lo,hi):
    scaled=np.clip((z-lo)/(hi-lo),0,1)
    image=cv2.applyColorMap(np.uint8(np.nan_to_num(scaled)*255),cv2.COLORMAP_TURBO)
    image[~np.isfinite(z)]=0
    return image


def tile(image,label,width=480):
    h,w=image.shape[:2];image=cv2.resize(image,(width,round(h*width/w)))
    top=np.zeros((30,width,3),np.uint8);cv2.putText(top,label,(7,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    return np.vstack((top,image))


def video(path,images,fps):
    writer=None;count=0
    try:
        for image in images:
            image=cv2.copyMakeBorder(image,0,image.shape[0]%2,0,image.shape[1]%2,cv2.BORDER_CONSTANT)
            if writer is None:
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(image.shape[1],image.shape[0]))
                if not writer.isOpened():raise RuntimeError('No video codec')
            writer.write(image);count+=1
    finally:
        if writer:writer.release()
    return count


def build_comparison():
    runs=[OUT/('rollercoaster_'+m) for m in ('dav2','vda')]
    loaded=[load_run(r) for r in runs];ranges=[x[0]['visualization_range'] for x in loaded]
    if any(x[0]['fps']!=24 for x in loaded):raise ValueError('Original FPS changed')
    count=len(loaded[0][1])
    if count!=70 or len(loaded[1][1])!=count:raise ValueError('Unexpected original asset frame count')
    panels=[];individual=[[],[]]
    for i,(a,b) in enumerate(zip(loaded[0][1],loaded[1][1])):
        if a['source_index']!=i or b['source_index']!=i or a['rgb_sha256']!=b['rgb_sha256']:raise ValueError('Different source frames')
        rgb=read_bgr(runs[0]/'rgb'/f'{a["id"]}.png')
        depths=[np.load(run/'depth'/f'{r["id"]}.npy') for run,r in zip(runs,(a,b))]
        for j,(run,r,z,bounds) in enumerate(zip(runs,(a,b),depths,ranges)):
            cv2.imwrite(str(run/'valid'/f'{r["id"]}.png'),(np.isfinite(z)&(z>=0)).astype(np.uint8)*255)
            individual[j].append(np.hstack((tile(rgb,'RGB',426),tile(color(z,*bounds),'Relative | fixed whole-clip range',426))))
        pieces=[tile(rgb,'RGB | original 24 FPS',426)]+[tile(color(z,*bounds),name+' | relative, fixed P2-P98',426) for z,bounds,name in zip(depths,ranges,('DAV2 Small','VDA Small'))]
        image=np.hstack(pieces)
        footer=np.zeros((36,image.shape[1],3),np.uint8)
        cv2.putText(footer,f'Genuine source frame {i+1}/{count} | colors are not a common metric scale',(7,25),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
        image=np.vstack((image,footer));panels.append(image)
        if i in (0,23,46,69):cv2.imwrite(str(OUT/f'rollercoaster_frame_{i:03d}.jpg'),image)
    video(OUT/'rollercoaster_comparison.mp4',panels,24)
    for j,run in enumerate(runs):
        video(run/'visualization/rgb_depth.mp4',individual[j],24)
        meta,records=loaded[j]
        meta['validity_definition']='Finite nonnegative relative output, including zero endpoints; not metric distance validity'
        write_json(run/'metadata.json',meta)
        for r in records: r['source']='third_party/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4'
        write_json(run/'manifest.json',records)
    # Finite zero relative predictions are rendered at the low end, not hidden.
    write_json(OUT/'video_comparison.json',dict(frames=70,requested_frames=100,fps=24,source_indices=list(range(70)),
        same_RGB_verified=True,zero_relative_values_preserved=True,normalization=dict(dav2=ranges[0],vda=ranges[1],scope='whole clip per method, no per-frame normalization'),
        limitation='Official linked source has 70 frames; course confirmation still needed for the 100-frame wording.'))


def blur(rgb,z,focus):
    soft=cv2.GaussianBlur(rgb,(0,0),12)
    weight=np.clip(abs(z-focus)/max(.3,focus*.3),0,1)[...,None]
    return np.uint8(np.clip(rgb*(1-weight)+soft*weight,0,255))


def build_NYU():
    folder=OUT/'nyu_application';folder.mkdir(exist_ok=True)
    baseline=ROOT/'outputs/full_benchmark_20261029/depth_pro_nyuv2'
    scores=list(csv.DictReader((baseline/'metrics_per_image.csv').open()))
    c_rows=list(csv.DictReader((ROOT/'outputs/full_benchmark_20261029/depth_pro_C_cuda/per_image.csv').open()))
    refined={r['sample_id']:r for r in c_rows if r['method']=='C_cuda'}
    refine=CUDAEdgeRefiner();selected=[]
    with NYUv2LabeledDataset(ROOT/'data/nyuv2/nyu_depth_v2_labeled.mat',ROOT/'data/nyuv2/splits.mat') as data:
        def read(row):
            s=data.get_test_sample(int(row['test_position']));file=baseline/'predictions'/f'{row["sample_id"]}.npz'
            with np.load(file) as z:raw=z['depth_m'];focal=float(z['focal_length_px'])
            return s,raw,refine(raw,s.rgb),focal,file
        first=max(scores,key=lambda r:float(r['abs_rel']));selected.append(('large_metric_error',first))
        for row in sorted(scores,key=lambda r:float(r['si_boundary_f1'])):
            if row['scene']==first['scene'] or float(row['abs_rel'])>.2:continue
            s=data.get_test_sample(int(row['test_position']))
            if depth_edge_map(s.depth_m[45:471,41:601],log_threshold=.10).sum()>500:
                selected.append(('boundary_recovery_difficulty',row));break
        used={r['scene'] for _,r in selected}
        candidates=[r for r in scores if r['scene'] not in used]
        third=min(candidates,key=lambda r:float(refined[r['sample_id']]['si_boundary_f1'])-float(r['si_boundary_f1']))
        selected.append(('refinement_boundary_regression',third))
        failures=OUT/'difficult_cases';failures.mkdir(exist_ok=True);records=[]
        for number,(category,row) in enumerate(selected,1):
            s,raw,new,focal,file=read(row);case=failures/f'case_{number}';case.mkdir(exist_ok=True)
            rgb=cv2.cvtColor(s.rgb,cv2.COLOR_RGB2BGR)
            values=[]
            for label,z in [('RAW',raw),('C',new)]:
                metrics=dict(compute_depth_metrics(z[45:471,41:601],s.depth_m[45:471,41:601],min_depth=.001,max_depth=10))
                metrics['si_boundary_f1']=compute_depth_pro_si_boundary_f1(np.clip(z[45:471,41:601],.001,10),s.depth_m[45:471,41:601])
                values.append(dict(method=label,**metrics))
            expected=refined[row['sample_id']]
            if abs(values[1]['abs_rel']-float(expected['abs_rel']))>1e-6 or abs(values[1]['si_boundary_f1']-float(expected['si_boundary_f1']))>1e-6:
                raise ValueError('Default C no longer matches the frozen baseline experiment')
            cv2.imwrite(str(case/'rgb.png'),rgb);np.savez_compressed(case/'depths.npz',RAW=raw,C=new,GT=s.depth_m)
            valid=(s.depth_m>.001)&(s.depth_m<10);cv2.imwrite(str(case/'GT_valid.png'),valid.astype(np.uint8)*255)
            error=np.minimum(abs(np.clip(raw,.001,10)-s.depth_m),2)
            overlay=rgb.copy();edge=depth_edge_map(s.depth_m,valid_mask=valid,log_threshold=.1);overlay[edge]=(0,255,0)
            panel=np.vstack((np.hstack([tile(rgb,'RGB',320),tile(color(s.depth_m,.5,10),'Filled GT | 0.5-10m',320),tile(color(raw,.5,10),'Depth Pro RAW | 0.5-10m',320)]),
                np.hstack([tile(color(new,.5,10),'RAW + C | same scale',320),tile(color(error,0,2),'RAW absolute error | 0-2m',320),tile(overlay,'GT edges (illustrative log jump 0.1)',320)])))
            cv2.imwrite(str(case/'comparison.png'),panel)
            # Choose a real high-error patch on the evaluation crop, no invented scene type.
            discrepancy=abs(np.log(np.maximum(raw,1e-6))-np.log(np.maximum(s.depth_m,1e-6)))
            mask=np.zeros(raw.shape,bool);mask[45:471,41:601]=True;discrepancy[~mask]=0
            smooth=cv2.blur(discrepancy.astype(np.float32),(81,81));y,x=np.unravel_index(np.argmax(smooth),smooth.shape)
            y=int(np.clip(y,105,411));x=int(np.clip(x,101,541));box=(y-60,y+60,x-60,x+60)
            imgs=[rgb,color(s.depth_m,.5,10),color(raw,.5,10),color(new,.5,10)]
            zoom=np.hstack([tile(im[box[0]:box[1],box[2]:box[3]],name,240) for im,name in zip(imgs,['RGB zoom','GT zoom','RAW zoom','C zoom'])])
            cv2.imwrite(str(case/'zoom.png'),zoom)
            record=dict(case=number,category=category,sample_id=row['sample_id'],test_position=int(row['test_position']),scene=row['scene'],
                selection='Post-hoc failure visualization, not parameter selection',source_prediction_sha256=sha256(file),zoom_top_bottom_left_right=list(box),metrics=values)
            write_json(case/'case.json',record);records.append(record)
        write_json(failures/'cases.json',records)
        # Accurate, provided NYU image for a simple depth-aware effect and point cloud.
        s,raw,new,focal,file=read(scores[0]);rgb=cv2.cvtColor(s.rgb,cv2.COLOR_RGB2BGR);h,w=raw.shape
        focus_points=[(int(w*.5),int(h*.6)),(int(w*.8),int(h*.25))]
        focus=[float(np.median(raw[max(0,y-3):y+4,max(0,x-3):x+4])) for x,y in focus_points]
        def app_panel(f):
            return np.vstack((np.hstack((tile(rgb,'Provided NYUv2 RGB',480),tile(color(new,.5,10),'C metric depth | 0.5-10m',480))),
                np.hstack((tile(blur(rgb,raw,f),f'Unchanged depth blur | focus {f:.2f}m',480),tile(blur(rgb,new,f),'C depth blur | same focus/blur strength',480)))))
        cv2.imwrite(str(folder/'rgb_depth_application.png'),app_panel(focus[0]))
        def frames():
            for i in range(120):
                fraction=.5-.5*np.cos(2*np.pi*i/120)
                yield app_panel(focus[0]*(1-fraction)+focus[1]*fraction)
        video(folder/'application_demo.mp4',frames(),24)
        cv2.imwrite(str(folder/'rgb.png'),rgb);np.savez_compressed(folder/'depths.npz',RAW=raw,C=new,GT=s.depth_m)
        write_ply(folder/'point_cloud.ply',new,s.rgb,dict(fx=focal,fy=focal,cx=(w-1)/2,cy=(h-1)/2),stride=4)
        write_json(folder/'application.json',dict(dataset='NYUv2 test split',sample_id=scores[0]['sample_id'],input_prediction_sha256=sha256(file),
            application='Depth-aware background blur; RAW and C share focus distances, sigma and RGB',focus_points=focus_points,focus_depths_from_RAW=focus,blur_sigma=12,
            point_cloud=dict(focal_source='Depth Pro prediction',principal_point='assumed image center',square_pixels=True,stride=4),
            limitation='Application is a visual demonstration, not independently calibrated metric measurement or proof of application accuracy improvement'))
    print('Built provided-image application and three real failure cases',flush=True)


def build_phone_extras():
    target=OUT/'self_recorded';target.mkdir(exist_ok=True);records=[]
    for name in ('landscape_static','portrait_static','static_with_people_moving','moving'):
        source=ROOT/'outputs/extended_video_validation_gpu518'/name
        if not (source/'baseline/manifest.json').exists() or not (source/'directions/E_half/manifest.json').exists():
            continue  # Optional personal excerpts are not needed for course reproduction.
        meta,frames=load_run(source/'baseline');first=np.load(source/'baseline/depth'/f'{frames[0]["id"]}.npy')
        h,w=first.shape;focus=float(np.median(first[h//3:2*h//3,w//3:2*w//3]))
        folder=target/name;folder.mkdir(exist_ok=True)
        def generate():
            for i,r in enumerate(frames):
                rgb=read_bgr(source/'baseline/rgb'/f'{r["id"]}.png')
                raw=np.load(source/'baseline/depth'/f'{r["id"]}.npy');new=np.load(source/'directions/E_half/depth'/f'{r["id"]}.npy')
                parts=[tile(rgb,'Self-recorded RGB',288),tile(blur(rgb,raw,focus),'RAW metric VDA blur',288),tile(blur(rgb,new,focus),'E_half blur | same focus',288)]
                image=np.hstack(parts)
                if i in (8,24,40,56):cv2.imwrite(str(folder/f'frame_{i:03d}.jpg'),image)
                yield image
        count=video(folder/'application_comparison.mp4',generate(),5)
        record=dict(clip=name,frames=count,fps=5,focus_depth_from_first_RAW=focus,blur_sigma=12,
            baseline_manifest_sha256=sha256(source/'baseline/manifest.json'),E_manifest_sha256=sha256(source/'directions/E_half/manifest.json'),
            role='Optional 64-frame excerpt, not the original full-rate capture and not a substitute for the designated clip',depth_GT=False)
        write_json(folder/'application.json',record);records.append(record)
    write_json(target/'manifest.json',records)
    print('Built four self-recorded application comparisons',flush=True)


if __name__=='__main__':
    import torch;torch.set_num_threads(4);cv2.setNumThreads(1)
    build_comparison();build_NYU();build_phone_extras()
