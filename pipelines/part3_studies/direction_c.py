"""C补充：两个固定经典对照、相同GT支持的定位/宽度/纹理诊断。"""
from collections import Counter
from pathlib import Path
import argparse
import csv
import sys
import time
import cv2
import numpy as np
import yaml
from scipy.ndimage import binary_dilation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pipelines.part3 import common as c
from pipelines.part3_studies import common as s
from src.part3_studies.classical import METHODS as CLASSICS, apply
from src.part3_studies.diagnostics import prepare, analyze
from src.analysis.scale_boundary import depth_edge_map, extract_profile

TABLES=('per_image','boundary','localization','profiles','texture')


def plot_case(output,key,rgb,prepared,depths,diagnostic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gt_edges=prepared['gt_maps'][diagnostic['boundary']['primary_ratio']]
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    axes=axes.ravel();axes[0].imshow(rgb);axes[0].set_title('RGB / course crop')
    axes[1].imshow(gt_edges,cmap='gray');axes[1].set_title('GT log-depth edges / ratio 1.05')
    for ax,(method,z) in zip(axes[2:],depths.items()):
        edges=depth_edge_map(z,valid_mask=prepared['valid'],log_threshold=np.log(1.05))
        hit=edges&binary_dilation(gt_edges,structure=np.ones((3,3),bool))
        missed=gt_edges&~binary_dilation(edges,structure=np.ones((3,3),bool))
        image=rgb.astype(np.float32)/255*.55
        image[missed]=[.1,.3,1];image[edges&~hit]=[1,.1,.1];image[hit]=[.1,1,.1]
        ax.imshow(image);ax.set_title(method)
    for ax in axes:ax.axis('off')
    fig.suptitle(key+' / green: matched prediction; red: unmatched prediction; blue: missed GT (1 px)')
    fig.tight_layout();fig.savefig(output/f'edges_{key}.png',dpi=120);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    axes[0].imshow(rgb);axes[0].set_title('RGB')
    axes[1].imshow(prepared['gt'],vmin=.5,vmax=10,cmap='turbo');axes[1].set_title('GT (0.5-10 m)')
    regions=np.zeros((*prepared['gt'].shape,3),np.float32)
    regions[prepared['texture_masks']['textured_flat']]=[1,.3,.1]
    regions[prepared['texture_masks']['untextured_flat']]=[.1,.6,1]
    axes[2].imshow(regions);axes[2].set_title('Orange: textured flat; blue: control')
    for ax in axes:ax.axis('off')
    fig.tight_layout();fig.savefig(output/f'texture_{key}.png',dpi=120);plt.close(fig)
    if prepared['profiles']:
        profile=prepared['profiles'][0];side=diagnostic['profiles']['plateau_samples']
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for name,z in [('GT',prepared['gt']),*depths.items()]:
            line=extract_profile(z,profile);x=np.arange(len(line))-profile['radius_px']
            axes[0].plot(x,line,label=name)
            lo,hi=np.median(line[:side]),np.median(line[-side:])
            if abs(hi-lo)>1e-10:axes[1].plot(x,(line-lo)/(hi-lo),label=name)
        axes[0].set_ylabel('Depth (m)');axes[1].set_ylabel('Own-plateau normalized shape (diagnostic)')
        for ax in axes:ax.set_xlabel('Offset along fixed GT cross-section (px)');ax.legend(fontsize=7)
        fig.suptitle(key+f" / crop (y={profile['y']}, x={profile['x']}, axis={profile['axis']})")
        fig.tight_layout();fig.savefig(output/f'profile_{key}.png',dpi=120);plt.close(fig)


def reference(cfg,base,model):
    folder=ROOT/base['output_root']/model/cfg['reference_name']
    record=c.checked(folder,'c_evaluation')
    if record['config']!=base or record['model']!=model:raise ValueError('Frozen C model/config differs')
    return folder,record


def collect(output,cfg,base,mc,model,source,ref,limit=None,verify=False):
    diagnostic=yaml.safe_load((ROOT/cfg['diagnostic_config']).read_text())
    existing={(r['sample_id'],r['method']):r for r in c.read_csv(ref/'per_image.csv')}
    reference_backend=c.read_json(ref/'run_metadata.json')['backend']
    saved={(r['sample_id'],r['method']):r for r in c.read_csv(output/'per_image.csv')} if verify else {}
    tables={k:[] for k in TABLES};keys=[]
    with c.dataset(mc) as data:
        for i in c.positions(len(data),limit):
            sample=data.get_test_sample(i);key=c.stem(sample);keys.append(key)
            raw,digest=c.native(source,sample)
            index=sample.matlab_index_1based-1
            labels=np.asarray(data.file['labels'][index]).T[c.CROP]
            instances=np.asarray(data.file['instances'][index]).T[c.CROP]
            prepared=prepare(sample.depth_m[c.CROP],sample.rgb[c.CROP],labels,instances,diagnostic,cfg['texture'])
            images={}
            for method in cfg['methods']:
                timing=0.;peak=None;backend='none'
                if method=='RAW':z=raw
                elif method in CLASSICS:
                    path=output/'predictions'/method/(key+'.npz')
                    if verify:
                        z=s.read_depth(path)
                        c.assert_close(z,apply(method,raw,sample.rgb,cfg['classical']),'Classic CPU replay/'+method)
                        timing=float(saved[(key,method)]['postprocess_seconds'])
                    else:
                        if i==0:apply(method,raw,sample.rgb,cfg['classical'])
                        start=time.perf_counter();z=apply(method,raw,sample.rgb,cfg['classical'])
                        timing=time.perf_counter()-start;c.valid_depth(z)
                        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,depth_m=z)
                    backend='cpu_float64_to_float32'
                else:
                    z=s.read_depth(ref/'predictions'/method/(key+'.npz'))
                    row=existing[(key,method)];timing=float(row['postprocess_seconds'])
                    peak=int(row['peak_gpu_memory_bytes']) if row['peak_gpu_memory_bytes'] else None
                    backend='reference_'+reference_backend
                values=c.score(z,sample.depth_m)
                if method not in CLASSICS:
                    if existing[(key,method)]['source_prediction_sha256']!=digest:raise ValueError('Reference RAW hash differs')
                    c.raw_matches(values,existing[(key,method)])
                extra,measures=analyze(z[c.CROP],prepared,diagnostic,cfg['texture'])
                ids=dict(sample_id=key,scene=sample.scene,method=method)
                tables['per_image'].append(dict(**ids,source_prediction_sha256=digest,
                    postprocess_seconds=timing,postprocess_backend=backend,peak_gpu_memory_bytes=peak,**values,**extra))
                for name,rows in measures.items():tables[name].extend(dict(**ids,**row) for row in rows)
                images[method]=z[c.CROP]
            if not verify and i in cfg['case_positions']:
                plot_case(output,key,sample.rgb[c.CROP],prepared,images,diagnostic)
            if i%25==0:print(f'C六方法诊断 {i+1}/654',flush=True)
    for method in CLASSICS:
        if {p.stem for p in (output/'predictions'/method).glob('*.npz')}!=set(keys):
            raise ValueError('Classical prediction coverage differs')
    return tables


def profile_summary(rows,method):
    current=[r for r in rows if r['method']==method]
    raw={(r['sample_id'],r['profile_id']):r for r in rows if r['method']=='RAW'}
    common=[]
    for r in current:
        baseline=raw[(r['sample_id'],r['profile_id'])]
        if r['status']=='measurable' and baseline['status']=='measurable':
            common.append(dict(sample_id=r['sample_id'],
                width_delta=r['width_10_90_px']-baseline['width_10_90_px'],
                localization_error_delta=r['absolute_midpoint_error_px']-baseline['absolute_midpoint_error_px']))
    return dict(selected_profiles=len(current),status_counts=dict(Counter(r['status'] for r in current)),
        metrics={field:s.describe(current,field) for field in ('width_10_90_px','width_excess_px',
                    'absolute_midpoint_error_px','contrast_ratio_to_gt')},
        jointly_measurable_with_RAW=dict(profiles=len(common),
            width_delta=s.describe(common,'width_delta'),
            localization_error_delta=s.describe(common,'localization_error_delta')))


def summary(tables,cfg):
    methods=cfg['methods'];rows=tables['per_image']
    result=dict(samples=len(rows)//len(methods),means=c.means(rows,c.METRICS),diagnostics={},
        paired_scene_deltas=s.paired(rows,methods,['abs_rel','si_boundary_f1','primary_f1',
            'gt_edge_distance_capped_px','texture_spurious_fraction','control_spurious_fraction'],
            cfg['bootstrap_resamples'],cfg['bootstrap_seed']),
        parameters=cfg['classical'],diagnostic_depth='Unclipped native-scale arrays; course scoring alone clips',
        timing='Classic CPU float64->float32, first image warmed once; reference C timings retained with original backend',
        limitation='Fixed descriptive controls, no tuning or guaranteed texture suppression; same inspected NYUv2 test split')
    for method in methods:
        current=[r for r in rows if r['method']==method]
        textures={region:[r for r in tables['texture'] if r['method']==method and r['region']==region]
                  for region in ('textured_flat','untextured_flat')}
        result['diagnostics'][method]=dict(
            primary={field:s.describe(current,field) for field in ('primary_precision','primary_recall','primary_f1',
                'gt_edge_distance_capped_px','prediction_edge_spurious_beyond_3px_fraction')},
            profiles=profile_summary(tables['profiles'],method),
            texture={region:dict(image_macro=s.describe(rr,'spurious_edge_fraction'),
                mask_pixels=sum(r['mask_pixels'] for r in rr),
                predicted_edge_pixels=sum(r['predicted_edge_pixels'] for r in rr)) for region,rr in textures.items()},
            processing=dict(backend=current[0]['postprocess_backend'],mean_seconds=float(np.mean([r['postprocess_seconds'] for r in current]))))
    return result


def write_tables(output,tables):
    for name,rows in tables.items():
        if rows:c.write_csv(output/(name+'.csv'),rows)
        else:
            with (output/(name+'.csv')).open('w',newline='') as f:
                csv.writer(f).writerow(['sample_id','scene','method'])


def run(output,cfg,base,mc,model,limit):
    ref,parent=reference(cfg,base,model)
    source,_,inputs=c.baseline_files(base,model,mc);inputs.update(c.resources(base,model,mc))
    record=s.begin(output,'study_c_diagnostics',cfg,model,inputs,[ref],limit=limit,
                   depth_type='metric',unit='meter',alignment='none',native_metric=True,
                   classical_backend='cpu',reference_backend=parent['backend'])
    try:
        tables=collect(output,cfg,base,mc,model,source,ref,limit)
        write_tables(output,tables);c.write_json(output/'summary.json',summary(tables,cfg))
        s.seal(output,record,limit is None and len(tables['per_image'])==654*6)
    except BaseException as exc:c.failed(output,exc);raise


def verify(output,cfg,base,mc,model):
    record=s.checked(output,'study_c_diagnostics',full=False)
    if record['model']!=model or record['config']!=cfg:raise ValueError('Study model/config differs')
    ref=ROOT/base['output_root']/model/cfg['reference_name']
    if str(ref.resolve())!=record['parents'][0]:raise ValueError('C reference differs')
    source=ROOT/base['models'][model]['baseline']
    tables=collect(output,cfg,base,mc,model,source,ref,record['limit'],verify=True)
    for name,rows in tables.items():s.table_equal(rows,c.read_csv(output/(name+'.csv')))
    if summary(tables,cfg)!=c.read_json(output/'summary.json'):raise ValueError('Diagnostic summary differs')
    return dict(passed=True,samples=len(tables['per_image'])//6,methods=cfg['methods'],
        classical_predictions_replayed=len(tables['per_image'])//3,all_diagnostics_and_GT_scores_recomputed=True,
        formal_complete=record['formal_complete'],reference_C_scope='Frozen input/artifact SHA and scores; old CPU formula replay remains original verify')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','run','verify'])
    p.add_argument('--model',required=True,choices=['depth_pro','unidepth_v2'])
    p.add_argument('--config',type=Path);p.add_argument('--output-dir',type=Path)
    p.add_argument('--run',type=Path);p.add_argument('--limit',type=int);p.add_argument('--report',type=Path)
    a=p.parse_args();cv2.setNumThreads(1)
    cfg,base,mc=s.config('c',a.model,a.config)
    if a.action=='check':
        c.resources(base,a.model,mc);c.baseline_files(base,a.model,mc);reference(cfg,base,a.model)
        with c.dataset(mc) as ds:
            if 'labels' not in ds.file or 'instances' not in ds.file:raise ValueError('GT annotation fields missing')
        print('完整RAW/C/两消融、GT标注和来源核对通过；不加载模型');return
    if a.action=='verify':
        if not a.run:p.error('--run is required')
        c.report(a.run,verify(a.run.resolve(),cfg,base,mc,a.model),a.report);return
    run((a.output_dir or ROOT/base['output_root']/a.model/cfg['output_name']).resolve(),cfg,base,mc,a.model,a.limit)


if __name__=='__main__':main()
