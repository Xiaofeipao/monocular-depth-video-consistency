"""单视频评价：全图逐像素变化、同光流时序变化与类型限定诊断。"""
from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np

from .common import read_json, write_json, utc_now, create_output_directory
from .clip_workflow import load_clip, load_comparison, identity, check_identity, verify_flow, seal
from .runner import verify_run
from .framewise import load_single_sources, measure_single
from .evaluation import _equivalent, _read_csv, _write_csv
from src.models.video_depth_anything_adapter import window_metadata

METRICS = {
    'flow_mae': ('flow', 'mean_pair_normalized_mae'),
    'flow_rmse': ('flow', 'mean_pair_normalized_rmse'),
    'pixel_mae': ('pixel', 'mean_pair_normalized_mae'),
    'pixel_rmse': ('pixel', 'mean_pair_normalized_rmse'),
    'background_std': ('background_flicker', 'normalized_std_mean'),
    'median_delta': ('median_summary', 'mean_abs_normalized_median_delta'),
    'median_log_ratio': ('median_summary', 'mean_abs_log_ratio'),
}


def pixel_pair(previous, current, span):
    """不读取光流或FB mask，比较全画面相同坐标；合法零值保留。"""
    a,b=np.asarray(previous),np.asarray(current)
    if a.ndim != 2 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('逐像素比较要求双方同形状、完整有限的二维数组')
    if span is not None and (not math.isfinite(span) or span <= 0):
        raise ValueError('固定跨度必须为正有限数或None')
    difference=b.astype(np.float64)-a.astype(np.float64)
    mae=float(np.abs(difference).mean());rmse=float(np.sqrt(np.square(difference).mean()))
    return {'pixels':int(a.size),'coverage':1.,'raw_mae':mae,'raw_rmse':rmse,
            'normalized_mae':mae/span if span else None,'normalized_rmse':rmse/span if span else None,
            'unavailable_reason':None if span else 'zero_reference_span'}


def improvement(a,b):
    return {'absolute_reduction':a-b if a is not None and b is not None else None,
            'relative_reduction':(a-b)/a if a is not None and b is not None and a > 0 else None,
            'reason':'unavailable_metric' if a is None or b is None else ('nonpositive_baseline' if a <= 0 else None)}


def load_sources(directory, part):
    directory=Path(directory).resolve();part=str(part)
    if part not in ('1','2'):raise ValueError('评价part只能是1或2')
    if part=='1':
        record,manifest=load_clip(directory);baseline=directory
        if record.get('part') != '1':raise ValueError('Part 1B需要自己的输出目录')
    else:
        record,manifest,baseline=load_comparison(directory)
    verify_run(baseline/'dav2');verify_flow(baseline)
    context=load_single_sources(baseline/'dav2',baseline/'flow')
    names=('dav2',) if part=='1' else ('dav2','vda')
    runs={'dav2':context['run']};raw={'dav2':context['raw']}
    bound=[identity(directory/'clip.json')]+context['bound_files']
    if part=='2':
        verify_run(directory/'vda');run=read_json(directory/'vda/run.json')
        if run['protocol_sha256'] != record['protocol']['sha256'] or run['manifest_sha256'] != record['manifest']['sha256']:
            raise ValueError('VDA没有使用本2B冻结协议或引用的同一RGB清单')
        if run['config'] != context['run']['config']:
            raise ValueError('两模型配置不同')
        for key in ('network_input_hw','actual_output_hw','device','precision','encoder','depth_type','unit','representation','alignment'):
            if run['model'].get(key) != context['run']['model'].get(key):
                raise ValueError(f'两模型不可比：{key}')
        if (run['model'].get('name') != 'vda' or run['model'].get('gt_used') or run['model'].get('alignment') != 'none'
                or run['model'].get('inference_mode') != 'offline_noncausal'
                or run['model'].get('windowing') != window_metadata(len(manifest['frames']))):
            raise ValueError('VDA必须为官方离线relative、无GT及正确窗口')
        runs['vda']=run;raw['vda']=[];bound.append(identity(directory/'vda/run.json'))
        for frame in run['frames']:
            path=directory/'vda'/frame['raw_path'];bound.append(identity(path))
            raw['vda'].append(np.load(path,mmap_mode='r',allow_pickle=False))
        check_evaluation(baseline,'1')
        bound.extend(record['baseline']['bindings'])
    for name in names:
        if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in raw[name]):
            raise ValueError('完整有限float32原始预测是正式评价前提；不删困难像素')
    # 全图就是背景诊断域；只在明确无人且场景静止时有科学意义。
    context['roi']=np.ones(context['shape'],bool) if record['clip']['type']=='fixed_static' else None
    return {'record':record,'directory':directory,'baseline_directory':baseline,'manifest':manifest,'manifest_path':context['manifest_path'],
            'shape':context['shape'],'flows':context['flows'],'context':context,'names':names,
            'raw':raw,'runs':runs,'bound_files':bound}


def measurements(sources, output=None, verification_dir=None):
    count=len(sources['manifest']['frames']);models={};pair_lists={};frame_lists={};artifacts=[]
    for name in sources['names']:
        dest=Path(output)/name if output is not None else None
        if dest is not None:dest.mkdir()
        context={**sources['context'],'raw':sources['raw'][name],'run':sources['runs'][name]}
        result,pairs,frames,saved=measure_single(context,dest,
            verification_dir=Path(verification_dir)/name if verification_dir is not None else None)
        result['flow']=result.pop('aggregate')
        result['reference']['region']='full_frame_complete_finite_reference'
        span=result['reference']['span']
        pixel_rows=[]
        for i,pair in enumerate(pairs,1):
            pixel=pixel_pair(sources['raw'][name][i-1],sources['raw'][name][i],span)
            pair.update({'pixel_'+k:v for k,v in pixel.items()});pixel_rows.append(pixel)
        result['pixel']={
            'included_pairs':len(pixel_rows) if span else 0,'total_pairs':len(pixel_rows),
            'pixels_per_pair':int(np.prod(sources['shape'])),'coverage':1.,
            'mean_pair_raw_mae':float(np.mean([p['raw_mae'] for p in pixel_rows])),
            'mean_pair_raw_rmse':float(np.mean([p['raw_rmse'] for p in pixel_rows])),
            'mean_pair_normalized_mae':float(np.mean([p['normalized_mae'] for p in pixel_rows])) if span else None,
            'mean_pair_normalized_rmse':float(np.mean([p['normalized_rmse'] for p in pixel_rows])) if span else None,
            'domain':'full_frame_same_pixel_coordinates; no_warp; no_FB_mask',
            'unavailable_reason':None if span else 'zero_reference_span'}
        changes=[abs(f['normalized_median_delta']) for f in frames if f['normalized_median_delta'] is not None]
        result['median_summary']['mean_abs_normalized_median_delta']=float(np.mean(changes)) if changes else None
        result['median_summary']['ratio_affine_limit']='原生比值受固定加性偏移影响；固定跨度变化不受该偏移影响，均不能分离真实运动'
        static=sources['record']['clip']['type']=='fixed_static'
        result['background_flicker']['domain']='full_frame' if static else None
        if not static:
            result['background_flicker']['unavailable_reason']='full_frame_contains_real_motion_or_camera_motion'
        result['scientific_protocol'].update(background_roi_role='no_manual_ROI; full_frame_background_only_for_fixed_static',
                                            pixel_domain='full_frame_no_flow_or_FB_mask',
                                            primary_diagnostic='pixel_and_background' if static else 'flow_and_median_proxy')
        models[name]=result;pair_lists[name]=pairs;frame_lists[name]=frames
        artifacts += [{**a,'path':name+'/'+a['path']} for a in saved]
    first=sources['names'][0]
    pairs=[];frames=[]
    for i in range(count-1):
        a=pair_lists[first][i]
        row={k:a[k] for k in ('previous_index','current_index','timestamp_seconds','dt_seconds','valid_pixels','coverage','quality_flag','exclusion_reason')}
        row['pixel_pixels']=a['pixel_pixels'];row['pixel_coverage']=a['pixel_coverage']
        for name in sources['names']:
            b=pair_lists[name][i]
            if b['valid_pixels'] != a['valid_pixels'] or b['coverage'] != a['coverage']:
                raise ValueError('两模型没有使用同一光流有效域')
            for field,original in [('flow_mae','normalized_mae'),('flow_rmse','normalized_rmse'),
                                   ('flow_raw_mae','raw_mae'),('flow_raw_rmse','raw_rmse'),
                                   ('pixel_mae','pixel_normalized_mae'),('pixel_rmse','pixel_normalized_rmse'),
                                   ('pixel_raw_mae','pixel_raw_mae'),('pixel_raw_rmse','pixel_raw_rmse')]:
                row[name+'_'+field]=b[original]
        if 'vda' in sources['names']:
            for kind in ('flow_mae','flow_rmse','pixel_mae','pixel_rmse'):
                row[kind+'_absolute_reduction']=improvement(row['dav2_'+kind],row['vda_'+kind])['absolute_reduction']
        pairs.append(row)
    for i in range(count):
        f=frame_lists[first][i];row={k:f[k] for k in ('index','source_index','timestamp_seconds')}
        for name in sources['names']:
            for key in ('raw_median','normalized_median_delta','median_ratio','abs_log_median_ratio','ratio_unavailable_reason'):
                row[name+'_'+key]=frame_lists[name][i][key]
        frames.append(row)
    directory=Path(output or verification_dir) if output is not None or verification_dir is not None else None
    if directory is not None and 'vda' in sources['names']:
        for i in range(1,count):
            relative=f'masks/pair_{i-1:06d}_{i:06d}.npy'
            if not np.array_equal(np.load(directory/'dav2'/relative),np.load(directory/'vda'/relative)):
                raise ValueError('两模型光流mask逐像素不同')
    reduced={}
    if 'vda' in sources['names']:
        base=read_json(sources['baseline_directory']/'part1/summary.json')
        _equivalent(models['dav2'],base['models']['dav2'],'Part 1B/2B DA baseline')
        reduced={k:improvement(models['dav2'][section][field],models['vda'][section][field]) for k,(section,field) in METRICS.items()}
    return {'models':models,'improvement':reduced},pairs,frames,artifacts


def analyze(directory, part):
    directory=Path(directory);part=str(part);sources=load_sources(directory,part)
    output=create_output_directory(directory/f'part{part}');status=output/'status.json'
    write_json(status,{'status':'running','started_at_utc':utc_now(),'command':sys.argv})
    try:
        scores,pairs,frames,artifacts=measurements(sources,output)
        for name,rows in [('pair_metrics.csv',pairs),('frame_metrics.csv',frames)]:
            _write_csv(output/name,rows);artifacts.append({**identity(output/name),'path':name})
        record={'schema_version':'clip_depth_analysis_v1','status':'complete','created_at_utc':utc_now(),'command':sys.argv,
                'part':part,'clip':sources['record']['clip'],'frames':len(frames),'pairs':len(pairs),
                'config':sources['record']['config'],'source_bindings':sources['bound_files'],'artifacts':artifacts,
                'timing':{n:sources['runs'][n]['timing'] for n in sources['names']},
                'model_metadata':{n:sources['runs'][n]['model'] for n in sources['names']},
                'flow_timing':read_json(sources['baseline_directory']/'flow/flow_manifest.json').get('timing'),**scores}
        identities={n:{'torch':sources['runs'][n].get('environment',{}).get('torch'),
                       'cuda_runtime':sources['runs'][n].get('environment',{}).get('cuda_runtime'),
                       'gpu_names':[g.get('name') for g in sources['runs'][n].get('environment',{}).get('visible_gpus',[])]}
                    for n in sources['names']}
        record['efficiency_comparability']={'recorded_identities':identities,
            'recorded_hardware_and_libraries_match':identities['dav2']==identities['vda'] if 'vda' in identities else None,
            'separate_invocations':True,'scope':'model_only; not_concurrent_load_controlled; offline_amortized_not_online_latency'}
        if sources['record']['config']['visualization'].get('generate_report',True):
            from .clip_report import generate
            record['report']=generate(output,record,sources,pairs,frames)
        write_json(output/'summary.json',record)
        write_json(status,{'status':'complete','summary':identity(output/'summary.json'),'finished_at_utc':utc_now()})
        return output/'summary.json'
    except BaseException as exc:
        write_json(status,{'status':'failed','error_type':type(exc).__name__,'error':str(exc),'finished_at_utc':utc_now()})
        raise


def check_evaluation(directory,part):
    directory=Path(directory);root=directory/f'part{part}'
    if str(part)=='2':load_comparison(directory)
    else:
        record,_=load_clip(directory)
        if record.get('part') != '1':raise ValueError('Part 1B需要自己的输出目录')
    status=read_json(root/'status.json')
    if status.get('status') != 'complete':raise ValueError('评价未完成，不能复用')
    check_identity(status['summary']);summary=read_json(root/'summary.json')
    if summary.get('status') != 'complete' or summary.get('part') != str(part):raise ValueError('评价身份不正确')
    for item in summary['source_bindings']:check_identity(item)
    for item in summary['artifacts']+summary.get('report',{}).get('assets',[]):
        check_identity({'path':root/item['path'],'sha256':item['sha256']})
    return summary


def verify(directory,part):
    directory=Path(directory);part=str(part);saved=check_evaluation(directory,part)
    sources=load_sources(directory,part);actual,pairs,frames,_=measurements(sources,verification_dir=directory/f'part{part}')
    _equivalent(saved['clip'],sources['record']['clip'],'clip identity')
    _equivalent(saved['config'],sources['record']['config'],'frozen configuration')
    _equivalent(saved['source_bindings'],sources['bound_files'],'source bindings')
    _equivalent(saved['flow_timing'],read_json(sources['baseline_directory']/'flow/flow_manifest.json').get('timing'),'flow timing')
    for key,value in actual.items():_equivalent(saved[key],value,key)
    _equivalent(_read_csv(directory/f'part{part}/pair_metrics.csv'),pairs,'pair CSV')
    _equivalent(_read_csv(directory/f'part{part}/frame_metrics.csv'),frames,'frame CSV')
    for name in sources['names']:
        _equivalent(saved['timing'][name],sources['runs'][name]['timing'],'timing')
        _equivalent(saved['model_metadata'][name],sources['runs'][name]['model'],'model metadata')
    record={'status':'verified','part':part,'summary':identity(directory/f'part{part}/summary.json'),
            'frames':len(frames),'pairs':len(pairs),'models_executed':False,'all_metrics_recomputed_from_raw':True,
            'masks_background_pixel_metrics_checked':True,'tolerance':{'relative':1e-6,'absolute':1e-8}}
    path=directory/f'verify{part}.json'
    if path.exists():_equivalent(read_json(path),record,'existing verification')
    else:write_json(path,record)
    return record


def summarize(root,part,output,clips=None):
    root=Path(root);part=str(part)
    directories=[root/c for c in clips] if clips else sorted(p for p in root.iterdir() if p.is_dir() and (p/f'part{part}/summary.json').exists())
    if not directories or len({p.resolve() for p in directories}) != len(directories):raise ValueError('没有已完成视频，或视频重复')
    rows=[];seen=set();sources=[];protocols={}
    names=('dav2',) if part=='1' else ('dav2','vda')
    for directory in directories:
        record,manifest=load_clip(directory);s=check_evaluation(directory,part);v=read_json(directory/f'verify{part}.json')
        if v.get('status') != 'verified' or v['summary'] != identity(directory/f'part{part}/summary.json'):
            raise ValueError('须先完成本视频的原始重算核验')
        if record['source_sha256'] in seen:raise ValueError('同一个源视频不能更名后重复计入汇总')
        seen.add(record['source_sha256']);condition=record['clip']['type']
        signature={k:record['config'][k] for k in ('input','runtime','inference','flow','evaluation','models','visualization')}
        if condition in protocols and protocols[condition] != signature:raise ValueError('不同科学协议不能合并为同一条件')
        protocols[condition]=signature
        row={'clip_id':record['clip']['clip_id'],'type':condition,'session':record['clip']['session'],
             'frames':s['frames'],'pairs':s['pairs'],'source_sha256':record['source_sha256']}
        for name in names:
            for key,(section,field) in METRICS.items():row[name+'_'+key]=s['models'][name][section][field]
        rows.append(row);sources.append(identity(directory/f'part{part}/summary.json'))
    groups=[]
    for condition in sorted({r['type'] for r in rows}):
        selected=[r for r in rows if r['type']==condition];means={};counts={}
        for name in names:
            for metric in METRICS:
                key=name+'_'+metric;values=[r[key] for r in selected if r[key] is not None]
                means[key]=float(np.mean(values)) if values else None;counts[key]=len(values)
        reduced={k:improvement(means['dav2_'+k],means['vda_'+k]) for k in METRICS} if part=='2' else {}
        groups.append({'type':condition,'clips':len(selected),'equal_clip_means':means,'evaluable_clips':counts,'improvement':reduced})
    output=create_output_directory(output);_write_csv(output/'clips.csv',rows)
    write_json(output/'summary.json',{'status':'complete','part':part,'created_at_utc':utc_now(),'groups':groups,
               'sources':sources,'clips_csv':identity(output/'clips.csv'),'models_executed':False,
               'aggregation':'equal_clip_within_type; N/A_not_zero; unknown_sessions_not_independent_evidence'})
    return {'status':'complete','clips':len(rows),'output':str(output)}
