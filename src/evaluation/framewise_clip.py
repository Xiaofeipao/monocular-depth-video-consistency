"""课程指定片段的DA-V2独立逐帧基线；与NYUv2对齐及VDA分开。"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import base64
import csv
import hashlib
import json
import sys
import time
from fractions import Fraction

import cv2
import numpy as np
import yaml

from src.video.common import PROJECT_ROOT as ROOT, read_json, write_json, sha256_file, utc_now, environment_info
from src.video.frames import _probe_video, validate_manifest
from src.evaluation.course_video_inputs import prepare as prepare_frames
from src.models.depth_anything_v2_adapter import DepthAnythingV2Adapter, _repository_identity, _runtime, network_input_shape
from src.metrics import video_diagnostics as diagnostic

DEFAULT_CONFIG=ROOT/'configs/part1/dav2_video.yaml'
SOURCES=['src/evaluation/framewise_clip.py','src/evaluation/course_video_inputs.py','src/metrics/video_diagnostics.py',
         'src/visualization/framewise_clip.py','src/models/depth_anything_v2_adapter.py',
         'src/video/common.py','src/video/frames.py','src/video/report.py','src/metrics/temporal.py',
         'pipelines/part1/dav2_video.py','scripts/part1/run_dav2_video.sh']


def absolute(path):return (ROOT/Path(path).expanduser()).resolve()
def identity(path):return {'path':str(Path(path).resolve()),'sha256':sha256_file(path)}
def checked(value):
    p=Path(value['path'])
    if sha256_file(p)!=value['sha256']:raise ValueError(f'文件SHA已改变：{p}')
    return p

def load_config(path=DEFAULT_CONFIG):
    c=yaml.safe_load(Path(path).read_text())
    if c.get('protocol')!='course_part1b_framewise_v1' or c['source']['kind']!='public':raise ValueError('只接受课程公开片段的逐帧协议')
    if c['input']['original_fps'] is not True or c['input']['requested_frames']!=100:raise ValueError('课程轨道要求原FPS、最多前100个真实帧')
    for value in (c['input']['max_long_edge'],c['inference']['input_size'],c['runtime']['torch_threads'],c['inference']['timing_repeats']):
        if type(value)!=int or value<1:raise ValueError('尺寸/线程/重复数必须为正整数')
    if c['inference']['warmup']<1:raise ValueError('至少预热一次')
    if c['runtime']['precision'] not in ('float16','float32'):raise ValueError('明确指定FP16/FP32')
    q=c['visualization']['percentiles']
    if len(q)!=2 or not 0<=q[0]<q[1]<=100 or c['visualization']['palette']!='TURBO':raise ValueError('非法固定显示协议')
    d=c['diagnostics']
    if d['fb_threshold_px']<0 or d['edge_threshold']<=0 or type(d['edge_tolerance_px'])!=int or d['edge_tolerance_px']<0:raise ValueError('非法诊断阈值')
    return c


def check_resources(c, *, require_device=True):
    video=absolute(c['source']['path']);checkpoint=absolute(c['model']['checkpoint'])
    if sha256_file(video)!=c['source']['sha256']:raise ValueError('源视频SHA不符；新素材使用新配置和空输出目录')
    if sha256_file(checkpoint)!=c['model']['sha256']:raise ValueError('DA-V2权重SHA不符')
    repo=_repository_identity(absolute(c['model']['repository']))
    if repo['upstream_git_sha']!=c['model']['commit'] or repo['repository_dirty']:raise ValueError('DA-V2官方代码版本或工作树不符')
    probe=_probe_video(video);pts=np.asarray(probe['timestamps']);interval=np.diff(pts)
    if len(pts)<2 or not np.all(interval>0):raise ValueError('至少两个顺序真实帧')
    fps=float(Fraction(probe['stream']['avg_frame_rate']))
    if not np.isfinite(fps) or fps<=0:raise ValueError('源视频没有有效原始FPS')
    # 本片为CFR；拒绝把VFR通过固定网格悄悄变成不同评估帧。
    if not np.allclose(interval,1/fps,rtol=1e-3,atol=2e-6):raise ValueError('检测到VFR，需保留全部PTS的专门展示协议，不能自动重采样')
    if require_device:_runtime(c['runtime']['device'],c['runtime']['precision'])
    return {'video':identity(video),'checkpoint':identity(checkpoint),'repository':repo,
            'source_frames':len(pts),'source_fps':fps,'source_hw':[int(probe['stream']['height']),int(probe['stream']['width'])],
            'selected_frames':min(100,len(pts)),'required_frames':100,'has_100_real_frames':len(pts)>=100,
            'environment':environment_info(),'models_executed':False}


def source_snapshot():return {p:sha256_file(ROOT/p) for p in SOURCES}

def validate_run(directory):
    directory=Path(directory).resolve();run=read_json(directory/'run.json')
    if run.get('schema')!='course_dav2_framewise_v1' or run.get('status')!='complete':raise ValueError('没有完整逐帧run，不能复用')
    protocol=read_json(checked(run['protocol']));checked(protocol['config_source'])
    if run['config']!=protocol['config']:raise ValueError('run与协议配置不同')
    checked(protocol['source']);checked(protocol['checkpoint'])
    for p,h in protocol['scientific_sources'].items():
        if sha256_file(ROOT/p)!=h:raise ValueError(f'当前科学源码改变：{p}')
        if hashlib.sha256(base64.b64decode(protocol['source_archive_base64'][p])).hexdigest()!=h:raise ValueError('源码归档改变')
    mp=checked(protocol['manifest']);manifest=validate_manifest(mp,verify_hashes=True)
    if manifest['source_kind']!='public' or manifest.get('roi_path') is not None:raise ValueError('必须为公开片段、无ROI')
    if manifest['source']['sha256']!=protocol['source']['sha256']:raise ValueError('源视频身份不一致')
    if manifest['resize']['max_long_edge']!=run['config']['input']['max_long_edge']:raise ValueError('缓存尺寸设置不同')
    count=protocol['frame_scope']['actual_frames'];shape=[manifest['height'],manifest['width']]
    if len(manifest['frames'])!=count or len(run['frames'])!=count or count!=min(100,manifest['source']['frame_count']):raise ValueError('原生帧数不完整')
    if run['frame_scope']!=protocol['frame_scope'] or manifest['fps']!=protocol['frame_scope']['original_fps']:raise ValueError('帧数/原FPS记录不一致')
    if manifest['selection']['selected_source_indices']!=list(range(count)):raise ValueError('不是原始前N帧，不能采样/重复/补帧')
    model=run['model']
    if (model.get('depth_type')!='relative' or model.get('unit')!='arbitrary' or model.get('alignment')!='none'
            or model.get('gt_used') or model.get('temporal_context') is not False or model.get('encoder')!='vits'
            or model.get('precision')!=run['config']['runtime']['precision']):raise ValueError('不是无GT的独立逐帧Small相对深度')
    if model.get('network_input_hw')!=list(network_input_shape(*shape,run['config']['inference']['input_size'])) or model.get('actual_output_hw')!=shape:
        raise ValueError('实际网络/输出分辨率不一致')
    raw=[];rgb=[]
    for i,(item,frame) in enumerate(zip(run['frames'],manifest['frames'])):
        if item['index']!=i or item['source_index']!=i or frame['source_index']!=i or item['timestamp_seconds']!=frame['timestamp_seconds']:
            raise ValueError('输出顺序/PTS不是同一真实输入')
        p=directory/item['raw_path']
        if p.resolve().parent!=(directory/'raw').resolve() or sha256_file(p)!=item['raw_sha256']:raise ValueError('原始预测路径或SHA改变')
        a=np.load(p,allow_pickle=False)
        if a.dtype!=np.float32 or list(a.shape)!=shape or not np.isfinite(a).all() or np.any(a<0):raise ValueError('预测须为完整非负有限float32')
        raw.append(a);rgb.append(cv2.cvtColor(cv2.imread(str(mp.parent/frame['rgb_path'])),cv2.COLOR_BGR2RGB))
    return run,protocol,manifest,np.stack(raw),rgb


def predict(config_path=DEFAULT_CONFIG, *, output=None, cache=None):
    config_path=absolute(config_path);c=load_config(config_path);resources=check_resources(c)
    directory=absolute(output or c['output']);cache=absolute(cache or c['input']['cache'])
    if directory.exists() or cache.exists():raise FileExistsError('不覆盖已有产物/缓存；完整运行可verify，重新实验请指定新的输出和缓存目录')
    directory.mkdir(parents=True);(directory/'raw').mkdir()
    record={'schema':'course_dav2_framewise_v1','status':'running','created_at_utc':utc_now(),'command':sys.argv,
            'config':deepcopy(c),'frames':[],'resources':resources}
    write_json(directory/'run.json',record);started=time.perf_counter()
    try:
        mp=prepare_frames(absolute(c['source']['path']),cache,fps=resources['source_fps'],
            max_long_edge=c['input']['max_long_edge'],requested_frames=100,source_url=c['source']['url'])
        manifest=validate_manifest(mp);count=len(manifest['frames'])
        if manifest['selection']['selected_source_indices']!=list(range(resources['selected_frames'])):raise ValueError('选帧不是连续原始前N帧')
        scope={'requested_frames':100,'actual_frames':count,'source_available_frames':resources['source_frames'],
               'original_fps':resources['source_fps'],'no_frame_duplication':True,'requirement_100_frames_satisfied':count==100,
               'short_source_status':'needs_course_confirmation' if count<100 else None,
               'interpretation':'actual first N original frames; no looping or interpolation'}
        protocol={'schema':'course_part1b_protocol_v1','created_at_utc':utc_now(),'command':sys.argv,'config':deepcopy(c),
                  'config_source':identity(config_path),'source':resources['video'],'checkpoint':resources['checkpoint'],
                  'manifest':identity(mp),'frame_scope':scope,'scientific_sources':source_snapshot(),
                  'raw_contract':{'depth_type':'relative','unit':'arbitrary','alignment':'none','gt_used':False,'temporal_context':False}}
        protocol['source_archive_base64']={p:base64.b64encode((ROOT/p).read_bytes()).decode() for p in SOURCES}
        write_json(directory/'protocol.json',protocol);record['protocol']=identity(directory/'protocol.json')
        import torch
        torch.set_num_threads(c['runtime']['torch_threads']);torch.manual_seed(c['runtime']['seed'])
        if c['runtime']['device'].startswith('cuda'):torch.cuda.set_device(c['runtime']['device'])
        start_load=time.perf_counter()
        model=DepthAnythingV2Adapter(repository=absolute(c['model']['repository']),checkpoint=absolute(c['model']['checkpoint']),
                 device=c['runtime']['device'],precision=c['runtime']['precision'],input_size=c['inference']['input_size'])
        load_seconds=time.perf_counter()-start_load
        def sync():
            if model.device.type=='cuda':torch.cuda.synchronize(model.device)
        first=cv2.cvtColor(cv2.imread(str(mp.parent/manifest['frames'][0]['rgb_path'])),cv2.COLOR_BGR2RGB)
        for _ in range(c['inference']['warmup']):model.infer(first)
        sync()
        if model.device.type=='cuda':torch.cuda.reset_peak_memory_stats(model.device)
        timings=[]
        for frame in manifest['frames']:
            rgb=cv2.cvtColor(cv2.imread(str(mp.parent/frame['rgb_path'])),cv2.COLOR_BGR2RGB);times=[]
            for repeat in range(c['inference']['timing_repeats']):
                sync();t=time.perf_counter();a=model.infer(rgb);sync();times.append(time.perf_counter()-t)
                if repeat==0:saved=a.copy()
            relative=f'raw/frame_{frame["index"]:06d}.npy';np.save(directory/relative,saved.astype(np.float32))
            record['frames'].append({'index':frame['index'],'source_index':frame['source_index'],'timestamp_seconds':frame['timestamp_seconds'],
                'raw_path':relative,'raw_sha256':sha256_file(directory/relative),'mean_seconds':float(np.mean(times)),
                'timing_repeats_seconds':times,'shape':list(saved.shape),'dtype':'float32'})
            timings.append(float(np.mean(times)))
            write_json(directory/'run.json',record)
        model.metadata['actual_output_hw']=[manifest['height'],manifest['width']]
        record.update(status='complete',finished_at_utc=utc_now(),model=deepcopy(model.metadata),frame_scope=scope,
                      environment=environment_info(),timing={'model_load_seconds':load_seconds,'warmup':c['inference']['warmup'],
                      'timing_repeats':c['inference']['timing_repeats'],'mean_seconds_per_frame':float(np.mean(timings)),
                      'median_seconds_per_frame':float(np.median(timings)),'p95_seconds_per_frame':float(np.percentile(timings,95)),
                      'fps':1/float(np.mean(timings)),'pipeline_seconds':time.perf_counter()-started,
                      'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated(model.device)) if model.device.type=='cuda' else None,
                      'peak_gpu_reserved_bytes':int(torch.cuda.max_memory_reserved(model.device)) if model.device.type=='cuda' else None,
                      'scope':'model preprocessing, transfer, forward, output resize and CPU transfer; excludes decode/save/report; mean over repeats'})
        write_json(directory/'run.json',record);return directory
    except BaseException as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc),finished_at_utc=utc_now())
        write_json(directory/'run.json',record);raise


def measure(raw,rgb,manifest,c):
    ref=diagnostic.reference(raw,c['visualization']['percentiles']);frames=[];pairs=[]
    for i,a in enumerate(raw):
        frames.append({'index':i,'timestamp_seconds':manifest['frames'][i]['timestamp_seconds'],'minimum':float(a.min()),
                       'maximum':float(a.max()),'median':float(np.median(a)),
                       'edge_fraction':float(diagnostic.edge_map(a,ref['span'],c['diagnostics']['edge_threshold']).mean())})
        if i:
            scores=diagnostic.pair(raw[i-1],a,rgb[i-1],rgb[i],ref['span'],c['diagnostics'])
            pairs.append({'previous_index':i-1,'current_index':i,'timestamp_seconds':frames[-1]['timestamp_seconds'],**scores})
    aggregate={}
    for key in ('pixel_mae_normalized','pixel_rmse_normalized','flow_mae_normalized','flow_rmse_normalized','flow_coverage','edge_instability'):
        values=[p[key] for p in pairs if p.get(key) is not None];aggregate[key]=float(np.mean(values)) if values else None
    cases=diagnostic.select_cases(pairs,c['diagnostics']['case_separation_frames'])
    affine=raw.astype(np.float64)*2+5
    error=float(np.max(np.abs((raw-ref['low'])/ref['span']-(affine-(2*ref['low']+5))/(2*ref['span'])))) if ref['span'] else None
    result={'reference':ref,'aggregate':aggregate,'case_candidates':cases,'frames':len(raw),'pairs':len(pairs),
            'scale_convention':{'unit':'arbitrary','meters_available':False,'gt_used':False,'test_alignment':'none',
                'positive_affine_demo':{'scale':2,'shift':5,'max_normalized_difference':error,'stored_as_prediction':False},
                'interpretation':'positive affine values can preserve the same relative display; no metric accuracy can be measured without GT'},
            'limits':{'pixel':'same coordinates include real camera/object motion',
                      'flow':'CPU Farneback, FB mask; approximate correspondence; neither temporal GT nor pure flicker',
                      'edge':'symmetric unmatched fraction after approximate flow and tolerance; not GT boundary accuracy',
                      'median':'scene composition and actual motion confound it; not metric scale drift'},
            'aggregation':'equal adjacent-pair means; unavailable values remain null'}
    return result,frames,pairs


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyze(directory, *, output=None):
    directory=Path(directory).resolve();run,protocol,manifest,raw,rgb=validate_run(directory)
    output=Path(output).resolve() if output else directory/'analysis'
    output.mkdir(parents=True,exist_ok=False);write_json(output/'status.json',{'status':'running','command':sys.argv})
    start=time.perf_counter()
    try:
        scores,frames,pairs=measure(raw,rgb,manifest,run['config'])
        write_csv(output/'frame_diagnostics.csv',frames);write_csv(output/'pair_diagnostics.csv',pairs)
        summary={'schema':'course_part1b_analysis_v1','status':'complete','created_at_utc':utc_now(),'command':sys.argv,
                 'run':identity(directory/'run.json'),'frame_scope':protocol['frame_scope'],
                 'model':run['model'],'timing':run['timing'],**scores}
        from src.visualization.framewise_clip import report
        assets=report(output,raw,rgb,frames,pairs,summary,run['config'],manifest['fps'])
        summary['assets']=[{'path':str(p.relative_to(output)),'sha256':sha256_file(p)} for p in assets]
        summary['analysis_seconds']=time.perf_counter()-start
        write_json(output/'summary.json',summary)
        handoff={'role':'future_Part_2B_read_only_input','manifest':protocol['manifest'],'dav2_run':identity(directory/'run.json'),
                 'raw_contract':protocol['raw_contract'],'frame_scope':protocol['frame_scope'],
                 'input_hw':[manifest['height'],manifest['width']],'network_input_hw':run['model']['network_input_hw'],
                 'precision':run['model']['precision'],'palette':'TURBO','dav2_fixed_normalization':scores['reference'],
                 'requirements':'same actual RGB frames/PTS and output canvas; each method fixed own whole-clip normalization; relative only',
                 'vda_executed':False}
        write_json(output/'handoff_2b.json',handoff)
        write_json(output/'status.json',{'status':'complete','summary':identity(output/'summary.json')})
        return output
    except BaseException as exc:
        write_json(output/'status.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)});raise


def verify(directory, *, analysis=None):
    directory=Path(directory).resolve();run,protocol,manifest,raw,rgb=validate_run(directory)
    output=Path(analysis).resolve() if analysis else directory/'analysis';status=read_json(output/'status.json')
    if status['status']!='complete':raise ValueError('分析未完成')
    summary=read_json(checked(status['summary']))
    if summary['run']!=identity(directory/'run.json'):raise ValueError('分析绑定不同的raw run')
    for item in summary['assets']:
        p=output/item['path']
        if not p.resolve().is_relative_to(output) or sha256_file(p)!=item['sha256']:raise ValueError('报告附件改变')
    actual,frames,pairs=measure(raw,rgb,manifest,run['config'])
    from src.video.evaluation import _equivalent,_read_csv
    for k,v in actual.items():_equivalent(summary[k],v,k)
    _equivalent(_read_csv(output/'frame_diagnostics.csv'),frames,'frame CSV')
    _equivalent(_read_csv(output/'pair_diagnostics.csv'),pairs,'pair CSV')
    _equivalent(summary['model'],run['model'],'model metadata');_equivalent(summary['timing'],run['timing'],'timing')
    _equivalent(summary['frame_scope'],protocol['frame_scope'],'frame scope')
    handoff=read_json(output/'handoff_2b.json')
    if handoff['manifest']!=protocol['manifest'] or handoff['dav2_run']!=identity(directory/'run.json') or handoff['dav2_fixed_normalization']!=actual['reference']:
        raise ValueError('2B输入交接记录改变')
    return {'status':'verified','run':identity(directory/'run.json'),'analysis':identity(output/'summary.json'),
            'frames':len(frames),'pairs':len(pairs),'models_executed':False,'raw_recomputed':True,
            'required_100_frames_satisfied':len(frames)==100,'tolerance':{'rtol':1e-6,'atol':1e-8}}
