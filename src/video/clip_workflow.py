"""以单段视频为单位的依赖调度；复用官方推理，不训练。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import base64
import hashlib
import json
import re
import sys
import uuid

import numpy as np
import yaml

from .common import (PROJECT_ROOT as ROOT, read_json, write_json, sha256_file,
                     utc_now, source_fingerprints, environment_info)
from .runner import (load_config, freeze_protocol, load_protocol, check_resources,
                     infer_run, verify_run, cache_flow)
from .frames import prepare_frames

DEFAULT_CONFIG = ROOT / 'configs/video/self_recorded.yaml'
DEFAULT_REGISTRY = ROOT / 'configs/video/clips.yaml'
OUTPUT_ROOT = ROOT / 'outputs/part1/video'
PART2_OUTPUT_ROOT = ROOT / 'outputs/part2/video'
CACHE_ROOT = ROOT / 'data/video_frames/self_recorded'
TYPES = ('fixed_static', 'fixed_dynamic', 'moving')
DEPENDENCIES = {
    'prepare': [], 'dav2': ['prepare'], 'flow': ['prepare'], 'vda': ['prepare'],
    'analyze1': ['dav2', 'flow'], 'verify1': ['analyze1'],
    'analyze2': ['vda'], 'verify2': ['analyze2'],
}


def absolute(path):
    return (ROOT / Path(path).expanduser()).resolve()


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path)}


def check_identity(item):
    path = Path(item['path'])
    if sha256_file(path) != item['sha256']:
        raise ValueError(f'文件已改变：{path}')
    return path


def seal(value):
    return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k != 'content_sha256'},
                                    ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def resolve_clip(clip_id, *, registry=DEFAULT_REGISTRY, video=None, video_type=None, session=None, output_root=OUTPUT_ROOT):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', clip_id):
        raise ValueError('clip ID只允许英文字母、数字、下划线和短横线，不能是路径')
    catalog = yaml.safe_load(absolute(registry).read_text())
    if catalog.get('schema_version') != 'self_recorded_clips_v1':
        raise ValueError('视频登记格式不正确')
    value = deepcopy(catalog.get('clips', {}).get(clip_id, {}))
    if not value and video is None and (absolute(output_root)/clip_id/'clip.json').is_file():
        frozen,_=load_clip(absolute(output_root)/clip_id)
        value=deepcopy(frozen['clip'])
    if video is not None:
        value['video'] = str(video)
    if video_type is not None:
        value['type'] = video_type
    if session is not None:
        value['session'] = session
    if not value.get('video') or value.get('type') not in TYPES:
        raise ValueError('新视频需要 --video 路径和 --type fixed_static/fixed_dynamic/moving，或先写入登记文件')
    if value.get('session') is not None and (not isinstance(value['session'], str) or not value['session'].strip()):
        raise ValueError('session必须为非空字符串或null，未知时不猜测')
    value.update(clip_id=clip_id, video=str(absolute(value['video'])))
    value.setdefault('session', None)
    value.setdefault('description', '用户明确登记的自录视频')
    if not Path(value['video']).is_file():
        raise FileNotFoundError(value['video'])
    return value


def task_plan(part, task='all', *, dependencies=True):
    if part not in ('1', '2'):
        raise ValueError('part只能是1或2，两个部分分别运行')
    if task == 'all' or task == 'verify':
        target = 'verify1' if part == '1' else 'verify2'
    elif task == 'analyze':
        target = 'analyze1' if part == '1' else 'analyze2'
    elif task in ('prepare', 'dav2', 'flow', 'vda'):
        target = task
    else:
        raise ValueError('未知任务')
    if (task == 'vda' and part == '1') or (task in ('dav2','flow') and part == '2'):
        raise ValueError('该任务不属于本部分；DA-V2与光流由Part 1B生成，VDA由Part 2B生成')
    result = []
    def visit(stage):
        if stage in result:
            return
        if dependencies:
            for parent in DEPENDENCIES[stage]:
                visit(parent)
        result.append(stage)
    visit(target)
    return result


def prepare_clip(clip, directory, cache, config, config_path):
    directory, cache = Path(directory), Path(cache)
    if directory.exists() or cache.exists():
        raise FileExistsError('拒绝覆盖已有视频目录或缓存；检查失败记录，或使用新ID/输出根目录')
    directory.mkdir(parents=True)
    try:
        check_resources(config)
        source_sha = sha256_file(clip['video'])
        provenance = {'recording_session': clip['session'], 'description': clip['description'],
                      'video_type': clip['type'], 'session_status': 'provided' if clip['session'] else 'unknown',
                      'camera_model': None, 'source': 'user_self_recorded'}
        manifest = prepare_frames(Path(clip['video']), cache, clip_id=clip['clip_id'], source_kind='self_recorded',
                                  camera_motion='moving' if clip['type'] == 'moving' else 'fixed', split='test',
                                  target_fps=config['input']['target_fps'], max_long_edge=config['input']['max_long_edge'],
                                  provenance=provenance)
        protocol = freeze_protocol(config, manifest, directory/'protocol.json', config_path=Path(config_path))
        sources = source_fingerprints()
        for relative in ('scripts/part1/video.sh', 'scripts/part2/video.sh',
                         'pipelines/part1/video_clip.py', 'pipelines/part2/video_clip.py'):
            sources[relative] = sha256_file(ROOT/relative)
        record = {'schema_version':'single_clip_workflow_v1', 'status':'ready', 'created_at_utc':utc_now(),
                  'command':sys.argv, 'part':'1', 'clip':clip, 'source_sha256':source_sha, 'manifest':identity(manifest),
                  'protocol':identity(protocol), 'config':deepcopy(config), 'config_source':identity(config_path),
                  'background_policy':'full_frame_only_if_fixed_static', 'pixel_policy':'all_pixels_no_flow_mask',
                  'project_source_sha256':sources,
                  'project_source_archive_base64':{k:base64.b64encode((ROOT/k).read_bytes()).decode('ascii') for k in sources}}
        record['content_sha256'] = seal(record)
        write_json(directory/'clip.json', record)
        return record
    except BaseException as exc:
        write_json(directory/'prepare_failure.json', {'status':'failed','time_utc':utc_now(),
                                                     'error_type':type(exc).__name__,'error':str(exc),'command':sys.argv})
        raise


def load_clip(directory):
    directory = Path(directory).resolve()
    value = read_json(directory/'clip.json')
    if value.get('schema_version') != 'single_clip_workflow_v1' or value.get('status') != 'ready' or seal(value) != value['content_sha256']:
        raise ValueError('视频登记未准备完成或已改变；不覆盖失败记录')
    if value['clip']['type'] not in TYPES or value['clip']['clip_id'] != directory.name:
        raise ValueError('视频类型/目录身份不一致')
    if sha256_file(value['clip']['video']) != value['source_sha256']:
        raise ValueError('原视频已替换，应使用新视频ID，不复用已有预测')
    for relative,digest in value['project_source_sha256'].items():
        if sha256_file(ROOT/relative) != digest or hashlib.sha256(base64.b64decode(value['project_source_archive_base64'][relative])).hexdigest() != digest:
            raise ValueError(f'冻结代码已改变：{relative}')
    check_identity(value['config_source'])
    protocol, manifest, mp = load_protocol(check_identity(value['protocol']))
    if identity(mp) != value['manifest'] or protocol['configuration'] != value['config']:
        raise ValueError('视频输入/协议身份不一致')
    if manifest.get('roi_path') is not None or manifest['source']['sha256'] != value['source_sha256']:
        raise ValueError('当前流程使用全图，不接收矩形ROI或不同源视频')
    motion='moving' if value['clip']['type']=='moving' else 'fixed'
    provenance=manifest['source'].get('provenance',{})
    if (manifest['clip_id'] != value['clip']['clip_id'] or manifest['source_kind'] != 'self_recorded'
            or manifest['split'] != 'test' or manifest['camera_motion'] != motion
            or provenance.get('video_type') != value['clip']['type']
            or provenance.get('recording_session') != value['clip']['session']
            or Path(manifest['source']['path']).resolve() != Path(value['clip']['video']).resolve()):
        raise ValueError('视频类型/来源/session与冻结帧清单不一致')
    return value, manifest


def completed_baseline(directory):
    """只读核验1B输入；不调用1B调度器，也不写入其核验或命令记录。"""
    from .clip_analysis import check_evaluation
    directory=Path(directory).resolve()
    if not (directory/'clip.json').is_file():
        raise FileNotFoundError(f'请先通过Part 1B入口完成该视频：{directory}')
    record,manifest=load_clip(directory)
    if record.get('part') != '1':
        raise ValueError('Part 2B必须引用独立完成的Part 1B目录')
    check_evaluation(directory,'1')
    verification=read_json(directory/'verify1.json')
    if verification.get('status') != 'verified' or verification['summary'] != identity(directory/'part1/summary.json'):
        raise ValueError('请先通过Part 1B入口完成本视频及raw重算核验')
    verify_run(directory/'dav2');verify_flow(directory)
    return record,manifest


def load_comparison(directory):
    record,manifest=load_clip(directory)
    if record.get('part') != '2':
        raise ValueError('这里需要Part 2B的独立输出目录')
    baseline=record['baseline']
    for item in baseline['bindings']:check_identity(item)
    parent,_=completed_baseline(baseline['directory'])
    if (parent['clip'] != record['clip'] or parent['config'] != record['config']
            or parent['manifest'] != record['manifest']):
        raise ValueError('Part 2B与引用的Part 1B输入/类型/科学配置不一致')
    return record,manifest,Path(baseline['directory'])


def prepare_comparison(baseline, directory, config, config_path):
    """独立冻结2B协议；RGB、DA与flow保持在1B，通过SHA引用。"""
    parent,_=completed_baseline(baseline)
    if config != parent['config']:
        raise ValueError('比较必须使用与Part 1B相同的科学配置')
    directory=Path(directory)
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True)
    try:
        protocol=freeze_protocol(config,Path(parent['manifest']['path']),directory/'protocol.json',config_path=config_path)
        record=deepcopy(parent)
        record.update(part='2',created_at_utc=utc_now(),command=sys.argv,
                      protocol=identity(protocol),config_source=identity(config_path),
                      baseline={'directory':str(Path(baseline).resolve()),'access':'read_only',
                                'bindings':[identity(Path(baseline)/p) for p in
                                            ('clip.json','protocol.json','dav2/run.json','flow/flow_manifest.json',
                                             'part1/summary.json','verify1.json')]})
        record['content_sha256']=seal(record)
        write_json(directory/'clip.json',record)
        return record
    except BaseException as exc:
        write_json(directory/'prepare_failure.json',{'status':'failed','time_utc':utc_now(),
                   'error_type':type(exc).__name__,'error':str(exc),'command':sys.argv})
        raise


def verify_flow(directory):
    directory=Path(directory); record,manifest=load_clip(directory)
    flow=read_json(directory/'flow/flow_manifest.json')
    if (flow.get('status') != 'complete' or flow['manifest_sha256'] != record['manifest']['sha256']
            or flow['protocol_sha256'] != record['protocol']['sha256'] or flow['frames_count'] != len(manifest['frames'])
            or flow['shape'] != [manifest['height'],manifest['width']]):
        raise ValueError('光流身份或完成状态不正确')
    pairs=flow['pairs']
    expected=record['config']['flow']
    for key in ('backend','iterations','max_long_edge'):
        if flow['flow'].get(key) != expected.get(key):
            raise ValueError(f'光流设置已改变：{key}')
    if expected['backend']=='raft_small' and flow['flow'].get('checkpoint_sha256') != expected['sha256']:
        raise ValueError('RAFT权重身份不一致')
    if [(p['previous_index'],p['current_index']) for p in pairs] != [(i-1,i) for i in range(1,len(manifest['frames']))]:
        raise ValueError('光流帧对缺失或顺序不正确')
    for pair in pairs:
        for direction in ('forward','backward'):
            path=check_identity({'path':directory/'flow'/pair[direction+'_path'],'sha256':pair[direction+'_sha256']})
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            if array.dtype != np.float32 or array.shape != (*flow['shape'],2) or not np.isfinite(array).all():
                raise ValueError('光流数值、形状或类型错误')
    return flow


def run_clip(clip, *, part='1', task='all', config_path=DEFAULT_CONFIG,
             output_root=OUTPUT_ROOT, cache_root=CACHE_ROOT, dependencies=True):
    if part != '1':raise ValueError('Part 1B调度器只运行1B；2B使用独立入口')
    config_path=absolute(config_path); config=load_config(config_path)
    directory=absolute(output_root)/clip['clip_id']; cache=absolute(cache_root)/clip['clip_id']
    stages=task_plan(part,task,dependencies=dependencies)
    if directory.exists():
        saved,_=load_clip(directory)
        if saved.get('part') != '1' or saved['clip'] != clip or saved['config'] != config or Path(saved['manifest']['path']).parent != cache:
            raise ValueError('请求与已冻结视频身份/配置/缓存位置不同；请使用新ID或新输出和缓存根目录')
    executed=[]; skipped=[]
    command={'status':'running','started_at_utc':utc_now(),'argv':sys.argv,'part':part,'task':task,
             'planned_stages':stages,'executed':executed,'skipped':skipped,
             'model_stages_attempted':[],'model_stages_completed':[]}
    log_path=directory/'commands'/f'{uuid.uuid4().hex}.json'
    try:
        for stage in stages:
            print(f'[{clip["clip_id"]}] {stage}', flush=True)
            if stage=='prepare':
                if not directory.exists():
                    prepare_clip(clip,directory,cache,config,config_path);executed.append(stage)
                else:
                    load_clip(directory);skipped.append(stage)
            elif stage in ('dav2','vda'):
                load_clip(directory)
                if (directory/stage).exists():
                    verify_run(directory/stage);skipped.append(stage)
                else:
                    command['model_stages_attempted'].append(stage)
                    infer_run(directory/'protocol.json',stage,directory/stage);executed.append(stage);command['model_stages_completed'].append(stage)
            elif stage=='flow':
                if (directory/'flow').exists():
                    verify_flow(directory);skipped.append(stage)
                else:
                    load_clip(directory);command['model_stages_attempted'].append('flow')
                    cache_flow(directory/'protocol.json',directory/'flow');executed.append(stage);command['model_stages_completed'].append('flow')
            elif stage.startswith('analyze'):
                from .clip_analysis import analyze, check_evaluation
                number=stage[-1]
                if (directory/f'part{number}').exists():
                    check_evaluation(directory,number);skipped.append(stage)
                else:
                    analyze(directory,number);executed.append(stage)
            else:
                from .clip_analysis import verify
                verify(directory,stage[-1]);executed.append(stage)
            if directory.exists():
                write_json(log_path,command)
        command.update(status='complete',finished_at_utc=utc_now())
        write_json(log_path,command)
        return {'status':'complete','clip_id':clip['clip_id'],'type':clip['type'],'directory':str(directory),
                'executed':executed,'skipped':skipped,'command_record':str(log_path)}
    except BaseException as exc:
        # A failed prepare may already have written its own immutable failure.
        # A task-only invocation without preparation does not create a bogus clip.
        if directory.exists():
            command.update(status='failed',error_type=type(exc).__name__,error=str(exc),finished_at_utc=utc_now())
            write_json(log_path,command)
        raise


def run_comparison(clip_id, *, task='all', config_path=DEFAULT_CONFIG, part1_root=OUTPUT_ROOT,
                   output_root=PART2_OUTPUT_ROOT, dependencies=True):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',clip_id):raise ValueError('无效视频ID')
    parent_root=absolute(part1_root);child_root=absolute(output_root)
    baseline=parent_root/clip_id;directory=child_root/clip_id
    if child_root==parent_root or parent_root in child_root.parents or child_root in parent_root.parents:
        raise ValueError('Part 1B和Part 2B输出目录必须分开，不能互相包含')
    # 任何2B阶段都先确认1B已完整完成；缺失时不创建假产物、不启动模型。
    parent,_=completed_baseline(baseline)
    config_path=absolute(config_path);config=load_config(config_path)
    if config != parent['config']:raise ValueError('Part 2B配置须与引用的Part 1B一致')
    if directory.exists():
        saved,_,bound_parent=load_comparison(directory)
        if bound_parent != baseline:raise ValueError('续跑不能更换Part 1B来源')
    stages=task_plan('2',task,dependencies=dependencies)
    command={'status':'running','started_at_utc':utc_now(),'argv':sys.argv,'part':'2','task':task,
             'baseline_directory':str(baseline),'planned_stages':stages,'executed':[],'skipped':[],
             'model_stages_attempted':[],'model_stages_completed':[]}
    log_path=directory/'commands'/f'{uuid.uuid4().hex}.json'
    try:
        for stage in stages:
            print(f'[{clip_id}] {stage}',flush=True)
            if stage=='prepare':
                if directory.exists():load_comparison(directory);command['skipped'].append(stage)
                else:prepare_comparison(baseline,directory,config,config_path);command['executed'].append(stage)
            elif stage=='vda':
                load_comparison(directory)
                if (directory/'vda').exists():verify_run(directory/'vda');command['skipped'].append(stage)
                else:
                    command['model_stages_attempted'].append('vda')
                    infer_run(directory/'protocol.json','vda',directory/'vda')
                    command['model_stages_completed'].append('vda');command['executed'].append(stage)
            elif stage=='analyze2':
                from .clip_analysis import analyze,check_evaluation
                if (directory/'part2').exists():check_evaluation(directory,'2');command['skipped'].append(stage)
                else:analyze(directory,'2');command['executed'].append(stage)
            elif stage=='verify2':
                from .clip_analysis import verify
                verify(directory,'2');command['executed'].append(stage)
            if directory.exists():write_json(log_path,command)
        command.update(status='complete',finished_at_utc=utc_now());write_json(log_path,command)
        return {'status':'complete','clip_id':clip_id,'directory':str(directory),'baseline_directory':str(baseline),
                'executed':command['executed'],'skipped':command['skipped'],'command_record':str(log_path)}
    except BaseException as exc:
        if directory.exists():
            command.update(status='failed',error_type=type(exc).__name__,error=str(exc),finished_at_utc=utc_now())
            write_json(log_path,command)
        raise


def check_local(config_path=DEFAULT_CONFIG, *, part='1'):
    config=load_config(absolute(config_path));resources=check_resources(config);env=environment_info()
    requested=config['runtime']['device']
    ready=env['cuda_available'] if requested.startswith('cuda') else requested=='cpu'
    if requested.startswith('cuda') and ready:
        index=int(requested.split(':')[1]) if ':' in requested else 0
        ready=index < len(env.get('visible_gpus',[]))
    return {'status':'ready' if ready else 'blocked','ready_for_requested_device':ready,
            'part':part,'requested_device':requested,'resources':resources,'environment':env,'models_executed':False}
