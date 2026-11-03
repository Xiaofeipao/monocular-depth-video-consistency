"""课程2B离线相对深度；独立运行VDA，再只读比较同帧DA/VDA。"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import base64
import hashlib
import sys
import time

import cv2
import numpy as np
import yaml

from src.evaluation import framewise_clip as framewise
from src.evaluation.course_video_inputs import prepare as prepare_frames
from src.models.depth_anything_v2_adapter import _repository_identity, _runtime, network_input_shape
from src.models.video_depth_anything_adapter import VideoDepthAnythingAdapter, window_metadata
from src.video.common import PROJECT_ROOT as ROOT, read_json, write_json, sha256_file, utc_now, environment_info
from src.video.frames import _probe_video, validate_manifest
from src.video.evaluation import _equivalent, _read_csv

DEFAULT_CONFIG = ROOT / 'configs/part2/vda_video.yaml'
SOURCES = sorted(set(framewise.SOURCES + [
    'src/evaluation/temporal_clip.py', 'src/visualization/temporal_clip.py',
    'src/models/video_depth_anything_adapter.py', 'src/video/evaluation.py',
    'pipelines/part2/vda_video.py', 'scripts/part2/run_vda_video.sh',
]))
METRICS = ('pixel_mae_normalized', 'pixel_rmse_normalized', 'flow_mae_normalized',
           'flow_rmse_normalized', 'edge_instability')
absolute, identity, checked = framewise.absolute, framewise.identity, framewise.checked


def load_config(path=DEFAULT_CONFIG):
    c = yaml.safe_load(absolute(path).read_text())
    if c.get('protocol') != 'course_part2b_offline_v1' or c['source']['kind'] != 'public':
        raise ValueError('只接受课程公开片段的VDA离线协议')
    if c['input']['requested_frames'] != 100 or c['input']['original_fps'] is not True:
        raise ValueError('必须使用前100个真实帧上限与原FPS')
    if c['inference']['mode'] != 'offline' or c['inference']['metric'] is not False:
        raise ValueError('本入口仅运行offline relative，不运行streaming或metric')
    for v in (c['input']['max_long_edge'], c['inference']['input_size'], c['inference']['warmup'],
              c['inference']['timing_repeats'], c['runtime']['torch_threads']):
        if type(v) is not int or v < 1:
            raise ValueError('尺寸、预热、重复和线程数须为正整数')
    if c['runtime']['precision'] not in ('float16', 'float32'):
        raise ValueError('精度必须显式指定；正式GPU轨道使用FP16')
    q = c['visualization']['percentiles']
    if len(q) != 2 or not 0 <= q[0] < q[1] <= 100 or c['visualization']['palette'] != 'TURBO':
        raise ValueError('非法整段固定显示协议')
    d = c['diagnostics']
    if (type(d['farneback']) is not bool or d['fb_threshold_px'] < 0 or d['edge_threshold'] <= 0
            or type(d['edge_tolerance_px']) is not int or d['edge_tolerance_px'] < 0
            or type(d['case_separation_frames']) is not int or d['case_separation_frames'] < 1):
        raise ValueError('非法辅助诊断设置')
    ids = []
    for case in c.get('review', {}).get('cases', []):
        name = case['id']
        if not name or not name.replace('_', '').isalnum() or name in ids:
            raise ValueError('局部观察ID必须唯一且不含路径')
        ids.append(name)
    return c


def check_resources(c, *, require_device=True):
    video, weight = absolute(c['source']['path']), absolute(c['model']['checkpoint'])
    if sha256_file(video) != c['source']['sha256'] or sha256_file(weight) != c['model']['sha256']:
        raise ValueError('视频或VDA权重SHA不符；不下载、不替换资源')
    repo = _repository_identity(absolute(c['model']['repository']))
    if repo['upstream_git_sha'] != c['model']['commit'] or repo['repository_dirty']:
        raise ValueError('VDA官方仓库版本或工作树不符')
    probe = _probe_video(video)
    pts = np.asarray(probe['timestamps'])
    fps = float(Fraction(probe['stream']['avg_frame_rate']))
    if len(pts) < 2 or not np.isfinite(fps) or fps <= 0 or not np.allclose(np.diff(pts), 1 / fps, rtol=1e-3, atol=2e-6):
        raise ValueError('需要至少两帧、正原FPS和连续CFR时间戳，不自动重采样VFR')
    if require_device:
        _runtime(c['runtime']['device'], c['runtime']['precision'])
    return {'video': identity(video), 'checkpoint': identity(weight), 'repository': repo,
            'source_frames': len(pts), 'source_fps': fps,
            'source_hw': [int(probe['stream']['height']), int(probe['stream']['width'])],
            'selected_frames': min(100, len(pts)), 'required_frames': 100,
            'has_100_real_frames': len(pts) >= 100,
            'environment': environment_info(), 'models_executed': False}


def check_manifest(mp, c, resources):
    m = validate_manifest(mp, verify_hashes=True)
    count = resources['selected_frames']
    if (m['source_kind'] != 'public' or m.get('roi_path') is not None
            or m['source']['sha256'] != resources['video']['sha256']
            or m['source']['frame_count'] != resources['source_frames']
            or m['fps'] != resources['source_fps']
            or m['resize']['max_long_edge'] != c['input']['max_long_edge']
            or m['selection']['selected_source_indices'] != list(range(count))
            or len(m['frames']) != count
            or not m['selection']['rule'].startswith('consecutive_original_decoded_frames')):
        raise ValueError('缓存不是指定源的同尺寸、原FPS、连续前N个真实帧；不覆盖缓存')
    # 显式检查每帧PTS，不能只靠FPS或帧数接受另一批时间点。
    pts = _probe_video(absolute(c['source']['path']))['timestamps'][:count]
    for i, (f, t) in enumerate(zip(m['frames'], pts)):
        if f['source_index'] != i or abs(f['timestamp_seconds'] - float(t - pts[0])) > 1e-9:
            raise ValueError('缓存不是相同的原始帧时间点')
    validate_cases(c, count, m['height'], m['width'])
    return m


def validate_cases(c, count, height, width):
    for case in c.get('review', {}).get('cases', []):
        frames, box = case['frames'], case['bbox_xyxy']
        if (not frames or any(type(i) is not int or not 0 <= i < count for i in frames)
                or frames != list(range(frames[0], frames[-1] + 1))):
            raise ValueError('局部观察须使用真实连续帧；新素材请另设观察窗口')
        if (len(box) != 4 or any(type(x) is not int for x in box)
                or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
            raise ValueError('局部观察框超出输出画布')


def predict(config_path=DEFAULT_CONFIG, *, output=None, cache=None):
    config_path = absolute(config_path)
    c = load_config(config_path)
    resources = check_resources(c)
    directory, cache = absolute(output or c['output']), absolute(cache or c['input']['cache'])
    if directory.exists() or directory == cache or directory.is_relative_to(cache) or cache.is_relative_to(directory):
        raise FileExistsError('输出必须为空新目录，并与RGB缓存分开；不覆盖任何已有运行')
    mp = cache / 'manifest.json'
    if cache.exists():
        manifest = check_manifest(mp, c, resources)
    else:
        # 先检查局部窗口，避免新素材尺寸/帧数不符时写出半套缓存。
        h, w = resources['source_hw']; factor = min(1, c['input']['max_long_edge'] / max(h, w))
        validate_cases(c, resources['selected_frames'], round(h * factor), round(w * factor))
        mp = prepare_frames(absolute(c['source']['path']), cache, requested_frames=100,
                            max_long_edge=c['input']['max_long_edge'], fps=resources['source_fps'], source_url=c['source']['url'])
        manifest = check_manifest(mp, c, resources)
    directory.mkdir(parents=True)
    (directory / 'raw').mkdir()
    record = {'schema': 'course_vda_offline_v1', 'status': 'running', 'created_at_utc': utc_now(),
              'command': sys.argv, 'config': deepcopy(c), 'resources': resources, 'frames': []}
    write_json(directory / 'run.json', record)
    started = time.perf_counter()
    try:
        count = len(manifest['frames'])
        scope = {'requested_frames': 100, 'actual_frames': count, 'source_available_frames': resources['source_frames'],
                 'original_fps': resources['source_fps'], 'no_frame_duplication': True,
                 'requirement_100_frames_satisfied': count == 100,
                 'short_source_status': 'needs_course_confirmation' if count < 100 else None,
                 'interpretation': 'actual first N original frames; official internal padding is not extra evaluated frames'}
        protocol = {'schema': 'course_part2b_protocol_v1', 'created_at_utc': utc_now(), 'command': sys.argv,
                    'config': deepcopy(c), 'config_source': identity(config_path), 'source': resources['video'],
                    'checkpoint': resources['checkpoint'], 'manifest': identity(mp), 'frame_scope': scope,
                    'scientific_sources': {p: sha256_file(ROOT / p) for p in SOURCES},
                    'raw_contract': {'depth_type': 'relative', 'unit': 'arbitrary', 'alignment': 'none',
                                     'gt_used': False, 'temporal_context': True, 'mode': 'offline'}}
        protocol['source_archive_base64'] = {p: base64.b64encode((ROOT / p).read_bytes()).decode() for p in SOURCES}
        write_json(directory / 'protocol.json', protocol)
        record['protocol'] = identity(directory / 'protocol.json')
        rgb = np.stack([cv2.cvtColor(cv2.imread(str(mp.parent / f['rgb_path'])), cv2.COLOR_BGR2RGB) for f in manifest['frames']])
        import torch
        torch.set_num_threads(c['runtime']['torch_threads'])
        torch.manual_seed(c['runtime']['seed'])
        if c['runtime']['device'].startswith('cuda'):
            torch.cuda.set_device(c['runtime']['device'])
        t = time.perf_counter()
        model = VideoDepthAnythingAdapter(repository=absolute(c['model']['repository']), checkpoint=absolute(c['model']['checkpoint']),
                                         device=c['runtime']['device'], precision=c['runtime']['precision'], input_size=c['inference']['input_size'])
        load_seconds = time.perf_counter() - t

        def sync():
            if model.device.type == 'cuda':
                torch.cuda.synchronize(model.device)

        for _ in range(c['inference']['warmup']):
            model.infer(rgb, manifest['fps'])
        sync()
        if model.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(model.device)
        timings = []
        for repeat in range(c['inference']['timing_repeats']):
            sync(); t = time.perf_counter()
            prediction = model.infer(rgb, manifest['fps'])
            sync(); timings.append(time.perf_counter() - t)
            if repeat == 0:
                saved = prediction.copy()
        if saved.shape != rgb.shape[:3] or saved.dtype != np.float32 or not np.isfinite(saved).all() or np.any(saved < 0):
            raise ValueError('VDA须返回完整原帧数、非负有限float32序列')
        for frame, a in zip(manifest['frames'], saved):
            relative = f'raw/frame_{frame["index"]:06d}.npy'
            np.save(directory / relative, a)
            record['frames'].append({'index': frame['index'], 'source_index': frame['source_index'],
                                     'timestamp_seconds': frame['timestamp_seconds'], 'raw_path': relative,
                                     'raw_sha256': sha256_file(directory / relative), 'shape': list(a.shape), 'dtype': 'float32'})
        # 与课程--save_npz一致，保留整段浮点序列；逐帧NPY用于独立核验与分析。
        np.savez_compressed(directory / 'depths.npz', depths=saved)
        model.metadata['actual_output_hw'] = [manifest['height'], manifest['width']]
        record.update(status='complete', finished_at_utc=utc_now(), frame_scope=scope,
                      model=deepcopy(model.metadata), raw_archive=identity(directory / 'depths.npz'), environment=environment_info(),
                      timing={'model_load_seconds': load_seconds, 'warmup': c['inference']['warmup'],
                              'timing_repeats': c['inference']['timing_repeats'], 'sequence_repeats_seconds': timings,
                              'mean_seconds_per_sequence': float(np.mean(timings)),
                              'mean_seconds_per_frame': float(np.mean(timings)) / count,
                              'median_seconds_per_frame': float(np.median(timings)) / count,
                              'p95_seconds_per_frame': float(np.percentile(timings, 95)) / count,
                              'fps': count / float(np.mean(timings)), 'pipeline_seconds': time.perf_counter() - started,
                              'peak_gpu_allocated_bytes': int(torch.cuda.max_memory_allocated(model.device)) if model.device.type == 'cuda' else None,
                              'peak_gpu_reserved_bytes': int(torch.cuda.max_memory_reserved(model.device)) if model.device.type == 'cuda' else None,
                              'scope': 'whole offline sequence: preprocessing, transfers, windows, stitching and output; excludes loading, decode/save/report; amortized per real output frame; not streaming latency',
                              'percentile_scope': 'percentiles across complete-sequence repeats divided by real frame count; not per-frame latency percentiles'})
        write_json(directory / 'run.json', record)
        return directory
    except BaseException as exc:
        record.update(status='failed', finished_at_utc=utc_now(), error_type=type(exc).__name__, error=str(exc))
        write_json(directory / 'run.json', record)
        raise


def validate_run(directory):
    directory = absolute(directory)
    run = read_json(directory / 'run.json')
    if run.get('schema') != 'course_vda_offline_v1' or run.get('status') != 'complete':
        raise ValueError('需要完整VDA离线run；失败阶段保留，不自动跳过')
    protocol = read_json(checked(run['protocol']))
    c = load_config(checked(protocol['config_source']))
    if run['config'] != c or protocol['config'] != c:
        raise ValueError('当前配置与冻结协议不同')
    checked(protocol['source']); checked(protocol['checkpoint'])
    for p, h in protocol['scientific_sources'].items():
        if sha256_file(ROOT / p) != h or hashlib.sha256(base64.b64decode(protocol['source_archive_base64'][p])).hexdigest() != h:
            raise ValueError(f'科学源码或归档改变：{p}')
    _equivalent(_repository_identity(absolute(c['model']['repository'])), run['resources']['repository'], 'official repository')
    mp = checked(protocol['manifest'])
    manifest = check_manifest(mp, c, run['resources'])
    count = len(manifest['frames']); shape = [manifest['height'], manifest['width']]
    scope = protocol['frame_scope']
    if (run['frame_scope'] != scope or scope['actual_frames'] != count or scope['requested_frames'] != 100
            or scope['source_available_frames'] != manifest['source']['frame_count']
            or scope['requirement_100_frames_satisfied'] != (count == 100)
            or scope['original_fps'] != manifest['fps'] or not scope['no_frame_duplication'] or len(run['frames']) != count):
        raise ValueError('真实帧数/FPS或课程门禁身份不一致')
    model = run['model']
    expected = {'depth_type': 'relative', 'unit': 'arbitrary', 'alignment': 'none', 'gt_used': False,
                'temporal_context': True, 'inference_mode': 'offline_noncausal', 'encoder': 'vits',
                'precision': c['runtime']['precision'], 'network_input_hw': list(network_input_shape(*shape, c['inference']['input_size'])),
                'actual_output_hw': shape}
    for k, v in expected.items():
        if model.get(k) != v:
            raise ValueError(f'VDA模型协议不符：{k}')
    windows = window_metadata(count)
    if model.get('windowing') != windows or model.get('observed_window_input_shapes') != [expected['network_input_hw']] * windows['window_count']:
        raise ValueError('VDA官方窗口记录或实际输入网格不符')
    raw, rgb = [], []
    for i, (item, frame) in enumerate(zip(run['frames'], manifest['frames'])):
        if item['index'] != i or item['source_index'] != i or item['timestamp_seconds'] != frame['timestamp_seconds']:
            raise ValueError('预测帧序或PTS改变')
        p = directory / item['raw_path']
        if p.resolve().parent != (directory / 'raw').resolve() or sha256_file(p) != item['raw_sha256']:
            raise ValueError('原始预测路径或SHA改变')
        a = np.load(p, allow_pickle=False)
        if a.dtype != np.float32 or list(a.shape) != shape or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('预测不是完整非负有限float32')
        raw.append(a)
        rgb.append(cv2.cvtColor(cv2.imread(str(mp.parent / frame['rgb_path'])), cv2.COLOR_BGR2RGB))
    raw = np.stack(raw)
    archive = checked(run['raw_archive'])
    if archive.resolve() != directory / 'depths.npz':
        raise ValueError('整段浮点档案路径不符')
    with np.load(archive, allow_pickle=False) as stored:
        if stored.files != ['depths'] or stored['depths'].dtype != np.float32 or not np.array_equal(stored['depths'], raw):
            raise ValueError('整段NPZ与逐帧原始预测不一致')
    return run, protocol, manifest, raw, rgb


def analyze(directory, *, output=None):
    directory = absolute(directory)
    run, protocol, manifest, raw, rgb = validate_run(directory)
    output = absolute(output) if output else directory / 'analysis'
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', {'status': 'running', 'command': sys.argv})
    try:
        scores, frames, pairs = framewise.measure(raw, rgb, manifest, run['config'])
        framewise.write_csv(output / 'frame_diagnostics.csv', frames)
        framewise.write_csv(output / 'pair_diagnostics.csv', pairs)
        summary = {'schema': 'course_part2b_analysis_v1', 'status': 'complete', 'created_at_utc': utc_now(),
                   'command': sys.argv, 'run': identity(directory / 'run.json'), 'frame_scope': protocol['frame_scope'],
                   'model': run['model'], 'timing': run['timing'], **scores}
        from src.visualization.temporal_clip import standalone_report
        standalone_report(output, raw, rgb, frames, pairs, summary, run['config'], manifest['fps'])
        summary['assets'] = [{'path': str(p.relative_to(output)), 'sha256': sha256_file(p)} for p in sorted(output.rglob('*'))
                             if p.is_file() and p.name not in ('status.json', 'summary.json')]
        write_json(output / 'summary.json', summary)
        write_json(output / 'status.json', {'status': 'complete', 'summary': identity(output / 'summary.json')})
        return output
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise


def check_analysis(directory, run, protocol, manifest, raw, rgb, analysis=None):
    output = absolute(analysis) if analysis else directory / 'analysis'
    status = read_json(output / 'status.json')
    if status['status'] != 'complete':
        raise ValueError('分析未完整完成')
    summary = read_json(checked(status['summary']))
    if summary['run'] != identity(directory / 'run.json'):
        raise ValueError('分析绑定了不同的VDA运行')
    for item in summary['assets']:
        p = output / item['path']
        if not p.resolve().is_relative_to(output) or sha256_file(p) != item['sha256']:
            raise ValueError('报告附件改变')
    actual, frames, pairs = framewise.measure(raw, rgb, manifest, run['config'])
    for k, v in actual.items():
        _equivalent(summary[k], v, k)
    _equivalent(_read_csv(output / 'frame_diagnostics.csv'), frames, 'frame CSV')
    _equivalent(_read_csv(output / 'pair_diagnostics.csv'), pairs, 'pair CSV')
    for k, v in [('model', run['model']), ('timing', run['timing']), ('frame_scope', protocol['frame_scope'])]:
        _equivalent(summary[k], v, k)
    return summary, frames, pairs, output


def verify(directory, *, analysis=None):
    directory = absolute(directory)
    run, protocol, manifest, raw, rgb = validate_run(directory)
    summary, frames, pairs, output = check_analysis(directory, run, protocol, manifest, raw, rgb, analysis)
    return {'status': 'verified', 'run': identity(directory / 'run.json'), 'analysis': identity(output / 'summary.json'),
            'frames': len(frames), 'pairs': len(pairs), 'models_executed': False, 'raw_recomputed': True,
            'required_100_frames_satisfied': len(frames) == 100, 'tolerance': {'rtol': 1e-6, 'atol': 1e-8}}


def comparison_inputs(dav2_directory, vda_directory, dav2_analysis=None, vda_analysis=None):
    da, vd = absolute(dav2_directory), absolute(vda_directory)
    if da == vd or da.is_relative_to(vd) or vd.is_relative_to(da):
        raise ValueError('1B/2B原始输出必须分开')
    dr, dp, dm, draw, rgb = framewise.validate_run(da)
    framewise.verify(da, analysis=dav2_analysis)
    vr, vp, vm, vraw, vrgb = validate_run(vd)
    vs, vf, vpair, va = check_analysis(vd, vr, vp, vm, vraw, vrgb, vda_analysis)
    da_analysis = absolute(dav2_analysis) if dav2_analysis else da / 'analysis'
    if (dp['source']['sha256'] != vp['source']['sha256'] or draw.shape != vraw.shape or dm['fps'] != vm['fps']
            or dr['model']['network_input_hw'] != vr['model']['network_input_hw']
            or dr['model']['precision'] != vr['model']['precision']
            or dr['config']['diagnostics'] != vr['config']['diagnostics']
            or dr['config']['visualization']['percentiles'] != vr['config']['visualization']['percentiles']
            or dr['config']['visualization']['palette'] != vr['config']['visualization']['palette']):
        raise ValueError('两模型的源、尺寸、精度、诊断或整段显示规则不一致，不能比较')
    keys = ('source_index', 'timestamp_seconds', 'rgb_sha256')
    if [[f[k] for k in keys] for f in dm['frames']] != [[f[k] for k in keys] for f in vm['frames']]:
        raise ValueError('两模型没有使用相同真实RGB帧和PTS')
    handoff = read_json(da_analysis / 'handoff_2b.json')
    if (handoff['manifest'] != dp['manifest'] or handoff['dav2_run'] != identity(da / 'run.json')
            or handoff['network_input_hw'] != vr['model']['network_input_hw'] or handoff['precision'] != vr['model']['precision']):
        raise ValueError('1B交接身份或比较输入不符')
    ds, df, dpair = framewise.measure(draw, rgb, dm, dr['config'])
    for a, b in zip(dpair, vpair):
        _equivalent(a['flow_coverage'], b['flow_coverage'], 'shared RGB flow coverage')
        if a.get('flow_pixels') != b.get('flow_pixels'):
            raise ValueError('两模型的光流有效像素域不一致')
    return {'dav2': {'run': dr, 'protocol': dp, 'summary': ds, 'frames': df, 'pairs': dpair, 'raw': draw,
                     'directory': da, 'analysis': da_analysis},
            'vda': {'run': vr, 'protocol': vp, 'summary': vs, 'frames': vf, 'pairs': vpair, 'raw': vraw,
                    'directory': vd, 'analysis': va}, 'rgb': rgb, 'manifest': dm}


def comparison_scores(inputs):
    a, b = inputs['dav2'], inputs['vda']
    rows, frames = [], []
    for left, right in zip(a['pairs'], b['pairs']):
        row = {k: left[k] for k in ('previous_index', 'current_index', 'timestamp_seconds', 'flow_coverage')}
        for metric in METRICS:
            x, y = left[metric], right[metric]
            row.update({f'dav2_{metric}': x, f'vda_{metric}': y,
                        f'dav2_minus_vda_{metric}': x - y if x is not None and y is not None else None})
        rows.append(row)
    for left, right in zip(a['frames'], b['frames']):
        frames.append({'index': left['index'], 'timestamp_seconds': left['timestamp_seconds'],
                       **{f'{name}_{k}': item[k] for name, item in [('dav2', left), ('vda', right)]
                          for k in ('minimum', 'maximum', 'median', 'edge_fraction')}})
    metrics = {}
    for metric in METRICS:
        usable = [r for r in rows if r[f'dav2_minus_vda_{metric}'] is not None]
        x = float(np.mean([r[f'dav2_{metric}'] for r in usable])) if usable else None
        y = float(np.mean([r[f'vda_{metric}'] for r in usable])) if usable else None
        metrics[metric] = {'dav2': x, 'vda': y, 'common_pair_count': len(usable),
                           'dav2_minus_vda': x - y if x is not None else None,
                           'reduction_percent': (x - y) / x * 100 if x is not None and x > 0 else None}
    # 显示共同的三个实际窗口；另外保留最大正差和最大负差候选，不预设改善。
    case_candidates = []
    for metric in ('flow_rmse_normalized', 'edge_instability'):
        usable = [r for r in rows if r[f'dav2_minus_vda_{metric}'] is not None]
        if usable:
            for role, selected in [('largest_dav2_minus_vda', max(usable, key=lambda r: r[f'dav2_minus_vda_{metric}'])),
                                   ('smallest_dav2_minus_vda', min(usable, key=lambda r: r[f'dav2_minus_vda_{metric}']))]:
                case_candidates.append({'metric': metric, 'role': role, 'current_index': selected['current_index'],
                                        'dav2_minus_vda': selected[f'dav2_minus_vda_{metric}']})
    result = {'metrics': metrics, 'reference': {name: inputs[name]['summary']['reference'] for name in ('dav2', 'vda')},
              'efficiency': {name: {'timing': inputs[name]['run']['timing'],
                                    'environment': inputs[name]['run'].get('environment'),
                                    'model': inputs[name]['run']['model']} for name in ('dav2', 'vda')},
              'frames': len(frames), 'pairs': len(rows), 'case_candidates': case_candidates,
              'common_flow_coverage_mean': float(np.mean([r['flow_coverage'] for r in rows if r['flow_coverage'] is not None]))
                  if any(r['flow_coverage'] is not None for r in rows) else None,
              'aggregation': 'equal pair means on common non-null pairs per metric; negative reductions retained; no GT accuracy claim',
              'normalization': 'same percentiles, own whole-clip fixed span per model; no framewise fitting; not shared meter units'}
    return result, frames, rows


def compare(dav2_directory, vda_directory, *, output, dav2_analysis=None, vda_analysis=None):
    output = absolute(output)
    for directory in (absolute(dav2_directory), absolute(vda_directory)):
        if output == directory or output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError('比较输出须与两条预测目录分开，避免修改或覆盖原始输入')
    if output.exists():
        raise FileExistsError(output)
    inputs = comparison_inputs(dav2_directory, vda_directory, dav2_analysis, vda_analysis)
    output.mkdir(parents=True)
    write_json(output / 'status.json', {'status': 'running', 'command': sys.argv})
    try:
        scores, frames, pairs = comparison_scores(inputs)
        sources = {name: {'run': identity(inputs[name]['directory'] / 'run.json'),
                          'analysis': identity(inputs[name]['analysis'] / 'summary.json')} for name in ('dav2', 'vda')}
        summary = {'schema': 'course_part2b_comparison_v1', 'status': 'complete', 'created_at_utc': utc_now(),
                   'command': sys.argv, 'sources': sources, 'frame_scope': inputs['vda']['protocol']['frame_scope'],
                   'scientific_sources': {p: sha256_file(ROOT / p) for p in SOURCES}, **scores}
        framewise.write_csv(output / 'frame_comparison.csv', frames)
        framewise.write_csv(output / 'pair_comparison.csv', pairs)
        from src.visualization.temporal_clip import comparison_report
        comparison_report(output, inputs, summary)
        summary['assets'] = [{'path': str(p.relative_to(output)), 'sha256': sha256_file(p)} for p in sorted(output.rglob('*'))
                             if p.is_file() and p.name not in ('status.json', 'summary.json')]
        write_json(output / 'summary.json', summary)
        write_json(output / 'status.json', {'status': 'complete', 'summary': identity(output / 'summary.json')})
        return output
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise


def verify_comparison(directory):
    directory = absolute(directory)
    status = read_json(directory / 'status.json')
    if status['status'] != 'complete':
        raise ValueError('比较未完整完成')
    summary = read_json(checked(status['summary']))
    for p, h in summary['scientific_sources'].items():
        if sha256_file(ROOT / p) != h:
            raise ValueError('比较源码改变')
    sources = summary['sources']
    for item in sources.values():
        checked(item['run']); checked(item['analysis'])
    inputs = comparison_inputs(Path(sources['dav2']['run']['path']).parent, Path(sources['vda']['run']['path']).parent,
                               Path(sources['dav2']['analysis']['path']).parent, Path(sources['vda']['analysis']['path']).parent)
    for item in summary['assets']:
        p = directory / item['path']
        if not p.resolve().is_relative_to(directory) or sha256_file(p) != item['sha256']:
            raise ValueError('比较附件改变')
    actual, frames, pairs = comparison_scores(inputs)
    for k, v in actual.items():
        _equivalent(summary[k], v, k)
    _equivalent(summary['frame_scope'], inputs['vda']['protocol']['frame_scope'], 'frame scope')
    _equivalent(_read_csv(directory / 'frame_comparison.csv'), frames, 'frame comparison CSV')
    _equivalent(_read_csv(directory / 'pair_comparison.csv'), pairs, 'pair comparison CSV')
    return {'status': 'verified', 'summary': identity(directory / 'summary.json'), 'models_executed': False,
            'both_raw_recomputed': True, 'frames': actual['frames'], 'pairs': actual['pairs'],
            'required_100_frames_satisfied': actual['frames'] == 100, 'tolerance': {'rtol': 1e-6, 'atol': 1e-8}}
