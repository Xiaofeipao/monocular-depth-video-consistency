"""课程2B：真实小文件和合成模型，检查独立离线任务及只读配对；不加载权重。"""
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from src.evaluation import framewise_clip as da
from src.evaluation import temporal_clip as vda
from src.models.depth_anything_v2_adapter import network_input_shape
from src.models.video_depth_anything_adapter import window_metadata
from src.video.common import read_json, write_json, sha256_file


@pytest.fixture
def clip(tmp_path, monkeypatch):
    source = tmp_path / 'source.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 24, (64, 36))
    for i in range(6):
        rgb = np.full((36, 64, 3), 30 + i * 10, np.uint8)
        rgb[:, 32:, 0] = 120
        writer.write(rgb)
    writer.release()
    weight = tmp_path / 'weight.pth'; weight.write_bytes(b'synthetic; not real weights')
    cache = tmp_path / 'rgb_cache'
    c = deepcopy(vda.load_config())
    c['source'].update(path=str(source), sha256=sha256_file(source))
    c['model'].update(repository=str(tmp_path / 'VDA'), checkpoint=str(weight), sha256=sha256_file(weight))
    c['runtime'].update(device='cpu', precision='float32')
    c['inference'].update(input_size=28, timing_repeats=1)
    c['input']['cache'] = str(cache); c['output'] = str(tmp_path / 'vda')
    c['review']['cases'] = [{'id': 'local', 'frames': [1, 2, 3], 'bbox_xyxy': [5, 5, 55, 30]}]
    c['visualization']['video'] = False
    path = tmp_path / 'vda.yaml'; path.write_text(yaml.safe_dump(c))
    dc = deepcopy(da.load_config())
    for key in ('source', 'runtime', 'input', 'visualization', 'diagnostics'):
        dc[key] = deepcopy(c[key])
    dc['model'].update(repository=str(tmp_path / 'DA'), checkpoint=str(weight), sha256=sha256_file(weight))
    dc['inference'].update(input_size=28, timing_repeats=1)
    dc['output'] = str(tmp_path / 'dav2')
    dpath = tmp_path / 'dav2.yaml'; dpath.write_text(yaml.safe_dump(dc))
    monkeypatch.setattr(vda, '_repository_identity', lambda p: {'upstream_git_sha': c['model']['commit'], 'repository_dirty': False})
    monkeypatch.setattr(da, '_repository_identity', lambda p: {'upstream_git_sha': dc['model']['commit'], 'repository_dirty': False})
    for task in (da, vda):
        monkeypatch.setattr(task, 'environment_info', lambda: {'kind': 'synthetic test', 'cuda_available': False})
    state = {'vda_calls': 0, 'da_calls': 0, 'zero': False, 'fail': False}

    class VDA:
        def __init__(self, **kw):
            self.device = torch.device('cpu')
            self.metadata = {'depth_type': 'relative', 'unit': 'arbitrary', 'alignment': 'none', 'gt_used': False,
                             'temporal_context': True, 'inference_mode': 'offline_noncausal', 'encoder': 'vits',
                             'precision': 'float32', 'parameters': 1, 'checkpoint_bytes': weight.stat().st_size}

        def infer(self, rgb, fps):
            state['vda_calls'] += 1
            if state['fail']:
                raise RuntimeError('synthetic sequence failure')
            shape = list(network_input_shape(*rgb.shape[1:3], 28))
            windows = window_metadata(len(rgb))
            self.metadata.update(windowing=windows, network_input_hw=shape,
                                 observed_window_input_shapes=[shape] * windows['window_count'])
            if state['zero']:
                return np.zeros(rgb.shape[:3], np.float32)
            yy, xx = np.indices(rgb.shape[1:3], dtype=np.float32)
            return np.stack([xx + yy + np.float32(frame.mean() / 200) for frame in rgb])

    class DA:
        def __init__(self, **kw):
            self.device = torch.device('cpu')
            self.metadata = {'depth_type': 'relative', 'unit': 'arbitrary', 'alignment': 'none', 'gt_used': False,
                             'temporal_context': False, 'encoder': 'vits', 'precision': 'float32',
                             'network_input_hw': list(network_input_shape(36, 64, 28)), 'parameters': 1}

        def infer(self, rgb):
            state['da_calls'] += 1
            yy, xx = np.indices(rgb.shape[:2], dtype=np.float32)
            return xx + yy + np.float32(rgb.mean() / 20)

    monkeypatch.setattr(vda, 'VideoDepthAnythingAdapter', VDA)
    monkeypatch.setattr(da, 'DepthAnythingV2Adapter', DA)
    return {'path': path, 'dpath': dpath, 'config': c, 'cache': cache, 'state': state, 'tmp': tmp_path}


def snapshots(directory):
    return {str(p.relative_to(directory)): sha256_file(p) for p in directory.rglob('*') if p.is_file()}


def complete_pair(clip):
    dav2 = da.predict(clip['dpath']); da.analyze(dav2)
    vroot = vda.predict(clip['path']); vda.analyze(vroot)
    return dav2, vroot


def test_independent_vda_creates_float_npz_and_same_four_diagnostics(clip):
    c = clip['config']; c['visualization']['video'] = True
    clip['path'].write_text(yaml.safe_dump(c))
    root = vda.predict(clip['path']); vda.analyze(root)
    result = vda.verify(root)
    assert clip['state']['da_calls'] == 0 and clip['state']['vda_calls'] == 2
    assert result['frames'] == 6 and result['pairs'] == 5 and not result['required_100_frames_satisfied']
    assert result['raw_recomputed'] and not result['models_executed']
    with np.load(root / 'depths.npz') as archive:
        assert archive['depths'].shape == (6, 36, 64)
    for name in ('diagnostics.png', 'diagnostics.pdf', 'contact.png', 'fixed_colorbar.png', 'rgb_relative_edges.mp4'):
        assert (root / 'analysis/visualization' / name).is_file()
    assert (root / 'analysis/review/local.png').is_file()
    assert read_json(root / 'analysis/summary.json')['reference']['scope'] == 'whole_clip_once'
    assert not read_json(root / 'analysis/visualization/display.json')['per_frame_normalization']
    with pytest.raises(FileExistsError):
        vda.predict(clip['path'])
    with pytest.raises(FileExistsError):
        vda.analyze(root)


def test_compare_reuses_rgb_preserves_both_runs_and_recomputes(clip):
    dav2 = da.predict(clip['dpath']); da.analyze(dav2)
    before = snapshots(dav2); cache_before = snapshots(clip['cache'])
    clip['config']['visualization']['video'] = True
    clip['path'].write_text(yaml.safe_dump(clip['config']))
    root = vda.predict(clip['path']); vda.analyze(root)
    assert snapshots(clip['cache']) == cache_before
    vbefore = snapshots(root); calls = deepcopy(clip['state'])
    target = vda.compare(dav2, root, output=clip['tmp'] / 'compare')
    result = vda.verify_comparison(target)
    assert result['both_raw_recomputed'] and not result['models_executed']
    assert result['pairs'] == 5
    assert snapshots(dav2) == before and snapshots(root) == vbefore
    assert clip['state'] == calls
    scores = read_json(target / 'summary.json')['metrics']
    assert scores['pixel_rmse_normalized']['reduction_percent'] > 0
    assert scores['pixel_rmse_normalized']['common_pair_count'] == 5
    assert (target / 'visualization/rgb_dav2_vda.mp4').is_file()
    assert (target / 'visualization/rgb_dav2_vda_edges.mp4').is_file()
    with pytest.raises(FileExistsError):
        vda.compare(dav2, root, output=target)
    with pytest.raises(ValueError):
        vda.compare(dav2, root, output=dav2 / 'forbidden')


def test_constant_vda_span_stays_na_instead_of_fake_zero_improvement(clip):
    clip['state']['zero'] = True
    dav2, root = complete_pair(clip)
    target = vda.compare(dav2, root, output=clip['tmp'] / 'compare')
    vda.verify_comparison(target)
    scores = read_json(target / 'summary.json')['metrics']
    assert scores['flow_rmse_normalized']['common_pair_count'] == 0
    assert scores['flow_rmse_normalized']['vda'] is None
    assert scores['flow_rmse_normalized']['reduction_percent'] is None


@pytest.mark.parametrize('kind', ['raw', 'npz', 'csv_resealed', 'window', 'rgb', 'config'])
def test_independent_tampering_is_rejected(clip, kind):
    root = vda.predict(clip['path']); vda.analyze(root)
    if kind == 'raw':
        np.save(root / 'raw/frame_000000.npy', np.zeros((36, 64), np.float32))
    elif kind == 'npz':
        np.savez_compressed(root / 'depths.npz', depths=np.zeros((6, 36, 64), np.float32))
    elif kind == 'rgb':
        (clip['cache'] / 'rgb/000000.png').write_bytes(b'changed')
    elif kind == 'config':
        clip['path'].write_text(clip['path'].read_text() + '# changed identity\n')
    elif kind == 'window':
        run = read_json(root / 'run.json'); run['model']['windowing']['stride'] = 1
        write_json(root / 'run.json', run)
    else:
        # 重新封装附件SHA也不能让错误CSV通过原生重算。
        p = root / 'analysis/pair_diagnostics.csv'; text = p.read_text().splitlines()
        row = text[1].split(','); row[3] = '999'; text[1] = ','.join(row); p.write_text('\n'.join(text) + '\n')
        summary = read_json(root / 'analysis/summary.json')
        for a in summary['assets']:
            if a['path'] == 'pair_diagnostics.csv':
                a['sha256'] = sha256_file(p)
        write_json(root / 'analysis/summary.json', summary)
        status = read_json(root / 'analysis/status.json'); status['summary']['sha256'] = sha256_file(root / 'analysis/summary.json')
        write_json(root / 'analysis/status.json', status)
    with pytest.raises(ValueError):
        vda.verify(root)


def test_comparison_rejects_different_diagnostic_protocol_before_writing(clip):
    c = clip['config']; c['diagnostics']['edge_threshold'] = .03
    clip['path'].write_text(yaml.safe_dump(c))
    dav2, root = complete_pair(clip)
    target = clip['tmp'] / 'compare'
    with pytest.raises(ValueError, match='不一致'):
        vda.compare(dav2, root, output=target)
    assert not target.exists()


def test_failure_is_recorded_and_never_overwritten(clip):
    clip['state']['fail'] = True
    with pytest.raises(RuntimeError, match='synthetic'):
        vda.predict(clip['path'])
    run = read_json(clip['tmp'] / 'vda/run.json')
    assert run['status'] == 'failed' and run['error_type'] == 'RuntimeError'
    with pytest.raises(FileExistsError):
        vda.predict(clip['path'])


@pytest.mark.parametrize('key,value', [('mode', 'streaming'), ('metric', True)])
def test_optional_modes_cannot_enter_this_relative_offline_track(clip, key, value):
    c = clip['config']; c['inference'][key] = value
    clip['path'].write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='offline relative'):
        vda.load_config(clip['path'])


def test_comparison_negative_result_and_zero_denominator_remain_explicit():
    def item(value):
        pair = {'previous_index': 0, 'current_index': 1, 'timestamp_seconds': 1., 'flow_coverage': .9}
        pair.update({k: value for k in vda.METRICS})
        return {'pairs': [pair], 'frames': [{'index': 0, 'timestamp_seconds': 0.,
                                           'minimum': 0., 'maximum': 1., 'median': 0., 'edge_fraction': .2}],
                'summary': {'reference': {'span': 1.}}, 'run': {'timing': {}, 'model': {}}}
    inputs = {'dav2': item(.1), 'vda': item(.2)}
    scores, _, _ = vda.comparison_scores(inputs)
    assert scores['metrics']['flow_rmse_normalized']['reduction_percent'] == pytest.approx(-100.)
    inputs['dav2'] = item(0.)
    scores, _, _ = vda.comparison_scores(inputs)
    assert scores['metrics']['flow_rmse_normalized']['reduction_percent'] is None
    assert scores['metrics']['flow_rmse_normalized']['dav2_minus_vda'] == pytest.approx(-.2)


def test_comparison_csv_resealed_hash_cannot_bypass_raw_recompute(clip):
    dav2, root = complete_pair(clip)
    target = vda.compare(dav2, root, output=clip['tmp'] / 'compare')
    p = target / 'pair_comparison.csv'; text = p.read_text().splitlines()
    row = text[1].split(','); row[4] = '999'; text[1] = ','.join(row)
    p.write_text('\n'.join(text) + '\n')
    summary = read_json(target / 'summary.json')
    for asset in summary['assets']:
        if asset['path'] == p.name:
            asset['sha256'] = sha256_file(p)
    write_json(target / 'summary.json', summary)
    status = read_json(target / 'status.json')
    status['summary']['sha256'] = sha256_file(target / 'summary.json')
    write_json(target / 'status.json', status)
    with pytest.raises(ValueError, match='numerical mismatch'):
        vda.verify_comparison(target)
