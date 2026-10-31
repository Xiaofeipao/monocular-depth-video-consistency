"""Synthetic benchmark lifecycle and corruption checks; no real model inference."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.nyuv2 import NYUv2Sample
from src.evaluation import relative_nyuv2 as bench
from src.video.common import read_json, write_json


@pytest.fixture
def config():
    c = bench.load_config(bench.PROJECT_ROOT / 'configs/part1/dav2_nyuv2.yaml')
    c['runtime'].update(device='cpu', precision='float32')
    c['statistics']['bootstrap_resamples'] = 20
    return c


@pytest.fixture
def fake(monkeypatch, config, tmp_path):
    from src.models import depth_anything_v2_adapter as adapter
    from src.visualization import nyuv2
    raw = np.broadcast_to(np.linspace(0, 2, 640, dtype=np.float32), (480, 640)).copy()
    gt = (1 / (0.4 * raw.astype(np.float64) + 0.2)).astype(np.float32)
    class Dataset:
        test_indices_1based = np.arange(1, 655)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get_test_sample(self, position):
            return NYUv2Sample(position, position + 1, np.zeros((480, 640, 3), np.uint8), gt, 'scene', 'room')
    calls = []
    class Model:
        device = torch.device('cpu')
        metadata = {'network_input_hw': [518, 700], 'parameters': 24785089}
        def __init__(self, **kwargs): pass
        def infer(self, rgb):
            calls.append(rgb.shape)
            return raw.copy()
    monkeypatch.setattr(bench, '_dataset', lambda c: Dataset())
    monkeypatch.setattr(bench, 'check_resources', lambda c: {
        'resources': {}, 'repository': {'repository': str(tmp_path), 'source_files_sha256': {}}, 'environment': {}})
    monkeypatch.setattr(adapter, 'DepthAnythingV2Adapter', Model)
    def panel(path, **kwargs):
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (12, 12)).save(path)
    monkeypatch.setattr(nyuv2, 'save_nyuv2_panel', panel)
    monkeypatch.setattr(nyuv2, 'save_contact_sheet', lambda path, panels: panel(path))
    return calls


def test_full_lifecycle_from_raw_recomputes_and_keeps_labels(config, fake, tmp_path):
    run, output = tmp_path / 'raw', tmp_path / 'evaluation'
    bench.predict(config, run, positions=[0, 653])
    assert len(fake) == 1 + 2 * 3
    bench.evaluate(run, output)
    before = len(fake)
    assert bench.verify(output)['images'] == 2
    assert len(fake) == before
    s = read_json(output / 'metrics_summary.json')
    assert s['label'] == 'aligned_relative' and not s['native_metric']
    assert s['benchmark_scope'] == 'subset_smoke_only'
    assert s['macro_mean']['abs_rel'] < 1e-6
    assert s['macro_mean']['delta1'] == 1
    assert read_json(run / 'run.json')['raw_contract']['gt_used'] is False
    with pytest.raises(FileExistsError): bench.predict(config, run, positions=[0, 653])
    with pytest.raises(FileExistsError): bench.evaluate(run, output)


@pytest.mark.parametrize('artifact', ['predictions', 'csv', 'aligned'])
def test_raw_and_derived_corruption_are_detected(config, fake, tmp_path, artifact):
    run, output = tmp_path / 'raw', tmp_path / 'evaluation'
    bench.predict(config, run, positions=[0])
    bench.evaluate(run, output)
    path = (next((run / 'predictions').glob('*.npz')) if artifact == 'predictions' else
            output / 'metrics_per_image.csv' if artifact == 'csv' else next((output / 'aligned').glob('*.npz')))
    with path.open('ab') as handle: handle.write(b'changed')
    with pytest.raises(ValueError, match='SHA|artifact'): bench.verify(output)


def test_resume_complete_is_noop_and_different_config_rejected(config, fake, tmp_path):
    run = tmp_path / 'raw'
    bench.predict(config, run, positions=[0])
    before = len(fake)
    bench.predict(config, run, positions=[0], resume=True)
    assert len(fake) == before
    c = deepcopy(config); c['inference']['input_size'] = 98
    with pytest.raises(ValueError, match='Resume'): bench.predict(c, run, positions=[0], resume=True)


@pytest.mark.parametrize('positions', [[], [0, 0], [-1], [654], [True]])
def test_invalid_position_selection(positions):
    with pytest.raises(ValueError): bench._positions(positions, 654)


def test_fixed_evaluation_protocol_and_output_protection(config, tmp_path):
    c = deepcopy(config); c['evaluation']['crop'][0] = 0
    path = tmp_path / 'bad.yaml'
    import yaml
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='protocol'): bench.load_config(path)
    with pytest.raises(ValueError, match='outputs'): bench._new_directory(bench.PROJECT_ROOT / 'data/never_create')


def test_cli_help_from_another_directory(tmp_path):
    import subprocess, sys
    result = subprocess.run([sys.executable, str(bench.PROJECT_ROOT / 'pipelines/part1/dav2_nyuv2.py'), '--help'],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert 'evaluate' in result.stdout and '--positions' in result.stdout
