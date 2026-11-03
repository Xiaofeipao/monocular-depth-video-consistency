"""Synthetic native-only lifecycle and RGB-only adapter regression checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from pipelines.part2 import run_key_analysis as analysis
from src.models.depth_pro_adapter import DepthProAdapter
from src.models.unidepth_v2_adapter import UniDepthV2Adapter


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    config = yaml.safe_load((analysis.PROJECT_ROOT / 'configs/part2/key_analysis.yaml').read_text())
    config['dataset']['expected_test_images'] = 3
    config['evaluation']['crop'] = dict(top=0, bottom=18, left=0, right=24)
    config['statistics']['bootstrap_resamples'] = 50
    config['absolute_scale_analysis']['depth_bins_m'] = [[0.001, 3.0], [3.0, 10.0]]
    config['boundary_analysis']['qualitative_selection'] = dict(depth_pro_advantage=1, unidepth_v2_advantage=1, both_low_si_f1=1)
    target = np.full((18, 24), 2, np.float32)
    target[:, 12:] = 4
    samples = [SimpleNamespace(depth_m=target, rgb=np.zeros((18, 24, 3), np.uint8), matlab_index_1based=i+11, scene=f'room_{i}', scene_type='room') for i in range(3)]
    class Dataset:
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return len(samples)
        def get_test_sample(self, i): return samples[i]
        def __enter__(self): return self
        def __exit__(self, *args): pass
    for method, multiplier in (('depth_pro', 0.8), ('unidepth_v2', 1.2)):
        root = tmp_path / method
        (root / 'predictions').mkdir(parents=True)
        baseline_config = tmp_path / f'{method}.yaml'
        baseline_config.write_text(yaml.safe_dump(config))
        (root / 'run_metadata.json').write_text(json.dumps(dict(sample_count=3, full_test_split=True, test_time_alignment='none', formal_training_run=False, config=str(baseline_config), config_sha256=analysis.sha256(baseline_config))))
        rows=[]
        for i, sample in enumerate(samples):
            stem = analysis.prediction_stem(i, sample.matlab_index_1based)
            prediction = target * np.float32(multiplier)
            # No camera arrays are needed by the analysis.
            np.savez_compressed(root / 'predictions' / f'{stem}.npz', depth_m=prediction)
            metrics = analysis.compute_depth_metrics(prediction, target, min_depth=0.001, max_depth=10.0)
            rows.append(dict(sample_id=stem, test_position=i, matlab_index_1based=sample.matlab_index_1based, scene=sample.scene, scene_type=sample.scene_type, si_boundary_f1=0.25 + 0.01*i, **{k:metrics[k] for k in analysis.DEPTH_METRICS}))
        analysis.write_csv(root/'metrics_per_image.csv', rows)
        (root / 'metrics_summary.json').write_text('{}\n')
        config['baseline_outputs'][method] = str(root)
    monkeypatch.setattr(analysis, 'save_scale_plots', lambda *a, **k:None)
    monkeypatch.setattr(analysis, 'save_boundary_plots', lambda *a, **k:None)
    monkeypatch.setattr(analysis, 'save_boundary_case_panel', lambda *a, **k:None)
    monkeypatch.setattr(analysis, 'save_contact_sheet', lambda *a, **k:None)
    return config, Dataset(), tmp_path/'result', Dataset


def test_native_bias_and_oracle_are_separate_without_camera_inputs(experiment):
    config, dataset, output, _ = experiment
    result = analysis.analyze(config, dataset, output)
    assert set(result) == {'absolute_scale','boundary'}
    for method, ratio in (('depth_pro', 0.8), ('unidepth_v2', 1.2)):
        scores = result['absolute_scale']['models'][method]
        assert scores['scale_ratio_median'] == pytest.approx(ratio)
        assert scores['native_metrics']['abs_rel'] == pytest.approx(0.2)
        assert scores['native_metrics']['rmse'] == pytest.approx(np.sqrt((0.4**2+0.8**2)/2))
        assert scores['oracle_scale_aligned_metrics']['abs_rel'] < 1e-6
    assert not (output/'camera_ablation').exists()
    assert not (output/'cross_domain').exists()
    assert len(analysis.read_csv(output/'boundary/boundary_per_image.csv')) == 18
    assert len(analysis.read_csv(output/'absolute_scale/depth_bin_per_image.csv')) == 12
    assert len(json.loads((output/'boundary/qualitative_selection.json').read_text())['cases']) == 3


@pytest.mark.parametrize('corruption', ['duplicate','identity','metrics','nonfinite','shape'])
def test_native_sources_reject_corruption(experiment, corruption):
    config, dataset, output, _ = experiment
    root = Path(config['baseline_outputs']['depth_pro'])
    rows=analysis.read_csv(root/'metrics_per_image.csv')
    if corruption == 'duplicate': rows[1]['test_position'] = rows[0]['test_position']
    elif corruption == 'identity': rows[0]['matlab_index_1based'] = 999
    elif corruption == 'metrics': rows[0]['abs_rel'] = 0.0
    else:
        path = next((root/'predictions').glob('test_0000*.npz'))
        value = np.ones((18,24), np.float32)
        if corruption == 'nonfinite': value[0,0] = np.nan
        else: value = value[:10]
        np.savez_compressed(path, depth_m=value)
    analysis.write_csv(root/'metrics_per_image.csv', rows)
    with pytest.raises(ValueError): analysis.analyze(config, dataset, output)


def test_main_lifecycle_no_inference_and_refuses_overwrite(experiment, monkeypatch, tmp_path):
    config, _, output, dataset_class = experiment
    config_path = tmp_path/'analysis.yaml'
    config_path.write_text(yaml.safe_dump(config))
    dataset_file = tmp_path/'data'
    dataset_file.write_bytes(b'synthetic')
    monkeypatch.setattr(analysis,'load_config',lambda p:deepcopy(config))
    monkeypatch.setattr(analysis,'resolve_and_verify',lambda *a:dataset_file)
    monkeypatch.setattr(analysis,'NYUv2LabeledDataset',dataset_class)
    monkeypatch.setattr(sys,'argv',['run_key_analysis.py','--config',str(config_path),'--output-dir',str(output)])
    analysis.main()
    metadata=json.loads((output/'run_metadata.json').read_text())
    assert metadata['status']=='complete'
    assert metadata['sample_count']==3
    assert metadata['inference_performed_this_invocation'] is False
    assert len(metadata['baseline_sources']['depth_pro']['prediction_sha256'])==3
    assert metadata['environment']['device']=='cpu'
    old = (output/'run_metadata.json').read_bytes()
    with pytest.raises(FileExistsError):analysis.main()
    assert (output/'run_metadata.json').read_bytes()==old
    output2=tmp_path/'failed'
    monkeypatch.setattr(sys,'argv',['run_key_analysis.py','--config',str(config_path),'--output-dir',str(output2)])
    rows=analysis.read_csv(Path(config['baseline_outputs']['depth_pro'])/'metrics_per_image.csv')
    rows[0]['abs_rel']=0
    analysis.write_csv(Path(config['baseline_outputs']['depth_pro'])/'metrics_per_image.csv',rows)
    with pytest.raises(ValueError, match='metrics differ'):analysis.main()
    failed=json.loads((output2/'run_metadata.json').read_text())
    assert failed['status']=='failed' and 'ValueError' in failed['error']


@pytest.mark.parametrize('field,value', [('alignment','median'),('crop',44),('count',653),('oracle',False)])
def test_config_rejects_protocol_changes(tmp_path, field, value):
    path=analysis.PROJECT_ROOT/'configs/part2/key_analysis.yaml'
    config=yaml.safe_load(path.read_text())
    assert analysis.load_config(path)==config
    if field=='alignment':config['evaluation']['test_time_alignment']=value
    elif field=='crop':config['evaluation']['crop']['top']=value
    elif field=='count':config['dataset']['expected_test_images']=value
    else:config['absolute_scale_analysis']['oracle_alignment']['forbidden_as_primary_result']=value
    changed=tmp_path/'changed.yaml';changed.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):analysis.load_config(changed)


def test_adapters_call_official_rgb_only_api_and_keep_camera_outputs():
    rgb=np.zeros((8,10,3),np.uint8)
    calls=[]
    def dp_infer(image):
        calls.append(tuple(image.shape))
        return {'depth':torch.full((8,10),2.),'focallength_px':torch.tensor(500.)}
    dp=DepthProAdapter.__new__(DepthProAdapter)
    dp.device=torch.device('cpu');dp.model=SimpleNamespace(infer=dp_infer)
    dp.transform=lambda image:torch.from_numpy(np.array(image).copy()).permute(2,0,1)
    depth,focal,_,peak=dp.infer(rgb)
    assert np.all(depth==2) and focal==500 and peak==0
    def ud_infer(image):
        calls.append(tuple(image.shape))
        return {'depth':torch.full((1,1,8,10),3.),'intrinsics':torch.eye(3)[None]}
    ud=UniDepthV2Adapter.__new__(UniDepthV2Adapter)
    ud.device=torch.device('cpu');ud.model=SimpleNamespace(infer=ud_infer)
    depth,intrinsics,_,peak=ud.infer(rgb)
    assert np.all(depth==3) and np.array_equal(intrinsics,np.eye(3)) and peak==0
    assert calls==[(3,8,10),(3,8,10)]
