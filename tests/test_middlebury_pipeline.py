"""小型合成数据验证两组固定基线的保存与完整原始重算，不运行正式场景搜索。"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from pipelines.part1 import run_middlebury as run
from src.stereo import protocol
from src.stereo.io import MiddleburyCalibration, MiddleburyScene
from src.stereo.matching import StereoMatchResult, left_right_consistency_mask
from src.visualization.stereo import save_stereo_panel

ROOT = Path(__file__).resolve().parents[1]


def synthetic(name):
    camera = np.diag([100., 100., 1.])
    gt = np.ones((4,8), np.float32); gt[0,-1] = 2.
    return MiddleburyScene(name, np.zeros((4,8,3),np.uint8), np.zeros((4,8,3),np.uint8),
        gt, np.zeros((4,8),bool), np.ones((4,8),bool),
        MiddleburyCalibration(camera,camera,1.,200.,8,4,16,False))


def fake_match(left, right, **kwargs):
    raw = np.ones(left.shape[:2],np.float32)
    right = -raw.copy()
    raw_valid = np.ones(raw.shape,bool); raw_valid[0,0] = False; raw[0,0] = np.nan
    lr = left_right_consistency_mask(raw,right,threshold_px=kwargs['lr_threshold_px'])
    valid = raw_valid & lr; final = raw.copy(); final[~valid] = np.nan
    return StereoMatchResult(raw,final,right,raw_valid,np.ones(raw.shape,bool),lr,valid,16,.01)


def fake_panel(path, **kwargs):
    # The visual encoding itself is tested separately with the real renderer.
    Path(path).write_bytes(b'synthetic panel')


@pytest.fixture
def setup_pipeline(tmp_path, monkeypatch):
    config = protocol.load_config(ROOT/'configs/part1/middlebury_sgbm.yaml')
    path = tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(config,sort_keys=False))
    for module in (protocol,run):
        monkeypatch.setattr(module,'data_hashes',lambda cfg,names:{n:'synthetic' for n in names})
    for module in (run,):
        monkeypatch.setattr(module,'load_middlebury_scene',lambda path:synthetic(path.name))
        monkeypatch.setattr(module,'match_stereo',fake_match)
    monkeypatch.setattr(run,'save_stereo_panel',fake_panel)
    monkeypatch.setattr(run,'save_auxiliary',lambda *args:None)
    return path,config


def test_two_fixed_experiments_and_raw_verify(tmp_path, setup_pipeline):
    path, config = setup_pipeline
    output = tmp_path/'run'
    run.main(['--config', str(path), '--output-dir', str(output)])
    assert run.verify_run(output, config)['prediction_files_checked'] == 30
    summary = json.loads((output/'metrics_summary.json').read_text())
    assert set(summary) == {'bm_fixed', 'sgbm_fixed'}
    for entry in summary.values():
        assert entry['all']['scene_count'] == 15
        assert entry['held_out']['scene_count'] == 10
        assert entry['all']['disparity_mae_px'] > 0
        assert entry['all']['valid_prediction_fraction'] == pytest.approx(28/32)
    with pytest.raises(FileExistsError):
        run.main(['--config', str(path), '--output-dir', str(output)])
    file = output/'Adirondack'/'bm_fixed'/'prediction.npz'
    with np.load(file) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays['depth_m'][0,1] += 1
    np.savez_compressed(file, **arrays)
    with pytest.raises(AssertionError):
        run.validate_saved_outputs(output_dir=output, scenes=[synthetic('Adirondack')],
            rows=[r for r in run.read_rows(output/'metrics_per_scene.csv') if r['scene']=='Adirondack'], config=config)
    with pytest.raises(ValueError, match='Artifact changed'):
        run.verify_run(output, config)


def test_fixed_only_needs_no_search_and_records_failures(tmp_path,setup_pipeline,monkeypatch):
    path,config=setup_pipeline; output=tmp_path/'fixed'
    run.main(['--config',str(path),'--output-dir',str(output),'--experiments','bm_fixed'])
    assert run.verify_run(output,config)['prediction_files_checked']==15
    failed=tmp_path/'failed'
    def error(*args,**kwargs): raise RuntimeError('synthetic matcher failure')
    monkeypatch.setattr(run,'match_stereo',error)
    with pytest.raises(RuntimeError):
        run.main(['--config',str(path),'--output-dir',str(failed),'--experiments','bm_fixed'])
    assert json.loads((failed/'status.json').read_text())['status']=='failed'


def test_undefined_scene_mae_is_not_silently_dropped():
    row=dict(scene='a',split='development',experiment='bm_fixed',algorithm='bm',parameter_source='fixed',
        runtime_seconds=1.,disparity_mae_px=1.,valid_prediction_fraction=.5,
        evaluation_pixels=2,valid_prediction_pixels=1,invalid_prediction_pixels=1)
    rows=[row,{**row,'scene':'b','disparity_mae_px':None,'valid_prediction_fraction':0.}]
    result=run.macro_summary(rows)['bm_fixed']['all']
    assert result['disparity_mae_px'] is None
    assert result['disparity_mae_px_defined_scenes']==1
    assert result['valid_prediction_fraction']==.25


def test_whole_frame_invalid_and_lr_masks_are_independent_of_gt(tmp_path):
    raw=np.array([[np.nan,1.,1.]],np.float32)
    save_stereo_panel(tmp_path/'visualization.png',scene_name='synthetic',method_name='BM',
        rgb=np.zeros((1,3,3),np.uint8),target_disparity_px=np.ones((1,3),np.float32),
        prediction_disparity_px=np.array([[np.nan,np.nan,1.]],np.float32),
        evaluation_mask=np.zeros((1,3),bool),raw_prediction_valid=np.array([[False,True,True]]),
        lr_consistent=np.array([[False,False,True]]),raw_disparity_px=raw,disparity_max_px=16.)
    assert np.asarray(Image.open(tmp_path/'matcher_invalid.png')).tolist()==[[255,0,0]]
    assert np.asarray(Image.open(tmp_path/'invalid.png')).tolist()==[[255,255,0]]
    assert np.asarray(Image.open(tmp_path/'lr_inconsistent.png')).tolist()==[[0,255,0]]
    assert np.asarray(Image.open(tmp_path/'lr_status.png')).tolist()==[[0,1,2]]


def test_visual_refresh_accepts_only_bound_display_changes(tmp_path, setup_pipeline):
    path, config = setup_pipeline
    output = tmp_path/'fixed_refresh'
    run.main(['--config',str(path),'--output-dir',str(output),'--experiments','bm_fixed'])
    record = json.loads((output/'run_metadata.json').read_text())
    target = 'Adirondack/bm_fixed/visualization.png'
    (output/target).write_bytes(b'authorized six-panel display')
    with pytest.raises(ValueError,match='Artifact changed'):
        run.verify_run(output,config)
    refresh = {'status':'complete',
        'original_run_metadata_sha256':protocol.sha256(output/'run_metadata.json'),
        'source_sha256':protocol.source_hashes(),
        'replacements':{target:{'original_sha256':record['artifacts'][target],
                                'sha256':protocol.sha256(output/target)}}}
    protocol.write_json(output/'visualization_refresh.json',refresh)
    assert run.verify_run(output,config)['prediction_files_checked']==15
    assert protocol.sha256(output/'run_metadata.json')==refresh['original_run_metadata_sha256']
    # A refresh never authorizes score/prediction changes or a different run.
    refresh['replacements']['metrics_summary.json']={
        'original_sha256':record['artifacts']['metrics_summary.json'],
        'sha256':protocol.sha256(output/'metrics_summary.json')}
    protocol.write_json(output/'visualization_refresh.json',refresh)
    with pytest.raises(ValueError,match='non-display artifacts'):
        run.verify_run(output,config)
