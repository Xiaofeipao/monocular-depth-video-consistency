"""开发/保留隔离、约束泛化、区域分母及原始重算的合成检查。"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.stereo_coverage import (pareto_frontier, regional_metrics, select_candidates,
                                           summarize)
from src.stereo.protocol import sha256, write_json
from pipelines.part1 import study_middlebury_coverage as study, tune_middlebury as tuning
from pipelines.part1 import run_middlebury as evaluation
from src.stereo.io import MiddleburyCalibration, MiddleburyScene
from src.stereo.matching import StereoMatchResult, left_right_consistency_mask


def candidate(index, mae, coverage):
    return dict(candidate=index, mae=mae, coverage=coverage)


def test_rules_apply_only_to_development_and_have_deterministic_ties():
    rows = [candidate(0, 2., .5), candidate(1, 1., .49), candidate(2, 1.9, .7),
            candidate(3, 1., .49), candidate(4, None, 0.)]
    chosen = select_candidates(rows, .011)
    assert chosen['matched_coverage']['candidate'] == 1
    assert chosen['matched_error']['candidate'] == 2
    assert 0 in [x['candidate'] for x in pareto_frontier([rows[0]])]
    assert {r['candidate'] for r in pareto_frontier(rows)} == {1, 2, 3}
    assert rows[0]['coverage'] == .5


@pytest.mark.parametrize('rows', [
    [candidate(0, None, 0.)], [candidate(0, 1., .5), candidate(0, .2, .8)],
    [candidate(0, float('nan'), .5)], [candidate(1, 1., .5)],
    [candidate(0, 1., float('inf'))],
])
def test_undefined_reference_and_invalid_candidates_are_rejected(rows):
    with pytest.raises(ValueError):
        select_candidates(rows)


def test_regional_counts_empty_domain_and_no_valid_prediction_remain_distinct():
    gt = np.array([[1., 2., np.nan]], np.float32)
    pred = np.array([[1., np.nan, 10.]], np.float32)
    mask = np.ones_like(gt, bool)
    row = regional_metrics(pred, gt, mask, np.isfinite(pred))
    assert row['mae'] == 0 and row['coverage'] == .5
    assert row['evaluation_pixels'] == 2 and row['invalid_prediction_pixels'] == 1
    assert row['bad1'] == .5
    empty = regional_metrics(pred, gt, ~mask, np.isfinite(pred))
    absent = regional_metrics(pred, gt, mask, ~mask)
    assert empty['coverage'] is None and absent['coverage'] == 0
    group = [dict(scene='a', method='bm', selection='fixed', region='all_GT', **row),
             dict(scene='b', method='bm', selection='fixed', region='all_GT', **absent)]
    summary = summarize(group)[0]
    assert summary['mae'] is None and summary['mae_defined_scenes'] == 1
    assert summary['coverage'] == .25 and summary['supported_scenes'] == 2


def synthetic(name):
    camera = np.diag([100., 100., 1.])
    gt = np.ones((4, 8), np.float32)
    return MiddleburyScene(name, np.zeros((4, 8, 3), np.uint8), np.zeros((4, 8, 3), np.uint8),
                           gt, np.ones((4, 8), bool), np.zeros((4, 8), bool),
                           MiddleburyCalibration(camera, camera, 1., 200., 8, 4, 16, False))


def fake_match(left, right, **kwargs):
    raw = np.ones(left.shape[:2], np.float32)
    back = -raw.copy()
    valid = np.ones(raw.shape, bool)
    lr = left_right_consistency_mask(raw, back, threshold_px=1.)
    final = raw.copy(); final[~lr] = np.nan
    return StereoMatchResult(raw, final, back, valid, valid, lr, lr, 16, .01)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    settings, config = study.load_settings(study.ROOT / 'configs/part1/middlebury_coverage.yaml')
    for method in ('bm', 'sgbm'):
        config['tuning'][method]['search'] = {
            k: [0, 10] if k == 'uniqueness_ratio' else [config['experiments'][method + '_fixed']['parameters'][k]]
            for k in config['tuning'][method]['search']}
    out = tmp_path / 'run'
    monkeypatch.setattr(study, 'data_hashes', lambda cfg, names: {n: 'synthetic' for n in names})
    held = set(config['split']['evaluation_scenes'])
    def load(path):
        if path.name in held:
            assert (out / 'selection.json').exists(), 'Held-out read before selection freeze'
        return synthetic(path.name)
    monkeypatch.setattr(study, 'load_middlebury_scene', load)
    monkeypatch.setattr(tuning, 'match_stereo', fake_match)
    monkeypatch.setattr(evaluation, 'match_stereo', fake_match)
    monkeypatch.setattr(evaluation, 'save_stereo_panel', lambda p, **kwargs: Path(p).write_bytes(b'panel'))
    monkeypatch.setattr(study, 'save_frontier', lambda p, *args: Path(p).write_bytes(b'frontier'))
    study.run(out, settings, config)
    return out, settings, config


def test_full_freeze_to_raw_replay_and_no_overwrite(completed):
    out, settings, config = completed
    report = study.verify(out, settings, config)
    assert report['prediction_files_checked'] == 60
    assert report['regional_rows_checked'] == 300
    assert report['development_rows_checked'] == 30
    assert report['lr_and_geometry_recomputed']
    with pytest.raises(FileExistsError):
        study.run(out, settings, config)


def test_artifact_and_repacked_regional_score_are_rejected(completed):
    out, settings, config = completed
    table = out / 'held_out.csv'
    before = table.read_text()
    table.write_text(before.replace(',all_GT,', ',not_a_region,', 1))
    with pytest.raises(ValueError, match='Artifact changed'):
        study.verify(out, settings, config)
    # 即使更新外层SHA，原始预测重算也会拒绝伪造的行身份。
    record = json.loads((out / 'run_metadata.json').read_text())
    record['artifacts']['held_out.csv'] = sha256(table)
    write_json(out / 'run_metadata.json', record)
    write_json(out / 'status.json', dict(status='complete', metadata_sha256=sha256(out / 'run_metadata.json')))
    with pytest.raises(ValueError, match='Regional table identity'):
        study.verify(out, settings, config)


def test_saved_disparity_mutation_is_detected(completed):
    out, settings, config = completed
    path = next((out / 'evaluation').rglob('prediction.npz'))
    with np.load(path) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays['disparity_px'][1, 2] += 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='Artifact changed'):
        study.verify(out, settings, config)
