#!/usr/bin/env python3
"""Direction B：795张开发预测→拟合一个尺度→冻结后评价654张test。"""
from pathlib import Path
import argparse
import hashlib
import os
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipelines.part3 import common as c
from src.refinement.global_scale import image_log_ratio, fit_global_scale, apply_global_scale


def adapter(model, mc, config):
    expected = config['models'][model]['prediction_env']
    if os.environ.get('CONDA_DEFAULT_ENV') != expected:
        raise RuntimeError('Development inference requires Conda environment ' + expected)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not begin a full development prediction')
    print('CUDA:', torch.cuda.get_device_name(0), 'torch:', torch.__version__, flush=True)
    if model == 'depth_pro':
        from src.models.depth_pro_adapter import DepthProAdapter
        return DepthProAdapter(repository=ROOT / mc['model']['repository'], checkpoint=ROOT / mc['model']['checkpoint'])
    from src.models.unidepth_v2_adapter import UniDepthV2Adapter
    return UniDepthV2Adapter(repository=ROOT / mc['model']['repository'], checkpoint_dir=(ROOT / mc['model']['checkpoint']).parent)


def predict(output, config, mc, model, limit=None):
    inputs = c.resources(config, model, mc)
    record = c.begin(output, 'b_development', config, model, inputs, limit=limit,
                     depth_type='metric', unit='meter', alignment='none', native_metric=True)
    try:
        model_instance = adapter(model, mc, config)
        rows = []
        (output / 'predictions').mkdir()
        with c.dataset(mc, development=True) as data:
            for i in c.positions(len(data), limit):
                sample = data.get_test_sample(i)
                z, _, seconds, memory = model_instance.infer(sample.rgb)
                z = cv2.resize(z, (640,480), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                c.valid_depth(z)
                key = c.development_stem(sample); p = output / 'predictions' / (key + '.npz')
                np.savez_compressed(p, depth_m=z)
                rows.append(dict(sample_id=key, position=i, matlab_index=sample.matlab_index_1based,
                                 rgb_sha256=hashlib.sha256(sample.rgb.tobytes()).hexdigest(),
                                 prediction_sha256=c.sha(p), inference_seconds=seconds,
                                 peak_gpu_memory_bytes=memory))
                if i % 50 == 0: print(f'{model} development {i + 1}/795', flush=True)
        c.write_csv(output / 'predictions.csv', rows)
        record['model_environment'] = model_instance.environment()
        c.seal(output, record, formal_complete=limit is None and len(rows) == 795)
    except BaseException as exc:
        c.failed(output, exc); raise


def development_rows(folder, mc, *, require_full=True):
    record = c.checked(folder, 'b_development', require_full=require_full)
    rows = c.read_csv(folder / 'predictions.csv')
    expected = 795 if require_full else len(c.positions(795, record['limit']))
    if len(rows) != expected or len({r['sample_id'] for r in rows}) != expected:
        raise ValueError('Development prediction coverage differs')
    keys = {r['sample_id'] for r in rows}
    if {p.stem for p in (folder / 'predictions').glob('*.npz')} != keys:
        raise ValueError('Unexpected or missing development arrays')
    output = []
    with c.dataset(mc, development=True) as data:
        for i, row in enumerate(rows):
            sample = data.get_test_sample(i); key = c.development_stem(sample)
            p = folder / 'predictions' / (key + '.npz')
            if (row['sample_id'] != key or int(row['position']) != i
                    or int(row['matlab_index']) != sample.matlab_index_1based
                    or row['rgb_sha256'] != hashlib.sha256(sample.rgb.tobytes()).hexdigest()
                    or row['prediction_sha256'] != c.sha(p)):
                raise ValueError('Development prediction identity differs')
            with np.load(p, allow_pickle=False) as arrays:
                z = c.valid_depth(arrays['depth_m'])
                ratio = image_log_ratio(z[c.CROP], sample.depth_m[c.CROP])
            output.append(dict(**row, log_ratio=ratio))
    return record, output


def fit(output, development, config, mc, model):
    parent, rows = development_rows(development, mc)
    if parent['model'] != model or parent['config'] != config:
        raise ValueError('Development model/config differs')
    scale = fit_global_scale([r['log_ratio'] for r in rows])
    record = c.begin(output, 'b_frozen_scale', config, model, parent['input_files'], parents=[development])
    try:
        c.write_csv(output / 'calibration_development.csv', rows)
        c.write_json(output / 'frozen_scale.json', dict(scale=scale, development_images=795,
                     test_images_used_for_fitting=0, formula=config['formula'],
                     development_csv_sha256=c.sha(output / 'calibration_development.csv')))
        c.seal(output, record, formal_complete=True)
    except BaseException as exc:
        c.failed(output, exc); raise


def verify_fit(folder, mc):
    record = c.checked(folder, 'b_frozen_scale')
    parent, replay = development_rows(Path(record['parents'][0]), mc)
    if parent['model'] != record['model'] or parent['config'] != record['config']:
        raise ValueError('Calibration source model/config differs')
    saved = c.read_csv(folder / 'calibration_development.csv')
    if len(saved) != len(replay): raise ValueError('Calibration row count differs')
    for original, row in zip(saved, replay, strict=True):
        for key in ('sample_id','prediction_sha256','rgb_sha256'):
            if original[key] != row[key]: raise ValueError('Calibration source identity differs')
        c.assert_close(float(original['log_ratio']), row['log_ratio'], 'Development log ratio')
    frozen = c.read_json(folder / 'frozen_scale.json')
    if (frozen['development_images'] != 795 or frozen['test_images_used_for_fitting'] != 0
            or frozen['development_csv_sha256'] != c.sha(folder / 'calibration_development.csv')):
        raise ValueError('Invalid calibration provenance')
    c.assert_close(frozen['scale'], fit_global_scale([r['log_ratio'] for r in replay]), 'Frozen global scale')
    return record, float(frozen['scale'])


def evaluate(output, frozen, config, mc, model, limit=None):
    parent, scale = verify_fit(frozen, mc)
    if parent['model'] != model or parent['config'] != config:
        raise ValueError('Frozen scale model/config differs')
    folder, baseline, inputs = c.baseline_files(config, model, mc)
    inputs.update(c.resources(config, model, mc))
    record = c.begin(output, 'b_evaluation', config, model, inputs, parents=[frozen], limit=limit,
                     scale=scale, depth_type='metric', unit='meter', alignment='none', native_metric=False,
                     calibration='single_scale_fitted_on_official_trainNdxs')
    try:
        rows = []; (output / 'predictions').mkdir()
        with c.dataset(mc) as data:
            for i in c.positions(len(data), limit):
                sample = data.get_test_sample(i); key = c.stem(sample)
                raw, source = c.native(folder, sample); adjusted = apply_global_scale(raw, scale)
                c.valid_depth(adjusted)
                np.savez_compressed(output / 'predictions' / (key + '.npz'), depth_m=adjusted)
                for method, z in [('RAW',raw), ('B_global',adjusted)]:
                    values = c.score(z, sample.depth_m, boundary=False)
                    if method == 'RAW': c.raw_matches(values, baseline[key])
                    rows.append(dict(sample_id=key, scene=sample.scene, method=method,
                                     source_prediction_sha256=source, **values))
        c.write_csv(output / 'per_image.csv', rows)
        c.write_json(output / 'summary.json', dict(samples=len(rows)//2, scale=scale,
                     means=c.means(rows, c.METRICS[:-1]),
                     limitation='Development-GT calibrated, not untouched zero-shot; no test fitting, no B+C or cross-domain claim.'))
        c.seal(output, record, formal_complete=limit is None and len(rows)==1308)
    except BaseException as exc:
        c.failed(output, exc); raise


def verify(folder, mc):
    record = c.checked(folder, require_full=False)
    if record['kind'] == 'b_development':
        _, rows = development_rows(folder, mc, require_full=False)
        return dict(passed=True, kind=record['kind'], prediction_files_checked=len(rows), formal_complete=record['formal_complete'])
    if record['kind'] == 'b_frozen_scale':
        _, scale = verify_fit(folder, mc)
        return dict(passed=True, kind=record['kind'], development_images=795, scale=scale, formal_complete=True)
    if record['kind'] != 'b_evaluation': raise ValueError('Not a B run')
    parent, scale = verify_fit(Path(record['parents'][0]), mc)
    if parent['model'] != record['model']: raise ValueError('Scale model differs')
    c.assert_close(scale, record['scale'], 'Evaluation scale')
    config, model = record['config'], record['model']
    source = ROOT / config['models'][model]['baseline']
    baseline = {r['sample_id']:r for r in c.read_csv(source / 'metrics_per_image.csv')}
    saved = c.read_csv(folder / 'per_image.csv'); replay = []
    expected = []
    with c.dataset(mc) as data:
        for i in c.positions(len(data), record['limit']):
            sample = data.get_test_sample(i); key = c.stem(sample); expected.append(key)
            raw, digest = c.native(source, sample); adjusted = apply_global_scale(raw, scale)
            with np.load(folder / 'predictions' / (key + '.npz'), allow_pickle=False) as z:
                c.assert_close(c.valid_depth(z['depth_m']), adjusted, 'B saved prediction')
            for method, prediction in [('RAW',raw),('B_global',adjusted)]:
                values = c.score(prediction, sample.depth_m, boundary=False)
                if method == 'RAW': c.raw_matches(values, baseline[key])
                replay.append(dict(sample_id=key, scene=sample.scene, method=method, source_prediction_sha256=digest, **values))
    if len(saved) != len(replay) or {p.stem for p in (folder / 'predictions').glob('*.npz')} != set(expected):
        raise ValueError('B evaluation coverage differs')
    for a, b in zip(saved, replay, strict=True):
        for key, value in b.items():
            if key in ('sample_id','scene','method','source_prediction_sha256'):
                if a[key] != value: raise ValueError('B score identity differs')
            else: c.assert_close(float(a[key]), value, 'B metric/' + key)
    if c.means(replay, c.METRICS[:-1]) != c.read_json(folder / 'summary.json')['means']:
        raise ValueError('B summary differs')
    return dict(passed=True, kind=record['kind'], samples=len(expected), scale=scale,
                raw_and_calibrated_predictions_recomputed=True, formal_complete=record['formal_complete'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check','predict-development','fit','evaluate','verify'])
    p.add_argument('--model', choices=['depth_pro','unidepth_v2'], required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/part3/direction_b.yaml')
    p.add_argument('--output-dir', type=Path); p.add_argument('--development-dir', type=Path)
    p.add_argument('--frozen-dir', type=Path); p.add_argument('--run', type=Path)
    p.add_argument('--limit', type=int); p.add_argument('--report', type=Path)
    a = p.parse_args(argv); cv2.setNumThreads(1)
    config, mc = c.load_config(a.config, a.model, 'b')
    base = ROOT / config['output_root'] / a.model
    if a.action == 'check':
        inputs = c.resources(config, a.model, mc); folder, _, files = c.baseline_files(config, a.model, mc)
        print('Resources/SHA, official 795/654 disjoint split and 654 native baseline identities passed:', folder)
        return
    if a.action == 'verify':
        if not a.run: p.error('verify requires --run')
        record = c.checked(a.run, require_full=False)
        if record['model'] != a.model or record['config'] != config: raise ValueError('Verification model/config differs')
        c.report(a.run, verify(a.run.resolve(), mc), a.report); return
    if a.limit is not None and a.action == 'fit': p.error('A global scale cannot be frozen from a subset')
    output = (a.output_dir or base / {'predict-development':'development_v1','fit':'frozen_scale_v1','evaluate':'evaluation_v1'}[a.action]).resolve()
    if a.action == 'predict-development': predict(output, config, mc, a.model, a.limit)
    elif a.action == 'fit': fit(output, (a.development_dir or base / 'development_v1').resolve(), config, mc, a.model)
    else: evaluate(output, (a.frozen_dir or base / 'frozen_scale_v1').resolve(), config, mc, a.model, a.limit)


if __name__ == '__main__': main()
