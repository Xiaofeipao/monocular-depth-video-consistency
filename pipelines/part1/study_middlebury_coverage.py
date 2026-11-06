#!/usr/bin/env python3
"""开发集误差—覆盖率研究：完整搜索、冻结、独立场景评估及原始重算。"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipelines.part1 import run_middlebury as evaluation
from pipelines.part1.tune_middlebury import measure
from src.analysis.stereo_coverage import (LABELS, REGIONS, pareto_frontier, regional_masks,
                                           regional_metrics, select_candidates, summarize)
from src.stereo.io import load_middlebury_scene
from src.stereo.protocol import data_hashes, environment, load_config, sha256, source_hashes, write_json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sources():
    names = ('src/analysis/stereo_coverage.py', 'pipelines/part1/study_middlebury_coverage.py',
             'scripts/part1/study_middlebury_coverage.sh')
    return {**source_hashes(), **{p: sha256(ROOT / p) for p in names}}


def load_settings(path):
    settings = yaml.safe_load(Path(path).read_text())
    if settings['protocol'] != 'middlebury_coverage_v1':
        raise ValueError('Unrecognized coverage protocol')
    tolerance = settings['coverage_tolerance']
    region = settings['regions']
    if (not np.isfinite(tolerance) or not 0 <= tolerance <= 1
            or not isinstance(region['texture_window'], int) or region['texture_window'] <= 0
            or region['texture_window'] % 2 != 1
            or any(not np.isfinite(region[k]) or region[k] < 0 for k in
                   ('texture_variance_threshold', 'boundary_sobel_threshold_px'))):
        raise ValueError('Invalid coverage or region settings')
    return settings, load_config(ROOT / settings['base_config'])


def grid(config, method):
    tune = config['tuning'][method]
    names = list(tune['search'])
    return [dict(config['experiments'][method + '_fixed']['parameters'])] + [
        {**tune['fixed'], **dict(zip(names, values)), 'lr_threshold_px': 1.}
        for values in itertools.product(*(tune['search'][k] for k in names))]


def write_candidates(path, rows):
    evaluation.write_csv(path, [{**r, 'parameters': json.dumps(r['parameters'], sort_keys=True)} for r in rows])


def read_candidates(path):
    with Path(path).open(newline='') as stream:
        return [dict(method=r['method'], candidate=int(r['candidate']),
                     mae=float(r['mae']) if r['mae'] else None, coverage=float(r['coverage']),
                     parameters=json.loads(r['parameters'])) for r in csv.DictReader(stream)]


def read_numeric(path, text_keys):
    with Path(path).open(newline='') as stream:
        return [{k: v if k in text_keys else float(v) if v else None for k, v in r.items()}
                for r in csv.DictReader(stream)]


def selection_from_rows(rows, settings, config):
    chosen = []
    for method in ('bm', 'sgbm'):
        group = [r for r in rows if r['method'] == method]
        expected = grid(config, method)
        if (len(group) != len(expected) or [r['candidate'] for r in group] != list(range(len(expected)))
                or [r['parameters'] for r in group] != expected):
            raise ValueError('Candidate grid is incomplete, duplicated or changed')
        for label, row in select_candidates(group, settings['coverage_tolerance']).items():
            chosen.append(dict(selection=label, **row))
    return dict(protocol=settings['protocol'], development=config['split']['development_scenes'],
                held_out=config['split']['evaluation_scenes'], coverage_tolerance=settings['coverage_tolerance'],
                tie_break='objective, secondary objective, candidate order', selected=chosen)


def save_frontier(path, rows, chosen, method):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    group = [r for r in rows if r['method'] == method and r['mae'] is not None]
    frontier = sorted(pareto_frontier(group), key=lambda r: r['coverage'])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([r['coverage'] for r in group], [r['mae'] for r in group], s=12, label='Development candidates')
    ax.plot([r['coverage'] for r in frontier], [r['mae'] for r in frontier], label='Pareto frontier')
    for row in chosen:
        if row['method'] == method:
            ax.scatter(row['coverage'], row['mae'], label=row['selection'])
    ax.set(xlabel='Valid prediction fraction', ylabel='Disparity MAE (px)', title=method.upper() + ' / 5 development scenes')
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def compare_reference(output):
    """只在结果生成后比较；参考文件不进入选参或评价。"""
    reference = output / 'provenance/reference'
    if not reference.exists():
        return None
    expected = json.loads((reference / 'selection.json').read_text())['selected']
    actual = json.loads((output / 'selection.json').read_text())['selected']
    for left, right in zip(expected, actual, strict=True):
        for key in ('method', 'selection', 'candidate'):
            if left[key] != right[key]:
                raise AssertionError('Reference selection differs: ' + key)
        lp = {**left['parameters'], 'lr_threshold_px': 1.}
        if lp != right['parameters']:
            raise AssertionError('Reference parameters differ')
    checked = 0
    for name, keys in [('development.csv', ('method', 'candidate')),
                       ('held_out.csv', ('scene', 'method', 'selection', 'region'))]:
        with (reference / name).open(newline='') as stream:
            original = list(csv.DictReader(stream))
        with (output / name).open(newline='') as stream:
            current = list(csv.DictReader(stream))
        index = {tuple(r[k] for k in keys): r for r in current}
        if len(index) != len(current) or len(original) != len(current):
            raise AssertionError('Reference row coverage differs')
        for row in original:
            pair = index[tuple(row[k] for k in keys)]
            for metric in ('mae', 'coverage'):
                evaluation.assert_equal(float(pair[metric]) if pair[metric] else None,
                                        float(row[metric]) if row[metric] else None, name + '/' + metric)
                checked += 1
    return dict(passed=True, rows_checked=checked // 2, numeric_values_checked=checked,
                selected_parameter_sets_checked=6, rtol=1e-6, atol=1e-8,
                note='Local float64 scoring; reference MAE used float32. No predictions changed to match scores.')


def verify(output, settings, config):
    status = json.loads((output / 'status.json').read_text())
    record = json.loads((output / 'run_metadata.json').read_text())
    if status.get('status') != 'complete' or status['metadata_sha256'] != sha256(output / 'run_metadata.json'):
        raise ValueError('Run incomplete or metadata changed')
    names = config['split']['development_scenes'] + config['split']['evaluation_scenes']
    if (record['settings'] != settings or record['config'] != config or record['source_sha256'] != sources()
            or record['input_sha256'] != data_hashes(config, names)):
        raise ValueError('Source, input data or configuration changed')
    for name, digest in record['artifacts'].items():
        if sha256(output / name) != digest:
            raise ValueError('Artifact changed: ' + name)
    candidates = read_candidates(output / 'development.csv')
    selection = selection_from_rows(candidates, settings, config)
    if selection != json.loads((output / 'selection.json').read_text()):
        raise ValueError('Frozen selection does not replay')
    details = read_numeric(output / 'development_per_scene.csv', {'method', 'scene'})
    expected = {(r['method'], r['candidate'], n) for r in candidates for n in selection['development']}
    if len(details) != len(expected) or {(r['method'], r['candidate'], r['scene']) for r in details} != expected:
        raise ValueError('Development scene coverage differs')
    for row in candidates:
        selected = [r for r in details if (r['method'], r['candidate']) == (row['method'], row['candidate'])]
        for source, target in [('disparity_mae_px', 'mae'), ('valid_prediction_fraction', 'coverage')]:
            values = [r[source] for r in selected]
            value = float(np.mean(values)) if all(v is not None for v in values) else None
            evaluation.assert_equal(value, row[target], 'Development macro score')
    scenes = [load_middlebury_scene(ROOT / config['dataset']['root'] / n) for n in selection['held_out']]
    rows = evaluation.read_rows(output / 'evaluation/metrics_per_scene.csv')
    experiments = {r['method'] + '_' + r['selection'] for r in selection['selected']}
    if (len(rows) != len(scenes) * len(experiments)
            or {(r['scene'], r['experiment']) for r in rows} != {(s.name, e) for s in scenes for e in experiments}):
        raise ValueError('Held-out prediction coverage differs')
    for row in rows:
        method = row['experiment'].split('_', 1)[0]
        selected = next(r for r in selection['selected'] if r['method'] == method
                        and row['experiment'] == method + '_' + r['selection'])
        meta = json.loads((output / 'evaluation' / row['scene'] / row['experiment'] / 'metadata.json').read_text())
        if (row['split'] != 'held_out' or row['algorithm'] != method
                or meta['matcher_parameters'] != selected['parameters'] or meta['algorithm'] != method):
            raise ValueError('Prediction identity or frozen parameters differ')
    report = evaluation.validate_saved_outputs(output_dir=output / 'evaluation', scenes=scenes,
                                               rows=rows, config=config)
    regions = collect_regions(output, scenes, selection, settings)
    saved = read_numeric(output / 'held_out.csv', {'scene', 'method', 'selection', 'region'})
    if len(saved) != len(regions):
        raise ValueError('Regional table coverage differs')
    for original, replay in zip(saved, regions, strict=True):
        for key, value in replay.items():
            if key in ('scene', 'method', 'selection', 'region'):
                if original[key] != value:
                    raise ValueError('Regional table identity changed')
            else:
                evaluation.assert_equal(original[key], value, 'Regional score/' + key)
    if summarize(regions) != json.loads((output / 'summary.json').read_text())['means']:
        raise ValueError('Regional macro summary differs')
    if compare_reference(output) != record['reference_comparison']:
        raise ValueError('Reference comparison differs')
    return dict(**report, regional_rows_checked=len(regions), development_rows_checked=len(details),
                selection_replayed=True, completed_at_utc=timestamp())


def collect_regions(output, scenes, selection, settings):
    rows = []
    for scene in scenes:
        masks = regional_masks(scene, settings['regions'])
        for selected in selection['selected']:
            method, label = selected['method'], selected['selection']
            with np.load(output / 'evaluation' / scene.name / (method + '_' + label) / 'prediction.npz',
                         allow_pickle=False) as saved:
                for region in REGIONS:
                    rows.append(dict(scene=scene.name, method=method, selection=label, region=region,
                                     **regional_metrics(saved['disparity_px'], scene.gt_disparity_px,
                                                        masks[region], saved['valid_mask'])))
    return rows


def run(output, settings, config, reference=None):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', dict(status='running', command=sys.argv, started_at_utc=timestamp()))
    try:
        initial = sources()
        names = config['split']['development_scenes'] + config['split']['evaluation_scenes']
        inputs = data_hashes(config, names)
        provenance = output / 'provenance'
        write_json(provenance / 'config.json', config)
        write_json(provenance / 'settings.json', settings)
        for name in initial:
            dest = provenance / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dest)
        dev = [load_middlebury_scene(ROOT / config['dataset']['root'] / n) for n in config['split']['development_scenes']]
        candidates, details = [], []
        for method in ('bm', 'sgbm'):
            params = grid(config, method)
            for index, values in enumerate(params):
                means, per_scene = measure(values, method, dev)
                candidates.append(dict(method=method, candidate=index, mae=means['disparity_mae_px'],
                                       coverage=means['valid_prediction_fraction'], parameters=values))
                details.extend(dict(method=method, candidate=index, **row) for row in per_scene)
                if index % 20 == 0:
                    print(f'{method} development {index + 1}/{len(params)}', flush=True)
        selection = selection_from_rows(candidates, settings, config)
        write_candidates(output / 'development.csv', candidates)
        evaluation.write_csv(output / 'development_per_scene.csv', details)
        # 先冻结，再加载任何保留场景的图像和GT数值。
        write_json(output / 'selection.json', selection)
        frozen = sha256(output / 'selection.json')
        chosen_at = timestamp()
        for method in ('bm', 'sgbm'):
            save_frontier(output / (method + '_pareto.png'), candidates, selection['selected'], method)
        evaluated_at = timestamp()
        scenes = [load_middlebury_scene(ROOT / config['dataset']['root'] / n) for n in selection['held_out']]
        eval_config = copy.deepcopy(config)
        for row in selection['selected']:
            eval_config['experiments'][row['method'] + '_' + row['selection']] = dict(
                algorithm=row['method'], parameter_source='frozen_coverage_development', parameters=row['parameters'])
        rows = []
        for scene in scenes:
            for row in selection['selected']:
                name = row['method'] + '_' + row['selection']
                rows.append(evaluation.evaluate_scene_method(scene, experiment=name, config=eval_config,
                                                             output_dir=output / 'evaluation'))
            print('held-out ' + scene.name, flush=True)
        evaluation.write_csv(output / 'evaluation/metrics_per_scene.csv', rows)
        regional = collect_regions(output, scenes, selection, settings)
        evaluation.write_csv(output / 'held_out.csv', regional)
        write_json(output / 'summary.json', dict(protocol=settings['protocol'], means=summarize(regional),
                    limitation='Matching constraints apply only on development; conditional MAE and coverage must be read together; regions overlap.'))
        if reference is not None:
            dest = provenance / 'reference'; dest.mkdir()
            for name in ('development.csv', 'held_out.csv', 'selection.json', 'summary.json'):
                shutil.copyfile(reference / name, dest / name)
        comparison = compare_reference(output)
        if initial != sources() or inputs != data_hashes(config, names) or frozen != sha256(output / 'selection.json'):
            raise ValueError('Sources, inputs or frozen selection changed during execution')
        record = dict(protocol=settings['protocol'], config=config, settings=settings, command=sys.argv,
                      source_sha256=initial, input_sha256=inputs, environment=environment(),
                      selection_frozen_at_utc=chosen_at, held_out_loading_started_at_utc=evaluated_at,
                      selection_sha256=frozen, created_at_utc=timestamp(), formal_training_run=False,
                      reference_comparison=comparison,
                      artifacts={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                                 if p.is_file() and p.name != 'status.json'})
        write_json(output / 'run_metadata.json', record)
        write_json(output / 'status.json', dict(status='complete', metadata_sha256=sha256(output / 'run_metadata.json')))
        report = verify(output, settings, config)
        write_json(output / 'verification.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except BaseException as exc:
        write_json(output / 'status.json', dict(status='failed', error_type=type(exc).__name__, error=str(exc)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/part1/middlebury_coverage.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--mode', choices=('run', 'verify'), default='run')
    parser.add_argument('--reference-dir', type=Path, help='可选：已有证据，只在本地结果生成后比较')
    parser.add_argument('--report', type=Path, help='verify报告另存位置；默认只打印，不覆盖已有记录')
    args = parser.parse_args(argv)
    cv2.setNumThreads(1)
    settings, config = load_settings(args.config)
    output = (args.output_dir or ROOT / settings['output_dir']).resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Outputs must be under outputs/')
    if args.mode == 'verify':
        if args.reference_dir:
            parser.error('verify使用运行时保存的参考副本，不读取外部参考')
        report = verify(output, settings, config)
        if args.report:
            if args.report.exists():
                raise FileExistsError(args.report)
            write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False))
    else:
        if args.report:
            parser.error('--report只用于verify')
        run(output, settings, config, args.reference_dir)


if __name__ == '__main__':
    main()
