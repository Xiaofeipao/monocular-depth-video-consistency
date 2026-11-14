#!/usr/bin/env python3
"""C v2：开发集场景划分→六候选选参/独立验证→冻结→正式对比/数值重放。"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import sys
import time
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipelines.part3 import common as c
from pipelines.part3_studies import common as study, direction_c as original
from pipelines.part3_c_v2 import common as v
from src.part3_c_v2.refine import Parameters, denoise, refine
from src.part3_studies.classical import apply as classical
from src.part3_studies.diagnostics import prepare as prepare_diagnostics, analyze


def output_path(cfg, model, stage):
    return ROOT/cfg['output_root']/model/(stage+'_v1')


def check(cfg, base, mc, diagnostic, model):
    """Quick read-only presence/source check, not a full SHA/score verification."""
    folder = ROOT/cfg['development_root']/model/'development_v1'
    record = c.read_json(folder/'run_metadata.json')
    if record['model'] != model or record['kind'] != 'b_development' or not record['formal_complete']:
        raise ValueError('Full correct-model native development predictions required')
    if record['source_sha256'] != c.sources():
        raise ValueError('Development frozen source differs')
    if len(list((folder/'predictions').glob('*.npz'))) != 795:
        raise ValueError('Require 795 development predictions')
    ref = ROOT/base['output_root']/model/'evaluation_v1'
    old = c.read_json(ref/'run_metadata.json')
    if old['source_sha256'] != c.sources() or old['model'] != model or not old['formal_complete']:
        raise ValueError('Full v1 C source/model differs')
    if len(list((ROOT/base['models'][model]['baseline']/'predictions').glob('*.npz'))) != 654:
        raise ValueError('Full native test baseline required')
    path = ROOT/base['output_root']/model/diagnostic['output_name']
    old = c.read_json(path/'run_metadata.json')
    if old['source_sha256'] != study.sources() or old['model'] != model or not old['formal_complete']:
        raise ValueError('Full classic/diagnostic v1 parent differs')
    return dict(passed=True, scope='Presence/count/source only; full hashes checked by prepare/evaluate',
                development_predictions=795, test_predictions=654, backend='CPU; no inference or training')


def prepare(output, cfg, base, mc, model):
    folder = ROOT/cfg['development_root']/model/'development_v1'
    parent = c.checked(folder, 'b_development')
    if parent['model'] != model:
        raise ValueError('Wrong development model')
    predictions = c.read_csv(folder/'predictions.csv')
    if len(predictions) != 795 or len({r['sample_id'] for r in predictions}) != 795:
        raise ValueError('Invalid development prediction identities')
    samples = []
    with c.dataset(mc, development=True) as data:
        for i, index in enumerate(data.test_indices_1based):
            samples.append(SimpleNamespace(test_position=i, matlab_index_1based=int(index),
                           scene=data._decode_reference_string('scenes', int(index)-1)))
    with c.dataset(mc) as data:
        test_scenes = {data._decode_reference_string('scenes', int(i)-1) for i in data.test_indices_1based}
        if {s.scene for s in samples} & test_scenes:
            raise ValueError('Development/test scene overlap')
    rows = v.partition(samples, cfg)
    for row, source in zip(rows, predictions, strict=True):
        if (row['sample_id'] != source['sample_id'] or row['position'] != int(source['position'])
                or row['matlab_index'] != int(source['matlab_index'])):
            raise ValueError('Development identity mismatch')
        row.update(rgb_sha256=source['rgb_sha256'], prediction_sha256=source['prediction_sha256'])
    inputs = {str(folder/'run_metadata.json'): c.sha(folder/'run_metadata.json'),
              str(folder/'predictions.csv'): c.sha(folder/'predictions.csv')}
    record = v.begin(output, 'c_v2_prepared', cfg, model, inputs, [folder],
                     development_predictions=str(folder), test_images_used_for_selection=0)
    try:
        c.write_csv(output/'development_manifest.csv', rows)
        c.write_json(output/'partition.json', {subset: dict(
            images=sum(r['subset'] == subset for r in rows),
            scenes=len({r['scene'] for r in rows if r['subset'] == subset}))
            for subset in ('tune', 'validation')})
        v.seal(output, record, True)
    except BaseException as exc:
        c.failed(output, exc); raise


def manifest(prepared, cfg, model):
    record = v.checked(prepared, 'c_v2_prepared')
    if record['model'] != model or record['config'] != cfg:
        raise ValueError('Prepared model/config differs')
    return c.read_csv(prepared/'development_manifest.csv'), Path(record['development_predictions'])


def preparation(sample, data, diagnostic):
    settings = yaml.safe_load((ROOT/diagnostic['diagnostic_config']).read_text())
    index = sample.matlab_index_1based-1
    labels = np.asarray(data.file['labels'][index]).T[c.CROP]
    instances = np.asarray(data.file['instances'][index]).T[c.CROP]
    result = prepare_diagnostics(sample.depth_m[c.CROP], sample.rgb[c.CROP], labels, instances,
                                 settings, diagnostic['texture'])
    return result, settings


def measures(z, sample, prepared, settings, diagnostic, method, digest, seconds=0):
    values = c.score(z, sample.depth_m)
    extra, tables = analyze(z[c.CROP], prepared, settings, diagnostic['texture'])
    ids = dict(sample_id=c.stem(sample),
               scene=sample.scene, method=method)
    row = dict(**ids, source_prediction_sha256=digest, postprocess_seconds=seconds,
               postprocess_backend='CPU_float64_to_float32', peak_gpu_memory_bytes=None,
               **values, **extra)
    return row, {name: [dict(**ids, **r) for r in rr] for name, rr in tables.items()}


def development_pass(rows, folder, cfg, mc, diagnostic, subset, selection=None, limit=None):
    chosen = [r for r in rows if r['subset'] == subset]
    if limit is not None:
        if not 1 <= limit <= len(chosen):
            raise ValueError('Development limit must be within each subset')
        chosen = chosen[:limit]
    methods = ['RAW', 'C', *[r['id'] for r in cfg['candidates']]] if subset == 'tune' else [
        'RAW', 'C', 'C_v2', 'C_v2_no_sharpen', 'C_v2_no_RGB_support']
    tables = {name: [] for name in original.TABLES}
    with c.dataset(mc, development=True) as data:
        for number, identity in enumerate(chosen):
            sample = data.get_test_sample(int(identity['position']))
            key = c.development_stem(sample)
            if (key != identity['sample_id'] or sample.scene != identity['scene']
                    or hashlib.sha256(sample.rgb.tobytes()).hexdigest() != identity['rgb_sha256']):
                raise ValueError('Development RGB/scene identity differs')
            path = folder/'predictions'/(key+'.npz')
            if c.sha(path) != identity['prediction_sha256']:
                raise ValueError('Development prediction SHA differs')
            raw = study.read_depth(path)
            smooth = denoise(raw, sample.rgb)
            prepared, settings = preparation(sample, data, diagnostic)
            for method in methods:
                if method == 'RAW':
                    z = raw
                elif method == 'C':
                    z = smooth
                elif subset == 'tune':
                    z = refine(raw, sample.rgb, v.parameters(cfg, method), smoothed=smooth)
                else:
                    z = refine(raw, sample.rgb, v.parameters(cfg, selection), method=method, smoothed=smooth)
                row, diagnostics = measures(z, sample, prepared, settings, diagnostic, method,
                                             identity['prediction_sha256'])
                # train identity is kept separate from the similarly indexed test split.
                row['sample_id'] = key
                tables['per_image'].append(row)
                for name, rr in diagnostics.items():
                    for r in rr:
                        r['sample_id'] = key
                    tables[name].extend(rr)
            if number % 25 == 0:
                print(f'{subset}: {number+1}/{len(chosen)}', flush=True)
    return tables, methods


def summarize(tables, cfg, methods):
    settings = dict(cfg, methods=methods, classical={})
    result = original.summary(tables, settings)
    result['parameters'] = dict(fixed=cfg['fixed'], candidates=cfg['candidates'])
    result['timing'] = 'CPU postprocessing only; no model inference. Development pass not timed.'
    result['limitation'] = 'Exploratory redesign after inspecting earlier test results; no fresh external confirmation.'
    common = v.common_profiles(tables['profiles'], methods)
    rows = common.pop('rows')
    common['metrics'] = {method: {field: study.describe([r for r in rows if r['method'] == method], field)
                                for field in ('width_10_90_px', 'absolute_midpoint_error_px')}
                         for method in methods}
    common['paired_scene_deltas'] = {}
    # paired() expects one row/image; average multiple profiles first.
    if rows:
        aggregate = []
        for method in methods:
            images = sorted({r['sample_id'] for r in rows if r['method'] == method})
            for key in images:
                group = [r for r in rows if r['method'] == method and r['sample_id'] == key]
                aggregate.append(dict(sample_id=key, scene=group[0]['scene'], method=method,
                    **{field: float(np.mean([r[field] for r in group]))
                       for field in ('width_10_90_px', 'absolute_midpoint_error_px')}))
        common['paired_scene_deltas'] = study.paired(aggregate, methods,
            ['width_10_90_px', 'absolute_midpoint_error_px'], cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    result['all_methods_common_profiles'] = common
    if 'C_v2' in methods:
        means = result['means']; width = common['metrics']
        w = {m: width[m]['width_10_90_px']['mean'] for m in methods}
        result['mean_targets'] = dict(
            si_above_RAW=means['C_v2']['si_boundary_f1'] > means['RAW']['si_boundary_f1'],
            si_above_C=means['C_v2']['si_boundary_f1'] > means['C']['si_boundary_f1'],
            width_below_RAW=w['C_v2'] < w['RAW'] if common['profiles'] else None,
            width_below_C=w['C_v2'] < w['C'] if common['profiles'] else None,
            interpretation='Mean signs only. Report CIs, location, coverage, texture, metric depth and failures too; no automatic acceptance/fallback.')
        compared = [dict(r, method='RAW' if r['method'] == 'C' else r['method'])
                    for r in tables['per_image'] if r['method'] != 'RAW']
        result['paired_scene_deltas_vs_C'] = study.paired(compared, [m for m in methods if m != 'C'],
            ['abs_rel', 'si_boundary_f1', 'primary_f1', 'texture_spurious_fraction'],
            cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    return result


def save_tables(output, tables):
    output.mkdir(parents=True, exist_ok=True)
    original.write_tables(output, tables)


def develop(output, prepared, cfg, mc, diagnostic, model, limit=None):
    rows, folder = manifest(prepared, cfg, model)
    record = v.begin(output, 'c_v2_development', cfg, model, {}, [prepared], limit=limit,
                     test_images_used_for_selection=0, validation_used_for_selection=False)
    try:
        tune, methods = development_pass(rows, folder, cfg, mc, diagnostic, 'tune', limit=limit)
        selection = v.rank_candidates(tune['per_image'], tune['profiles'], [r['id'] for r in cfg['candidates']])
        save_tables(output/'tune', tune)
        c.write_json(output/'tune/summary.json', summarize(tune, cfg, methods))
        # Chosen from tune, and written BEFORE any validation scores are computed.
        c.write_json(output/'selection_before_validation.json', selection)
        validation, methods = development_pass(rows, folder, cfg, mc, diagnostic, 'validation',
                                               selection['selected'], limit=limit)
        save_tables(output/'validation', validation)
        c.write_json(output/'validation/summary.json', summarize(validation, cfg, methods))
        v.seal(output, record, limit is None)
    except BaseException as exc:
        c.failed(output, exc); raise


def freeze(output, development, cfg, model):
    parent = v.checked(development, 'c_v2_development')
    if parent['model'] != model or parent['config'] != cfg:
        raise ValueError('Development model/config differs')
    selection = c.read_json(development/'selection_before_validation.json')
    tables = {name: c.read_csv(development/'tune'/(name+'.csv')) for name in original.TABLES}
    # Convert numeric CSV fields needed for common support and ranking.
    rows = numeric_tables(tables)
    replay = v.rank_candidates(rows['per_image'], rows['profiles'], [r['id'] for r in cfg['candidates']])
    v.equivalent(selection, replay, 'development-only selection')
    record = v.begin(output, 'c_v2_frozen', cfg, model, {}, [development],
                     test_images_used_for_selection=0, validation_used_for_selection=False)
    try:
        c.write_json(output/'frozen_parameters.json', dict(model=model, selected=selection['selected'],
            parameters=asdict(v.parameters(cfg, selection['selected'])), selection=selection,
            validation_mean_targets=c.read_json(development/'validation/summary.json')['mean_targets'],
            test_images_used_for_selection=0, validation_used_for_selection=False,
            failed_validation_does_not_replace_method_with_RAW=True))
        v.seal(output, record, True)
    except BaseException as exc:
        c.failed(output, exc); raise


def numeric_tables(tables):
    string_fields = {'sample_id', 'scene', 'method', 'source_prediction_sha256',
                     'postprocess_backend', 'status', 'axis', 'region'}
    return {name: [{key: (value if key in string_fields else None if value == '' else value == 'True' if value in ('True', 'False') else float(value))
                    for key, value in row.items()} for row in rows] for name, rows in tables.items()}


def evaluation_pass(output, cfg, base, mc, diagnostic, model, source, ref, classics, params,
                    limit=None, verify=False):
    tables = {name: [] for name in original.TABLES}
    saved = {(r['sample_id'], r['method']): r for r in c.read_csv(output/'per_image.csv')} if verify else {}
    old_scores = {(r['sample_id'], r['method']): r for r in c.read_csv(classics/'per_image.csv')}
    count = 0
    with c.dataset(mc) as data:
        for i in c.positions(len(data), limit):
            sample = data.get_test_sample(i); key = c.stem(sample)
            raw, digest = c.native(source, sample)
            prepared, settings = preparation(sample, data, diagnostic)
            smooth = study.read_depth(ref/'predictions/C'/(key+'.npz'))
            images = {}
            for method in cfg['methods']:
                seconds = 0.
                if method == 'RAW':
                    z = raw
                elif method == 'C':
                    z = smooth
                elif method in ('Gaussian_log', 'RGB_guided_log'):
                    z = study.read_depth(classics/'predictions'/method/(key+'.npz'))
                    if verify:
                        c.assert_close(z, classical(method, raw, sample.rgb, diagnostic['classical']), 'Classic CPU replay')
                else:
                    path = output/'predictions'/method/(key+'.npz')
                    if verify:
                        z = study.read_depth(path)
                        c.assert_close(z, refine(raw, sample.rgb, params, method=method), 'C v2 CPU replay')
                        seconds = float(saved[key, method]['postprocess_seconds'])
                    else:
                        # Recompute the complete v2 method from RAW; timing includes denoising.
                        start = time.perf_counter()
                        z = refine(raw, sample.rgb, params, method=method)
                        seconds = time.perf_counter()-start
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(path, depth_m=c.valid_depth(z))
                row, measures_ = measures(z, sample, prepared, settings, diagnostic, method, digest, seconds)
                if method in ('RAW', 'C', 'Gaussian_log', 'RGB_guided_log'):
                    previous = old_scores[key, method]
                    if previous['source_prediction_sha256'] != digest:
                        raise ValueError('Old C/native source digest differs')
                    c.raw_matches({name: row[name] for name in c.METRICS}, previous)
                    row['postprocess_seconds'] = float(previous['postprocess_seconds'])
                    row['postprocess_backend'] = previous['postprocess_backend']
                    row['peak_gpu_memory_bytes'] = int(previous['peak_gpu_memory_bytes']) if previous['peak_gpu_memory_bytes'] else None
                tables['per_image'].append(row)
                for name, rr in measures_.items():
                    tables[name].extend(rr)
                images[method] = z[c.CROP]
            if not verify and i in cfg['case_positions']:
                plot(output, key, sample.rgb[c.CROP], prepared, images, settings)
            count += 1
            if i % 25 == 0:
                print(f'C v2 evaluation {i+1}/{len(data)}', flush=True)
    for method in ('C_v2', 'C_v2_no_sharpen', 'C_v2_no_RGB_support'):
        files = list((output/'predictions'/method).glob('*.npz'))
        if len(files) != count:
            raise ValueError('Missing/extra C v2 prediction arrays')
    return tables


def plot(output, key, rgb, prepared, images, settings):
    # Original helper has six panels; split groups explicitly so v2 is never omitted.
    for name, methods in [('main', ['RAW', 'C', 'C_v2']),
                          ('controls', ['RAW', 'C_v2', 'C_v2_no_sharpen', 'C_v2_no_RGB_support',
                                        'Gaussian_log', 'RGB_guided_log'])]:
        destination = output/'visualization'/name
        destination.mkdir(parents=True, exist_ok=True)
        original.plot_case(destination, key, rgb, prepared, {m: images[m] for m in methods}, settings)


def evaluate(output, frozen, cfg, base, mc, diagnostic, model, limit=None):
    parent = v.checked(frozen, 'c_v2_frozen')
    if parent['model'] != model or parent['config'] != cfg:
        raise ValueError('Frozen model/config differs')
    source, _, inputs = c.baseline_files(base, model, mc)
    inputs.update(c.resources(base, model, mc))
    ref, _ = original.reference(diagnostic, base, model)
    classics = ROOT/base['output_root']/model/diagnostic['output_name']
    old = study.checked(classics, 'study_c_diagnostics')
    if old['model'] != model or old['config'] != diagnostic:
        raise ValueError('Classic model/config differs')
    params = Parameters(**c.read_json(frozen/'frozen_parameters.json')['parameters'])
    record = v.begin(output, 'c_v2_evaluation', cfg, model, inputs, [frozen, ref, classics], limit=limit)
    try:
        tables = evaluation_pass(output, cfg, base, mc, diagnostic, model, source, ref, classics, params, limit)
        save_tables(output, tables)
        c.write_json(output/'summary.json', summarize(tables, cfg, cfg['methods']))
        v.seal(output, record, limit is None)
    except BaseException as exc:
        c.failed(output, exc); raise


def verify(output, cfg, base, mc, diagnostic, model):
    record = v.checked(output, full=False)
    if record['model'] != model or record['config'] != cfg:
        raise ValueError('Run model/config differs')
    if record['kind'] == 'c_v2_development':
        prepared = Path(record['parents'][0])
        rows, folder = manifest(prepared, cfg, model)
        tune, methods = development_pass(rows, folder, cfg, mc, diagnostic, 'tune', limit=record['limit'])
        selection = v.rank_candidates(tune['per_image'], tune['profiles'], [r['id'] for r in cfg['candidates']])
        v.equivalent(selection, c.read_json(output/'selection_before_validation.json'), 'selection replay')
        validation, vm = development_pass(rows, folder, cfg, mc, diagnostic, 'validation', selection['selected'], record['limit'])
        for name, tables, names in [('tune', tune, methods), ('validation', validation, vm)]:
            for table, rows in tables.items():
                study.table_equal(rows, c.read_csv(output/name/(table+'.csv')))
            v.equivalent(summarize(tables, cfg, names), c.read_json(output/name/'summary.json'), 'development summary')
    elif record['kind'] == 'c_v2_evaluation':
        frozen, ref, classics = map(Path, record['parents'])
        params = Parameters(**c.read_json(frozen/'frozen_parameters.json')['parameters'])
        source = ROOT/base['models'][model]['baseline']
        tables = evaluation_pass(output, cfg, base, mc, diagnostic, model, source, ref, classics,
                                 params, record['limit'], verify=True)
        for name, rows in tables.items():
            study.table_equal(rows, c.read_csv(output/(name+'.csv')))
        v.equivalent(summarize(tables, cfg, cfg['methods']), c.read_json(output/'summary.json'), 'evaluation summary')
    else:
        raise ValueError('verify expects development or evaluation run')
    return dict(passed=True, kind=record['kind'], formal_complete=record['formal_complete'],
                selected_only_on_development=True, all_v2_predictions_and_GT_diagnostics_replayed=True,
                previous_C_scope='SHA and metric replay; old joint-filter CPU formula remains its separate v1 verify')


def smoke(output, cfg, mc, diagnostic, model):
    """Two development images, one predeclared candidate; never parameter selection."""
    folder = ROOT/cfg['development_root']/model/'development_v1'
    source = c.read_json(folder/'run_metadata.json')
    if source['model'] != model or source['source_sha256'] != c.sources():
        raise ValueError('Wrong smoke development source')
    ids = c.read_csv(folder/'predictions.csv')[:2]
    inputs = {str(folder/'run_metadata.json'): c.sha(folder/'run_metadata.json')}
    for row in ids:
        path = folder/'predictions'/(row['sample_id']+'.npz')
        if c.sha(path) != row['prediction_sha256']:
            raise ValueError('Smoke raw checksum differs')
        inputs[str(path)] = row['prediction_sha256']
    mc_path = ROOT/cfg['base_config']
    inputs[str(mc_path)] = c.sha(mc_path)
    record = v.begin(output, 'c_v2_smoke', cfg, model, inputs,
                     samples=2, candidate='blend050_steps3', parameter_selection=False,
                     source_scope='Only two native development arrays; not the full development parent audit')
    try:
        methods = ['RAW', 'C', 'C_v2', 'C_v2_no_sharpen', 'C_v2_no_RGB_support']
        tables = {name: [] for name in original.TABLES}
        with c.dataset(mc, development=True) as data:
            for row in ids:
                sample = data.get_test_sample(int(row['position']))
                if hashlib.sha256(sample.rgb.tobytes()).hexdigest() != row['rgb_sha256']:
                    raise ValueError('Smoke RGB differs')
                key = c.development_stem(sample)
                raw = study.read_depth(folder/'predictions'/(key+'.npz'))
                smooth = denoise(raw, sample.rgb)
                prepared, settings = preparation(sample, data, diagnostic)
                images = {}
                for method in methods:
                    z = raw if method == 'RAW' else smooth if method == 'C' else refine(
                        raw, sample.rgb, v.parameters(cfg, 'blend050_steps3'), method=method, smoothed=smooth)
                    metric, diag = measures(z, sample, prepared, settings, diagnostic, method, row['prediction_sha256'])
                    metric['sample_id'] = key;tables['per_image'].append(metric)
                    for name, rr in diag.items():
                        for r in rr:r['sample_id'] = key
                        tables[name].extend(rr)
                    images[method] = z[c.CROP]
                destination = output/'visualization'
                destination.mkdir(exist_ok=True)
                original.plot_case(destination, key, sample.rgb[c.CROP], prepared, images, settings)
        save_tables(output, tables)
        c.write_json(output/'summary.json', summarize(tables, cfg, methods))
        v.seal(output, record, False)
    except BaseException as exc:
        c.failed(output, exc);raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'smoke', 'prepare', 'develop', 'freeze', 'evaluate', 'verify'])
    parser.add_argument('--model', choices=['depth_pro', 'unidepth_v2'], required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/part3_c_v2/direction_c.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--development', type=Path)
    parser.add_argument('--frozen', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(); cv2.setNumThreads(1)
    cfg, base, mc, diagnostic = v.configuration(args.config, args.model)
    if args.limit is not None and args.action not in ('develop', 'evaluate'):
        parser.error('--limit only applies to develop/evaluate')
    output = (args.output_dir or output_path(cfg, args.model,
        {'prepare': 'prepared', 'develop': 'development', 'freeze': 'frozen', 'evaluate': 'evaluation'}.get(args.action, args.action))).resolve()
    if args.action == 'check':
        print(check(cfg, base, mc, diagnostic, args.model)); return
    if args.action == 'smoke':
        smoke(output, cfg, mc, diagnostic, args.model); return
    if args.action == 'verify':
        if not args.run:
            parser.error('--run is required')
        c.report(args.run, verify(args.run.resolve(), cfg, base, mc, diagnostic, args.model), args.report); return
    if args.action == 'prepare':
        prepare(output, cfg, base, mc, args.model)
    elif args.action == 'develop':
        develop(output, (args.prepared or output_path(cfg, args.model, 'prepared')).resolve(),
                cfg, mc, diagnostic, args.model, args.limit)
    elif args.action == 'freeze':
        freeze(output, (args.development or output_path(cfg, args.model, 'development')).resolve(), cfg, args.model)
    else:
        evaluate(output, (args.frozen or output_path(cfg, args.model, 'frozen')).resolve(),
                 cfg, base, mc, diagnostic, args.model, args.limit)


if __name__ == '__main__':
    main()
