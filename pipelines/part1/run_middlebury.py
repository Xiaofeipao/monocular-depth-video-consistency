#!/usr/bin/env python3
"""Middlebury四组经典双目：课程评价、全图诊断与保存产物独立核验。"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.output_schema import validate_depth_output
from src.stereo.geometry import middlebury_disparity_to_depth
from src.stereo.io import load_middlebury_scene
from src.stereo.matching import StereoMatcherConfig, match_stereo, left_right_consistency_mask
from src.stereo.metrics import compute_disparity_metrics, compute_stereo_depth_metrics
from src.stereo.protocol import (PROTOCOL, load_config, sha256, source_hashes,
    data_hashes, environment, write_json, load_selection, archive_protocol)
from src.stereo.sensitivity import analyze_depth_sensitivity, analyze_prediction_error_by_distance
from src.visualization.stereo import (save_contact_sheet, save_empirical_depth_error_plot,
    save_sensitivity_plot, save_stereo_failure_zoom, save_stereo_panel)


def scene_split(config, name):
    return 'development' if name in config['split']['development_scenes'] else 'held_out'


def to_depth(disparity, calibration):
    return middlebury_disparity_to_depth(disparity,
        focal_length_px=calibration.focal_length_px, baseline_mm=calibration.baseline_mm,
        doffs_px=calibration.doffs_px)


def score(scene, disparity, depth, valid, config):
    target_depth, target_valid = to_depth(scene.gt_disparity_px, scene.calibration)
    return {
        **compute_disparity_metrics(disparity, scene.gt_disparity_px, prediction_valid_mask=valid),
        **compute_stereo_depth_metrics(depth, target_depth.astype(np.float32),
            evaluation_mask=target_valid, depth_bin_edges_m=config['evaluation']['depth_bin_edges_m']),
    }


def evaluate_scene_method(scene, *, experiment, config, output_dir):
    setting = config['experiments'][experiment]
    values = dict(setting['parameters']); lr = values.pop('lr_threshold_px')
    result = match_stereo(scene.left_rgb, scene.right_rgb,
        calibration_num_disparities=scene.calibration.num_disparities,
        config=StereoMatcherConfig(method=setting['algorithm'], **values), lr_threshold_px=lr)
    depth, geometry_valid = to_depth(result.left_disparity_px, scene.calibration)
    valid = result.valid_mask & geometry_valid
    disparity = result.left_disparity_px.copy(); disparity[~valid] = np.nan
    depth = depth.astype(np.float32); depth[~valid] = np.nan
    calibration = scene.calibration
    metadata = {
        'protocol': PROTOCOL, 'scene': scene.name, 'experiment': experiment,
        'algorithm': setting['algorithm'], 'parameter_source': setting['parameter_source'],
        'depth_type': 'metric', 'unit': 'meter', 'alignment': 'none',
        'invalid_representation': 'nan_with_boolean_mask',
        'focal_length_px': calibration.focal_length_px, 'baseline_mm': calibration.baseline_mm,
        'baseline_m': calibration.baseline_m, 'doffs_px': calibration.doffs_px,
        'formula': config['geometry']['formula'], 'opencv_fixed_point_scale': 16.,
        'num_disparities': result.num_disparities, 'lr_threshold_px': lr,
        'matcher_parameters': setting['parameters'],
        'evaluation_domain': 'all finite positive GT disparities; not restricted by mask0nocc',
        'runtime_seconds': result.runtime_seconds,
    }
    validate_depth_output(depth, metadata, valid_mask=valid)
    directory = Path(output_dir)/scene.name/experiment
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory/'prediction.npz', disparity_px=disparity,
        raw_disparity_px=result.raw_left_disparity_px,
        right_disparity_px=result.right_disparity_px, depth_m=depth, valid_mask=valid,
        raw_left_valid=result.raw_left_valid, raw_right_valid=result.raw_right_valid,
        lr_consistent=result.lr_consistent)
    write_json(directory/'metadata.json', metadata)
    save_stereo_panel(directory/'visualization.png', scene_name=scene.name,
        method_name=experiment.upper(), rgb=scene.left_rgb,
        target_disparity_px=scene.gt_disparity_px, prediction_disparity_px=disparity,
        evaluation_mask=np.isfinite(scene.gt_disparity_px) & (scene.gt_disparity_px > 0),
        raw_prediction_valid=result.raw_left_valid, lr_consistent=result.lr_consistent,
        raw_disparity_px=result.raw_left_disparity_px, disparity_max_px=float(result.num_disparities))
    return {'scene': scene.name, 'split': scene_split(config, scene.name),
        'experiment': experiment, 'algorithm': setting['algorithm'],
        'parameter_source': setting['parameter_source'], 'runtime_seconds': result.runtime_seconds,
        **score(scene, disparity, depth, valid, config)}


def macro_summary(rows):
    """All scenes have equal weight; undefined primary MAE must not disappear."""
    summaries = {}
    for experiment in sorted({r['experiment'] for r in rows}):
        method_rows = [r for r in rows if r['experiment'] == experiment]
        summaries[experiment] = {k: method_rows[0][k] for k in ('algorithm', 'parameter_source')}
        for split in ('development', 'held_out', 'all'):
            selected = [r for r in method_rows if split == 'all' or r['split'] == split]
            primary = ('runtime_seconds', 'disparity_mae_px', 'valid_prediction_fraction',
                       *(key for key in ('bad1','bad2','bad4') if key in method_rows[0]))
            values = {'scene_count': len(selected)}
            for key in primary:
                available = [float(r[key]) for r in selected if r[key] is not None]
                values[key] = float(np.mean(available)) if selected and len(available)==len(selected) else None
                values[key+'_defined_scenes'] = len(available)
            # Auxiliary distance bins may be absent in individual scenes: expose support.
            for key in (k for k in method_rows[0] if k.startswith('depth_') and not k.endswith('_pixels')):
                available = [float(r[key]) for r in selected if r[key] is not None]
                values[key] = float(np.mean(available)) if available else None
                values[key+'_defined_scenes'] = len(available)
            for key in ('evaluation_pixels', 'valid_prediction_pixels', 'invalid_prediction_pixels'):
                values[key] = sum(int(r[key]) for r in selected)
            summaries[experiment][split] = values
    return summaries


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    text_keys = {'scene','split','experiment','algorithm','parameter_source'}
    return [{k: v if k in text_keys else float(v) if v else None for k,v in r.items()} for r in rows]


def assert_equal(actual, expected, label):
    if actual is None or expected is None:
        if actual is not None or expected is not None: raise AssertionError(label)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=label)


def validate_saved_outputs(*, output_dir, scenes, rows, config):
    """Rebuild LR, depth, valid mask and all reported accuracy metrics from raw arrays."""
    scene_by_name = {s.name:s for s in scenes}; checked = 0
    for row in rows:
        scene = scene_by_name[row['scene']]; experiment = row['experiment']
        directory = Path(output_dir)/scene.name/experiment
        metadata = json.loads((directory/'metadata.json').read_text())
        with np.load(directory/'prediction.npz', allow_pickle=False) as archive:
            arrays = {k:archive[k] for k in archive.files}
        for key, array in arrays.items():
            if array.shape != scene.gt_disparity_px.shape: raise AssertionError(f'Shape: {key}')
            expected_dtype = np.bool_ if key in ('valid_mask','raw_left_valid','raw_right_valid','lr_consistent') else np.float32
            if array.dtype != expected_dtype: raise AssertionError(f'Dtype: {key}')
        raw = arrays['raw_disparity_px']; right = arrays['right_disparity_px']
        np.testing.assert_array_equal(np.isfinite(raw), arrays['raw_left_valid'])
        np.testing.assert_array_equal(np.isfinite(right), arrays['raw_right_valid'])
        lr = left_right_consistency_mask(raw, right, threshold_px=metadata['lr_threshold_px'])
        np.testing.assert_array_equal(lr, arrays['lr_consistent'])
        depth, geometry_valid = to_depth(raw, scene.calibration)
        valid = arrays['raw_left_valid'] & lr & geometry_valid
        np.testing.assert_array_equal(valid, arrays['valid_mask'])
        expected_disparity = raw.copy(); expected_disparity[~valid] = np.nan
        depth = depth.astype(np.float32); depth[~valid] = np.nan
        assert_equal(expected_disparity, arrays['disparity_px'], 'Final disparity')
        assert_equal(depth, arrays['depth_m'], 'Metric conversion')
        validate_depth_output(arrays['depth_m'], metadata, valid_mask=valid)
        for key, value in score(scene, arrays['disparity_px'], arrays['depth_m'], valid, config).items():
            assert_equal(value, row[key], f'{scene.name}/{experiment}/{key}'); checked += 1
    return {'passed':True, 'prediction_files_checked':len(rows), 'metric_values_checked':checked,
            'lr_and_geometry_recomputed':True, 'relative_tolerance':1e-6, 'absolute_tolerance':1e-8}


def save_auxiliary(output, scenes, config, experiments):
    sensitivity = analyze_depth_sensitivity(scenes, **{
        k:config['sensitivity'][k] for k in ('disparity_errors_px','depth_bin_width_m')})
    write_csv(output/'sensitivity.csv', sensitivity)
    save_sensitivity_plot(output/'sensitivity.png', sensitivity)
    empirical = []
    for experiment in experiments:
        predictions = {}
        for scene in scenes:
            with np.load(output/scene.name/experiment/'prediction.npz') as archive:
                predictions[scene.name] = archive['disparity_px'].copy()
        empirical.extend({'experiment':experiment, **r} for r in analyze_prediction_error_by_distance(
            scenes, predictions, depth_bin_width_m=config['sensitivity']['depth_bin_width_m']))
        save_contact_sheet(output/f'visualization_contact_sheet_{experiment}.jpg',
            [output/s.name/experiment/'visualization.png' for s in scenes])
    write_csv(output/'empirical_depth_error_by_distance.csv', empirical)
    save_empirical_depth_error_plot(output/'empirical_depth_error_by_distance.png', empirical)
    zoom = config['visualization']['failure_zoom']; experiment = zoom['experiment']
    if experiment in experiments:
        scene = next(s for s in scenes if s.name == zoom['scene'])
        with np.load(output/scene.name/experiment/'prediction.npz') as archive:
            disparity = archive['disparity_px']
        save_stereo_failure_zoom(output/f'failure_zoom_{scene.name}_{experiment}.png', scene_name=scene.name,
            method_name=experiment.upper(), rgb=scene.left_rgb, target_disparity_px=scene.gt_disparity_px,
            prediction_disparity_px=disparity,
            evaluation_mask=np.isfinite(scene.gt_disparity_px) & (scene.gt_disparity_px > 0),
            crop_xyxy=tuple(zoom['crop_xyxy']), disparity_max_px=float(scene.calibration.num_disparities))


def visualization_artifact_hashes(output, record):
    """Accept an explicitly recorded display refresh without rewriting run history."""
    import ast
    original = record['source_sha256']
    current = source_hashes()
    path = output/'visualization_refresh.json'
    if not path.exists():
        if current != original:
            raise ValueError('Scientific source differs from this run')
        return record['artifacts']
    refresh = json.loads(path.read_text())
    allowed_sources = {'src/visualization/stereo.py', 'pipelines/part1/run_middlebury.py'}
    different = {key for key in set(current)|set(original) if current.get(key)!=original.get(key)}
    if (refresh.get('status')!='complete'
            or refresh.get('original_run_metadata_sha256')!=sha256(output/'run_metadata.json')
            or refresh.get('source_sha256')!=current
            or not different <= allowed_sources):
        raise ValueError('Display refresh source or run binding differs')
    # Only the verifier may change in the pipeline; matching/scoring stay exact.
    relative = 'pipelines/part1/run_middlebury.py'
    archived = output/'provenance/source'/relative
    if sha256(archived) != original[relative]:
        raise ValueError('Archived scientific pipeline changed')
    def scientific_ast(file):
        module = ast.parse(file.read_text())
        module.body = [node for node in module.body if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {'verify_run','visualization_artifact_hashes'})]
        return ast.dump(module, include_attributes=False)
    if scientific_ast(archived) != scientific_ast(PROJECT_ROOT/relative):
        raise ValueError('Display refresh cannot change matching or scoring')
    names = record['config']['split']['development_scenes']+record['config']['split']['evaluation_scenes']
    allowed_artifacts = {f'{name}/{experiment}/visualization.png'
        for name in names for experiment in record['experiments']}
    allowed_artifacts.update(f'visualization_contact_sheet_{experiment}.jpg' for experiment in record['experiments'])
    replacements = refresh.get('replacements',{})
    if not replacements or not set(replacements) <= allowed_artifacts:
        raise ValueError('Display refresh contains non-display artifacts')
    hashes = dict(record['artifacts'])
    for relative, entry in replacements.items():
        if entry['original_sha256'] != hashes.get(relative):
            raise ValueError('Display refresh original artifact binding differs')
        hashes[relative] = entry['sha256']
    return hashes


def verify_run(output, config):
    status = json.loads((output/'status.json').read_text())
    if status.get('status') != 'complete' or status.get('metadata_sha256') != sha256(output/'run_metadata.json'):
        raise ValueError('Run incomplete or metadata changed')
    record = json.loads((output/'run_metadata.json').read_text())
    if record['config'] != config:
        raise ValueError('Configuration differs from this run')
    artifact_hashes = visualization_artifact_hashes(output, record)
    names = config['split']['development_scenes']+config['split']['evaluation_scenes']
    if record['input_sha256'] != data_hashes(config, names): raise ValueError('Input data changed')
    for relative, digest in artifact_hashes.items():
        if sha256(output/relative) != digest: raise ValueError(f'Artifact changed: {relative}')
    scenes = [load_middlebury_scene(PROJECT_ROOT/config['dataset']['root']/n) for n in names]
    rows = read_rows(output/'metrics_per_scene.csv')
    expected = {(n,e) for n in names for e in record['experiments']}
    if len(rows)!=len(expected) or {(r['scene'],r['experiment']) for r in rows}!=expected:
        raise ValueError('Scene/experiment coverage is incomplete or duplicated')
    for row in rows:
        if row['split']!=scene_split(config,row['scene']): raise ValueError('Scene split differs')
    report = validate_saved_outputs(output_dir=output, scenes=scenes, rows=rows, config=config)
    if macro_summary(rows) != json.loads((output/'metrics_summary.json').read_text()):
        raise ValueError('Macro summaries differ from per-scene metrics')
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output/'verification.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/part1/middlebury_sgbm.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--tuning-dir', type=Path)
    parser.add_argument('--experiments', nargs='+')
    parser.add_argument('--mode', choices=('run','verify'), default='run')
    args = parser.parse_args(argv); config = load_config(args.config)
    output = (args.output_dir or PROJECT_ROOT/config['outputs']['root']/config['outputs']['evaluation_run']).resolve()
    if output.is_relative_to(PROJECT_ROOT) and not output.is_relative_to(PROJECT_ROOT/'outputs'):
        raise ValueError('Outputs must be under outputs/')
    if args.mode == 'verify':
        if args.experiments or args.tuning_dir: raise ValueError('Verify uses the saved experiment and parameter records')
        print(json.dumps(verify_run(output, config), indent=2)); return
    experiments = args.experiments or list(config['experiments'])
    if len(set(experiments))!=len(experiments) or set(experiments)-set(config['experiments']):
        raise ValueError('Unknown or repeated experiment')
    if output.exists(): raise FileExistsError(f'Refusing to overwrite: {output}')
    tuning = args.tuning_dir or PROJECT_ROOT/config['outputs']['root']/config['outputs']['tuning_run']
    selections = {}; resolved = copy.deepcopy(config)
    for experiment in experiments:
        setting = resolved['experiments'][experiment]
        if experiment.endswith('_tuned'):
            method = setting['algorithm']; record = load_selection(config, method, tuning)
            selections[method] = record; setting['parameters'] = record['selected_parameters']
    names = config['split']['development_scenes']+config['split']['evaluation_scenes']
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'status.json', {'status':'running','command':sys.argv})
    try:
        started = datetime.now(timezone.utc).isoformat()
        sources = archive_protocol(output/'provenance',config)
        shutil.copyfile(args.config,output/'provenance'/'config.yaml')
        inputs = data_hashes(config,names)
        for method, selection in selections.items():
            write_json(output/'provenance'/f'{method}_selection.json',selection)
        scenes = [load_middlebury_scene(PROJECT_ROOT/config['dataset']['root']/n) for n in names]
        rows = []
        for i,scene in enumerate(scenes,1):
            for experiment in experiments:
                row = evaluate_scene_method(scene, experiment=experiment, config=resolved, output_dir=output)
                rows.append(row)
                print(f'[{i}/{len(scenes)}] {scene.name}/{experiment}: '
                      f'MAE={row["disparity_mae_px"]}, valid fraction={row["valid_prediction_fraction"]:.6f}',flush=True)
        write_csv(output/'metrics_per_scene.csv',rows)
        write_json(output/'metrics_summary.json',macro_summary(rows))
        save_auxiliary(output, scenes, config, experiments)
        write_json(output/'reload_validation.json',validate_saved_outputs(
            output_dir=output, scenes=scenes, rows=rows, config=config))
        if sources!=source_hashes() or inputs!=data_hashes(config,names):
            raise ValueError('Source or input changed during the run')
        record = {'protocol':PROTOCOL, 'started_at_utc':started,
            'completed_at_utc':datetime.now(timezone.utc).isoformat(), 'command':sys.argv,
            'config':config, 'config_file_sha256':sha256(args.config), 'source_sha256':sources,
            'input_sha256':inputs, 'experiments':experiments, 'scene_count':len(scenes),
            'resolved_parameters':{e:resolved['experiments'][e]['parameters'] for e in experiments},
            'environment':environment(), 'formal_training_run':False,
            'input_sizes_wh':{s.name:[s.calibration.width,s.calibration.height] for s in scenes},
            'disparity_precision':'OpenCV int16 / 16 -> float32 pixel disparity',
            'geometry_precision':'float64 calculation -> float32 stored meters',
            'artifacts':{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*'))
                         if p.is_file() and p!=output/'status.json'}}
        write_json(output/'run_metadata.json',record)
        write_json(output/'status.json',{'status':'complete','metadata_sha256':sha256(output/'run_metadata.json')})
        print(json.dumps(verify_run(output, config),indent=2))
    except BaseException as exc:
        write_json(output/'status.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise


if __name__ == '__main__':
    main()
