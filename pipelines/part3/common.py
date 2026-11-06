"""B/C共享输入、固定课程协议、来源绑定和不可覆盖的运行记录。"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.nyuv2 import NYUv2LabeledDataset, load_nyuv2_test_indices
from src.metrics.depth import compute_depth_metrics
from src.metrics.boundary import compute_depth_pro_si_boundary_f1

METRICS = ('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'delta1', 'delta2', 'delta3', 'si_boundary_f1')
CROP = np.s_[45:471, 41:601]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def load_config(path, model, direction):
    config = yaml.safe_load(Path(path).read_text())
    expected = 'part3_global_scale_v1' if direction == 'b' else 'part3_joint_bilateral_v1'
    if config['protocol'] != expected or model not in config['models']:
        raise ValueError('B/C protocol or model differs')
    model_config = yaml.safe_load((ROOT / config['models'][model]['config']).read_text())
    ev = model_config['evaluation']
    if (model_config['dataset']['split_variable'] != 'testNdxs'
            or model_config['dataset']['expected_test_images'] != 654
            or model_config['model']['depth_type'] != 'metric'
            or model_config['model']['alignment'] != 'none'
            or ev['test_time_alignment'] != 'none' or ev['native_scale'] != 'metric'
            or [ev['crop'][k] for k in ('top','bottom','left','right')] != [45,471,41,601]
            or ev['valid_depth_m'] != dict(minimum=.001, maximum=10., bounds='exclusive')
            or not ev['clip_prediction_to_valid_range'] or ev['aggregation'] != 'per_image_macro_mean'):
        raise ValueError('Part 2 baseline must use the frozen native metric course protocol')
    if direction == 'b' and config['formula'] != 'exp(median_over_795_images(median_valid_pixels(log(GT/pred))))':
        raise ValueError('B requires the declared development-only global scale formula')
    if direction == 'c':
        if config['parameters'] != dict(radius=2, sigma_rgb=.10, sigma_log=.18, sigma_space=1.5):
            raise ValueError('C reproduces the fixed main-branch defaults; parameters cannot be silently changed')
        if config['methods'] != ['RAW','C','C_no_RGB','C_no_depth_gate']:
            raise ValueError('C requires RAW, full C and both fixed ablations')
        if config['bootstrap_resamples'] < 1:
            raise ValueError('Positive bootstrap count required')
    return config, model_config


def resources(config, model, mc):
    paths = {}
    for section, key, digest_key in [('dataset','labeled_mat','labeled_mat_sha256'),
                                    ('dataset','split_file','split_file_sha256'),
                                    ('model','checkpoint','checkpoint_sha256')]:
        p = ROOT / mc[section][key]
        digest = sha(p)
        if digest != mc[section][digest_key]:
            raise ValueError('Resource SHA differs: ' + str(p))
        paths[str(p)] = digest
    if model == 'unidepth_v2':
        p = ROOT / mc['model']['config']
        paths[str(p)] = sha(p)
        if paths[str(p)] != mc['model']['config_sha256']:
            raise ValueError('UniDepth model config changed')
    split = ROOT / mc['dataset']['split_file']
    train, test = (load_nyuv2_test_indices(split, variable=v) for v in ('trainNdxs','testNdxs'))
    if len(train) != 795 or len(test) != 654 or set(train) & set(test):
        raise ValueError('Require disjoint official 795/654 splits')
    versions = yaml.safe_load((ROOT / 'configs/versions.yaml').read_text())['third_party']
    upstream = ROOT / mc['model']['repository']
    revision = subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'], text=True).strip()
    if revision != versions['unidepth' if model == 'unidepth_v2' else model]['commit']:
        raise ValueError('Upstream commit differs')
    return paths


def dataset(mc, development=False):
    return NYUv2LabeledDataset(ROOT / mc['dataset']['labeled_mat'], ROOT / mc['dataset']['split_file'],
                               split_variable='trainNdxs' if development else 'testNdxs')


def stem(sample):
    return f'test_{sample.test_position:04d}_mat_{sample.matlab_index_1based:04d}'


def development_stem(sample):
    return f'train_{sample.test_position:04d}_mat_{sample.matlab_index_1based:04d}'


def valid_depth(depth):
    depth = np.asarray(depth)
    if depth.dtype != np.float32 or depth.shape != (480,640) or not np.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError('Require finite positive full 480x640 float32 native metric depth')
    return depth


def baseline_files(config, model, mc):
    folder = (ROOT / config['models'][model]['baseline']).resolve()
    summary = read_json(folder / 'metrics_summary.json')
    if (summary.get('image_count') != 654 or summary.get('depth_type') != 'metric'
            or summary.get('unit') != 'meter' or summary.get('alignment') != 'none'
            or summary.get('extra_camera_information') != 'none'):
        raise ValueError('Baseline must be a full RGB-only native metric run')
    indices = load_nyuv2_test_indices(ROOT / mc['dataset']['split_file'])
    keys = [f'test_{i:04d}_mat_{int(index):04d}' for i, index in enumerate(indices)]
    if {p.stem for p in (folder / 'predictions').glob('*.npz')} != set(keys):
        raise ValueError('Baseline prediction identities/count differ')
    table = read_csv(folder / 'metrics_per_image.csv')
    indexed = {r['sample_id']: r for r in table}
    if len(table) != 654 or set(indexed) != set(keys):
        raise ValueError('Baseline metric table identities differ')
    files = {str(p): sha(p) for p in [folder / 'metrics_summary.json', folder / 'metrics_per_image.csv', folder / 'run_metadata.json']}
    for i, key in enumerate(keys):
        p, m = folder / 'predictions' / (key + '.npz'), folder / 'metadata' / (key + '.json')
        meta = read_json(m)
        if (meta.get('sample_id') != key or meta.get('matlab_index_1based') != int(indices[i])
                or meta.get('depth_type') != 'metric' or meta.get('unit') != 'meter'
                or meta.get('alignment') != 'none' or meta.get('external_focal_length_provided', False)
                or meta.get('external_intrinsics_provided', False)):
            raise ValueError('Baseline metadata is not the declared native prediction: ' + key)
        files[str(p)], files[str(m)] = sha(p), sha(m)
    return folder, indexed, files


def native(folder, sample):
    p = folder / 'predictions' / (stem(sample) + '.npz')
    with np.load(p, allow_pickle=False) as archive:
        depth = valid_depth(archive['depth_m']).copy()
    return depth, sha(p)


def score(depth, target, boundary=True):
    values = dict(compute_depth_metrics(depth[CROP], target[CROP], min_depth=.001, max_depth=10.))
    if boundary:
        sys.path.insert(0, str(ROOT / 'ml-depth-pro/src'))
        values['si_boundary_f1'] = compute_depth_pro_si_boundary_f1(np.clip(depth[CROP], .001, 10), target[CROP])
    return values


def assert_close(actual, expected, label):
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8, equal_nan=False, err_msg=label)


def raw_matches(scores, row):
    for key in scores:
        if key in row:
            assert_close(scores[key], float(row[key]), 'RAW vs frozen Part 2/' + key)


def means(rows, metric_names):
    methods = list(dict.fromkeys(r['method'] for r in rows))
    return {method: {k: float(np.mean([float(r[k]) for r in rows if r['method'] == method])) for k in metric_names}
            for method in methods}


def sources():
    files = [p for d in ['src/refinement','pipelines/part3','scripts/part3','configs/part3']
             for p in (ROOT / d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.sh','.yaml','.json','.txt')]
    files += [ROOT / p for p in ['src/data/nyuv2.py','src/metrics/depth.py','src/metrics/boundary.py',
                                'src/models/depth_pro_adapter.py','src/models/unidepth_v2_adapter.py',
                                'configs/part2/depth_pro_nyuv2.yaml','configs/part2/unidepth_v2_nyuv2.yaml',
                                'configs/versions.yaml','ml-depth-pro/src/depth_pro/eval/boundary_metrics.py']]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def environment():
    import importlib.metadata
    packages = {}
    for name in ('numpy','torch','opencv-python','h5py','scipy'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                conda_env=os.environ.get('CONDA_DEFAULT_ENV'), packages=packages)


def begin(path, kind, config, model, input_files, parents=(), **extra):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Project outputs must be under outputs/')
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / 'status.json', dict(status='running', kind=kind))
    record = dict(kind=kind, config=config, model=model, input_files=input_files,
                  parents=[str(Path(p).resolve()) for p in parents], source_sha256=sources(),
                  command=sys.argv, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  environment=environment(), formal_training_run=False, **extra)
    write_json(path / 'provenance/config.json', config)
    for name in record['source_sha256']:
        dest = path / 'provenance/source' / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dest)
    return record


def seal(path, record, *, formal_complete):
    if record['source_sha256'] != sources():
        raise ValueError('Source changed during execution')
    for file, digest in record['input_files'].items():
        if sha(file) != digest:
            raise ValueError('Input changed during execution: ' + file)
    record['formal_complete'] = formal_complete
    record['artifacts'] = {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*'))
                           if p.is_file() and p.name not in ('status.json','run_metadata.json','verification.json')}
    write_json(path / 'run_metadata.json', record)
    write_json(path / 'status.json', dict(status='complete' if formal_complete else 'smoke_complete',
                                         metadata_sha256=sha(path / 'run_metadata.json')))


def checked(path, kind=None, *, require_full=True):
    path = Path(path).resolve()
    status, record = read_json(path / 'status.json'), read_json(path / 'run_metadata.json')
    if status.get('status') not in ('complete','smoke_complete') or status.get('metadata_sha256') != sha(path / 'run_metadata.json'):
        raise ValueError('Incomplete run or changed metadata')
    if kind and record['kind'] != kind:
        raise ValueError('Wrong parent run type')
    if require_full and not record['formal_complete']:
        raise ValueError('Smoke/subset run cannot supply formal calibration or evaluation')
    if record['source_sha256'] != sources():
        raise ValueError('Scientific source changed')
    for name, digest in record['artifacts'].items():
        if sha(path / name) != digest:
            raise ValueError('Artifact changed: ' + name)
    for name, digest in record['input_files'].items():
        if sha(name) != digest:
            raise ValueError('Parent/resource changed: ' + name)
    for parent in record['parents']:
        checked(parent)
    return record


def failed(path, exc):
    if (path / 'status.json').exists():
        write_json(path / 'status.json', dict(status='failed', error_type=type(exc).__name__, error=str(exc)))


def positions(count, limit):
    if limit is not None and not 1 <= limit <= count:
        raise ValueError('Limit must be within the split')
    return range(count if limit is None else limit)


def report(path, value, destination=None):
    # 已有科学记录保持只读；可显式另存核验。
    if destination:
        destination = Path(destination)
        if destination.exists(): raise FileExistsError(destination)
        write_json(destination, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
