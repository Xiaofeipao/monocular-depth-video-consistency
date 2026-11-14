"""C v2来源封装、场景划分、开发集选参与共同剖面统计。"""
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil
import sys
import numpy as np
import yaml
from pipelines.part3 import common as c
from pipelines.part3_studies import common as study
from src.part3_c_v2.refine import Parameters

ROOT = c.ROOT
METHODS = ['RAW', 'C', 'Gaussian_log', 'RGB_guided_log',
           'C_v2', 'C_v2_no_sharpen', 'C_v2_no_RGB_support']


def equivalent(actual, expected, path='summary'):
    """Identity/status exact, recomputed numerical values use declared tolerance."""
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or set(actual) != set(expected):
            raise ValueError('Summary fields differ: '+path)
        for key in actual:
            equivalent(actual[key], expected[key], path+'/'+str(key))
    elif isinstance(actual, (list, tuple)):
        if not isinstance(expected, (list, tuple)) or len(actual) != len(expected):
            raise ValueError('Summary length differs: '+path)
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            equivalent(a, b, path+'/'+str(i))
    elif isinstance(actual, (float, np.floating)):
        if not isinstance(expected, (float, int)) or isinstance(expected, bool):
            raise ValueError('Summary numeric type differs: '+path)
        c.assert_close(actual, expected, path)
    elif actual != expected or (isinstance(actual, bool) != isinstance(expected, bool)):
        raise ValueError('Summary identity/status differs: '+path)


def configuration(path, model):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['protocol'] != 'part3_c_bounded_shock_v2' or cfg['methods'] != METHODS:
        raise ValueError('Wrong v2 protocol/method coverage')
    base, mc = c.load_config(ROOT/cfg['base_config'], model, 'c')
    diagnostic, _, _ = study.config('c', model, ROOT/cfg['diagnostics_reference'])
    if not 0 < cfg['development_scene_fraction'] < 1:
        raise ValueError('Scene fraction must be strictly between zero and one')
    if cfg['selection'] != 'equal_rank_sum_SI_width_primaryF1_texture_then_SI_then_id':
        raise ValueError('Unrecognized development-only selection rule')
    ids = [row['id'] for row in cfg['candidates']]
    if not ids or len(ids) != len(set(ids)) or 'RAW' in ids:
        raise ValueError('Distinct non-RAW candidate identities required')
    for row in cfg['candidates']:
        if set(row) != {'id', 'denoise_blend', 'iterations'}:
            raise ValueError('Unexpected candidate field')
        parameters(cfg, row['id'])
    if cfg['bootstrap_resamples'] < 1:
        raise ValueError('Positive bootstrap count required')
    return cfg, base, mc, diagnostic


def parameters(cfg, identity):
    row = next(r for r in cfg['candidates'] if r['id'] == identity)
    return Parameters(**cfg['fixed'], **{k: row[k] for k in ('denoise_blend', 'iterations')})


def partition(samples, cfg):
    scenes = sorted({s.scene for s in samples}, key=lambda s: hashlib.sha256(
        (cfg['development_scene_seed']+'\0'+s).encode()).hexdigest())
    if len(scenes) < 2:
        raise ValueError('At least two development scenes required')
    count = max(1, min(len(scenes)-1, int(len(scenes)*cfg['development_scene_fraction'])))
    design = set(cfg.get('design_scenes', [])) & set(scenes)
    if len(design) > count:
        raise ValueError('Too many reserved design scenes')
    remaining = [scene for scene in scenes if scene not in design]
    tuning = design | set(remaining[:count-len(design)])
    return [dict(position=s.test_position, matlab_index=s.matlab_index_1based,
                 sample_id=c.development_stem(s), scene=s.scene,
                 subset='tune' if s.scene in tuning else 'validation') for s in samples]


def sources():
    result = study.sources()
    for directory in ('src/part3_c_v2', 'pipelines/part3_c_v2',
                      'scripts/part3_c_v2', 'configs/part3_c_v2'):
        for p in sorted((ROOT/directory).rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.sh', '.yaml') and '__pycache__' not in p.parts:
                result[str(p.relative_to(ROOT))] = c.sha(p)
    return result


def begin(output, kind, cfg, model, inputs, parents=(), **extra):
    output = Path(output).resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'outputs/part3/direction_c_v2'):
        raise ValueError('Project C v2 outputs must be under outputs/part3/direction_c_v2/')
    output.mkdir(parents=True, exist_ok=False)
    c.write_json(output/'status.json', dict(status='running', kind=kind))
    record = dict(kind=kind, config=cfg, model=model, input_files=inputs,
                  parents=[str(Path(p).resolve()) for p in parents], source_sha256=sources(),
                  created_at_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                  environment=c.environment(), backend='cpu_float64_to_float32',
                  depth_type='metric', unit='meter', alignment='none',
                  native_metric=False, postprocess='RGB-supported bounded log-depth shock',
                  formal_training_run=False, test_results_previously_inspected=True, **extra)
    for name in record['source_sha256']:
        dest = output/'provenance/source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    c.write_json(output/'provenance/config.json', cfg)
    return record


def seal(output, record, full):
    if sources() != record['source_sha256']:
        raise ValueError('C v2 source changed during run')
    for name, digest in record['input_files'].items():
        if c.sha(name) != digest:
            raise ValueError('C v2 input changed: '+name)
    record['formal_complete'] = bool(full)
    record['artifacts'] = {str(p.relative_to(output)): c.sha(p) for p in sorted(output.rglob('*'))
                           if p.is_file() and p.name not in ('status.json', 'run_metadata.json')}
    c.write_json(output/'run_metadata.json', record)
    c.write_json(output/'status.json', dict(status='complete' if full else 'subset_complete',
                                           metadata_sha256=c.sha(output/'run_metadata.json')))


def checked(output, kind=None, full=True):
    output = Path(output).resolve()
    record = c.read_json(output/'run_metadata.json')
    if not record['kind'].startswith('c_v2_'):
        return study.checked(output, kind, full=full)
    status = c.read_json(output/'status.json')
    if (status['status'] not in ('complete', 'subset_complete')
            or status['metadata_sha256'] != c.sha(output/'run_metadata.json')):
        raise ValueError('Incomplete C v2 run/changed metadata')
    if kind and record['kind'] != kind:
        raise ValueError('Wrong C v2 parent kind')
    if full and not record['formal_complete']:
        raise ValueError('Subset cannot supply formal C v2 parameters')
    if record['source_sha256'] != sources():
        raise ValueError('C v2 scientific source changed')
    actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
    if actual != set(record['artifacts']) | {'status.json', 'run_metadata.json'}:
        raise ValueError('Unexpected/missing C v2 artifact')
    for name, digest in record['artifacts'].items():
        if c.sha(output/name) != digest:
            raise ValueError('C v2 artifact changed: '+name)
    for name, digest in record['input_files'].items():
        if c.sha(name) != digest:
            raise ValueError('C v2 input changed: '+name)
    for parent in record['parents']:
        checked(parent)
    return record


def common_profiles(profiles, methods):
    """All methods use exactly the same measurable GT-fixed cross-sections."""
    index = {(r['sample_id'], r['profile_id'], r['method']): r for r in profiles}
    identities = sorted({(r['sample_id'], r['profile_id']) for r in profiles})
    keys = [(key, number) for key, number in identities if all(
        (key, number, m) in index and index[key, number, m]['status'] == 'measurable' for m in methods)]
    shared = [index[key, number, method] for key, number in keys for method in methods]
    return dict(profiles=len(keys), images=len({key for key, _ in keys}),
                scene_count=len({r['scene'] for r in shared}), rows=shared)


def rank_candidates(rows, profiles, identities):
    # RAW establishes common support but is never a selectable fallback.
    common = common_profiles(profiles, ['RAW', *identities])
    if not common['profiles']:
        raise ValueError('No common measurable development profiles; do not freeze parameters')
    table = []
    for identity in identities:
        current = [r for r in rows if r['method'] == identity]
        table.append(dict(candidate=identity,
            si_boundary_f1=study.describe(current, 'si_boundary_f1')['mean'],
            primary_f1=study.describe(current, 'primary_f1')['mean'],
            texture_spurious_fraction=study.describe(current, 'texture_spurious_fraction')['mean'],
            width_10_90_px=study.describe([r for r in common['rows'] if r['method'] == identity],
                                         'width_10_90_px')['mean']))
    for field, larger in [('si_boundary_f1', True), ('primary_f1', True),
                          ('width_10_90_px', False), ('texture_spurious_fraction', False)]:
        values = sorted({r[field] for r in table if r[field] is not None}, reverse=larger)
        for row in table:
            row[field+'_rank'] = values.index(row[field])+1 if row[field] is not None else len(identities)+1
    for row in table:
        row['rank_sum'] = sum(value for key, value in row.items() if key.endswith('_rank'))
    table.sort(key=lambda r: (r['rank_sum'], -r['si_boundary_f1'], r['candidate']))
    return dict(selected=table[0]['candidate'], ranking=table,
                common_profiles=common['profiles'], common_images=common['images'],
                policy='Four equal dense ranks; ties use SI then candidate id; never fallback to RAW')
