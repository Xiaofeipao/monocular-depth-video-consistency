#!/usr/bin/env python3
"""Direction C：固定5×5联合滤波、两项消融、场景统计与原始预测重算。"""
from pathlib import Path
import argparse
import json
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipelines.part3 import common as c
from src.refinement.ablation import make_refiner


def scene_statistics(rows, resamples=10000, seed=42):
    raw = {r['sample_id']: r for r in rows if r['method'] == 'RAW'}
    full = {r['sample_id']: r for r in rows if r['method'] == 'C'}
    if set(raw) != set(full): raise ValueError('Unpaired C/RAW rows')
    keys = sorted(raw); scenes = sorted({raw[k]['scene'] for k in keys})
    groups = np.array([scenes.index(raw[k]['scene']) for k in keys])
    counts = np.bincount(groups, minlength=len(scenes))
    draws = np.random.default_rng(seed).integers(0, len(scenes), (resamples, len(scenes)))
    denominator = counts[draws].sum(axis=1)
    result = {}
    for metric, maximize in [('abs_rel',False), ('rmse',False), ('delta1',True), ('si_boundary_f1',True)]:
        delta = np.array([float(full[k][metric]) - float(raw[k][metric]) for k in keys])
        totals = np.bincount(groups, weights=delta, minlength=len(scenes))
        lower, upper = np.quantile(totals[draws].sum(axis=1) / denominator, [.025,.975])
        result[metric] = dict(delta_C_minus_RAW=float(delta.mean()), ci95=[float(lower),float(upper)],
                              image_win_fraction=float(np.mean(delta > 0 if maximize else delta < 0)))
    return dict(images=len(keys), scenes=len(scenes), resamples=resamples, seed=seed,
                unit='Whole scenes; image-weighted ratio within each draw', metrics=result,
                limitation='Exploratory conditional interval, not cross-domain superiority or test-set parameter selection')


def choose_cases(rows):
    raw = {r['sample_id']:r for r in rows if r['method']=='RAW'}
    full = {r['sample_id']:r for r in rows if r['method']=='C'}
    orderings = [sorted(raw, key=lambda k:-float(raw[k]['abs_rel'])),
                 sorted(raw, key=lambda k:float(full[k]['si_boundary_f1'])-float(raw[k]['si_boundary_f1'])),
                 sorted(raw, key=lambda k:-(float(full[k]['si_boundary_f1'])-float(raw[k]['si_boundary_f1'])))]
    chosen = []
    for category, order in zip(('large_raw_metric_error','lowest_boundary_gain','highest_boundary_gain'), orderings):
        key = next((k for k in order if k not in {r['sample_id'] for r in chosen}), None)
        if key:
            chosen.append(dict(sample_id=key, category=category, RAW=raw[key], C=full[key],
                               selection='Post-hoc illustration only, never parameter selection'))
    return chosen


def save_cases(output, rows, source, mc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen = choose_cases(rows)
    c.write_json(output / 'case_selection.json', chosen)
    with c.dataset(mc) as data:
        for item in chosen:
            key = item['sample_id']; sample = data.get_test_sample(int(key.split('_')[1]))
            raw, _ = c.native(source, sample)
            with np.load(output / 'predictions/C' / (key + '.npz')) as p:
                improved = p['depth_m']
            fig, axes = plt.subplots(1,4, figsize=(16,4))
            axes[0].imshow(sample.rgb); axes[0].set_title('RGB')
            for ax, z, title in zip(axes[1:], (sample.depth_m,raw,improved), ('Filled GT (m)','RAW (m)','C (m)')):
                ax.imshow(z, vmin=.5, vmax=10, cmap='turbo'); ax.set_title(title)
            for ax in axes: ax.axis('off')
            fig.suptitle(key + ' / ' + item['category']); fig.tight_layout()
            fig.savefig(output / (key + '.png'), dpi=120); plt.close(fig)


def run(output, config, mc, model, backend, limit=None):
    inputs = c.resources(config, model, mc)
    source, baseline, files = c.baseline_files(config, model, mc); inputs.update(files)
    refiners = {m:make_refiner(m, backend) for m in config['methods'] if m != 'RAW'}
    torch = None
    if backend == 'cuda':
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        cuda = dict(device=torch.cuda.get_device_name(0), precision='float32', runtime=torch.version.cuda)
    else: cuda = dict(device='cpu', precision='float32', runtime=None)
    record = c.begin(output, 'c_evaluation', config, model, inputs, backend=backend, device=cuda, limit=limit,
                     depth_type='metric', unit='meter', alignment='none', native_metric=True,
                     timing_scope='Postprocessing incl. host/device transfers; excludes model, I/O and scoring; first sample warmup per method')
    try:
        rows = []
        for method in refiners: (output / 'predictions' / method).mkdir(parents=True)
        with c.dataset(mc) as data:
            for i in c.positions(len(data), limit):
                sample = data.get_test_sample(i); key = c.stem(sample)
                raw, digest = c.native(source, sample)
                for method in config['methods']:
                    seconds, peak = 0., None
                    if method == 'RAW': z = raw
                    else:
                        refine = refiners[method]
                        if i == 0: refine(raw, sample.rgb)
                        if torch:
                            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter(); z = refine(raw, sample.rgb)
                        if torch: torch.cuda.synchronize()
                        seconds = time.perf_counter() - start
                        if torch: peak = int(torch.cuda.max_memory_allocated())
                        c.valid_depth(z)
                        np.savez_compressed(output / 'predictions' / method / (key + '.npz'), depth_m=z)
                    values = c.score(z, sample.depth_m)
                    if method == 'RAW': c.raw_matches(values, baseline[key])
                    rows.append(dict(sample_id=key, scene=sample.scene, method=method,
                                     source_prediction_sha256=digest, postprocess_seconds=seconds,
                                     peak_gpu_memory_bytes=peak, **values))
                if i % 50 == 0: print(f'{model} C {i + 1}/654', flush=True)
        c.write_csv(output / 'per_image.csv', rows)
        c.write_json(output / 'summary.json', dict(samples=len(rows)//4, means=c.means(rows,c.METRICS),
                     parameters=config['parameters'], methods=config['methods'],
                     mean_postprocess_seconds=c.means(rows,['postprocess_seconds']),
                     limitation='Fixed settings; no parameter search, no B+C, no implied temporal or cross-domain gain'))
        c.write_json(output / 'scene_statistics.json', scene_statistics(rows, config['bootstrap_resamples'], config['bootstrap_seed']))
        save_cases(output, rows, source, mc)
        c.seal(output, record, formal_complete=limit is None and len(rows)==2616)
    except BaseException as exc:
        c.failed(output, exc); raise


def verify(output, mc):
    record = c.checked(output, 'c_evaluation', require_full=False)
    config, model = record['config'], record['model']
    source = ROOT / config['models'][model]['baseline']
    baseline = {r['sample_id']:r for r in c.read_csv(source / 'metrics_per_image.csv')}
    saved = c.read_csv(output / 'per_image.csv')
    count = len(c.positions(654, record['limit']))
    if len(saved) != count * 4:
        raise ValueError('Incomplete C score table')
    index = {(r['sample_id'],r['method']):r for r in saved}
    if len(index) != len(saved): raise ValueError('Duplicated C score identities')
    # 用CPU的同公式独立重做滤波；验证CUDA输出允许微小归约差异。
    refiners = {m:make_refiner(m,'cpu') for m in config['methods'] if m != 'RAW'}
    expected_keys = []
    with c.dataset(mc) as data:
        for i in range(count):
            sample = data.get_test_sample(i); key = c.stem(sample); expected_keys.append(key)
            raw, digest = c.native(source, sample)
            for method in config['methods']:
                row = index[(key,method)]
                if row['source_prediction_sha256'] != digest or row['scene'] != sample.scene:
                    raise ValueError('C input identity differs')
                if method == 'RAW': z = raw
                else:
                    expected = refiners[method](raw, sample.rgb)
                    with np.load(output / 'predictions' / method / (key + '.npz'), allow_pickle=False) as archive:
                        z = c.valid_depth(archive['depth_m'])
                    c.assert_close(z, expected, 'C same-formula CPU replay/' + method)
                values = c.score(z, sample.depth_m)
                if method == 'RAW': c.raw_matches(values, baseline[key])
                for metric, value in values.items():
                    c.assert_close(float(row[metric]),value,'C GT score/' + metric)
    for method in refiners:
        if {p.stem for p in (output / 'predictions' / method).glob('*.npz')} != set(expected_keys):
            raise ValueError('C prediction file coverage differs')
    summary = c.read_json(output / 'summary.json')
    for method, metrics in c.means(saved, c.METRICS).items():
        for metric, value in metrics.items(): c.assert_close(value,summary['means'][method][metric],'C macro mean')
    if scene_statistics(saved,config['bootstrap_resamples'],config['bootstrap_seed']) != c.read_json(output / 'scene_statistics.json'):
        raise ValueError('Scene statistics differ')
    return dict(passed=True, samples=count, prediction_files_checked=count*3,
                raw_and_ablations_recomputed=True, cpu_formula_replayed=True,
                formal_complete=record['formal_complete'], rtol=1e-6, atol=1e-8)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check','run','verify'])
    p.add_argument('--model', required=True, choices=['depth_pro','unidepth_v2'])
    p.add_argument('--config', type=Path, default=ROOT / 'configs/part3/direction_c.yaml')
    p.add_argument('--backend', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--output-dir', type=Path); p.add_argument('--run', type=Path)
    p.add_argument('--limit', type=int); p.add_argument('--report', type=Path)
    a = p.parse_args(argv); cv2.setNumThreads(1)
    config, mc = c.load_config(a.config, a.model, 'c')
    if a.action == 'check':
        c.resources(config,a.model,mc); source,_,_ = c.baseline_files(config,a.model,mc)
        print('Dataset/SHA, official splits and 654 native baseline identities passed:',source); return
    if a.action == 'verify':
        if not a.run: p.error('verify requires --run')
        record = c.checked(a.run,require_full=False)
        if record['model'] != a.model or record['config'] != config: raise ValueError('Verification model/config differs')
        c.report(a.run,verify(a.run.resolve(),mc),a.report); return
    output = (a.output_dir or ROOT / config['output_root'] / a.model / 'evaluation_v1').resolve()
    run(output, config, mc, a.model, a.backend, a.limit)


if __name__ == '__main__': main()
