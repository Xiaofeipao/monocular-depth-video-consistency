#!/usr/bin/env python3
"""独立复核两条native米制基线的绝对尺度与物体边界；不运行模型。"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import yaml
from scipy.ndimage import maximum_filter
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.analysis.scale_boundary import (METRICS, annotation_edges, choose_profiles,
    depth_edge_map, edge_comparison, error_statistics, measure_profile, region_masks,
    scale_statistics)
from src.data.nyuv2 import NYUv2LabeledDataset
from src.metrics.depth import compute_depth_metrics

MODELS = ('depth_pro', 'unidepth_v2')
TABLES = ('per_image', 'depth_bins', 'regions', 'boundary', 'localization', 'profiles')
SOURCE_FILES = [
    'pipelines/part2/analyze_scale_boundary.py', 'src/analysis/scale_boundary.py',
    'src/visualization/scale_boundary.py', 'src/data/nyuv2.py',
    'src/metrics/depth.py', 'src/metrics/boundary.py',
    'scripts/part2/analyze_scale_boundary.sh',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path):
    c = yaml.safe_load(Path(path).read_text())
    e = c['evaluation']
    if (c['protocol'] != 'nyuv2_scale_boundary_v1' or c['dataset']['expected_images'] != 654
        or c['dataset']['split_variable'] != 'testNdxs' or e['crop'] != [45,471,41,601]
        or (e['minimum_m'],e['maximum_m']) != (.001,10.)
        or e['alignment'] != 'none' or e['aggregation'] != 'per_image_macro_mean'
        or set(c['baselines']) != set(MODELS)
        or c['scale']['oracle'] != 'raw_per_image_median_ratio_diagnostic_only'
        or c['statistics']['unit'] != 'scene_cluster_preserving_image_weights'):
        raise ValueError('配置必须保持标准NYUv2原生米制协议与诊断区分')
    b, p = c['boundary'], c['profiles']
    if (b['primary_ratio'] not in b['ratios'] or any(r<=1 for r in b['ratios'])
        or b['matching_distance'] != 'chebyshev' or b['tolerance_pixels'] != [1,2,3]
        or b['localization_distance'] != 'euclidean_capped'
        or b['distance_cap_px'] <= 3 or b['object_annotation_support_px'] < 0
        or p['radius_px'] <= p['plateau_samples'] or p['plateau_samples'] < 2
        or not p['require_object_annotation_support'] or p['maximum_per_image'] < 1
        or p['minimum_gt_ratio'] <= 1 or p['minimum_prediction_ratio'] <= 1
        or p['maximum_gt_width_px'] <= 0):
        raise ValueError('无效的边界/剖面协议')
    bins = c['scale']['depth_bins_m']
    if (bins[0][0] != .001 or bins[-1][1] != 10.
        or any(a>=b for a,b in bins) or any(a[1]!=b[0] for a,b in zip(bins,bins[1:]))):
        raise ValueError('距离分桶必须完整且不重叠')
    if not (0<c['statistics']['confidence']<1) or c['statistics']['bootstrap_resamples']<100:
        raise ValueError('无效的统计配置')
    return c


def prepare_sources(c):
    """Read-only input verification; original key_analysis is not a dependency."""
    sources = {'dataset': {}, 'baselines': {}}
    for key in ('labeled_mat','split_file'):
        path = ROOT/c['dataset'][key]
        actual = sha(path)
        if actual != c['dataset'][key+'_sha256']:
            raise ValueError(f'Dataset SHA mismatch: {path}')
        sources['dataset'][key] = {'path':str(path.resolve()),'sha256':actual}
    rows = {}
    for model in MODELS:
        root = ROOT/c['baselines'][model]
        meta = json.loads((root/'run_metadata.json').read_text())
        if (meta['sample_count']!=c['dataset']['expected_images'] or not meta['full_test_split']
            or meta['test_time_alignment']!='none' or meta['formal_training_run']):
            raise ValueError(f'基线不完整或不是native: {model}')
        cfg_path = Path(meta['config'])
        if sha(cfg_path)!=meta['config_sha256']:
            raise ValueError(f'Baseline config SHA mismatch: {model}')
        cfg = yaml.safe_load(cfg_path.read_text())
        e = cfg['evaluation']
        crop = e['crop']
        if ([crop[k] for k in ('top','bottom','left','right')]!=c['evaluation']['crop']
            or e['valid_depth_m']!={'minimum':.001,'maximum':10.,'bounds':'exclusive'}
            or not e['clip_prediction_to_valid_range'] or e['test_time_alignment']!='none'
            or e['aggregation']!='per_image_macro_mean'
            or cfg['dataset']['split_variable']!='testNdxs'
            or any(cfg['dataset'][k+'_sha256']!=c['dataset'][k+'_sha256'] for k in ('labeled_mat','split_file'))
            or cfg['model'].get('provide_external_intrinsics',False)
            or cfg['model'].get('provide_external_focal_length',False)):
            raise ValueError(f'Baseline protocol mismatch: {model}')
        data = read_csv(root/'metrics_per_image.csv')
        lookup = {int(r['test_position']):r for r in data}
        n = c['dataset']['expected_images']
        predictions = sorted((root/'predictions').glob('*.npz'))
        if len(data)!=n or set(lookup)!=set(range(n)) or len(predictions)!=n:
            raise ValueError(f'Baseline missing/duplicate samples: {model}')
        summary = json.loads((root/'metrics_summary.json').read_text())
        for k in (*METRICS,'si_boundary_f1'):
            if not np.isclose(np.mean([float(r[k]) for r in data]),summary['macro_mean'][k],rtol=1e-6,atol=1e-8):
                raise ValueError(f'Baseline summary differs from per-image metrics: {model}/{k}')
        sources['baselines'][model] = {
            'root':str(root.resolve()),'config_path':str(cfg_path.resolve()),'config_sha256':sha(cfg_path),
            'records':{name:sha(root/name) for name in ('run_metadata.json','metrics_per_image.csv','metrics_summary.json')},
            'predictions':{p.name:sha(p) for p in predictions}}
        rows[model] = lookup
    return sources, rows


def load_depth(c, model, sample):
    stem = f'test_{sample.test_position:04d}_mat_{sample.matlab_index_1based:04d}'
    path = ROOT/c['baselines'][model]/'predictions'/f'{stem}.npz'
    with np.load(path) as f:
        p = f['depth_m'].copy()
    if p.dtype!=np.float32 or p.shape!=sample.depth_m.shape or not np.isfinite(p).all() or np.any(p<=0):
        raise ValueError(f'Invalid full-size float32 native prediction: {path}')
    return p


def evaluate(c, baseline_rows, positions=None):
    tables = {k:[] for k in TABLES}
    top,bottom,left,right = c['evaluation']['crop']
    crop = np.s_[top:bottom,left:right]
    minimum,maximum = c['evaluation']['minimum_m'],c['evaluation']['maximum_m']
    with NYUv2LabeledDataset(ROOT/c['dataset']['labeled_mat'],ROOT/c['dataset']['split_file'],split_variable='testNdxs') as ds:
        if len(ds)!=c['dataset']['expected_images']:
            raise ValueError('Unexpected split size')
        if 'labels' not in ds.file or 'instances' not in ds.file:
            raise ValueError('物体分析需要原MAT的labels与instances')
        positions = list(range(len(ds))) if positions is None else list(positions)
        for count, position in enumerate(positions):
            sample = ds.get_test_sample(position)
            stem = f'test_{position:04d}_mat_{sample.matlab_index_1based:04d}'
            gt = sample.depth_m[crop]
            valid = np.isfinite(gt)&(gt>minimum)&(gt<maximum)
            labels = np.asarray(ds.file['labels'][sample.matlab_index_1based-1]).T[crop]
            instances = np.asarray(ds.file['instances'][sample.matlab_index_1based-1]).T[crop]
            semantic = annotation_edges(labels,instances,valid)
            support = maximum_filter(semantic,size=2*c['boundary']['object_annotation_support_px']+1)>0
            gt_maps = {ratio:depth_edge_map(gt,valid_mask=valid,log_threshold=np.log(ratio)) for ratio in c['boundary']['ratios']}
            primary_edges = gt_maps[c['boundary']['primary_ratio']]
            object_edges = primary_edges & support
            regions = region_masks(primary_edges,object_edges,valid)
            profiles = choose_profiles(gt,valid,support,c['profiles'])
            for model in MODELS:
                source = baseline_rows[model][position]
                if (source['sample_id']!=stem or int(source['matlab_index_1based'])!=sample.matlab_index_1based
                    or source['scene']!=sample.scene):
                    raise ValueError(f'Baseline image identity differs: {model}/{stem}')
                prediction = load_depth(c,model,sample)[crop]
                ids = dict(model=model,sample_id=stem,test_position=position,
                           matlab_index_1based=sample.matlab_index_1based,scene=sample.scene)
                scale, tracks = scale_statistics(prediction,gt,valid,minimum,maximum)
                primary = compute_depth_metrics(prediction,gt,min_depth=minimum,max_depth=maximum,clip_prediction=True)
                for metric in METRICS:
                    if not np.isfinite(float(source[metric])) or not np.isclose(primary[metric],float(source[metric]),rtol=1e-6,atol=1e-8):
                        raise ValueError(f'Native score mismatch: {model}/{stem}/{metric}')
                tables['per_image'].append({**ids,**scale,**{f'primary_{k}':primary[k] for k in METRICS},
                    'si_boundary_f1_from_frozen_baseline':float(source['si_boundary_f1']),
                    'labelled_fraction':float(np.mean((labels>0)[valid])),
                    'object_supported_gt_edge_pixels':int(object_edges.sum()),
                    'gt_selected_profile_count':len(profiles)})
                for track, depth in tracks.items():
                    for lo,hi in c['scale']['depth_bins_m']:
                        mask = valid&(gt>=lo)&(gt<hi)
                        tables['depth_bins'].append({**ids,'track':track,'lower_m':lo,'upper_m':hi,**error_statistics(depth,gt,mask)})
                    total = error_statistics(depth,gt,valid)
                    for region,mask in regions.items():
                        stats = error_statistics(depth,gt,mask)
                        tables['regions'].append({**ids,'track':track,'region':region,**stats,
                            'pixel_fraction':stats['pixel_count']/scale['valid_pixels'],
                            'squared_error_share':(stats['squared_error_sum'] or 0)/total['squared_error_sum'] if total['squared_error_sum'] else None,
                            'absolute_relative_error_share':(stats['absolute_relative_error_sum'] or 0)/total['absolute_relative_error_sum'] if total['absolute_relative_error_sum'] else None})
                for track in ('native_raw','native_clipped'):
                    depth = tracks[track]
                    for ratio in c['boundary']['ratios']:
                        predicted_edges = depth_edge_map(depth,valid_mask=valid,log_threshold=np.log(ratio))
                        b, distances = edge_comparison(predicted_edges,gt_maps[ratio],valid,gt_maps[ratio]&support,
                            c['boundary']['tolerance_pixels'],c['boundary']['distance_cap_px'])
                        for item in b:
                            tables['boundary'].append({**ids,'track':track,'ratio':ratio,**item})
                        tables['localization'].append({**ids,'track':track,'ratio':ratio,**distances})
                for number, profile in enumerate(profiles):
                    tables['profiles'].append({**ids,'profile_id':number,**profile,
                        **measure_profile(tracks['native_raw'],gt,profile,c['profiles'])})
            if count%25==0 or count==len(positions)-1:
                print(f'[{count+1}/{len(positions)}] 尺度、物体边界与剖面 {stem}',flush=True)
    return tables


class ClusterStatistics:
    def __init__(self, config):
        self.config = config
        self.draws = {}

    def describe(self, rows, metric):
        selected = [r for r in rows if r.get(metric) is not None]
        if not selected:
            return dict(mean=None,lower=None,upper=None,image_count=0,scene_count=0)
        # A profile table has several rows/image: aggregate within each image first.
        images = {}
        for r in selected:
            images.setdefault((r['scene'],r['sample_id']),[]).append(float(r[metric]))
        values = [(scene,float(np.mean(v))) for (scene,_),v in images.items()]
        scenes = sorted({scene for scene,_ in values})
        sums = np.array([sum(x for s,x in values if s==scene) for scene in scenes])
        counts = np.array([sum(s==scene for s,_ in values) for scene in scenes])
        n = len(scenes)
        if n not in self.draws:
            rng = np.random.default_rng(self.config['seed'])
            self.draws[n] = rng.integers(0,n,size=(self.config['bootstrap_resamples'],n))
        draw = self.draws[n]
        means = sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
        alpha = (1-self.config['confidence'])/2
        return dict(mean=float(np.mean([x for _,x in values])),
                    lower=float(np.quantile(means,alpha)),upper=float(np.quantile(means,1-alpha)),
                    image_count=len(images),scene_count=n)


def aggregate(tables,c):
    stats = ClusterStatistics(c['statistics'])
    summary = dict(protocol=c['protocol'],primary_alignment='none',
        oracle_tracks_are_diagnostic_only=True,
        si_metric_source='frozen baseline CSV; not recomputed by this analysis',
        interval_unit=c['statistics']['unit'],bootstrap_resamples=c['statistics']['bootstrap_resamples'],
        empty_region_policy='N/A; macro over images with support, report coverage',
        profiles_policy='GT-only selection, own-plateau shape normalization, image-mean aggregation',models={},paired={},correlations={})
    for model in MODELS:
        rows = [r for r in tables['per_image'] if r['model']==model]
        metrics = [f'primary_{k}' for k in METRICS]+[
            'raw_scale_ratio','clipped_scale_ratio','raw_abs_log_scale',
            'raw_out_of_range_fraction','raw_above_maximum_fraction','raw_below_minimum_fraction',
            'log_bias','log_mse','log_bias_squared','centered_log_mse',
            'native_raw_abs_rel','native_raw_rmse_m','native_raw_mae_m','native_raw_bias_m',
            'native_clipped_abs_rel','native_clipped_mae_m','native_clipped_bias_m',
            'oracle_raw_median_clipped_abs_rel','oracle_raw_median_clipped_rmse_m',
            'si_boundary_f1_from_frozen_baseline']
        report = {k:stats.describe(rows,k) for k in metrics}
        ratio = np.array([r['raw_scale_ratio'] for r in rows])
        lower,upper = c['scale']['within_ratio']
        mean_mse = np.mean([r['log_mse'] for r in rows])
        report['scale_distribution'] = dict(median=float(np.median(ratio)),
            p05=float(np.quantile(ratio,.05)),p95=float(np.quantile(ratio,.95)),
            under_fraction=float(np.mean(ratio<lower)),within_fraction=float(np.mean((ratio>=lower)&(ratio<=upper))),
            over_fraction=float(np.mean(ratio>upper)),
            global_log_bias_squared_share=float(np.mean([r['log_bias_squared'] for r in rows])/mean_mse) if mean_mse else None,
            maximum_absolute_log_decomposition_residual=max(abs(r['log_decomposition_residual']) for r in rows))
        report['raw_pixel_tail'] = dict(valid_pixels=sum(r['valid_pixels'] for r in rows),
            images_with_any_out_of_range=sum(r['raw_out_of_range_fraction']>0 for r in rows),
            minimum_prediction_m=min(r['raw_min_m'] for r in rows),maximum_prediction_m=max(r['raw_max_m'] for r in rows))
        summary['models'][model] = report
        a = [r['raw_abs_log_scale'] for r in rows]
        b = [r['native_clipped_abs_rel'] for r in rows]
        rho = spearmanr(a,b).statistic if len(set(a))>1 and len(set(b))>1 else None
        summary['correlations'][model] = {'absolute_log_scale_vs_absrel_spearman':float(rho) if rho is not None and np.isfinite(rho) else None,'interpretation':'descriptive, not causal'}
    grouped = {}
    for table,keys,metrics in [
        ('depth_bins',('model','track','lower_m','upper_m'),('abs_rel','mae_m','rmse_m','bias_m','relative_bias')),
        ('regions',('model','track','region'),('abs_rel','mae_m','rmse_m','bias_m','pixel_fraction','squared_error_share','absolute_relative_error_share')),
        ('boundary',('model','track','ratio','tolerance_px'),('precision','recall','f1','object_edge_recall')),
        ('localization',('model','track','ratio'),('prediction_edge_distance_capped_px','gt_edge_distance_capped_px','gt_edge_miss_beyond_3px_fraction','prediction_edge_spurious_beyond_3px_fraction')),
    ]:
        rows_out = []
        groups = {}
        for row in tables[table]:
            groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
        for key,rows in groups.items():
            result = dict(zip(keys,key))
            result['image_count'] = len(rows)
            if 'pixel_count' in rows[0]:
                result['supported_image_count'] = sum(r['pixel_count']>0 for r in rows)
                result['pixel_count'] = sum(r['pixel_count'] for r in rows)
            for metric in metrics:
                s = stats.describe(rows,metric)
                result.update({metric+'_'+k:v for k,v in s.items()})
            rows_out.append(result)
        grouped[table] = rows_out
    profile_summary = {}
    for model in MODELS:
        rows = [r for r in tables['profiles'] if r['model']==model]
        status = {s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}
        profile_summary[model] = dict(profile_count=len(rows),images_with_profiles=len({r['sample_id'] for r in rows}),
            status_counts=status,measurable_fraction=sum(r['status']=='measurable' for r in rows)/len(rows) if rows else None,
            **{m:stats.describe(rows,m) for m in ('absolute_midpoint_error_px','signed_midpoint_error_px','width_10_90_px','width_excess_px','contrast_ratio_to_gt','profile_mae_m')})
    summary['profiles'] = profile_summary
    # Paired estimates use the same image IDs and preserve the image macro mean.
    specs = [('per_image',None,('primary_abs_rel','primary_rmse','primary_delta1','raw_abs_log_scale')),
             ('boundary',dict(track='native_raw',ratio=c['boundary']['primary_ratio'],tolerance_px=3),('f1','object_edge_recall')),
             ('regions',dict(track='native_clipped',region='edge_le3px'),('abs_rel','rmse_m')),
             ('profiles',None,('absolute_midpoint_error_px','width_excess_px'))]
    for table,filters,metrics in specs:
        selected = [r for r in tables[table] if not filters or all(r[k]==v for k,v in filters.items())]
        # Profiles are paired per cross-section before averaging within image.
        index = {(r['model'],r['sample_id'],r.get('profile_id',0)):r for r in selected}
        for metric in metrics:
            rows=[]
            for key,a in index.items():
                if key[0]!='depth_pro':continue
                b=index.get(('unidepth_v2',key[1],key[2]))
                if b and a.get(metric) is not None and b.get(metric) is not None:
                    rows.append({'scene':a['scene'],'sample_id':a['sample_id'],metric:a[metric]-b[metric]})
            summary['paired'][f'{table}_{metric}_depth_pro_minus_unidepth'] = stats.describe(rows,metric)
    return summary,grouped


def save_tables(output,tables):
    for name,rows in tables.items():
        write_csv(output/'tables'/f'{name}.csv',rows)


def assert_sources_unchanged(sources):
    for data in sources['dataset'].values():
        if sha(data['path'])!=data['sha256']:raise ValueError('Dataset changed during analysis')
    for source in sources['baselines'].values():
        root=Path(source['root'])
        if sha(source['config_path'])!=source['config_sha256']:raise ValueError('Baseline config changed')
        for name,digest in source['records'].items():
            if sha(root/name)!=digest:raise ValueError(f'Baseline record changed: {name}')
        for name,digest in source['predictions'].items():
            if sha(root/'predictions'/name)!=digest:raise ValueError(f'Prediction changed: {name}')


def numeric_tables_equal(expected,actual):
    if set(expected)!=set(actual):raise ValueError('Table coverage changed')
    comparisons=0
    for name in expected:
        if len(expected[name])!=len(actual[name]):raise ValueError(f'Row count differs: {name}')
        for a,b in zip(expected[name],actual[name]):
            if set(a)!=set(b):raise ValueError(f'Column coverage differs: {name}')
            for key in a:
                x,y=a[key],b[key]
                if x is None:
                    if y not in (None,''):raise ValueError(f'Expected N/A: {name}/{key}')
                elif isinstance(x,int) and not isinstance(x,bool):
                    if not float(y).is_integer() or int(x)!=int(float(y)):raise ValueError(f'Integer count differs: {name}/{key}')
                    comparisons+=1
                elif isinstance(x,float):
                    if not np.isclose(float(x),float(y),rtol=1e-6,atol=1e-8):raise ValueError(f'Numeric mismatch: {name}/{key}')
                    comparisons+=1
                elif str(x)!=str(y):raise ValueError(f'Identity/status differs: {name}/{key}')
    return comparisons


def summaries_close(expected, actual, path='summary'):
    """Numerical replay uses tolerances; identities and coverage remain exact."""
    if isinstance(expected, dict):
        if not isinstance(actual,dict) or set(expected)!=set(actual):
            raise ValueError(f'Summary keys differ: {path}')
        for key in expected:
            summaries_close(expected[key],actual[key],f'{path}/{key}')
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(expected)!=len(actual):
            raise ValueError(f'Summary coverage differs: {path}')
        for i,(a,b) in enumerate(zip(expected,actual)):
            summaries_close(a,b,f'{path}/{i}')
    elif isinstance(expected,int) and not isinstance(expected,bool):
        if type(actual) is not int or expected!=actual:
            raise ValueError(f'Summary integer coverage differs: {path}')
    elif isinstance(expected,float):
        if not isinstance(actual,(float,int)) or isinstance(actual,bool) or not np.isclose(expected,actual,rtol=1e-6,atol=1e-8):
            raise ValueError(f'Summary numeric mismatch: {path}')
    elif expected != actual:
        raise ValueError(f'Summary value differs: {path}')


def execute(c,config_path,output):
    if output.exists():raise FileExistsError(f'输出目录已存在，请另选目录: {output}')
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    meta=dict(status='running',created_at_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv,config_sha256=sha(config_path),formal_training_run=False,
        inference_performed=False,analysis_device='cpu',full_test_split=False,
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                         packages={k:version(k) for k in ('scipy','matplotlib','h5py','PyYAML','Pillow')}),source_sha256={})
    dump_json(output/'run_metadata.json',meta)
    try:
        shutil.copy2(config_path,output/'config.yaml')
        for relative in SOURCE_FILES:
            p=ROOT/relative;dest=output/'provenance'/relative
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            meta['source_sha256'][relative]=sha(p)
        sources,baseline_rows=prepare_sources(c)
        dump_json(output/'inputs.json',sources)
        tables=evaluate(c,baseline_rows)
        save_tables(output,tables)
        summary,grouped=aggregate(tables,c)
        for name,rows in grouped.items():write_csv(output/'summary'/f'{name}.csv',rows)
        dump_json(output/'summary.json',summary)
        from src.visualization.scale_boundary import make_report
        make_report(c,tables,grouped,summary,output)
        assert_sources_unchanged(sources)
        meta.update(status='complete',full_test_split=True,image_count=c['dataset']['expected_images'],
                    table_rows={k:len(v) for k,v in tables.items()})
        artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name not in ('run_metadata.json','manifest.json','verification.json')}
        dump_json(output/'manifest.json',artifacts)
    except Exception as e:
        meta.update(status='failed',error=f'{type(e).__name__}: {e}')
        raise
    finally:
        meta.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start)
        dump_json(output/'run_metadata.json',meta)
    return meta


def verify(output,full=True):
    meta=json.loads((output/'run_metadata.json').read_text())
    if meta['status']!='complete' or not meta['full_test_split']:raise ValueError('Run is not complete')
    manifest=json.loads((output/'manifest.json').read_text())
    for name,digest in manifest.items():
        if sha(output/name)!=digest:raise ValueError(f'Artifact SHA mismatch: {name}')
    sources=json.loads((output/'inputs.json').read_text())
    assert_sources_unchanged(sources)
    config=yaml.safe_load((output/'config.yaml').read_text())
    for relative,digest in meta['source_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError(f'评分/绘图源码已变化；请使用保存的源码复现: {relative}')
    comparisons=0
    if full:
        _,rows=prepare_sources(config)
        actual=evaluate(config,rows)
        saved={name:read_csv(output/'tables'/f'{name}.csv') for name in TABLES}
        comparisons=numeric_tables_equal(actual,saved)
        summary,grouped=aggregate(actual,config)
        summaries_close(summary,json.loads((output/'summary.json').read_text()))
        for name,data in grouped.items():numeric_tables_equal({name:data},{name:read_csv(output/'summary'/f'{name}.csv')})
    result=dict(all_passed=True,full_numeric_replay=full,numeric_values_checked=comparisons,
                artifact_count=len(manifest),numeric_rtol=1e-6,numeric_atol=1e-8,verified_at_utc=datetime.now(timezone.utc).isoformat())
    dump_json(output/'verification.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/part2/scale_boundary.yaml')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--mode',choices=('run','verify'),default='run')
    parser.add_argument('--hashes-only',action='store_true',help='verify时只核对来源与文件SHA，不作全量数值重放')
    args=parser.parse_args()
    c=load_config(args.config)
    output=(args.output_dir or ROOT/c['outputs']['root']).resolve()
    result=execute(c,args.config,output) if args.mode=='run' else verify(output,not args.hashes_only)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
