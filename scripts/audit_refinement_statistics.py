"""Paired NYUv2 refinement audit, resampling whole scenes rather than frames."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',default='docs/evidence/full_benchmark_20261029')
    p.add_argument('--output',default='docs/evidence/paper_comparison/refinement_scene_bootstrap.json')
    p.add_argument('--resamples',type=int,default=10000)
    a=p.parse_args()
    if a.resamples<1: p.error('Positive resample count required')
    root=Path(a.evidence)
    output={}; inputs={}
    for stem in ('depth_pro','unidepth_v2'):
        source=root/f'{stem}_C_cuda/per_image.csv'
        for path in (source,root/f'{stem}_nyuv2/metrics_per_image.csv'):
            inputs[path.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows=list(csv.DictReader(source.open(encoding='utf-8')))
        original={r['sample_id']:r for r in csv.DictReader((root/f'{stem}_nyuv2/metrics_per_image.csv').open(encoding='utf-8'))}
        paired={}
        for r in rows: paired.setdefault(r['sample_id'],{})[r['method']]=r
        if len(rows)!=1308 or len(paired)!=654 or set(paired)!=set(original): raise ValueError('654 paired images required')
        for pair in paired.values():
            if set(pair)!={'RAW','C_cuda'} or pair['RAW']['source_prediction_sha256']!=pair['C_cuda']['source_prediction_sha256']:
                raise ValueError('Mismatched refinement pair')
        keys=sorted(paired)
        scenes=sorted({original[k]['scene'] for k in keys})
        groups=np.array([scenes.index(original[k]['scene']) for k in keys])
        counts=np.bincount(groups,minlength=len(scenes))
        rng=np.random.default_rng(42)
        # A draw samples G whole scenes with replacement, retaining every image
        # in a drawn scene; ratio of total metric sums to total image counts.
        draws=rng.integers(0,len(scenes),(a.resamples,len(scenes)))
        sampled_counts=counts[draws].sum(axis=1)
        metrics={}
        for metric,maximize in [('abs_rel',False),('rmse',False),('delta1',True),('si_boundary_f1',True)]:
            raw=np.array([float(paired[k]['RAW'][metric]) for k in keys])
            new=np.array([float(paired[k]['C_cuda'][metric]) for k in keys])
            for k in keys:
                if abs(float(paired[k]['RAW'][metric])-float(original[k][metric]))>1e-10:
                    raise ValueError('Frozen baseline metrics mismatch')
            diff=new-raw
            totals=np.bincount(groups,weights=diff,minlength=len(scenes))
            samples=totals[draws].sum(axis=1)/sampled_counts
            lo,hi=np.quantile(samples,[.025,.975])
            metrics[metric]=dict(RAW_mean=float(raw.mean()),refined_mean=float(new.mean()),delta_refined_minus_RAW=float(diff.mean()),
                                paired_scene_bootstrap_ci95=[float(lo),float(hi)],
                                descriptive_image_win_fraction=float(np.mean(diff>0 if maximize else diff<0)),
                                CI_excludes_zero_in_favorable_direction=bool(lo>0 if maximize else hi<0))
        output[stem]=dict(images=len(keys),scenes=len(scenes),metrics=metrics)
    report=dict(seed=42,resamples=a.resamples,input_sha256=inputs,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),unit='NYUv2 scene, sampled with replacement; image-weighted ratio estimator within each draw',
                caveats=['Conditional on this frozen test set and protocol; not evidence of cross-domain superiority.','Scenes may still share acquisition/environment conditions.','Exploratory intervals without multiple-comparison adjustment; no test-set tuning performed.','Statistical detectability is separate from practical effect size.'],models=output)
    dest=Path(a.output); dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
