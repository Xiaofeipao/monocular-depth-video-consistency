"""Compare two saved per-image metric runs; does not infer prediction causes."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    def load(name):
        rows=list(csv.DictReader(Path(name).open(encoding='utf-8-sig')))
        indexed={r['sample_id']:r for r in rows}
        if len(indexed)!=654 or len(rows)!=654:raise ValueError('Need 654 unique test images')
        return indexed
    left,right=load(a.reference),load(a.candidate)
    if left.keys()!=right.keys():raise ValueError('Test image identities differ')
    result=[]
    for key in left:
        for metric in ('abs_rel','rmse','delta1','si_boundary_f1'):
            x,y=float(left[key][metric]),float(right[key][metric])
            if not np.isfinite([x,y]).all():raise ValueError('Non-finite score')
            result.append(dict(sample_id=key,metric=metric,reference=x,candidate=y,difference=y-x))
    with (out/'per_image_difference.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    summary={m:dict(mean_difference=float(np.mean([r['difference'] for r in result if r['metric']==m])),max_absolute_difference=float(max(abs(r['difference']) for r in result if r['metric']==m))) for m in ('abs_rel','rmse','delta1','si_boundary_f1')}
    hashes={role:hashlib.sha256(Path(path).read_bytes()).hexdigest() for role,path in [('reference',a.reference),('candidate',a.candidate)]}
    (out/'summary.json').write_text(json.dumps(dict(samples=654,metrics=summary,input_sha256=hashes,
        limitation='Metric-only audit; agreement does not establish identical prediction arrays. Without both raw runs, GPU/library/interpolation causes remain unresolved.'),indent=2),encoding='utf-8')

if __name__=='__main__':main()
