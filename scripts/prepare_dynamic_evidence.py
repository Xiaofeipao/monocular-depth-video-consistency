"""Export small, source-bound dynamic-clip evidence; no model execution."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    summary=json.loads((run/'summary.json').read_text());rows=list(csv.DictReader((run/'per_pair.csv').open(encoding='utf-8')))
    assert len(rows)==7104
    for name,backends in summary['scores'].items():
        for backend,regions in backends.items():
            for region,methods in regions.items():
                for method,score in methods.items():
                    rs=[float(r['normalized_warp_mae']) for r in rows if r['sequence']==name and r['backend']==backend and r['region']==region and r['method']==method and r['normalized_warp_mae']!='']
                    assert len(rs)==score['evaluable_pairs'] and np.isclose(np.mean(rs),score['normalized_warp_mae'])
    for name in ['protocol.json','models.json','inputs.json','summary.json','per_pair.csv','silhouette.csv','timing.csv','FILES.sha256.json']:
        shutil.copyfile(run/name,out/name)
    for name in ('bmx-trees','tennis'):
        (out/name).mkdir()
        for filename in ['normalization.json','display.json']:shutil.copyfile(run/name/filename,out/name/filename)
    hashes={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out/'export_checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print('Exported small evidence; all 96 aggregate groups independently recomputed from 7104 pair/region/method rows')

if __name__=='__main__':main()
