"""Read-only replay of saved dynamic-clip flow scores; does not run any model."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--data',default='data/course_provided/dynamic_clips');a=p.parse_args(argv)
    root=Path(a.run);data=Path(a.data);cv2.setNumThreads(1)
    hashes=json.loads((root/'FILES.sha256.json').read_text())
    for rel,h in hashes.items():
        f=(root/rel).resolve()
        if not f.is_relative_to(root.resolve()):raise ValueError('Path outside run')
        assert hashlib.sha256(f.read_bytes()).hexdigest()==h,rel
    inputs=json.loads((root/'inputs.json').read_text())
    for name,rs in inputs.items():
        for r in rs:
            assert hashlib.sha256((data/name/r['rgb']).read_bytes()).hexdigest()==r['rgb_sha256']
            assert hashlib.sha256((data/(name+'_mask')/r['mask']).read_bytes()).hexdigest()==r['mask_sha256']
    rows=list(csv.DictReader((root/'per_pair.csv').open()));checked=0
    assert len(rows)==(79+69)*2*4*6
    group={}
    for row in rows:group.setdefault((row['sequence'],row['backend'],int(row['current'])),[]).append(row)
    for (name,backend,i),items in group.items():
        with np.load(root/name/('flow_'+backend)/f'{i:05d}.npz') as f:
            back=f['backward'];common=f['common_mask'];forward=f['forward']
        h,w=common.shape;y,x=np.indices((h,w),dtype=np.float32);mx=x+back[...,0];my=y+back[...,1]
        def remap(z):return cv2.remap(z.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        sampled=remap(forward);err=np.linalg.norm(back+sampled,axis=-1)
        assert np.all(err[common]<=1.00001)
        current=cv2.imread(str(data/(name+'_mask')/f'{i:05d}.png'),0)>0
        previous=remap(cv2.imread(str(data/(name+'_mask')/f'{i-1:05d}.png'),0)>0)
        b=current.astype(np.uint8);kernel=np.ones((5,5),np.uint8)
        regions={'all':np.ones((h,w),bool),'foreground':current&(previous>.999),'background':~current&(previous<.001),'boundary':cv2.dilate(b,kernel)!=cv2.erode(b,kernel)}
        spans=json.loads((root/name/'normalization.json').read_text())['spans']
        for row in items:
            m=row['method'];mask=regions[row['region']]&common;n=int(mask.sum());assert n==int(row['valid_pixels'])
            if n<8:assert row['normalized_warp_mae']=='';continue
            z=np.load(root/name/m/f'{i:05d}.npy');prev=remap(np.load(root/name/m/f'{i-1:05d}.npy'))
            value=float(np.mean(abs(z[mask]-prev[mask]))/spans[m])
            assert np.isclose(value,float(row['normalized_warp_mae']),rtol=1e-6,atol=1e-8),row
            checked+=1
    print(json.dumps(dict(status='passed',hashed_files=len(hashes),input_frames=150,replayed_scores=checked,models_executed=False)))

if __name__=='__main__':main()
