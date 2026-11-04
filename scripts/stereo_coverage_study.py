"""Development-only coverage/MAE selection, followed by frozen held-out scoring."""
import argparse,itertools,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,yaml
from src.stereo.io import load_middlebury_scene
from src.stereo.matching import StereoMatcherConfig,match_stereo
from src.io import write_csv,write_json,sha256

def score(z,gt,mask):
    valid=mask&np.isfinite(gt)&(gt>0); pred=valid&np.isfinite(z)&(z>0)
    return dict(mae=float(np.abs(z[pred]-gt[pred]).mean()) if pred.any() else None,
                coverage=float(pred.sum()/valid.sum()) if valid.any() else None)

def frontier(rows):
    return [r for r in rows if not any(q['mae']<=r['mae'] and q['coverage']>=r['coverage']
        and (q['mae']<r['mae'] or q['coverage']>r['coverage']) for q in rows)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    config=yaml.safe_load(Path('configs/part1/middlebury_sgbm.yaml').read_text(encoding='utf-8'))
    dev=config['split']['development_scenes'];test=config['split']['evaluation_scenes']
    assert len(dev)==5 and len(test)==10 and not set(dev)&set(test)
    root=Path(config['dataset']['root']); scenes={n:load_middlebury_scene(root/n) for n in dev}
    def predict(scene,method,params):
        return match_stereo(scene.left_rgb,scene.right_rgb,calibration_num_disparities=scene.calibration.num_disparities,
            config=StereoMatcherConfig(method=method,**{k:v for k,v in params.items() if k!='lr_threshold_px'}),lr_threshold_px=1.).left_disparity_px
    candidates=[];selected=[]
    for method in ('bm','sgbm'):
        t=config['tuning'][method]; keys=list(t['search']); fixed=config['experiments'][method+'_fixed']['parameters']
        params=[fixed]+[{**t['fixed'],**dict(zip(keys,v))} for v in itertools.product(*(t['search'][k] for k in keys))]
        rows=[]
        for i,pr in enumerate(params):
            ms=[score(predict(s,method,pr),s.gt_disparity_px,np.ones(s.gt_disparity_px.shape,bool)) for s in scenes.values()]
            if any(m['mae'] is None for m in ms):continue
            rows.append(dict(method=method,candidate=i,mae=float(np.mean([m['mae'] for m in ms])),coverage=float(np.mean([m['coverage'] for m in ms])),parameters=pr))
            if i%20==0:print(method,i,len(params),flush=True)
        base=rows[0]; good=[r for r in rows if abs(r['coverage']-base['coverage'])<=.01]
        same_error=[r for r in rows if r['mae']<=base['mae']]
        choices={'fixed':base,'matched_coverage':min(good,key=lambda r:(r['mae'],-r['coverage'])),
                 'matched_error':max(same_error,key=lambda r:(r['coverage'],-r['mae']))}
        for label,r in choices.items():selected.append(dict(selection=label,**r))
        pareto=frontier(rows)
        candidates.extend(rows)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots();ax.scatter([r['coverage'] for r in rows],[r['mae'] for r in rows],s=10,label='Development candidates')
        pf=sorted(pareto,key=lambda r:r['coverage']);ax.plot([r['coverage'] for r in pf],[r['mae'] for r in pf],label='Pareto frontier')
        for label,r in choices.items():ax.scatter(r['coverage'],r['mae'],label=label)
        ax.set(xlabel='Valid prediction fraction',ylabel='MAE on valid predictions (px)',title=method+' / 5 development scenes');ax.legend();fig.tight_layout();fig.savefig(out/(method+'_pareto.png'));plt.close(fig)
    # Freeze decisions before loading any held-out images or GT.
    write_json(out/'selection.json',dict(development=dev,held_out=test,coverage_tolerance=.01,
        rule='min MAE within +/-1 percentage point of fixed coverage; max coverage with MAE <= fixed; all selected only on development',selected=selected))
    write_csv(out/'development.csv',[{**r,'parameters':json.dumps(r['parameters'],sort_keys=True)} for r in candidates])
    rows=[]
    for name in test:
        s=load_middlebury_scene(root/name);gt=s.gt_disparity_px
        gray=cv2.cvtColor(s.left_rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
        texture=cv2.boxFilter(gray*gray,-1,(7,7))-cv2.boxFilter(gray,-1,(7,7))**2
        valid=np.isfinite(gt)&(gt>0); safe=np.where(valid,gt,0)
        edge=(np.abs(cv2.Sobel(safe,cv2.CV_32F,1,0))+np.abs(cv2.Sobel(safe,cv2.CV_32F,0,1)))>4
        regions={'all_GT':np.ones(gt.shape,bool),'non_occluded':s.non_occluded_mask,
                 'occluded':s.occluded_mask,'low_texture':texture<25,'depth_boundary':edge}
        for r in selected:
            z=predict(s,r['method'],r['parameters'])
            for region,mask in regions.items():rows.append(dict(scene=name,method=r['method'],selection=r['selection'],region=region,**score(z,gt,mask)))
    write_csv(out/'held_out.csv',rows)
    means=[]
    for method in ('bm','sgbm'):
        for label in ('fixed','matched_coverage','matched_error'):
            for region in regions:
                rs=[r for r in rows if r['method']==method and r['selection']==label and r['region']==region and r['mae'] is not None]
                means.append(dict(method=method,selection=label,region=region,scenes=len(rs),mae=float(np.mean([r['mae'] for r in rs])),coverage=float(np.mean([r['coverage'] for r in rs]))))
    write_json(out/'summary.json',dict(status='complete',means=means,selection_sha256=sha256(out/'selection.json'),
        source_sha256=sha256(__file__),limitation='MAE is conditional on valid predictions; coverage matching is development-only and not guaranteed on held-out scenes; regional masks may overlap.'))

if __name__=='__main__':main()
