"""补充实验来源/状态/配对统计；不改变冻结B/C的源码集合或核验逻辑。"""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import numpy as np
import yaml
from pipelines.part3 import common as old

ROOT = old.ROOT


def config(direction, model, path=None):
    path = path or ROOT / f'configs/part3_studies/direction_{direction}.yaml'
    cfg = yaml.safe_load(Path(path).read_text())
    expected = 'part3_b_three_estimators_v1' if direction == 'b' else 'part3_c_classical_diagnostics_v1'
    if cfg['protocol'] != expected: raise ValueError('Wrong study protocol')
    base, mc = old.load_config(ROOT / cfg['base_config'], model, direction)
    methods = ['RAW','B_median_log','B_mean_log','B_absrel'] if direction == 'b' else [
        'RAW','C','C_no_RGB','C_no_depth_gate','Gaussian_log','RGB_guided_log']
    if cfg['methods'] != methods: raise ValueError('Required method coverage differs')
    if direction == 'c':
        if cfg['classical'] != dict(gaussian=dict(radius=2,sigma=1.5),guided=dict(radius=2,epsilon=.01)):
            raise ValueError('Classical controls are fixed before evaluation')
        if cfg['texture'] != dict(radius=3,maximum_gt_ratio=1.02,minimum_rgb_gradient=.05,
                                  maximum_control_gradient=.01,prediction_edge_ratio=1.05):
            raise ValueError('Texture protocol is fixed')
        # 严格复用2A门槛，不能通过修改评价阈值追逐收益。
        diagnostic=yaml.safe_load((ROOT/cfg['diagnostic_config']).read_text())
        if diagnostic['evaluation'] != dict(crop=[45,471,41,601],minimum_m=.001,maximum_m=10.,
                                             alignment='none',aggregation='per_image_macro_mean'):
            raise ValueError('Diagnostic protocol differs')
    return cfg,base,mc


def sources():
    files = dict(old.sources())
    for folder in ('src/part3_studies','pipelines/part3_studies','scripts/part3_studies','configs/part3_studies'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.sh','.yaml'):
                files[str(p.relative_to(ROOT))]=old.sha(p)
    for name in ('src/analysis/scale_boundary.py','configs/part2/scale_boundary.yaml'):
        files[name]=old.sha(ROOT/name)
    return files


def begin(output,kind,cfg,model,inputs,parents=(),**extra):
    output=Path(output).resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'outputs'):
        raise ValueError('Project output must be under outputs/')
    output.mkdir(parents=True,exist_ok=False)
    old.write_json(output/'status.json',dict(status='running',kind=kind))
    record=dict(kind=kind,config=cfg,model=model,input_files=inputs,
                parents=[str(Path(p).resolve()) for p in parents],source_sha256=sources(),
                created_at_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
                environment=old.environment(),formal_training_run=False,
                test_results_previously_inspected=True,**extra)
    old.write_json(output/'provenance/config.json',cfg)
    for name in record['source_sha256']:
        dest=output/'provenance/source'/name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    return record


def seal(output,record,full):
    if sources()!=record['source_sha256']:raise ValueError('Study source changed during run')
    for name,digest in record['input_files'].items():
        if old.sha(name)!=digest:raise ValueError('Input changed: '+name)
    record['formal_complete']=bool(full)
    record['artifacts']={str(p.relative_to(output)):old.sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p.name not in ('status.json','run_metadata.json','verification.json')}
    old.write_json(output/'run_metadata.json',record)
    old.write_json(output/'status.json',dict(status='complete' if full else 'smoke_complete',
                                           metadata_sha256=old.sha(output/'run_metadata.json')))


def checked(output,kind=None,full=True):
    output=Path(output).resolve();record=old.read_json(output/'run_metadata.json')
    if not record['kind'].startswith('study_'):return old.checked(output,kind,require_full=full)
    status=old.read_json(output/'status.json')
    if (status['status'] not in ('complete','smoke_complete')
        or status['metadata_sha256']!=old.sha(output/'run_metadata.json')):
        raise ValueError('Incomplete study or changed metadata')
    if kind and kind!=record['kind']:raise ValueError('Wrong study kind')
    if full and not record['formal_complete']:raise ValueError('Subset cannot provide formal parent')
    if sources()!=record['source_sha256']:raise ValueError('Study scientific source changed')
    for name,digest in record['artifacts'].items():
        if old.sha(output/name)!=digest:raise ValueError('Artifact changed: '+name)
    for name,digest in record['input_files'].items():
        if old.sha(name)!=digest:raise ValueError('Input changed: '+name)
    for parent in record['parents']:checked(parent)
    return record


def read_depth(path):
    with np.load(path,allow_pickle=False) as z:return old.valid_depth(z['depth_m']).copy()


def table_equal(expected,saved):
    if len(expected)!=len(saved):raise ValueError('Table length differs')
    for a,b in zip(expected,saved,strict=True):
        if set(a)!=set(b):raise ValueError('Table fields differ')
        for key,value in a.items():
            if value is None:
                if b[key] not in ('',None):raise ValueError('N/A became a value: '+key)
            elif isinstance(value,(int,float,np.number)) and not isinstance(value,bool):
                old.assert_close(value,float(b[key]),'Recomputed table/'+key)
            elif str(value)!=str(b[key]):raise ValueError('Table identity differs: '+key)


def describe(rows,field):
    # 先图内平均（尤其多个剖面），再图像等权；明确可测覆盖。
    images={}
    for r in rows:
        value=r.get(field)
        if value is not None and value!='':images.setdefault(r['sample_id'],[]).append(float(value))
    values=[np.mean(v) for v in images.values()]
    return dict(mean=float(np.mean(values)) if values else None,images=len(values),
                measurements=sum(len(v) for v in images.values()))


def paired(rows,methods,fields,resamples=10000,seed=42):
    raw={r['sample_id']:r for r in rows if r['method']=='RAW'}
    result={}
    for method in methods:
        if method=='RAW':continue
        current={r['sample_id']:r for r in rows if r['method']==method}
        if set(current)!=set(raw):raise ValueError('Unpaired method coverage')
        result[method]={}
        for field in fields:
            keys=[k for k in sorted(raw) if current[k].get(field) not in (None,'') and raw[k].get(field) not in (None,'')]
            if not keys:
                result[method][field]=dict(mean_delta=None,ci95=None,images=0,scenes=0);continue
            scenes=sorted({raw[k]['scene'] for k in keys})
            groups={s:i for i,s in enumerate(scenes)}
            ids=np.array([groups[raw[k]['scene']] for k in keys])
            delta=np.array([float(current[k][field])-float(raw[k][field]) for k in keys])
            sums=np.bincount(ids,weights=delta,minlength=len(scenes));counts=np.bincount(ids,minlength=len(scenes))
            draw=np.random.default_rng(seed).integers(0,len(scenes),(resamples,len(scenes)))
            interval=np.quantile(sums[draw].sum(axis=1)/counts[draw].sum(axis=1),[.025,.975])
            result[method][field]=dict(mean_delta=float(delta.mean()),ci95=interval.tolist(),images=len(keys),scenes=len(scenes))
    return result
