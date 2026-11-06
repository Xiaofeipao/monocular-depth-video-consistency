"""B开发/测试隔离、C同公式与组件移除、来源和原始预测重算。"""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.part3 import common as c, direction_b as b, direction_c as d
from src.refinement.edge_aware import refine_edges
from src.refinement.ablation import cpu_ablation, make_refiner
from src.refinement.global_scale import image_log_ratio, fit_global_scale, apply_global_scale


def test_c_exact_cpu_port_and_holes():
    source = c.ROOT / 'github_repository/monocular-depth-video-consistency/src/exploration/edge_aware.py'
    port=c.ROOT/'src/refinement/edge_aware.py'
    origin=c.read_json(c.ROOT/'src/refinement/ORIGIN.json')
    assert hashlib.sha256(port.read_bytes()).hexdigest()==origin['files']['src/exploration/edge_aware.py']['lf_sha256']
    namespace = {}
    if source.exists(): exec(compile(source.read_text(), str(source), 'exec'), namespace)
    rng = np.random.default_rng(7)
    z = rng.uniform(1,4,(7,9)).astype(np.float32); z[1,2] = np.nan
    image = rng.integers(0,256,(7,9,3),dtype=np.uint8)
    expected = refine_edges(z,image)
    if namespace: np.testing.assert_array_equal(expected,namespace['refine_edges'](z,image))
    np.testing.assert_array_equal(cpu_ablation(z,image),expected)
    for method in ('C','C_no_RGB','C_no_depth_gate'):
        out = make_refiner(method,'cpu')(z,image)
        np.testing.assert_array_equal(np.isfinite(out), np.isfinite(z))
        assert (out[np.isfinite(out)] > 0).all()


def test_scale_formula_and_invalid_gt():
    gt = np.array([[.001,10.,np.nan,4.,0]],np.float32)
    pred = np.array([[1.,1.,1.,2.,1.]],np.float32)
    ratio = image_log_ratio(pred,gt)
    assert ratio == float(np.log(np.float32(2)))
    logs = np.full(795,ratio); logs[:200] = 9
    scale = fit_global_scale(logs)
    assert scale == float(np.exp(np.median(logs)))
    np.testing.assert_allclose(apply_global_scale(pred,scale), pred*scale)
    with pytest.raises(ValueError): image_log_ratio(pred, np.zeros_like(gt))


@pytest.mark.parametrize('ratios', [[], [0.]*794, [0.]*796, [float('nan')]*795])
def test_subset_or_nonfinite_scale_fit_rejected(ratios):
    with pytest.raises(ValueError): fit_global_scale(ratios)


def test_configuration_and_bootstrap_pairing():
    config, _ = c.load_config(c.ROOT/'configs/part3/direction_c.yaml','depth_pro','c')
    assert config['parameters']['radius']==2
    rows=[]
    for i in range(4):
        for method, value in [('RAW',.2),('C',.1)]:
            rows.append(dict(sample_id=str(i),scene='shared' if i<3 else 'other',method=method,
                             abs_rel=value,rmse=value,delta1=1-value,si_boundary_f1=1-value))
    stats = d.scene_statistics(rows,100,42)
    assert stats['scenes']==2 and stats['images']==4
    assert stats['metrics']['abs_rel']['ci95'][1]<0
    assert d.choose_cases(rows)[0]['sample_id']=='0'


@pytest.fixture
def mini(tmp_path, monkeypatch):
    # 只缩小测试输入；生产配置、科学协议和真实运行不改变。
    def valid(z):
        assert z.dtype==np.float32 and z.shape==(4,8)
        assert np.isfinite(z).all() and (z>0).all()
        return z
    monkeypatch.setattr(c,'valid_depth',valid)
    monkeypatch.setattr(c,'CROP',np.s_[:,:])
    monkeypatch.setattr(c,'resources',lambda *args:{})
    class Dataset:
        def __init__(self, development):self.development=development
        def __len__(self):return 795 if self.development else 654
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get_test_sample(self,i):
            return SimpleNamespace(test_position=i,matlab_index_1based=i+(2000 if self.development else 1),
                                   rgb=np.zeros((4,8,3),np.uint8),depth_m=np.full((4,8),3.,np.float32),scene='scene_'+str(i%2))
    monkeypatch.setattr(c,'dataset',lambda mc,development=False:Dataset(development))
    def score(z,target,boundary=True):
        error=float(np.mean(abs(z-target)/target))
        result={k:error for k in c.METRICS[:-1]};result['valid_pixels']=32
        if boundary:result['si_boundary_f1']=.5
        return result
    monkeypatch.setattr(c,'score',score)
    source=tmp_path/'baseline';(source/'predictions').mkdir(parents=True)
    baseline={}
    for i in range(2):
        sample=Dataset(False).get_test_sample(i);key=c.stem(sample)
        z=np.full((4,8),2.,np.float32)
        np.savez_compressed(source/'predictions'/(key+'.npz'),depth_m=z)
        baseline[key]=dict(sample_id=key,**score(z,sample.depth_m))
    c.write_csv(source/'metrics_per_image.csv',list(baseline.values()))
    files={str(p):c.sha(p) for p in source.rglob('*') if p.is_file()}
    monkeypatch.setattr(c,'baseline_files',lambda *args:(source,baseline,dict(files)))
    class Adapter:
        def infer(self,rgb):return np.full((4,8),2.,np.float32),500.,.01,123
        def environment(self):return dict(device='synthetic')
    monkeypatch.setattr(b,'adapter',lambda *args:Adapter())
    # 生产总是480×640；合成测试保持小网格。
    monkeypatch.setattr(b.cv2,'resize',lambda z,*args,**kwargs:z.copy())
    monkeypatch.setattr(d,'save_cases',lambda path,rows,*args:c.write_json(path/'case_selection.json',d.choose_cases(rows)))
    return tmp_path,source


def test_b_complete_development_freeze_and_partial_eval_replay(mini):
    tmp, source=mini
    config,mc=c.load_config(c.ROOT/'configs/part3/direction_b.yaml','depth_pro','b')
    config['models']['depth_pro']['baseline']=str(source)
    dev=tmp/'dev';frozen=tmp/'frozen';out=tmp/'eval'
    b.predict(dev,config,mc,'depth_pro')
    b.fit(frozen,dev,config,mc,'depth_pro')
    assert b.verify_fit(frozen,mc)[1] == pytest.approx(1.5,rel=1e-6)
    b.evaluate(out,frozen,config,mc,'depth_pro',limit=2)
    report=b.verify(out,mc)
    assert report['passed'] and report['samples']==2 and not report['formal_complete']
    assert not c.read_json(out/'run_metadata.json')['native_metric']
    with pytest.raises(FileExistsError):b.fit(frozen,dev,config,mc,'depth_pro')


def test_b_smoke_cannot_freeze_and_development_tamper_rejected(mini):
    tmp,_=mini;config,mc=c.load_config(c.ROOT/'configs/part3/direction_b.yaml','depth_pro','b')
    dev=tmp/'dev';b.predict(dev,config,mc,'depth_pro',limit=2)
    assert b.verify(dev,mc)['prediction_files_checked']==2
    with pytest.raises(ValueError,match='Smoke/subset'):b.fit(tmp/'freeze',dev,config,mc,'depth_pro')
    table=dev/'predictions.csv';table.write_text(table.read_text().replace('train_0000','train_9999'))
    with pytest.raises(ValueError,match='Artifact changed'):b.verify(dev,mc)


def test_c_full_method_and_two_ablations_raw_replay(mini):
    tmp, source=mini;config,mc=c.load_config(c.ROOT/'configs/part3/direction_c.yaml','depth_pro','c')
    config['models']['depth_pro']['baseline']=str(source)
    output=tmp/'c';d.run(output,config,mc,'depth_pro','cpu',limit=2)
    report=d.verify(output,mc)
    assert report['prediction_files_checked']==6 and not report['formal_complete']
    with pytest.raises(ValueError,match='Smoke/subset'):c.checked(output)
    # 外层SHA重封装仍不能绕过预测重算。
    p=output/'predictions/C/test_0000_mat_0001.npz'
    np.savez_compressed(p,depth_m=np.full((4,8),4.,np.float32))
    record=c.read_json(output/'run_metadata.json');record['artifacts'][str(p.relative_to(output))]=c.sha(p)
    c.write_json(output/'run_metadata.json',record)
    status=c.read_json(output/'status.json');status['metadata_sha256']=c.sha(output/'run_metadata.json');c.write_json(output/'status.json',status)
    with pytest.raises(AssertionError):d.verify(output,mc)


def test_cuda_cpu_and_component_parity_after_cuda_check():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA not available')
    rng=np.random.default_rng(4)
    z=rng.uniform(1,4,(7,9)).astype(np.float32)
    rgb=rng.integers(0,256,(7,9,3),dtype=np.uint8)
    for method in ('C','C_no_RGB','C_no_depth_gate'):
        np.testing.assert_allclose(make_refiner(method,'cuda')(z,rgb),make_refiner(method,'cpu')(z,rgb),rtol=1e-6,atol=1e-8)
