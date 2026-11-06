"""补充实验的科学目标、经典公式、诊断失效与来源保护回归。"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import cv2
from src.part3_studies.estimators import weighted_median, fit_rules, calibration_values
from src.part3_studies.classical import guided_log, apply
from src.part3_studies.diagnostics import texture_masks, prepare, analyze
from pipelines.part3_studies import common as s, direction_b as b, direction_c as d
from pipelines.part3 import common as c


def test_absrel_weighted_median_minimizes_image_macro_objective():
    # 两图像素数不同；每图先平均。像素池化会给出不同结果。
    ratios=[np.array([.5]),np.array([2.,2.,2.,2.])]
    values=np.concatenate([1/a for a in ratios])
    weights=np.concatenate([a/len(a) for a in ratios])
    scale=weighted_median(values,weights)
    assert scale==.5
    objective=lambda x:np.mean([np.mean(abs(x*a-1)) for a in ratios])
    for x in np.linspace(.1,3,100):assert objective(scale)<=objective(x)+1e-12
    assert weighted_median(np.array([1.,2.]),np.ones(2))==1.
    with pytest.raises(ValueError):weighted_median([1.,2.],[1.,0.])


def test_log_mean_and_median_are_distinct_and_full_development_required():
    rows=[dict(median_log_ratio=x,mean_log_ratio=x) for x in [0.,0.,np.log(8)]]
    rules=fit_rules(rows,np.array([1.,2.,3.]),np.ones(3),expected_images=3)
    assert rules['B_median_log']==1
    assert rules['B_mean_log']==pytest.approx(2)
    with pytest.raises(ValueError):fit_rules(rows,np.ones(3),np.ones(3))
    gt=np.array([[.001,10.,2.,np.nan]],np.float32)
    p,g=calibration_values(np.ones((1,4)),gt)
    assert p.tolist()==[1.] and g.tolist()==[2.]


def test_classical_constant_and_constant_guidance_closed_form():
    depth=np.full((9,11),3.,np.float32);rgb=np.full((9,11,3),127,np.uint8)
    cfg=dict(gaussian=dict(radius=2,sigma=1.5),guided=dict(radius=2,epsilon=.01))
    for method in ('Gaussian_log','RGB_guided_log'):
        np.testing.assert_allclose(apply(method,depth,rgb,cfg),depth,rtol=1e-6)
    z=np.linspace(.5,6,99).reshape(9,11).astype(np.float32)
    expected=np.exp(cv2.boxFilter(cv2.boxFilter(np.log(z.astype(np.float64)),-1,(5,5),
        borderType=cv2.BORDER_REPLICATE),-1,(5,5),borderType=cv2.BORDER_REPLICATE))
    np.testing.assert_allclose(guided_log(z,rgb),expected,rtol=1e-6)
    with pytest.raises(ValueError):guided_log(z*0,rgb)


def test_colored_guided_matches_slow_window_definition():
    # 独立逐像素窗口回归，避免只测同一向量化实现。
    rng=np.random.default_rng(8);rgb=rng.integers(0,256,(5,6,3),dtype=np.uint8)
    z=rng.uniform(1,4,(5,6)).astype(np.float32);I=rgb.astype(float)/255;p=np.log(z)
    padI=np.pad(I,((1,1),(1,1),(0,0)),mode='edge');padp=np.pad(p,1,mode='edge')
    a=np.zeros_like(I);bias=np.zeros_like(p)
    for y in range(5):
        for x in range(6):
            block=padI[y:y+3,x:x+3].reshape(-1,3);target=padp[y:y+3,x:x+3].ravel()
            mi=block.mean(0);mp=target.mean()
            cov=(block-mi).T@(block-mi)/9+.02*np.eye(3)
            a[y,x]=np.linalg.solve(cov,((block-mi)*(target-mp)[:,None]).mean(0))
            bias[y,x]=mp-a[y,x]@mi
    pa=np.pad(a,((1,1),(1,1),(0,0)),mode='edge');pb=np.pad(bias,1,mode='edge')
    expected=np.zeros_like(p)
    for y in range(5):
        for x in range(6):expected[y,x]=np.exp(pa[y:y+3,x:x+3].mean((0,1))@I[y,x]+pb[y:y+3,x:x+3].mean())
    np.testing.assert_allclose(guided_log(z,rgb,1,.02),expected,rtol=1e-6)


def test_texture_masks_have_fixed_gt_support_and_empty_is_na():
    cfg,_,_=s.config('c','depth_pro')
    dc=__import__('yaml').safe_load((c.ROOT/cfg['diagnostic_config']).read_text())
    gt=np.full((41,61),2.,np.float32)
    rgb=np.zeros((41,61,3),np.uint8);rgb[:,::4]=255;rgb[:,1::4]=255
    valid=np.ones_like(gt,bool);edges=np.zeros_like(valid)
    masks=texture_masks(gt,rgb,valid,edges,cfg['texture'])
    assert masks['textured_flat'].sum()>0
    assert not np.any(masks['textured_flat']&masks['untextured_flat'])
    prep=prepare(gt,np.zeros_like(rgb),np.zeros_like(gt),np.zeros_like(gt),dc,cfg['texture'])
    extra,tables=analyze(gt,prep,dc,cfg['texture'])
    assert extra['texture_spurious_fraction'] is None
    assert tables['texture'][0]['mask_pixels']==0
    assert extra['selected_profiles']==0 and extra['measurable_profiles']==0
    prep=prepare(gt,rgb,np.zeros_like(gt),np.zeros_like(gt),dc,cfg['texture'])
    clean,_=analyze(gt,prep,dc,cfg['texture'])
    corr=gt.copy();corr[:,1::4]*=1.2;corr[:,2::4]*=1.2
    dirty,_=analyze(corr,prep,dc,cfg['texture'])
    assert dirty['texture_spurious_fraction']>clean['texture_spurious_fraction']


def test_profile_common_support_keeps_failure_counts():
    rows=[]
    for method,status,width in [('RAW','measurable',4.),('C','weak_contrast',None)]:
        rows.append(dict(sample_id='one',scene='s',profile_id=0,method=method,status=status,
            width_10_90_px=width,width_excess_px=width,absolute_midpoint_error_px=width,contrast_ratio_to_gt=.5))
    report=d.profile_summary(rows,'C')
    assert report['status_counts']=={'weak_contrast':1}
    assert report['jointly_measurable_with_RAW']['profiles']==0
    assert report['metrics']['width_10_90_px']['mean'] is None


def test_new_sources_do_not_change_frozen_source_inventory():
    path=c.ROOT/'outputs/part3/direction_c/depth_pro/evaluation_v1/run_metadata.json'
    if path.exists():assert c.sources()==c.read_json(path)['source_sha256']
    assert any(k.startswith('src/part3_studies/') for k in s.sources())
    assert not any(k.startswith('src/part3_studies/') for k in c.sources())


def test_fixed_gt_profile_detects_blur_and_preserves_shared_coordinates():
    cfg,_,_=s.config('c','depth_pro')
    dc=__import__('yaml').safe_load((c.ROOT/cfg['diagnostic_config']).read_text())
    gt=np.ones((41,81),np.float32);gt[:,41:]=2
    labels=np.ones_like(gt);labels[:,41:]=2
    prep=prepare(gt,np.zeros((*gt.shape,3),np.uint8),labels,np.ones_like(gt),dc,cfg['texture'])
    assert prep['profiles']
    _,raw=analyze(gt,prep,dc,cfg['texture'])
    blurred=cv2.GaussianBlur(gt,(13,13),2,borderType=cv2.BORDER_REPLICATE)
    _,smooth=analyze(blurred,prep,dc,cfg['texture'])
    assert raw['profiles'][0]['status']==smooth['profiles'][0]['status']=='measurable'
    assert smooth['profiles'][0]['width_10_90_px']>raw['profiles'][0]['width_10_90_px']
    assert (raw['profiles'][0]['x'],raw['profiles'][0]['y'])==(smooth['profiles'][0]['x'],smooth['profiles'][0]['y'])


def test_fit_all_795_synthetic_predictions_and_refit(mini):
    import hashlib
    tmp,source,Dataset=mini
    cfg,base,mc=s.config('b','depth_pro');cfg['development_root']=str(tmp/'b')
    base['models']['depth_pro']['baseline']=str(source)
    development=Path(cfg['development_root'])/'depth_pro/development_v1'
    rec=c.begin(development,'b_development',base,'depth_pro',{})
    (development/'predictions').mkdir()
    rows=[]
    for i in range(795):
        sample=Dataset(True).get_test_sample(i);key=c.development_stem(sample)
        path=development/'predictions'/(key+'.npz')
        np.savez_compressed(path,depth_m=np.full((41,61),2.,np.float32))
        rows.append(dict(sample_id=key,position=i,matlab_index=sample.matlab_index_1based,
            rgb_sha256=hashlib.sha256(sample.rgb.tobytes()).hexdigest(),prediction_sha256=c.sha(path)))
    c.write_csv(development/'predictions.csv',rows);c.seal(development,rec,formal_complete=True)
    ref=development.parent/'frozen_scale_v1'
    rec=c.begin(ref,'b_frozen_scale',base,'depth_pro',{},parents=[development])
    c.write_json(ref/'frozen_scale.json',dict(scale=1.5));c.seal(ref,rec,formal_complete=True)
    output=tmp/'new_fit';b.fit(output,cfg,base,mc,'depth_pro')
    report=b.verify(output,cfg,base,mc,'depth_pro')
    assert report['all_three_scales_refitted'] and report['formal_complete']
    assert len(c.read_csv(output/'development_statistics.csv'))==795
    for scale in c.read_json(output/'frozen_scales.json')['scales'].values():assert scale==pytest.approx(1.5,rel=1e-6)
    assert c.read_json(output/'run_metadata.json')['test_images_used_for_fitting']==0


@pytest.fixture
def mini(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'CROP',np.s_[:,:])
    def valid(z):
        assert z.dtype==np.float32 and z.shape==(41,61)
        assert np.isfinite(z).all() and (z>0).all();return z
    monkeypatch.setattr(c,'valid_depth',valid)
    class Dataset:
        def __init__(self,dev):
            self.dev=dev
            self.file={k:np.ones((654,61,41),np.uint8) for k in ('labels','instances')}
        def __len__(self):return 795 if self.dev else 654
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def get_test_sample(self,i):
            return SimpleNamespace(test_position=i,matlab_index_1based=i+(2000 if self.dev else 1),
                rgb=np.zeros((41,61,3),np.uint8),depth_m=np.full((41,61),3.,np.float32),scene='s'+str(i%2))
    monkeypatch.setattr(c,'dataset',lambda mc,development=False:Dataset(development))
    monkeypatch.setattr(c,'resources',lambda *a:{})
    def score(z,gt,boundary=True):
        value=float(np.mean(abs(z-gt)/gt));r={k:value for k in c.METRICS[:-1]};r['valid_pixels']=z.size
        if boundary:r['si_boundary_f1']=.5
        return r
    monkeypatch.setattr(c,'score',score)
    source=tmp_path/'baseline';(source/'predictions').mkdir(parents=True)
    baseline={}
    for i in range(2):
        sample=Dataset(False).get_test_sample(i);key=c.stem(sample)
        z=np.full((41,61),2.,np.float32);np.savez_compressed(source/'predictions'/(key+'.npz'),depth_m=z)
        baseline[key]=dict(sample_id=key,**score(z,sample.depth_m))
    c.write_csv(source/'metrics_per_image.csv',list(baseline.values()))
    monkeypatch.setattr(c,'baseline_files',lambda *a:(source,baseline,{}))
    return tmp_path,source,Dataset


def test_b_saved_arrays_gt_replay_and_no_overwrite(mini):
    tmp,source,_=mini;cfg,base,mc=s.config('b','depth_pro');base['models']['depth_pro']['baseline']=str(source)
    frozen=tmp/'fit';record=s.begin(frozen,'study_b_fit',cfg,'depth_pro',{},development_images=795)
    c.write_json(frozen/'frozen_scales.json',dict(scales={m:1.5 for m in b.METHODS}))
    s.seal(frozen,record,True)
    out=tmp/'eval';b.evaluate(out,frozen,cfg,base,mc,'depth_pro',2)
    assert b.verify(out,cfg,base,mc,'depth_pro')['passed']
    with pytest.raises(FileExistsError):b.evaluate(out,frozen,cfg,base,mc,'depth_pro',2)
    file=out/'predictions/B_absrel/test_0000_mat_0001.npz'
    np.savez_compressed(file,depth_m=np.full((41,61),5.,np.float32))
    with pytest.raises(ValueError,match='Artifact changed'):b.verify(out,cfg,base,mc,'depth_pro')


def test_c_complete_six_method_tables_replay_and_smoke_gate(mini,monkeypatch):
    tmp,source,Dataset=mini;cfg,base,mc=s.config('c','depth_pro');base['models']['depth_pro']['baseline']=str(source)
    base['output_root']=str(tmp/'reference');ref=Path(base['output_root'])/'depth_pro'/cfg['reference_name'];ref.mkdir(parents=True)
    rows=[]
    for i in range(2):
        sample=Dataset(False).get_test_sample(i);z,digest=c.native(source,sample)
        for method in cfg['methods'][:4]:
            if method!='RAW':
                p=ref/'predictions'/method/(c.stem(sample)+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(p,depth_m=z)
            rows.append(dict(sample_id=c.stem(sample),scene=sample.scene,method=method,
                source_prediction_sha256=digest,postprocess_seconds=0,peak_gpu_memory_bytes='',**c.score(z,sample.depth_m)))
    c.write_csv(ref/'per_image.csv',rows);c.write_json(ref/'run_metadata.json',dict(backend='cpu'))
    parent=dict(model='depth_pro',config=base,backend='cpu')
    monkeypatch.setattr(d,'reference',lambda *a:(ref,parent))
    real_checked=s.checked
    # 合成父目录不冒充正式来源；只在测试夹具跳过它的来源封装。
    monkeypatch.setattr(s,'checked',lambda path,*a,**kw:parent if Path(path)==ref else real_checked(path,*a,**kw))
    monkeypatch.setattr(d,'plot_case',lambda *a:None)
    out=tmp/'out';d.run(out,cfg,base,mc,'depth_pro',2)
    assert len(c.read_csv(out/'per_image.csv'))==12
    assert d.verify(out,cfg,base,mc,'depth_pro')['passed']
    with pytest.raises(ValueError,match='Subset'):s.checked(out)
    # 即使重新封装SHA，重算仍可发现错误的经典预测。
    file=out/'predictions/Gaussian_log/test_0000_mat_0001.npz'
    np.savez_compressed(file,depth_m=np.full((41,61),4.,np.float32))
    record=c.read_json(out/'run_metadata.json');record['artifacts'][str(file.relative_to(out))]=c.sha(file)
    c.write_json(out/'run_metadata.json',record)
    status=c.read_json(out/'status.json');status['metadata_sha256']=c.sha(out/'run_metadata.json');c.write_json(out/'status.json',status)
    with pytest.raises(AssertionError):d.verify(out,cfg,base,mc,'depth_pro')
