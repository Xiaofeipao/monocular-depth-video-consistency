"""Known scale/edge geometry and independent analysis lifecycle, without models."""
from copy import deepcopy
from pathlib import Path
import json

import h5py
import numpy as np
import pytest
import yaml
from scipy.io import savemat
from scipy.ndimage import gaussian_filter1d

from src.analysis.scale_boundary import (annotation_edges,edge_comparison,depth_edge_map,
    scale_statistics,error_statistics,region_masks,profile_shape,choose_profiles,measure_profile)
from pipelines.part2 import analyze_scale_boundary as pipeline


def config():
    return yaml.safe_load((pipeline.ROOT/'configs/part2/scale_boundary.yaml').read_text())


def test_unclipped_scale_and_oracle_do_not_inherit_scoring_clamp():
    gt=np.full((8,10),2,np.float32);p=gt*12;valid=np.ones(gt.shape,bool)
    row,tracks=scale_statistics(p,gt,valid)
    assert row['raw_scale_ratio']==12
    assert row['clipped_scale_ratio']==5
    assert row['raw_above_maximum_fraction']==1
    assert row['native_raw_abs_rel']==11
    assert row['native_clipped_abs_rel']==4
    assert row['oracle_raw_median_clipped_abs_rel']==0
    assert np.array_equal(p,gt*12)
    assert abs(row['log_decomposition_residual'])<1e-12


def test_log_error_identity_with_local_distortion():
    gt=np.full((10,10),2.);p=gt.copy();p[:5]*=.5;p[5:]*=2
    row,_=scale_statistics(p,gt,np.ones(gt.shape,bool))
    assert abs(row['log_bias'])<1e-12
    assert row['centered_log_mse']>0
    assert row['log_bias_squared']+row['centered_log_mse']==pytest.approx(row['log_mse'])


def test_annotations_combine_class_and_instance_and_keep_instance_zero():
    labels=np.array([[1,1,2,2,0]])
    instances=np.array([[0,1,1,1,2]])
    edge=annotation_edges(labels,instances,np.ones(labels.shape,bool))
    assert edge.tolist()==[[True,True,True,False,False]]


def test_empty_regions_are_na_and_partition_covers_the_same_valid_domain():
    gt=np.ones((10,12))*2;valid=np.ones(gt.shape,bool);none=np.zeros(gt.shape,bool)
    masks=region_masks(none,none,valid)
    assert error_statistics(gt,gt,masks['edge_le3px'])['abs_rel'] is None
    assert masks['interior_gt5px'].all()
    assert np.array_equal(sum(masks[k].astype(int) for k in ['edge_0_1px','edge_1_3px','edge_3_5px','interior_gt5px']),valid.astype(int))


def test_tolerance_and_empty_edge_conventions_match_existing_metric():
    from src.metrics.boundary import compute_depth_boundary_metrics
    gt=np.ones((20,24))*2;gt[:,12:]=4;p=np.ones_like(gt)*2;p[:,14:]=4;valid=np.ones(gt.shape,bool)
    g=depth_edge_map(gt,log_threshold=np.log(1.05));e=depth_edge_map(p,log_threshold=np.log(1.05))
    rows,dist=edge_comparison(e,g,valid,g,[1,2,3],10)
    for row in rows:
        original=compute_depth_boundary_metrics(p,gt,log_threshold=np.log(1.05),tolerance_px=row['tolerance_px'])
        assert row['f1']==original['boundary_f1']
        assert row['object_edge_recall']==row['recall']
    assert dist['gt_edge_distance_capped_px']>0
    rows,dist=edge_comparison(np.zeros_like(g),g,valid,g,[3],10)
    assert rows[0]['f1']==0 and rows[0]['object_edge_recall']==0
    assert dist['gt_edge_distance_capped_px']==10
    rows,_=edge_comparison(np.zeros_like(g),np.zeros_like(g),valid,np.zeros_like(g),[3],10)
    assert rows[0]['f1']==1 and rows[0]['object_edge_recall'] is None


def test_profile_separates_shift_blur_contrast_and_depth_order():
    c=config()['profiles'];r=c['radius_px'];gt=np.ones((5,2*r+1))*2;gt[:,r+1:]=4
    profile=dict(y=2,x=r,axis='x',radius_px=r)
    shape=profile_shape(gt[2],c['plateau_samples'],c['minimum_gt_ratio'])
    profile.update(gt_midpoint_index=shape['midpoint_index'],gt_width_10_90_px=shape['width_10_90_px'],gt_log_contrast=shape['log_contrast'])
    same=measure_profile(gt*2,gt,profile,c)
    assert same['absolute_midpoint_error_px']==0 and same['width_excess_px']==0
    assert same['profile_mae_m']>0
    moved=np.ones_like(gt)*2;moved[:,r+4:]=4
    assert measure_profile(moved,gt,profile,c)['signed_midpoint_error_px']==pytest.approx(3)
    smooth=gaussian_filter1d(gt,2,axis=1)
    assert measure_profile(smooth,gt,profile,c)['width_excess_px']>0
    reverse=6-gt
    assert measure_profile(reverse,gt,profile,c)['status']=='wrong_depth_order'
    weak=np.ones_like(gt)*2
    assert measure_profile(weak,gt,profile,c)['status']=='weak_contrast'
    assert profile_shape(weak[2],5,1.05)['status']=='weak_contrast'
    oscillating=np.r_[np.ones(8)*2,[4,2,4,2,4],np.ones(8)*4]
    out=profile_shape(oscillating,5,1.05)
    assert out['status']=='ambiguous_crossings' and out['width_10_90_px'] is None


def test_profiles_are_gt_only_and_require_annotation_support():
    c=config()['profiles'];gt=np.ones((50,100))*2;gt[:,50:]=4;valid=np.ones(gt.shape,bool)
    assert not choose_profiles(gt,valid,np.zeros_like(valid),c)
    profiles=choose_profiles(gt,valid,valid,c)
    assert profiles and len(profiles)<=3
    assert all(p['gt_width_10_90_px']<=6 for p in profiles)


def test_cluster_bootstrap_preserves_image_weights_and_profile_image_average():
    stats=pipeline.ClusterStatistics(dict(seed=42,bootstrap_resamples=100,confidence=.95))
    rows=[dict(scene='a',sample_id='1',m=0),dict(scene='a',sample_id='2',m=0),dict(scene='b',sample_id='3',m=100)]
    assert stats.describe(rows,'m')['mean']==pytest.approx(100/3)
    rows.append(dict(scene='b',sample_id='3',m=0))
    assert stats.describe(rows,'m')['mean']==pytest.approx(50/3)
    assert stats.describe([],'m')['mean'] is None


@pytest.fixture
def synthetic(tmp_path):
    c=config();c['dataset']['expected_images']=2
    c['evaluation']['crop']=[0,32,0,64]
    c['statistics']['bootstrap_resamples']=100
    c['cases']['fixed_test_positions']=[0,1];c['cases']['automatically_selected']=0
    gt=np.ones((32,64),np.float32)*2;gt[:,32:]=4
    mat=tmp_path/'dataset.mat';split=tmp_path/'splits.mat'
    with h5py.File(mat,'w') as f:
        f['images']=np.zeros((2,3,64,32),np.uint8)
        f['depths']=np.stack([gt.T,gt.T])
        labels=np.ones((32,64),np.uint16);labels[:,32:]=2
        f['labels']=np.stack([labels.T,labels.T]);f['instances']=np.zeros((2,64,32),np.uint8)
        for field,strings in [('scenes',['room_a','room_b']),('sceneTypes',['room','room'])]:
            refs=np.zeros((1,2),h5py.ref_dtype)
            for i,text in enumerate(strings):
                d=f.create_dataset(f'{field}_{i}',data=np.array([ord(x) for x in text],np.uint16));refs[0,i]=d.ref
            f[field]=refs
    savemat(split,{'testNdxs':np.array([[1],[2]])})
    c['dataset'].update(labeled_mat=str(mat),split_file=str(split),labeled_mat_sha256=pipeline.sha(mat),split_file_sha256=pipeline.sha(split))
    for model,multiplier in [('depth_pro',.8),('unidepth_v2',1.2)]:
        root=tmp_path/model;(root/'predictions').mkdir(parents=True)
        baseline_config={'dataset':{'expected_test_images':2,'split_variable':'testNdxs',**{k:c['dataset'][k] for k in ['labeled_mat_sha256','split_file_sha256']}},
            'evaluation':dict(crop=dict(top=0,bottom=32,left=0,right=64),valid_depth_m=dict(minimum=.001,maximum=10.,bounds='exclusive'),clip_prediction_to_valid_range=True,test_time_alignment='none',aggregation='per_image_macro_mean'),
            'model':dict(provide_external_intrinsics=False,provide_external_focal_length=False)}
        cfg=tmp_path/f'{model}.yaml';cfg.write_text(yaml.safe_dump(baseline_config))
        pipeline.dump_json(root/'run_metadata.json',dict(sample_count=2,full_test_split=True,test_time_alignment='none',formal_training_run=False,config=str(cfg),config_sha256=pipeline.sha(cfg)))
        rows=[]
        for i in range(2):
            stem=f'test_{i:04d}_mat_{i+1:04d}';p=gt*np.float32(multiplier)
            np.savez_compressed(root/'predictions'/f'{stem}.npz',depth_m=p)
            scores=pipeline.compute_depth_metrics(p,gt,min_depth=.001,max_depth=10.)
            rows.append(dict(sample_id=stem,test_position=i,matlab_index_1based=i+1,scene=f'room_{"a" if i==0 else "b"}',si_boundary_f1=.2,**{k:scores[k] for k in pipeline.METRICS}))
        pipeline.write_csv(root/'metrics_per_image.csv',rows)
        pipeline.dump_json(root/'metrics_summary.json',{'macro_mean':{k:float(np.mean([r[k] for r in rows])) for k in (*pipeline.METRICS,'si_boundary_f1')}})
        c['baselines'][model]=str(root)
    cfg=tmp_path/'analysis.yaml';cfg.write_text(yaml.safe_dump(c))
    return c,cfg,tmp_path/'result'


def test_complete_native_analysis_render_and_full_replay(synthetic):
    c,cfg,output=synthetic
    before={str(p):pipeline.sha(p) for m in c['baselines'].values() for p in Path(m).rglob('*') if p.is_file()}
    meta=pipeline.execute(c,cfg,output)
    assert meta['status']=='complete' and not meta['inference_performed']
    assert meta['table_rows']['per_image']==4
    assert (output/'report.md').exists() and (output/'figures/profile_geometry.png').exists()
    result=pipeline.verify(output,full=True)
    assert result['all_passed'] and result['numeric_values_checked']>1000
    assert all(pipeline.sha(p)==digest for p,digest in before.items())
    old=(output/'run_metadata.json').read_bytes()
    with pytest.raises(FileExistsError):pipeline.execute(c,cfg,output)
    assert (output/'run_metadata.json').read_bytes()==old
    rows=pipeline.read_csv(output/'tables/per_image.csv');rows[0]['raw_scale_ratio']='999'
    pipeline.write_csv(output/'tables/per_image.csv',rows)
    with pytest.raises(ValueError,match='Artifact SHA'):pipeline.verify(output)


def test_failure_is_recorded_and_baseline_protocol_is_checked(synthetic):
    c,cfg,output=synthetic
    root=Path(c['baselines']['depth_pro']);rows=pipeline.read_csv(root/'metrics_per_image.csv')
    rows[0]['sample_id']='wrong';pipeline.write_csv(root/'metrics_per_image.csv',rows)
    with pytest.raises(ValueError,match='identity'):
        pipeline.execute(c,cfg,output)
    assert json.loads((output/'run_metadata.json').read_text())['status']=='failed'


@pytest.mark.parametrize('change',['crop','alignment','oracle','split','counts'])
def test_formal_config_rejects_protocol_drift(tmp_path,change):
    c=config()
    if change=='crop':c['evaluation']['crop'][0]=44
    if change=='alignment':c['evaluation']['alignment']='median'
    if change=='oracle':c['scale']['oracle']='primary'
    if change=='split':c['dataset']['split_variable']='trainNdxs'
    if change=='counts':c['dataset']['expected_images']=653
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):pipeline.load_config(p)


def test_replay_uses_numeric_tolerances_with_exact_coverage():
    pipeline.summaries_close({'point':1.,'coverage':2,'missing':None,'role':'native'},
                             {'point':1.+1e-8,'coverage':2,'missing':None,'role':'native'})
    with pytest.raises(ValueError):pipeline.summaries_close({'point':1.},{'point':1.1})
    with pytest.raises(ValueError):pipeline.summaries_close({'point':1.},{'other':1.})
    with pytest.raises(ValueError):pipeline.summaries_close({'role':'native'},{'role':'oracle'})

    with pytest.raises(ValueError):pipeline.summaries_close({'image_count':1000000},{'image_count':1000001})
