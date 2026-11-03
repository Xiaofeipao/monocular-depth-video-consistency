"""课程1B：原始帧序、无GT尺度/时间诊断与真实文件生命周期；模型仅合成。"""
from pathlib import Path
from copy import deepcopy
import json
import cv2
import numpy as np
import pytest
import torch
import yaml
from src.evaluation import framewise_clip as task
from src.evaluation.course_video_inputs import prepare
from src.metrics import video_diagnostics as d
from src.video.common import sha256_file,read_json,write_json
from src.models.depth_anything_v2_adapter import network_input_shape


@pytest.fixture
def clip(tmp_path,monkeypatch):
    source=tmp_path/'source.mp4';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),24,(64,36))
    for i in range(6):
        a=np.zeros((36,64,3),np.uint8);a[:,:,:]=30+i*10;a[:,32:,0]=120;writer.write(a)
    writer.release()
    weight=tmp_path/'weights.pth';weight.write_bytes(b'synthetic fixture; not a real checkpoint')
    c=deepcopy(task.load_config());c['source'].update(path=str(source),sha256=sha256_file(source))
    c['model'].update(checkpoint=str(weight),sha256=sha256_file(weight),repository=str(tmp_path/'DA'))
    c['runtime'].update(device='cpu',precision='float32');c['inference'].update(input_size=28,timing_repeats=1)
    c['visualization']['video']=False;c['output']=str(tmp_path/'run');c['input']['cache']=str(tmp_path/'rgb_cache')
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(c))
    monkeypatch.setattr(task,'_repository_identity',lambda p:{'upstream_git_sha':c['model']['commit'],'repository_dirty':False})
    monkeypatch.setattr(task,'environment_info',lambda:{'kind':'synthetic model test','cuda_available':False})
    calls=[]
    class Model:
        def __init__(self,**kwargs):
            self.device=torch.device('cpu');self.metadata={'depth_type':'relative','unit':'arbitrary','alignment':'none',
                'gt_used':False,'temporal_context':False,'encoder':'vits','precision':'float32','parameters':1,
                'network_input_hw':list(network_input_shape(36,64,28))}
        def infer(self,rgb):
            calls.append(float(rgb.mean()));yy,xx=np.indices(rgb.shape[:2],dtype=np.float32)
            return xx+yy+np.float32(rgb.mean()/20)
    monkeypatch.setattr(task,'DepthAnythingV2Adapter',Model)
    return {'source':source,'config':c,'path':p,'output':tmp_path/'run','cache':tmp_path/'rgb_cache','calls':calls}


def test_original_24fps_indices_have_no_pts_rounding_drops(clip):
    p=prepare(clip['source'],clip['cache'],requested_frames=100,max_long_edge=640,fps=24.,source_url='synthetic')
    m=read_json(p)
    assert [f['source_index'] for f in m['frames']]==list(range(6)) and m['fps']==24
    assert m['selection']['rule'].startswith('consecutive_original')
    assert len({f['rgb_sha256'] for f in m['frames']})==6


def test_relative_affine_convention_and_zero_reference():
    raw=np.arange(48,dtype=np.float32).reshape(3,4,4);r=d.reference(raw);s=r['span']
    a=d.difference(raw[0],raw[1],s);b=d.difference(raw[0]*2+5,raw[1]*2+5,s*2)
    assert a['mae_normalized']==pytest.approx(b['mae_normalized'])
    assert a['rmse_normalized']==pytest.approx(b['rmse_normalized'])
    assert d.reference(np.ones((3,4,4)))['span'] is None
    assert d.difference(np.zeros((4,4)),np.ones((4,4)),None)['mae_normalized'] is None
    with pytest.raises(ValueError):d.difference(raw[0],raw[1],0)
    with pytest.raises(ValueError):d.difference(np.full((4,4),np.nan),raw[1],s)


def test_exact_translation_changes_screen_pixels_but_not_correspondence():
    yy,xx=np.indices((24,32),dtype=np.float32);a=xx+yy*2;b=np.roll(a,1,axis=1)
    forward=np.zeros((24,32,2),np.float32);forward[...,0]=1;back=-forward
    settings={'fb_threshold_px':1,'edge_threshold':.02,'edge_tolerance_px':2}
    score=d.flow_pair(a,b,forward,back,50,settings)
    assert score['flow_rmse_normalized']==0 and score['flow_coverage']<1
    assert d.difference(a,b,50)['rmse_normalized']>0


def test_short_clip_run_and_full_recompute_without_vda(clip):
    root=task.predict(clip['path']);assert len(clip['calls'])==7
    task.analyze(root);verified=task.verify(root)
    assert verified['frames']==6 and verified['pairs']==5 and not verified['required_100_frames_satisfied']
    run=read_json(root/'run.json');assert run['resources']['source_fps']==24
    assert run['frame_scope']['short_source_status']=='needs_course_confirmation'
    summary=read_json(root/'analysis/summary.json')
    assert not summary['scale_convention']['meters_available']
    assert summary['scale_convention']['positive_affine_demo']['max_normalized_difference']<1e-6
    assert summary['reference']['scope']=='whole_clip_once' and not summary['reference']['per_frame_normalization']
    handoff=read_json(root/'analysis/handoff_2b.json');assert not handoff['vda_executed']
    assert (root/'analysis/visualization/contact.png').is_file()
    with pytest.raises(FileExistsError):task.predict(clip['path'])
    with pytest.raises(FileExistsError):task.analyze(root)


@pytest.mark.parametrize('name',['raw','csv','source','handoff','scope','network'])
def test_tampering_is_rejected(clip,name):
    root=task.predict(clip['path']);task.analyze(root)
    if name=='raw':np.save(root/'raw/frame_000000.npy',np.zeros((36,64),np.float32))
    elif name=='csv':
        p=root/'analysis/pair_diagnostics.csv';p.write_text(p.read_text()+'altered')
    elif name=='source':clip['source'].write_bytes(b'changed')
    elif name=='handoff':
        p=root/'analysis/handoff_2b.json';v=read_json(p);v['manifest']['sha256']='0'*64;write_json(p,v)
    else:
        p=root/'run.json';v=read_json(p)
        if name=='scope':v['frame_scope']['actual_frames']=100
        else:v['model']['network_input_hw']=[14,14]
        write_json(p,v)
    with pytest.raises(ValueError):task.verify(root)


def test_source_or_checkpoint_mismatch_fails_before_creating_run(clip):
    c=deepcopy(clip['config']);c['source']['sha256']='0'*64;clip['path'].write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):task.predict(clip['path'])
    assert not clip['output'].exists() and not clip['calls']
