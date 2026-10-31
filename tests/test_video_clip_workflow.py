"""单视频工作流、无光流指标及真实文件重算测试；不运行任何深度/RAFT模型。"""
from copy import deepcopy
from pathlib import Path
import json

import cv2
import numpy as np
import pytest
import yaml

from src.video import clip_workflow as workflow, clip_analysis as analysis, runner
from src.video.common import read_json,write_json,sha256_file,source_fingerprints
from src.models.video_depth_anything_adapter import window_metadata
from pipelines.part1 import video_clip as cli1
from pipelines.part2 import video_clip as cli2
from pipelines.video.clip import build_parser


@pytest.fixture
def synthetic(tmp_path,monkeypatch):
    """真实短MP4/PNG/NPY与冻结/评价，模型输出由明确标注的合成信号代替。"""
    monkeypatch.setattr(runner,'check_resources',lambda _: {})
    monkeypatch.setattr(workflow,'check_resources',lambda _: {})
    config=deepcopy(runner.load_config(workflow.DEFAULT_CONFIG))
    config['runtime'].update(device='cpu',precision='float32')
    config['input'].update(target_fps=10.,max_long_edge=32)
    config['flow'].update(backend='farneback',iterations=2)
    config['visualization'].update(generate_report=False,generate_video=False)
    config_path=tmp_path/'config.yaml';config_path.write_text(yaml.safe_dump(config))
    source=tmp_path/'source.mp4';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10.,(32,24))
    assert writer.isOpened()
    for i in range(4):
        image=np.zeros((24,32,3),np.uint8);image[:,:,0]=np.arange(32)*7;image[:,:,1]=i*25;writer.write(image)
    writer.release()
    registry=tmp_path/'clips.yaml';registry.write_text(yaml.safe_dump({'schema_version':'self_recorded_clips_v1','clips':{
        'sample':{'video':str(source),'type':'fixed_static','session':None,'description':'合成视频测试，不是正式实验'}}}))
    output=tmp_path/'part1';output2=tmp_path/'part2';cache=tmp_path/'cache';calls=[]
    def infer(protocol,name,dest):
        p,manifest,mp=runner.load_protocol(protocol);dest=Path(dest);dest.mkdir();(dest/'raw').mkdir()
        h,w=manifest['height'],manifest['width'];yy,xx=np.indices((h,w),dtype=np.float32);base=xx+2*yy
        record={'status':'complete','manifest_path':str(mp),'manifest_sha256':sha256_file(mp),
                'protocol_path':str(protocol),'protocol_sha256':sha256_file(protocol),'config':p['configuration'],
                'project_source_sha256':source_fingerprints(),'frames':[],
                'environment':{'torch':'synthetic','cuda_runtime':None,'visible_gpus':[]},
                'model':{'name':name,'depth_type':'relative','representation':'inverse_depth_like','unit':'arbitrary',
                         'alignment':'none','gt_used':False,'device':'cpu','encoder':'vits','precision':'float32',
                         'network_input_hw':[28,42],'actual_output_hw':[h,w],'parameters':10,'checkpoint_bytes':40},
                'timing':{'output_frames':len(manifest['frames']),'model_fps':10.,'amortized_ms_per_output_frame':100.,
                          'peak_gpu_allocated_bytes':None,'peak_gpu_reserved_bytes':None}}
        if name=='vda':record['model'].update(inference_mode='offline_noncausal',windowing=window_metadata(len(manifest['frames'])),window_starts=[0])
        for frame in manifest['frames']:
            i=frame['index'];value=base+float(i if name=='dav2' else i*.25)
            relative=f'raw/frame_{i:06d}.npy';np.save(dest/relative,value.astype(np.float32))
            record['frames'].append({'index':i,'source_index':frame['source_index'],'timestamp_seconds':frame['timestamp_seconds'],
                                     'shape':[h,w],'dtype':'float32','raw_path':relative,'raw_sha256':sha256_file(dest/relative)})
        write_json(dest/'run.json',record);calls.append(name);return dest/'run.json'
    def flow(protocol,dest):
        p,manifest,mp=runner.load_protocol(protocol);dest=Path(dest);dest.mkdir();(dest/'pairs').mkdir()
        h,w=manifest['height'],manifest['width']
        record={'status':'complete','manifest_path':str(mp),'manifest_sha256':sha256_file(mp),
                'protocol_path':str(protocol),'protocol_sha256':sha256_file(protocol),
                'frames_count':len(manifest['frames']),'shape':[h,w],
                'flow':{'backend':'farneback','iterations':2,'max_long_edge':None},'pairs':[],
                'timing':{'synthetic':True},'project_source_sha256':source_fingerprints()}
        for i in range(1,len(manifest['frames'])):
            pair={'previous_index':i-1,'current_index':i}
            for direction in ('forward','backward'):
                relative=f'pairs/{i}_{direction}.npy';np.save(dest/relative,np.zeros((h,w,2),np.float32))
                pair[direction+'_path']=relative;pair[direction+'_sha256']=sha256_file(dest/relative)
            record['pairs'].append(pair)
        write_json(dest/'flow_manifest.json',record);calls.append('flow');return dest/'flow_manifest.json'
    monkeypatch.setattr(workflow,'infer_run',infer);monkeypatch.setattr(workflow,'cache_flow',flow)
    def run(part='both',task='all',video_type=None,**kwargs):
        clip=workflow.resolve_clip('sample',registry=registry,video_type=video_type)
        # 测试辅助显式依次调用两独立调度器；正式入口不存在both模式。
        if part in ('1','both'):
            result=workflow.run_clip(clip,task=task,config_path=config_path,output_root=output,cache_root=cache,**kwargs)
        if part in ('2','both'):
            result=workflow.run_comparison('sample',task=task,config_path=config_path,part1_root=output,output_root=output2,**kwargs)
        return result
    return {'run':run,'root':output/'sample','output':output,'root2':output2/'sample','output2':output2,'cache':cache,'config':config,'config_path':config_path,
            'registry':registry,'source':source,'calls':calls}


def test_pixel_formula_uses_full_frame_including_zero_and_no_flow():
    a=np.array([[0.,1.],[2.,3.]],np.float32);b=a+np.array([[0.,2.],[0.,4.]])
    result=analysis.pixel_pair(a,b,2.)
    assert result['normalized_mae']==pytest.approx(.75)
    assert result['normalized_rmse']==pytest.approx(np.sqrt(5)/2)
    assert result['pixels']==4 and result['coverage']==1.
    assert analysis.pixel_pair(a+7,b+7,2.)==result
    assert analysis.pixel_pair(a*3+7,b*3+7,6.)['normalized_mae']==pytest.approx(.75)


@pytest.mark.parametrize('span',[None])
def test_constant_reference_is_not_a_false_perfect_score(span):
    result=analysis.pixel_pair(np.ones((2,3)),np.ones((2,3))*2,span)
    assert result['raw_mae']==1 and result['normalized_mae'] is None
    assert result['unavailable_reason']=='zero_reference_span'


@pytest.mark.parametrize('bad',['nan','inf','shape','span'])
def test_bad_pixel_inputs_fail(bad):
    a=np.zeros((2,3));b=a.copy();span=1.
    if bad=='nan':b[0,0]=np.nan
    if bad=='inf':b[0,0]=np.inf
    if bad=='shape':b=np.zeros((2,2))
    if bad=='span':span=0
    with pytest.raises(ValueError):analysis.pixel_pair(a,b,span)


def test_single_video_complete_two_parts_and_independent_reconstruction(synthetic):
    result=synthetic['run']();root=synthetic['root']
    assert result['status']=='complete' and set(synthetic['calls'])=={'dav2','flow','vda'}
    for part in ('1','2'):
        current=root if part=='1' else synthetic['root2']
        s=analysis.check_evaluation(current,part);v=read_json(current/f'verify{part}.json')
        assert s['frames']==4 and s['pairs']==3 and v['all_metrics_recomputed_from_raw']
        assert s['models']['dav2']['background_flicker']['domain']=='full_frame'
        assert s['models']['dav2']['background_flicker']['pixels']==24*32
        assert s['models']['dav2']['pixel']['coverage']==1.
        analysis.verify(current,part)
    s=read_json(synthetic['root2']/'part2/summary.json')
    assert s['improvement']['flow_mae']['relative_reduction']==pytest.approx(.75)
    assert s['improvement']['pixel_mae']['relative_reduction']==pytest.approx(.75)
    assert s['models']['dav2']==read_json(root/'part1/summary.json')['models']['dav2']
    assert synthetic['source'].is_file() and read_json(root/'clip.json')['clip']['session'] is None


def test_step2_is_independent_and_step4_reuses_it_without_repredicting(synthetic):
    synthetic['run'](part='1');assert synthetic['calls']==['dav2','flow']
    before={str(p):sha256_file(p) for p in synthetic['root'].rglob('*') if p.is_file()}
    synthetic['run'](part='2');assert synthetic['calls']==['dav2','flow','vda']
    after={str(p):sha256_file(p) for p in synthetic['root'].rglob('*') if p.is_file()}
    assert after==before
    assert not (synthetic['root2']/'dav2').exists() and not (synthetic['root2']/'flow').exists()
    assert not (synthetic['root']/'vda').exists() and not (synthetic['root']/'part2').exists()
    synthetic['run'](part='2');assert synthetic['calls']==['dav2','flow','vda']


@pytest.mark.parametrize('kind',['fixed_dynamic','moving'])
def test_motion_types_never_label_full_frame_std_as_background(synthetic,kind):
    synthetic['run'](video_type=kind)
    s=read_json(synthetic['root2']/'part2/summary.json')
    assert not s['models']['dav2']['background_flicker']['applicable']
    assert s['models']['dav2']['background_flicker']['normalized_std_mean'] is None
    assert s['models']['dav2']['pixel']['mean_pair_normalized_mae']>0
    assert s['models']['dav2']['scientific_protocol']['primary_diagnostic']=='flow_and_median_proxy'
    assert not (synthetic['root2']/'part2/dav2/background_normalized_std.npy').exists()


def test_task_only_dav2_does_not_run_flow_vda_or_analysis(synthetic):
    r=synthetic['run'](part='1',task='dav2')
    assert r['executed']==['prepare','dav2'] and synthetic['calls']==['dav2']
    assert not (synthetic['root']/'flow').exists() and not (synthetic['root']/'part1').exists()


def test_no_dependencies_is_strict_and_missing_inputs_fail(synthetic):
    with pytest.raises(FileNotFoundError):synthetic['run'](part='1',task='analyze',dependencies=False)
    assert not synthetic['calls']
    synthetic['run'](part='1',task='prepare')
    with pytest.raises(FileNotFoundError):synthetic['run'](part='1',task='analyze',dependencies=False)
    assert read_json(next((synthetic['root']/'commands').glob('*.json')))['status'] in ('complete','failed')


def test_source_replacement_and_type_or_setting_changes_reject_resume(synthetic):
    synthetic['run'](part='1',task='prepare')
    with pytest.raises(ValueError):synthetic['run'](video_type='moving')
    config=read_json(synthetic['root']/'clip.json')['config'];config['evaluation']['fb_threshold_px']=2
    synthetic['config_path'].write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):synthetic['run']()


@pytest.mark.parametrize('target',['pixel_csv','mask','raw','summary','source'])
def test_real_tampering_is_detected(synthetic,target):
    synthetic['run']();root=synthetic['root2']
    if target=='pixel_csv':
        p=root/'part2/pair_metrics.csv';p.write_text(p.read_text().replace('0.25','99.25'))
    elif target=='mask':
        p=next((root/'part2/vda/masks').glob('*.npy'));np.save(p,np.zeros((24,32),bool))
    elif target=='raw':
        p=next((root/'vda/raw').glob('*.npy'));np.save(p,np.zeros((24,32),np.float32))
    elif target=='summary':
        p=root/'part2/summary.json';value=read_json(p);value['models']['dav2']['pixel']['mean_pair_normalized_mae']=999;write_json(p,value)
    else:synthetic['source'].write_bytes(b'changed video')
    with pytest.raises(ValueError):analysis.verify(root,'2')


def test_translation_has_nonzero_pixel_change_even_when_flow_error_is_zero(synthetic):
    synthetic['run'](part='1',task='dav2');synthetic['run'](part='1',task='flow')
    root=synthetic['root'];run=read_json(root/'dav2/run.json');flow=read_json(root/'flow/flow_manifest.json')
    yy,xx=np.indices((24,32),dtype=np.float32);base=xx+2*yy
    for i,frame in enumerate(run['frames']):
        path=root/'dav2'/frame['raw_path'];np.save(path,np.roll(base,i,axis=1));frame['raw_sha256']=sha256_file(path)
    for pair in flow['pairs']:
        for direction,sign in [('forward',1),('backward',-1)]:
            value=np.zeros((24,32,2),np.float32);value[...,0]=sign;path=root/'flow'/pair[direction+'_path'];np.save(path,value);pair[direction+'_sha256']=sha256_file(path)
    write_json(root/'dav2/run.json',run);write_json(root/'flow/flow_manifest.json',flow)
    synthetic['run'](part='1')
    s=read_json(root/'part1/summary.json')['models']['dav2']
    assert s['flow']['mean_pair_normalized_mae']==0
    p=analysis.pixel_pair(base,np.roll(base,1,axis=1),s['reference']['span'])
    assert p['normalized_mae']>0
    # The first true translation pair is exact on the non-occluded support.
    rows=analysis._read_csv(root/'part1/pair_metrics.csv')
    assert rows[0]['dav2_flow_mae']==0 and rows[0]['dav2_pixel_mae']>0
    assert rows[0]['pixel_pixels']==24*32 and rows[0]['coverage']<1


def test_summary_supports_arbitrary_clip_counts_and_keeps_na(synthetic,tmp_path):
    synthetic['run'](video_type='moving')
    analysis.summarize(synthetic['output2'],'2',tmp_path/'summary')
    s=read_json(tmp_path/'summary/summary.json')
    assert s['groups'][0]['clips']==1 and s['groups'][0]['type']=='moving'
    assert s['groups'][0]['equal_clip_means']['dav2_background_std'] is None
    with pytest.raises(ValueError):analysis.summarize(synthetic['output2'],'2',tmp_path/'other',['sample','sample'])
    with pytest.raises(FileExistsError):analysis.summarize(synthetic['output2'],'2',tmp_path/'summary')


def test_real_report_has_unwarped_and_flow_videos_and_fixed_scales(synthetic):
    c=synthetic['config'];c['visualization'].update(generate_report=True,generate_video=True)
    synthetic['config_path'].write_text(yaml.safe_dump(c));synthetic['run']()
    s=read_json(synthetic['root2']/'part2/summary.json');r=s['report']
    videos=[a for a in r['assets'] if a['path'].endswith('.mp4')]
    assert len(videos)==4 and sorted(v['frames'] for v in videos)==[3,3,4,4]
    assert not r['per_frame_normalization'] and not r['manual_roi']
    assert r['pixel_domain']=='full_frame_no_warp_no_FB_mask'
    assert (synthetic['root2']/'part2/report/background_std.png').exists()
    assert analysis.verify(synthetic['root2'],'2')['status']=='verified'


def test_official_registry_dynamic_count_and_plan_does_not_execute(synthetic,capsys):
    assert len(yaml.safe_load(workflow.DEFAULT_REGISTRY.read_text())['clips'])==6
    assert cli2.main(['plan','--clip','sample'])==0
    assert not synthetic['calls'] and not synthetic['root'].exists()
    assert json.loads(capsys.readouterr().out)['stages'][-1]=='verify2'
    with pytest.raises(ValueError):workflow.resolve_clip('../escape',registry=synthetic['registry'])
    with pytest.raises(ValueError):workflow.resolve_clip('new',registry=synthetic['registry'])
    with pytest.raises(ValueError):workflow.task_plan('1','vda')


def test_unregistered_video_is_remembered_for_the_next_part(synthetic):
    clip=workflow.resolve_clip('new_video',registry=synthetic['registry'],video=synthetic['source'],video_type='moving')
    workflow.run_clip(clip,part='1',task='prepare',config_path=synthetic['config_path'],output_root=synthetic['output'],cache_root=synthetic['cache'])
    remembered=workflow.resolve_clip('new_video',registry=synthetic['registry'],output_root=synthetic['output'])
    assert remembered==clip


def test_failed_model_stage_is_retained_and_cannot_be_silently_skipped(synthetic,monkeypatch):
    def fail(protocol,name,dest):
        dest.mkdir();write_json(dest/'run.json',{'status':'failed','reason':'synthetic intentional failure'});raise RuntimeError('synthetic failure')
    monkeypatch.setattr(workflow,'infer_run',fail)
    with pytest.raises(RuntimeError):synthetic['run'](part='1',task='dav2')
    assert read_json(synthetic['root']/'dav2/run.json')['status']=='failed'
    with pytest.raises(ValueError):synthetic['run'](part='1',task='dav2')
    assert all(read_json(p)['status']=='failed' for p in (synthetic['root']/'commands').glob('*.json'))


@pytest.mark.parametrize('field',['precision','network_input_hw','windowing'])
def test_pairing_rejects_incomparable_vda_metadata(synthetic,field):
    synthetic['run'](part='1');synthetic['run'](part='2',task='vda')
    path=synthetic['root2']/'vda/run.json';r=read_json(path)
    if field=='precision':r['model'][field]='float16'
    elif field=='network_input_hw':r['model'][field]=[28,56]
    else:r['model'][field]['window_count']=100
    write_json(path,r)
    with pytest.raises(ValueError):analysis.analyze(synthetic['root2'],'2')


def test_full_frame_background_includes_changes_outside_a_small_rectangle(synthetic):
    synthetic['run'](part='1',task='dav2');synthetic['run'](part='1',task='flow')
    root=synthetic['root'];r=read_json(root/'dav2/run.json')
    for i,entry in enumerate(r['frames']):
        p=root/'dav2'/entry['raw_path'];a=np.load(p);a[:,:16]-=i;a[:,16:]+=i*8;np.save(p,a);entry['raw_sha256']=sha256_file(p)
    write_json(root/'dav2/run.json',r);synthetic['run'](part='1')
    s=read_json(root/'part1/summary.json')['models']['dav2']
    assert s['background_flicker']['pixels']==24*32
    std=np.load(root/'part1/dav2/background_normalized_std.npy')
    assert np.all(std[:,:16]==0) and np.all(std[:,16:]>0)
    assert s['background_flicker']['normalized_std_mean']==pytest.approx(float(std.mean()))


@pytest.mark.parametrize('state',['missing','prepared','predicted'])
def test_part2_requires_completed_part1_without_creating_outputs(synthetic,state):
    if state=='prepared':synthetic['run'](part='1',task='prepare')
    if state=='predicted':synthetic['run'](part='1',task='dav2')
    before=list(synthetic['calls'])
    with pytest.raises((FileNotFoundError,ValueError)):synthetic['run'](part='2')
    assert synthetic['calls']==before and not synthetic['root2'].exists()


def test_public_entries_reject_other_parts_tasks_and_combined_switch():
    for part,other in [('1','vda'),('2','dav2'),('2','flow')]:
        with pytest.raises(SystemExit):build_parser(part).parse_args(['run','--clip','sample','--task',other])
        with pytest.raises(SystemExit):build_parser(part).parse_args(['run','--clip','sample','--part','both'])
    assert build_parser('1').parse_args(['run','--clip','a']).output_root==workflow.OUTPUT_ROOT
    assert build_parser('2').parse_args(['run','--clip','a']).output_root==workflow.PART2_OUTPUT_ROOT
    with pytest.raises(SystemExit):build_parser('2').parse_args(['run','--clip','a','--video','a.mp4'])
    with pytest.raises(ValueError):workflow.run_clip({},part='2')
    with pytest.raises(ValueError):workflow.task_plan('both')
    assert workflow.task_plan('2')==['prepare','vda','analyze2','verify2']


def test_part2_refuses_overlap_and_wrong_parent_config(synthetic):
    synthetic['run'](part='1')
    for output in (synthetic['output'],synthetic['output']/'nested',synthetic['output'].parent):
        with pytest.raises(ValueError):workflow.run_comparison('sample',part1_root=synthetic['output'],
            output_root=output,config_path=synthetic['config_path'])
    c=deepcopy(synthetic['config']);c['evaluation']['minimum_coverage']=.1
    other=synthetic['config_path'].with_name('other.yaml');other.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):workflow.run_comparison('sample',part1_root=synthetic['output'],
        output_root=synthetic['output2'],config_path=other)
    assert not synthetic['root2'].exists()


def test_part2_binding_detects_parent_verification_tampering(synthetic):
    synthetic['run']()
    p=synthetic['root']/'verify1.json';r=read_json(p);r['frames']=999;write_json(p,r)
    with pytest.raises(ValueError):analysis.verify(synthetic['root2'],'2')


def test_new_clip_part2_uses_frozen_parent_without_registry(synthetic):
    clip=workflow.resolve_clip('new_video',registry=synthetic['registry'],video=synthetic['source'],video_type='moving')
    workflow.run_clip(clip,config_path=synthetic['config_path'],output_root=synthetic['output'],cache_root=synthetic['cache'])
    result=workflow.run_comparison('new_video',config_path=synthetic['config_path'],
                                  part1_root=synthetic['output'],output_root=synthetic['output2'])
    assert result['status']=='complete' and synthetic['calls']==['dav2','flow','vda']
    r,_,baseline=workflow.load_comparison(synthetic['output2']/'new_video')
    assert r['clip']==clip and baseline==synthetic['output']/'new_video'



def test_independent_cli_run_paths_and_part2_does_not_write_part1(synthetic):
    shared=['--clip','sample','--config',str(synthetic['config_path'])]
    assert cli1.main(['run',*shared,'--registry',str(synthetic['registry']),
                      '--output-root',str(synthetic['output']),'--cache-root',str(synthetic['cache'])])==0
    before={str(p):sha256_file(p) for p in synthetic['root'].rglob('*') if p.is_file()}
    assert cli2.main(['run',*shared,'--part1-root',str(synthetic['output']),
                      '--output-root',str(synthetic['output2'])])==0
    assert before=={str(p):sha256_file(p) for p in synthetic['root'].rglob('*') if p.is_file()}
    assert synthetic['calls']==['dav2','flow','vda']
    assert read_json(synthetic['root2']/'verify2.json')['status']=='verified'
    assert not (synthetic['root']/'vda').exists()
