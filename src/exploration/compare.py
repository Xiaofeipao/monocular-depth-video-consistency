"""Compare preserved Part 3 directions on one frozen metric video run."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ..evaluate import evaluate_run
from ..flow import FarnebackFlow,RAFTFlow
from ..geometry import measure_point
from ..io import RunWriter,load_run,load_depth,read_bgr,sha256,write_csv,write_json
from ..metrics import temporal_metrics,evaluation_mask
from ..temporal import TemporalFusion,FusionConfig
from ..visualization import render_run
from .application import ObstacleWarning,warning_metrics
from .calibration import fit_scale,apply_scale
from .edge_aware import refine_edges,CUDAEdgeRefiner
from .efficient import ReducedFlow
from .uncertainty import UncertaintyFusion


METHODS = ('RAW','A','B','C','D','E_half','E_adaptive','F','B+C+D')
OPTIMIZED_METHODS = ('A_conservative','C_fast','C_cuda','D_guarded','E_local','F_coverage')
ALL_METHODS = METHODS + OPTIMIZED_METHODS


def load_calibration(path, test_records):
    path=Path(path)
    config=json.loads(path.read_text(encoding='utf-8'))
    if config.get('source')!='independent_measured_landmarks':
        raise ValueError('B requires independent_measured_landmarks, not benchmark GT')
    run=(path.parent/config['run']).resolve()
    meta,records=load_run(run)
    if meta['depth_type']!='metric' or meta['representation']!='depth':
        raise ValueError('Calibration run must contain metric depth')
    by_id={r['sample_id']:r for r in records}
    test_ids={r['sample_id'] for r in test_records}
    predicted,measured=[],[]
    seen=set()
    for point in config['points']:
        sample_id=point['sample_id']
        if sample_id in test_ids:
            raise ValueError('Calibration/test frame IDs overlap: forbidden test-time calibration')
        record=by_id[sample_id]
        observation=(sample_id,int(point['x']),int(point['y']))
        if observation in seen:
            raise ValueError('Duplicate calibration landmark is not independent')
        seen.add(observation)
        file=run/'depth'/f"{record['id']}.npy"
        if sha256(file)!=record['depth_sha256']:
            raise ValueError('Calibration prediction has changed')
        depth=load_depth(file)
        # Focal length cancels for axial-depth measurements.
        k=dict(fx=1.,fy=1.,cx=0.,cy=0.)
        predicted.append(float(measure_point(depth,int(point['x']),int(point['y']),k)[2]))
        measured.append(point['depth_m'])
    return {**fit_scale(predicted,measured),'config_sha256':sha256(path),'calibration_run_manifest_sha256':sha256(run/'manifest.json')}


class DirectionProcessor:
    def __init__(self, method, calibration=None):
        if method not in ALL_METHODS:
            raise ValueError('Unknown direction')
        if ('B' in method) and calibration is None:
            raise ValueError('B requires independent measured calibration')
        self.method,self.calibration=method,calibration
        self.cuda_refiner=CUDAEdgeRefiner() if method=='C_cuda' else None
        self.flow=ReducedFlow(.5,adaptive=method in ('E_adaptive','E_local'),local_motion=method=='E_local') if method.startswith('E_') else FarnebackFlow()
        self.temporal=TemporalFusion(FusionConfig(alpha=.2 if method=='A_conservative' else .4)) if method in ('A','A_conservative','E_half','E_adaptive','E_local') else None
        self.uncertainty=(UncertaintyFusion(alpha_max=.25,max_log_update=.01,edge_threshold=.08) if method=='D_guarded' else UncertaintyFusion()) if method in ('D','D_guarded','B+C+D') else None
        self.previous=None
        self.weights=[]

    def step(self, depth, rgb):
        method=self.method
        result=depth.copy()
        debug={}
        if method in ('B','B+C+D'):
            result=apply_scale(result,self.calibration['scale'])
        if method in ('C','C_fast','B+C+D'):
            result=refine_edges(result,rgb,radius=1 if method=='C_fast' else 2)
        if self.cuda_refiner is not None:
            result=self.cuda_refiner(result,rgb)
        if self.temporal or self.uncertainty:
            forward=backward=None
            if self.previous is not None:
                forward,backward=self.flow(self.previous,rgb)
            if self.temporal:
                result,debug=self.temporal.step(result,rgb,forward,backward)
            else:
                result,debug=self.uncertainty.step(result,rgb,forward,backward)
        self.previous=rgb
        if 'weight' in debug:
            self.weights.append(float(np.mean(debug['weight'])))
        return np.asarray(result,np.float32),debug


def select_winners(rows):
    """Task-specific optima, with explicit accuracy/boundary constraints."""
    baseline=next(r for r in rows if r['method']=='RAW')
    output={}
    for metric,maximum in [('AbsRel',False),('edge_F1',True),('temporal_log_error',False),('postprocess_fps',True),('warning_F1',True)]:
        candidates=[r for r in rows if r.get(metric) is not None]
        if candidates:
            optimum=(max if maximum else min)(r[metric] for r in candidates)
            output[metric]=[r['method'] for r in candidates if np.isclose(r[metric],optimum,rtol=1e-6,atol=1e-9)]
    eligible=[r for r in rows if r['method'] not in ('RAW','F','F_coverage') and r.get('temporal_log_error') is not None
              and r.get('AbsRel') is not None and r['AbsRel']<=baseline['AbsRel']+.01
              and r.get('delta1',0)>=baseline.get('delta1',0)-.005
              and r.get('edge_F1',0)>=baseline.get('edge_F1',0)-.01]
    output['accuracy_boundary_constrained_temporal_best']=(min(eligible,key=lambda r:r['temporal_log_error'])['method'] if eligible else None)
    pose_eligible=[r for r in eligible if r.get('pose_temporal_log_error') is not None]
    output['accuracy_boundary_constrained_pose_best']=(min(pose_eligible,key=lambda r:r['pose_temporal_log_error'])['method'] if pose_eligible else None)
    output['constraint']='AbsRel <= RAW+0.01, delta1 >= RAW-0.005, edge_F1 >= RAW-0.01; B variants use extra measured information'
    return output


def compare(run, output, manifest=None, calibration=None, methods=METHODS, repeats=3,
            evaluation_flow='farneback', flow_checkpoint=None, roi=None, warning_threshold=1.5):
    run,output=Path(run),Path(output)
    if repeats<1:
        raise ValueError('At least one timed repeat required')
    meta,records=load_run(run)
    if meta['depth_type']!='metric' or meta['representation']!='depth' or len(records)<2 or not meta.get('fps'):
        raise ValueError('Comparison needs a frozen metric video run')
    if roi is not None:
        h,w=records[0]['shape']
        if not (0<=roi[0]<roi[1]<=h and 0<=roi[2]<roi[3]<=w):
            raise ValueError('Invalid application ROI')
    calibrated=load_calibration(calibration,records) if calibration else None
    frames,depths=[],[]
    for record in records:
        key=record['id']
        for kind,extension,hash_key in [('rgb','png','rgb_sha256'),('depth','npy','depth_sha256')]:
            if sha256(run/kind/f'{key}.{extension}')!=record[hash_key]:
                raise ValueError('Frozen run hash mismatch')
        frames.append(read_bgr(run/'rgb'/f'{key}.png'))
        depths.append(load_depth(run/'depth'/f'{key}.npy'))
    dataset=None
    targets=None
    if manifest:
        manifest=Path(manifest)
        dataset=json.loads(manifest.read_text(encoding='utf-8'))
        by_id={s['id']:s for s in dataset['samples']}
        if len(by_id)!=len(records) or set(by_id)!={r['sample_id'] for r in records}:
            raise ValueError('Manifest must match the exact video test split')
        targets=[]
        for record in records:
            sample=by_id[record['sample_id']]
            target=load_depth(manifest.parent/sample['depth'],sample.get('depth_scale',1.))
            if target.shape!=depths[0].shape:
                raise ValueError('Application GT must already be registered at saved RGB resolution')
            mask=load_depth(manifest.parent/sample['mask'])>0 if sample.get('mask') else None
            protocol=dataset['protocol']
            valid=evaluation_mask(target,protocol['min_depth'],protocol['max_depth'],mask,protocol.get('crop'))
            target=np.where(valid,target,np.nan)
            targets.append(target)
    output.mkdir(parents=True,exist_ok=False)
    eval_backend=FarnebackFlow() if evaluation_flow=='farneback' else RAFTFlow('cpu',flow_checkpoint)
    fields=[eval_backend(frames[i-1],frames[i]) for i in range(1,len(frames))]
    rows,skipped=[],{}
    synthetic=bool(meta.get('synthetic'))
    for method in methods:
        if method not in ALL_METHODS:
            raise ValueError(f'Unknown method {method}')
        if 'B' in method and calibrated is None:
            skipped[method]='No independent measured calibration; test GT is not substituted'
            continue
        # One warm-up pass, then independent full-sequence repeats. Flow is
        # recomputed inside every processing pass and included in its timing.
        times=[]
        for repeat in range(repeats+1):
            processor=DirectionProcessor(method,calibrated)
            warning=ObstacleWarning(warning_threshold,confirmations=2 if method in ('F','F_coverage') else 1,
                                    exit_margin_m=.15 if method in ('F','F_coverage') else 0.,
                                    min_coverage=.5 if method=='F_coverage' else 0.)
            begin=time.perf_counter()
            results,alerts=[],[]
            for depth,rgb in zip(depths,frames):
                result,_=processor.step(depth,rgb)
                results.append(result)
                alerts.append(warning.step(result,roi))
            elapsed=time.perf_counter()-begin
            if repeat:
                times.append(elapsed)
        for raw,result in zip(depths,results):
            valid=np.isfinite(raw)&(raw>0)
            if not np.array_equal(valid,np.isfinite(result)&(result>0)):
                raise ValueError('Method changed the evaluation validity mask')
        derived={**meta,'model':f"{meta['model']}+direction_{method}",'direction':method,
                 'parent_manifest_sha256':sha256(run/'manifest.json'),'synthetic':synthetic,
                 'calibration':calibrated if 'B' in method else None,
                 'extra_information':'independent_measured_landmarks' if 'B' in method else 'none'}
        writer=RunWriter(output/method,derived,{'method':method,'repeats':repeats,'warmup_passes':1})
        for record,result,rgb in zip(records,results,frames):
            extra={k:record[k] for k in ('sample_id','focal_px','intrinsics_source') if k in record}
            writer.add(result,rgb,record['source'],record['source_index'],extra)
        runtime={'scope':'postprocessing + warning compute, including its flow; excludes backbone/decode/I/O/evaluation',
                 'repeat_seconds':times,'mean_seconds':float(np.mean(times)),
                 'postprocess_fps':len(records)/float(np.mean(times)),'cpu':method!='C_cuda',
                 'device':'cuda' if method=='C_cuda' else 'cpu',
                 'host_device_transfers_included':method=='C_cuda'}
        write_json(writer.path/'processing_diagnostics.json',{
            'history_resets':processor.temporal.resets if processor.temporal else None,
            'mean_history_weight':float(np.mean(processor.weights)) if processor.weights else None,
            'flow_resolution_scales':getattr(processor.flow,'used_scales',None)})
        writer.finish(runtime)
        render_run(writer.path,writer.records,*meta['visualization_range'],meta['unit'],meta['fps'],f'{method}'+(' SYNTHETIC' if synthetic else ''))
        temporal=[temporal_metrics(results[i-1],results[i],*fields[i-1]) for i in range(1,len(results))]
        write_json(writer.path/'warnings.json',{'roi':roi,'threshold_m':warning_threshold,'hysteresis':method in ('F','F_coverage'),'min_coverage':.5 if method=='F_coverage' else 0.,'frames':alerts})
        row={'method':method,'extra_information':derived['extra_information'],'AbsRel':None,'RMSE':None,'delta1':None,'edge_F1':None,
             'temporal_log_error':None,'local_flicker':None,'temporal_coverage':None,
            'postprocess_fps':runtime['postprocess_fps'],'warning_F1':None,'warning_recall':None,'warning_accuracy':None,
            'pose_temporal_log_error':None,'pose_temporal_MAE_m':None}
        for metric,out_key in [('temporal_log_error','temporal_log_error'),('local_flicker','local_flicker'),('coverage','temporal_coverage')]:
            values=[r[metric] for r in temporal if r[metric] is not None]
            row[out_key]=float(np.mean(values)) if values else None
        write_json(writer.path/'temporal_comparison.json',{'evaluation_flow':evaluation_flow,'transitions':temporal})
        if manifest:
            # Video frames are correlated; reserve confidence intervals for
            # independent clips rather than bootstrap individual frames.
            scores=evaluate_run(writer.path,manifest,bootstrap=0)
            for key in ('AbsRel','RMSE','delta1','edge_F1'):
                row[key]=float(np.mean([r[key] for r in scores]))
            if dataset.get('pose_convention') and all('pose_world_camera' in s for s in dataset['samples']):
                from ..evaluate_rgbd import evaluate_pose_run
                pose=evaluate_pose_run(writer.path,manifest)['metric_means']
                for key in ('pose_temporal_log_error','pose_temporal_MAE_m'):
                    row[key]=pose.get(key)
            regions=[d if roi is None else d[roi[0]:roi[1],roi[2]:roi[3]] for d in targets]
            if not all(np.isfinite(region).all() for region in regions):
                # A percentile over sparse LiDAR GT cannot be compared to a
                # percentile over dense predictions on a different pixel set.
                write_json(writer.path/'application_metrics.json',{
                    'status':'skipped','reason':'Application ROI lacks dense valid GT in every frame; select a fully labelled ROI.'})
                rows.append(row)
                continue
            ground_alerts=[ObstacleWarning(warning_threshold,confirmations=1,exit_margin_m=0.).step(d,roi)['warning'] for d in targets]
            app=warning_metrics([r['warning'] for r in alerts],ground_alerts)
            write_json(writer.path/'application_metrics.json',app)
            for key in ('warning_F1','warning_recall','warning_accuracy'):
                row[key]=app[key]
        rows.append(row)
    report={'synthetic':synthetic,'dataset':dataset.get('dataset') if dataset else None,
            'source_manifest_sha256':sha256(run/'manifest.json'),'methods':rows,'skipped':skipped,
            'winners':select_winners(rows) if any(r['method']=='RAW' for r in rows) else {},
            'limitations':['Synthetic results cannot establish best real-data direction.' if synthetic else 'One run does not establish cross-domain superiority.',
                          'B variants use independent calibration information and must be reported separately.',
                          'E measures flow/postprocessing speed, not foundation-model acceleration.',
                          'F changes warnings only; depth arrays remain RAW.',
                          'No per-transition statistical significance; formal comparison needs independent clips.',
                          'A/D/E use Farneback for fusion; scoring with Farneback is not an independent-flow crosscheck.'],
            'evaluation_flow':evaluation_flow,'application_roi':roi,'warning_threshold_m':warning_threshold}
    write_csv(output/'comparison.csv',rows)
    write_json(output/'comparison.json',report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--manifest')
    p.add_argument('--calibration')
    p.add_argument('--methods',nargs='+',choices=ALL_METHODS,default=list(METHODS))
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--evaluation-flow',choices=['farneback','raft'],default='farneback')
    p.add_argument('--flow-checkpoint')
    p.add_argument('--application-roi',nargs=4,type=int)
    p.add_argument('--warning-threshold',type=float,default=1.5)
    a=p.parse_args(argv)
    report=compare(a.run,a.output,a.manifest,a.calibration,a.methods,a.repeats,a.evaluation_flow,
                   a.flow_checkpoint,a.application_roi,a.warning_threshold)
    print(json.dumps({'synthetic':report['synthetic'],'winners':report['winners'],'skipped':report['skipped']},indent=2))


if __name__=='__main__':
    main()
