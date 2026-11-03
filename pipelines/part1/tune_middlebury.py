#!/usr/bin/env python3
"""在五个固定开发场景搜索BM/SGBM，冻结MAE与有效比例约束下的选择。"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
from datetime import datetime,timezone

import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.stereo.io import load_middlebury_scene
from src.stereo.matching import StereoMatcherConfig,match_stereo
from src.stereo.geometry import middlebury_disparity_to_depth
from src.stereo.metrics import compute_disparity_metrics
from src.stereo.protocol import (PROTOCOL,load_config,sha256,source_hashes,data_hashes,
                                 environment,write_json,choose_candidate,archive_protocol)


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def measure(parameters,method,scenes):
    values=dict(parameters);lr=values.pop('lr_threshold_px',1.)
    matcher=StereoMatcherConfig(method=method,**values);rows=[]
    for scene in scenes:
        result=match_stereo(scene.left_rgb,scene.right_rgb,
                            calibration_num_disparities=scene.calibration.num_disparities,
                            config=matcher,lr_threshold_px=lr)
        _, geometry_valid=middlebury_disparity_to_depth(result.left_disparity_px,
            focal_length_px=scene.calibration.focal_length_px,
            baseline_mm=scene.calibration.baseline_mm,doffs_px=scene.calibration.doffs_px)
        rows.append({'scene':scene.name,**compute_disparity_metrics(result.left_disparity_px,
            scene.gt_disparity_px,prediction_valid_mask=result.valid_mask & geometry_valid)})
    means={key:float(np.mean([r[key] for r in rows])) if all(r[key] is not None for r in rows) else None
           for key in ('disparity_mae_px','valid_prediction_fraction')}
    return means,rows


def tune_method(*,method,config,scenes,output_dir):
    if [s.name for s in scenes]!=config['split']['development_scenes']:
        raise ValueError('Search can only use the declared development scenes in order')
    directory=Path(output_dir)/method;directory.mkdir(parents=True,exist_ok=False)
    status=directory/'status.json'
    write_json(status,{'status':'running','command':sys.argv,'started_at_utc':datetime.now(timezone.utc).isoformat()})
    try:
        sources=archive_protocol(directory/'provenance',config)
        inputs=data_hashes(config,config['split']['development_scenes'])
        tuning=config['tuning'][method];names=list(tuning['search'])
        baseline,baseline_rows=measure(config['experiments'][method+'_fixed']['parameters'],method,scenes)
        rows=[];per_scene=[]
        combinations=list(itertools.product(*(tuning['search'][n] for n in names)))
        for i,combination in enumerate(combinations):
            candidate=dict(zip(names,combination));params={**tuning['fixed'],**candidate,'lr_threshold_px':1.}
            score,details=measure(params,method,scenes)
            rows.append({**candidate,**score})
            per_scene.extend({'candidate_index':i,**candidate,**r} for r in details)
            print(f'[{method} {i+1}/{len(combinations)}] {candidate} {score}',flush=True)
        best=choose_candidate(rows,baseline,names)
        write_csv(directory/'candidates.csv',rows);write_csv(directory/'metrics_per_scene.csv',per_scene)
        write_csv(directory/'fixed_baseline_per_scene.csv',baseline_rows)
        if sources!=source_hashes() or inputs!=data_hashes(config,config['split']['development_scenes']):
            raise ValueError('Source or development input changed during the search')
        record={'protocol':PROTOCOL,'method':method,'status':'complete','command':sys.argv,
                'created_at_utc':datetime.now(timezone.utc).isoformat(),'config':config,
                'development_scenes':config['split']['development_scenes'],
                'candidate_count':len(rows),'fixed_baseline':baseline,'best':best,
                'selected_parameters':{**tuning['fixed'],**{n:best[n] for n in names},'lr_threshold_px':1.},
                'selection_rule':config['tuning']['objective'],'source_sha256':sources,
                'input_sha256':inputs,
                'environment':environment(),'formal_training_run':False,
                'artifacts':{str(p.relative_to(directory)):sha256(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='status.json'}}
        write_json(directory/'best.json',record)
        write_json(status,{'status':'complete','best_sha256':sha256(directory/'best.json')})
        return record
    except BaseException as exc:
        write_json(status,{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/part1/middlebury_sgbm.yaml')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--methods',nargs='+',choices=('bm','sgbm'),default=['bm','sgbm'])
    args=parser.parse_args(argv);config=load_config(args.config)
    if len(set(args.methods))!=len(args.methods):raise ValueError('Repeated method')
    output=(args.output_dir or ROOT/config['outputs']['root']/config['outputs']['tuning_run']).resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'outputs'):raise ValueError('Outputs must be under outputs/')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'status.json',{'status':'running','command':sys.argv,'config_sha256':sha256(args.config)})
    try:
        scenes=[load_middlebury_scene(ROOT/config['dataset']['root']/name) for name in config['split']['development_scenes']]
        records={m:tune_method(method=m,config=config,scenes=scenes,output_dir=output) for m in args.methods}
        write_json(output/'summary.json',{m:{'selected_parameters':r['selected_parameters'],'best':r['best']} for m,r in records.items()})
        write_json(output/'status.json',{'status':'complete','summary_sha256':sha256(output/'summary.json')})
    except BaseException as exc:
        write_json(output/'status.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise


if __name__=='__main__':main()
