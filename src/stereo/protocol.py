"""Part 1A固定基线协议与来源绑定。"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import platform
import sys
import os
import shutil

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'middlebury_fixed_v1'
EXPERIMENTS = ('bm_fixed','sgbm_fixed')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_config(path):
    config=yaml.safe_load(Path(path).read_text())
    names=config['split']['development_scenes']+config['split']['evaluation_scenes']
    if config['protocol_status'] != PROTOCOL or len(names)!=config['dataset']['expected_scenes'] or len(set(names))!=len(names):
        raise ValueError('Invalid course protocol or scene split')
    if len(names)!=15 or len(config['split']['development_scenes'])!=5 or set(config['experiments'])!=set(EXPERIMENTS):
        raise ValueError('Expected two fixed experiments, five development and ten held-out scenes')
    ev=config['evaluation']
    if (ev['mask_name']!='finite_positive_gt' or ev['aggregation']!='per_scene_macro_mean'
            or ev['primary_metrics']!=['disparity_mae_px','valid_prediction_fraction']
            or config['geometry']['opencv_disparity_scale']!=16.):
        raise ValueError('Disparity protocol differs from the declared course evaluation')
    if (ev.get('auxiliary_metrics') != ['bad1','bad2','bad4','depth_abs_rel','depth_rmse_m']
            or ev.get('bad_pixel_rule') != 'absolute_error_strictly_greater_than_threshold; invalid_counts_as_bad; full_gt_denominator'):
        raise ValueError('Auxiliary metric definitions differ')
    if (config['geometry']['calibration_baseline_unit'] != 'millimeter'
            or config['geometry']['output_depth_unit'] != 'meter'
            or config['geometry']['formula'] != 'z_m = focal_length_px * baseline_mm / (disparity_px + doffs_px) / 1000'
            or config['development_search']['lr_threshold_px'] != 1.
            or config['dataset']['resolution'] != 'quarter'):
        raise ValueError('Calibration units, resolution or fixed LR rule differs')
    for method in ('bm','sgbm'):
        if config['experiments'][method+'_fixed']['parameters']['lr_threshold_px']!=1.:
            raise ValueError('LR threshold must be fixed at one pixel')
        reference = config['experiments'][method+'_fixed']['parameters']
        tuning = config['development_search'][method]
        if any(not values or reference[name] not in values or len(values)!=len(set(values))
               for name,values in tuning['search'].items()):
            raise ValueError('Candidate grid must be unique and include the fixed reference')
        if any(reference[name] != value for name,value in tuning['fixed'].items()):
            raise ValueError('Fixed and searched experiments must share unsearched parameters')
    return config


def source_hashes():
    files=list((ROOT/'src/stereo').glob('*.py'))+[ROOT/'src/visualization/stereo.py',
        ROOT/'pipelines/part1/run_middlebury.py',ROOT/'scripts/part1/run_middlebury.sh']
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(files)}


def archive_protocol(directory, config):
    """Save the actual scientific implementation and parsed protocol, without inputs."""
    directory = Path(directory)
    write_json(directory/'config.json', config)
    hashes = source_hashes()
    for relative in hashes:
        target = directory/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
    return hashes


def data_hashes(config,names):
    base=ROOT/config['dataset']['root']
    return {str(Path(name)/file):sha256(base/name/file) for name in names
            for file in ('im0.png','im1.png','disp0GT.pfm','calib.txt','mask0nocc.png')}


def environment():
    cpu=platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')),cpu)
    return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'numpy':np.__version__,'opencv':cv2.__version__,'opencv_threads':cv2.getNumThreads(),
            'device':'cpu','cpu_model':cpu,'logical_cpu_count':os.cpu_count(),'gpu_used':False,'peak_gpu_memory_bytes':None,
            'timing_scope':'two OpenCV matcher.compute calls; excludes decoding, LR check, geometry and plotting'}


