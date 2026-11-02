"""Numerical regression against the project's frozen pre-study Git source."""
import hashlib
import subprocess
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.exploration.uncertainty import UncertaintyFusion
from src.metrics.pose import pose_temporal_metrics
from src.io import sha256,write_json

BASE='29c81327c95d638626cf5ed1d435b69a5f640295'


def legacy(path,package):
    source=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT,text=True)
    scope={'__name__':package+'.legacy','__package__':package}
    exec(compile(source,path,'exec'),scope)
    return scope,hashlib.sha256(source.encode()).hexdigest()


def main():
    fusion,old_fusion_sha=legacy('src/exploration/uncertainty.py','src.exploration')
    pose,old_pose_sha=legacy('src/metrics/pose.py','src.metrics')
    rng=np.random.default_rng(15)
    for config in ({},{'alpha_max':.25,'max_log_update':.01,'edge_threshold':.08}):
        old=fusion['UncertaintyFusion'](**config); new=UncertaintyFusion(**config)
        for i in range(8):
            z=(2*np.exp(rng.normal(0,.02,(24,32)))).astype(np.float32)
            rgb=np.full((24,32,3),100 if i<5 else 250,np.uint8)
            flow=np.zeros((24,32,2),np.float32)
            a,_=old.step(z,rgb,flow,flow); b,_=new.step(z,rgb,flow,flow)
            np.testing.assert_array_equal(a,b)
    z=np.linspace(1,3,24*32).reshape(24,32); z[2:4,5:7]=np.nan
    k=dict(fx=30,fy=30,cx=15.5,cy=11.5)
    for tx in (0,.02,.1):
        args=(np.nan_to_num(z,nan=2)*1.03,np.nan_to_num(z,nan=2)*1.04,z,z,
              [0,0,0,0,0,0,1],[tx,0,0,0,0,0,1],k)
        assert pose_temporal_metrics(*args)==pose['pose_temporal_metrics'](*args)
    write_json(ROOT/'docs/evidence/d_research/legacy_parity.json',dict(all_passed=True,
        baseline_git_sha=BASE,seed=15,fusion_frames=16,pose_cases=3,
        old_normalized_source_sha256=dict(fusion=old_fusion_sha,pose=old_pose_sha),
        current_source_sha256={p:sha256(ROOT/p) for p in ('src/exploration/uncertainty.py','src/metrics/pose.py')},
        scope='Exact outputs on deterministic regression cases, not proof for all possible inputs'))
    print('Legacy parity passed: 16 fusion frames + 3 pose cases')


if __name__=='__main__': main()
