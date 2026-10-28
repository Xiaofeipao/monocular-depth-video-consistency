"""Part 1: run all Middlebury scenes (or a single rectified scene)."""
import argparse
from pathlib import Path
import time

import numpy as np

from .geometry import read_middlebury_calibration,read_pfm,stereo_depth,disparity_to_depth
from .io import RunWriter,read_bgr,write_image,write_csv,write_json
from .metrics import depth_metrics,summarize
from .visualization import render_run


def sensitivity(focal,baseline,doffs,distances=None):
    distances = np.linspace(.5,50,100) if distances is None else distances
    rows = []
    for z in distances:
        disparity = focal*baseline/z-doffs
        if disparity <= 0:
            continue
        for perturbation in (-2.,-1.,-.5,-.25,.25,.5,1.,2.):
            measured = float(disparity_to_depth(np.array(disparity+perturbation),focal,baseline,doffs))
            rows.append({"depth_m":float(z),"disparity_px":float(disparity),"perturbation_px":perturbation,
                "exact_error_m":abs(measured-float(z)) if np.isfinite(measured) else None,
                "first_order_error_m":float(z*z/(focal*baseline)*abs(perturbation))})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,help="Scene folder or root containing im0.png, im1.png, calib.txt")
    p.add_argument("--output",required=True)
    p.add_argument("--method",choices=["bm","sgbm"],default="sgbm")
    p.add_argument("--num-disparities",type=int,help="Default: use scene calibration ndisp rounded to 16")
    p.add_argument("--block-size",type=int,default=5)
    p.add_argument("--vis-max",type=float,default=10)
    a = p.parse_args(argv)
    root = Path(a.input)
    scenes = sorted({x.parent for x in root.rglob("im0.png")})
    if not scenes:
        p.error("No Middlebury scenes found")
    writer = RunWriter(a.output,{"model":a.method,"depth_type":"metric","unit":"meter",
        "representation":"depth","fps":None,"visualization_range":[0,a.vis_max]},vars(a))
    rows = []
    start = time.perf_counter()
    for scene in scenes:
        calibration = read_middlebury_calibration(scene/"calib.txt")
        ndisp = a.num_disparities or ((calibration['ndisp']+15)//16*16)
        left,right = read_bgr(scene/"im0.png"),read_bgr(scene/"im1.png")
        disp,depth,valid = stereo_depth(left,right,calibration['focal_px'],calibration['baseline_m'],calibration['doffs'],
            a.method,ndisp,a.block_size)
        writer.add(depth,left,scene,None,{"sample_id":scene.name,"calibration":calibration})
        key = writer.records[-1]['id']
        np.save(writer.path/"depth"/f"{key}_disparity.npy",disp)
        write_image(writer.path/"visualization"/f"{key}_invalid.png",(~valid).astype(np.uint8)*255)
        write_csv(writer.path/"visualization"/f"{key}_sensitivity.csv",sensitivity(calibration['focal_px'],calibration['baseline_m'],calibration['doffs']))
        if (scene/"disp0GT.pfm").exists():
            gt_disp = read_pfm(scene/"disp0GT.pfm")
            gt_valid = np.isfinite(gt_disp) & (gt_disp>0)
            if (scene/"mask0nocc.png").exists():
                gt_valid &= read_bgr(scene/"mask0nocc.png")[...,0] > 0
            matched = gt_valid & valid
            if not matched.any() or not gt_valid.any():
                raise ValueError(f"No evaluated pixels in {scene}")
            error = np.abs(disp[matched]-gt_disp[matched])
            gt_depth = disparity_to_depth(gt_disp,calibration['focal_px'],calibration['baseline_m'],calibration['doffs'])
            scores,_,_ = depth_metrics(depth,gt_depth,minimum=1e-6,maximum=1e6,mask=matched)
            rows.append({"sample_id":scene.name,"coverage":float(matched.sum()/gt_valid.sum()),
                "disparity_MAE":float(error.mean()),"disparity_RMSE":float(np.sqrt(np.mean(error**2))),
                **{f"bad{n}_matched":float(np.mean(error>n)) for n in (1,2,4)},
                # Rejected predictions count as failures in this full-mask score.
                **{f"bad{n}_full":float(((~valid & gt_valid).sum()+np.sum(error>n))/gt_valid.sum()) for n in (1,2,4)},
                **scores})
    elapsed = time.perf_counter()-start
    writer.finish({"seconds":elapsed,"frames":len(scenes),"fps":len(scenes)/elapsed})
    render_run(writer.path,writer.records,0,a.vis_max,"meter",label=a.method)
    if rows:
        write_csv(writer.path/"metrics_per_sample.csv",rows)
        write_json(writer.path/"metrics_summary.json",{"protocol":"depth/disparity errors on matched pixels, coverage and full-mask bad-pixel rates also reported",
            "metrics":summarize(rows)})
    print(f"Processed {len(scenes)} scenes")


if __name__ == "__main__":
    main()
