"""Flow-aligned video proxies on a frozen RGB sequence; never per-frame align."""
import argparse
from pathlib import Path

import numpy as np

from .flow import FarnebackFlow,RAFTFlow
from .io import load_run,read_bgr,write_json,write_csv
from .metrics import temporal_metrics,summarize


def evaluate_video(run,flow_backend="farneback",checkpoint=None,device="cpu",static_roi=None,cache=None):
    run = Path(run)
    meta,records = load_run(run)
    if len(records)<2 or meta.get("fps") is None:
        raise ValueError("Video evaluation needs at least two ordered frames and FPS")
    flow = FarnebackFlow() if flow_backend == "farneback" else RAFTFlow(device,checkpoint)
    previous = rgb_previous = None
    rows,roi_values = [],[]
    cumulative = 0.
    for i,record in enumerate(records):
        key = record['id']
        rgb = read_bgr(run/"rgb"/f"{key}.png")
        raw = np.load(run/"depth"/f"{key}.npy",allow_pickle=False)
        valid = np.isfinite(raw)&(raw>0)
        depth = np.full(raw.shape,np.nan,np.float32)
        if meta["representation"] == "inverse_depth":
            np.divide(1,raw,out=depth,where=valid)
        else:
            depth[valid] = raw[valid]
        if static_roi:
            top,bottom,left,right = static_roi
            if not (0<=top<bottom<=depth.shape[0] and 0<=left<right<=depth.shape[1]):
                raise ValueError("Invalid static ROI")
            region = depth[top:bottom,left:right]
            roi_values.append(region.copy())
        if previous is not None:
            cache_key = (records[i-1]['rgb_sha256'],record['rgb_sha256'],flow.name,checkpoint)
            if cache is not None and cache_key in cache:
                forward,backward = cache[cache_key]
            else:
                forward,backward = flow(rgb_previous,rgb)
                if cache is not None:
                    cache[cache_key] = forward,backward
            result = temporal_metrics(previous,depth,forward,backward)
            if result['log_scale_ratio'] is not None:
                cumulative += result['log_scale_ratio']
            rows.append({"frame_id":key,**result,"cumulative_log_scale_proxy":cumulative})
        previous,rgb_previous = depth,rgb
    suffix = flow.name
    if static_roi:
        suffix += "_roi_"+"_".join(map(str,static_roi))
    destination = run/"evaluation"/suffix
    destination.mkdir(parents=True,exist_ok=False)
    write_csv(destination/"temporal_per_frame.csv",rows)
    descriptive = summarize(rows)
    # Adjacent transitions are correlated. Per-transition bootstrap intervals
    # would suggest unjustified statistical confidence from a single clip.
    for item in descriptive.values():
        item.pop('ci95',None)
    summary = {"flow_backend":flow.name,"flow_checkpoint":str(checkpoint) if checkpoint else None,
               "interpretation":"flow-aligned proxy, not pose-compensated geometry; scale change may be real camera motion",
               "depth_type":meta['depth_type'],"representation":meta['representation'],
               "aggregation":"descriptive transition means; confidence intervals require independent clips",
               "metrics":descriptive,"signed_cumulative_log_scale_proxy":cumulative,
               "absolute_log_scale_variation":float(sum(abs(r['log_scale_ratio']) for r in rows if r['log_scale_ratio'] is not None))}
    if static_roi:
        stack = np.stack(roi_values)
        # Only pixels valid in EVERY frame contribute; do not hide failed pixels.
        shared = np.isfinite(stack).all(axis=0)
        if not shared.any():
            raise ValueError("No static ROI pixels valid throughout the clip")
        values = stack[:,shared]
        summary['static_roi'] = static_roi
        summary['static_pixel_std_mean'] = float(values.std(axis=0).mean())
        medians = np.median(values,axis=1)
        summary['static_median_depth_cv'] = float(medians.std()/medians.mean())
        summary['static_valid_coverage'] = float(shared.mean())
    write_json(destination/"temporal_summary.json",summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    p.add_argument("--flow",choices=["farneback","raft"],default="farneback")
    p.add_argument("--flow-checkpoint")
    p.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    p.add_argument("--static-roi",type=int,nargs=4,metavar=("TOP","BOTTOM","LEFT","RIGHT"))
    a = p.parse_args(argv)
    evaluate_video(a.run,a.flow,a.flow_checkpoint,a.device,a.static_roi)


if __name__ == "__main__":
    main()
