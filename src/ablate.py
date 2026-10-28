"""Run A0-A6 using identical raw predictions and cached optical flow."""
import argparse
from pathlib import Path

from .evaluate_video import evaluate_video
from .evaluate import evaluate_run
from .io import write_json
from .refine import refine_run
from .temporal import FusionConfig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--alpha",type=float,default=.4)
    p.add_argument("--manifest",help="Optional GT manifest for this exact video sequence")
    p.add_argument("--fusion-flow",choices=["farneback","raft"],default="farneback")
    p.add_argument("--evaluation-flow",choices=["farneback","raft"],default="farneback")
    p.add_argument("--flow-checkpoint")
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--static-roi",type=int,nargs=4)
    a = p.parse_args(argv)
    root = Path(a.output)
    root.mkdir(parents=True,exist_ok=False)
    fusion_cache,evaluation_cache = {},{}
    results = {}
    for i in range(7):
        destination = root/f"A{i}"
        refine_run(a.run,destination,FusionConfig(ablation=i,alpha=a.alpha),a.fusion_flow,
                   a.device,a.flow_checkpoint if a.fusion_flow == "raft" else None,fusion_cache)
        summary = evaluate_video(destination,a.evaluation_flow,
            a.flow_checkpoint if a.evaluation_flow == "raft" else None,a.device,a.static_roi,evaluation_cache)
        if a.manifest:
            evaluate_run(destination,a.manifest)
        results[f"A{i}"] = summary
    write_json(root/"ablation_summary.json",{
        "fusion_flow":a.fusion_flow,"evaluation_flow":a.evaluation_flow,
        "independent_flow_crosscheck":a.fusion_flow != a.evaluation_flow,
        "static_roi_crosscheck":a.static_roi,"results":results})


if __name__ == "__main__":
    main()
