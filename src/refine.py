"""Part 3: read frozen metric predictions, never rerun or modify the model."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from .flow import FarnebackFlow,RAFTFlow
from .io import RunWriter,load_run,read_bgr,sha256
from .temporal import FusionConfig,TemporalFusion
from .visualization import render_run


def refine_run(run,output,config,flow_backend="farneback",device="cpu",flow_checkpoint=None,cache=None):
    run = Path(run)
    if Path(output).resolve() == run.resolve():
        raise ValueError("Refinement must use a new output directory")
    meta,records = load_run(run)
    if meta["depth_type"] != "metric" or meta["representation"] != "depth":
        raise ValueError("Fusion currently accepts metric depth only; relative inverse-depth needs a frozen calibration")
    if len(records)<2 or meta.get("fps") is None:
        raise ValueError("Refinement requires an ordered video run")
    flow = FarnebackFlow() if flow_backend == "farneback" else RAFTFlow(device,flow_checkpoint)
    metadata = {**meta,"model":f"{meta['model']}+A{config.ablation}","parent_manifest_sha256":sha256(run/"manifest.json"),
                "fusion":asdict(config),"flow_backend":flow.name,
                "flow_checkpoint_sha256":sha256(flow_checkpoint) if flow_checkpoint else None}
    writer = RunWriter(output,metadata,{"source_run":str(run),"fusion":asdict(config)})
    fusion = TemporalFusion(config)
    previous = None
    start = time.perf_counter()
    for i,record in enumerate(records):
        key = record["id"]
        rgb = read_bgr(run/"rgb"/f"{key}.png")
        depth = np.load(run/"depth"/f"{key}.npy",allow_pickle=False)
        forward = backward = None
        if previous is not None and config.ablation >= 2:
            cache_key = (sha256(run/"rgb"/f"{records[i-1]['id']}.png"),sha256(run/"rgb"/f"{key}.png"),
                         flow.name,metadata['flow_checkpoint_sha256'])
            if cache is not None and cache_key in cache:
                forward,backward = cache[cache_key]
            else:
                forward,backward = flow(previous,rgb)
                if cache is not None:
                    cache[cache_key] = forward,backward
        result,debug = fusion.step(depth,rgb,forward,backward)
        writer.add(result,rgb,record["source"],record["source_index"],
                   {k:record[k] for k in ("sample_id","focal_px","intrinsics_source") if k in record})
        debug_dir = writer.path/"diagnostics"
        debug_dir.mkdir(exist_ok=True)
        np.savez_compressed(debug_dir/f"{key}.npz",**debug)
        previous = rgb
    elapsed = time.perf_counter()-start
    render_run(writer.path,writer.records,*meta["visualization_range"],meta["unit"],meta["fps"],metadata["model"])
    writer.finish({"postprocess_seconds":elapsed,"postprocess_fps":len(records)/elapsed,
                   "frames":len(records),"resets":fusion.resets,"flow_cache_reused":cache is not None,
                   "includes_model_inference":False,"includes_visualization":False})
    return writer.path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--ablation",type=int,choices=range(7),default=6)
    p.add_argument("--alpha",type=float,default=.4)
    p.add_argument("--flow",choices=["farneback","raft"],default="farneback")
    p.add_argument("--flow-checkpoint")
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--domain",choices=["depth","inverse"],default="inverse")
    a = p.parse_args(argv)
    refine_run(a.run,a.output,FusionConfig(ablation=a.ablation,alpha=a.alpha,domain=a.domain),a.flow,a.device,a.flow_checkpoint)


if __name__ == "__main__":
    main()
