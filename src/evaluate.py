"""Evaluate a frozen prediction run against an explicit dataset protocol."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .io import load_run,load_depth,read_bgr,sha256,write_csv,write_json
from .metrics import depth_metrics,boundary_metrics,summarize


def evaluate_run(run,manifest,alignment="none",clip=None,bootstrap=1000):
    run,manifest = Path(run),Path(manifest)
    meta,records = load_run(run)
    dataset = json.loads(manifest.read_text(encoding="utf-8"))
    protocol = dataset["protocol"]
    if clip is None:
        clip=bool(protocol.get('clip_predictions',False))
    by_id = {r["sample_id"]:r for r in records}
    ids = [r["id"] for r in dataset["samples"]]
    if len(by_id) != len(records) or len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError("Prediction/sample IDs must match the entire frozen split exactly")
    rows = []
    for sample in dataset["samples"]:
        record = by_id[sample["id"]]
        if sha256(run/"depth"/(record["id"]+".npy")) != record["depth_sha256"]:
            raise ValueError("Saved prediction hash changed after the run was frozen")
        saved_rgb = read_bgr(run/"rgb"/(record["id"]+".png"))
        input_rgb = read_bgr(manifest.parent/sample["rgb"])
        if not np.array_equal(saved_rgb,input_rgb):
            raise ValueError(f"Sample ID matches but RGB content differs: {sample['id']}")
        pred = load_depth(run/"depth"/(record["id"]+".npy"))
        gt = load_depth(manifest.parent/sample["depth"],sample.get("depth_scale",1.))
        if pred.shape != gt.shape:
            pred = cv2.resize(pred,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR)
        mask = load_depth(manifest.parent/sample["mask"]) > 0 if sample.get("mask") else None
        scores,aligned,valid = depth_metrics(pred,gt,minimum=protocol["min_depth"],maximum=protocol["max_depth"],
            mask=mask,crop=protocol.get("crop"),alignment=alignment,depth_type=meta["depth_type"],
            representation=meta["representation"],clip=clip,bounds=protocol.get('depth_bounds','inclusive'))
        scores.update(boundary_metrics(aligned,gt,valid,protocol.get("edge_threshold",.08),protocol.get("edge_tolerance",2)))
        rows.append({"sample_id":sample["id"],**scores})
    tag = "native" if alignment == "none" else f"aligned_{alignment}"
    if clip:
        tag += "_clipped"
    # Evaluation does not change frozen depth/metadata.
    destination = run/"evaluation"/tag
    destination.mkdir(parents=True,exist_ok=False)
    write_csv(destination/"metrics_per_sample.csv",rows)
    write_json(destination/"metrics_summary.json",{
        "depth_type":meta["depth_type"],"alignment":alignment,"clip_predictions":clip,
        "dataset_sha256":sha256(manifest),"protocol":protocol,
        "aggregation":"per_image_macro_mean","boundary_metric":"log_depth_jump_tolerant_edges (not Depth Pro SI F1)",
        "bootstrap_samples":bootstrap,"metrics":summarize(rows,bootstrap=bootstrap)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--alignment",choices=["none","median","scale_shift"],default="none")
    p.add_argument("--clip-predictions",action="store_true",default=None,help="By default follow manifest protocol; enable only if supplied protocol requires clipping")
    a = p.parse_args(argv)
    rows = evaluate_run(a.run,a.manifest,a.alignment,a.clip_predictions)
    print(f"Evaluated {len(rows)} samples")


if __name__ == "__main__":
    main()
