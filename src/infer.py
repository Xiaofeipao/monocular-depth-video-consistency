"""python -m src.infer: image, image manifest, or offline video inference."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .io import RunWriter,read_bgr,read_frames,sha256
from .models import DepthModel
from .visualization import render_run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,help="Image, video, frame directory or dataset JSON manifest")
    p.add_argument("--output",required=True,help="New run directory (must not exist)")
    p.add_argument("--model",required=True,choices=["dav2","depth-pro","vda"])
    p.add_argument("--repo",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--encoder",choices=["vits","vitb"],default="vits")
    p.add_argument("--metric",action="store_true",help="Use VDA metric checkpoint")
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--fp16",action="store_true")
    p.add_argument("--input-size",type=int,default=518)
    p.add_argument("--fps",type=float)
    p.add_argument("--max-frames",type=int)
    p.add_argument("--max-side",type=int,help="Shared external resize for ALL compared models")
    p.add_argument("--vis-min",type=float,required=True)
    p.add_argument("--vis-max",type=float,required=True)
    p.add_argument("--warmup",type=int,default=0,help="Extra image predictions before timed inference")
    a = p.parse_args(argv)
    if a.vis_max <= a.vis_min or a.warmup < 0:
        p.error("Invalid visualization range or warmup")
    if Path(a.output).exists():
        p.error("Output exists; choose a new run ID")
    source = Path(a.input)
    start = time.perf_counter()
    manifest = None
    if source.suffix.lower() == ".json":
        manifest = json.loads(source.read_text(encoding="utf-8"))
        frames = [read_bgr(source.parent/r["rgb"]) for r in manifest["samples"]]
        indices,fps = list(range(len(frames))),None
    elif source.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
        frames,indices,fps = [read_bgr(source)],[0],None
    else:
        frames,indices,fps = read_frames(source,a.fps,a.max_frames,a.max_side)
    if not frames:
        p.error("No samples")
    if manifest and (a.max_side or a.fps or a.max_frames):
        p.error("Manifest images are uncropped and unresized; sampling flags apply to videos only")
    model = DepthModel(a.model,a.repo,a.checkpoint,a.device,a.encoder,a.metric,a.input_size,a.fp16)
    if a.model == "vda" and fps is None:
        p.error("VDA requires video or frame directory with --fps")
    decode_load_seconds = time.perf_counter()-start
    for _ in range(a.warmup):
        if a.model == "vda":
            model.video(frames[:min(8,len(frames))],fps)
        else:
            model.image(frames[0])
    model.synchronize()
    if a.device == "cuda":
        model.torch.cuda.reset_peak_memory_stats()
    metadata = {**model.metadata,"fps":fps,"frame_indices":indices,
                "visualization_range":[a.vis_min,a.vis_max],"input_manifest_sha256":sha256(source) if source.is_file() else None,
                "inference_mode":"offline" if a.model == "vda" else "framewise"}
    writer = RunWriter(a.output,metadata,vars(a))
    inference_times = []
    start_pipeline = time.perf_counter()
    if a.model == "vda":
        begin = time.perf_counter()
        depths = model.video(frames,fps)
        model.synchronize()
        inference_times = [(time.perf_counter()-begin)/len(frames)]
        predictions = ((d,{}) for d in depths)
    else:
        def predict():
            for frame in frames:
                model.synchronize()
                begin = time.perf_counter()
                result = model.image(frame)
                model.synchronize()
                inference_times.append(time.perf_counter()-begin)
                yield result
        predictions = predict()
    for i,(depth,extra) in enumerate(predictions):
        if depth.shape != frames[i].shape[:2] or not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError("Model smoke check failed: dimensions, NaN/Inf or negative output")
        # Relative heads can return zero; preserve raw outputs and their valid masks.
        extra["sample_id"] = manifest["samples"][i]["id"] if manifest else f"{indices[i]:06d}"
        writer.add(depth,frames[i],source,indices[i],extra)
    model_seconds = sum(inference_times)*(len(frames) if a.model == "vda" else 1)
    render_run(writer.path,writer.records,a.vis_min,a.vis_max,model.metadata["unit"],fps,a.model)
    elapsed = decode_load_seconds+time.perf_counter()-start_pipeline
    timing = {"model_seconds":model_seconds,"model_fps":len(frames)/model_seconds,
              "model_latency_ms":float(1000*np.mean(inference_times)),
              "p50_ms":float(1000*np.quantile(inference_times,.5)),"p95_ms":float(1000*np.quantile(inference_times,.95)),
              "latency_kind":"offline_amortized" if a.model == "vda" else "per_frame",
              "end_to_end_seconds":elapsed,"end_to_end_fps":len(frames)/elapsed,
              "peak_gpu_bytes":int(model.torch.cuda.max_memory_allocated()) if a.device == "cuda" else None,
              "frames":len(frames),"warmup":a.warmup,"includes_model_load":True}
    writer.finish(timing)
    print(f"Saved {len(frames)} raw predictions to {writer.path}")


if __name__ == "__main__":
    main()
