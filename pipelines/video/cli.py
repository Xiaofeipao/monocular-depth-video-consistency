#!/usr/bin/env python3
"""Shared Part 1B/2B command line; all scientific settings are frozen first."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.common import environment_info, read_json, sha256_file, write_json
from src.video.runner import (
    cache_flow, check_resources, freeze_protocol, infer_run, load_config,
    load_protocol, validate_config, verify_run,
)


def _config(args):
    config = deepcopy(load_config(args.config))
    overrides = {"device": ("runtime", "device"), "precision": ("runtime", "precision"),
                 "torch_threads": ("runtime", "torch_threads"),
                 "input_size": ("inference", "input_size"),
                 "warmup": ("inference", "warmup"),
                 "timing_repeats": ("inference", "timing_repeats"),
                 "flow_backend": ("flow", "backend")}
    for attr, (section, key) in overrides.items():
        if getattr(args, attr, None) is not None:
            config[section][key] = getattr(args, attr)
    return validate_config(config)


def _config_arguments(parser):
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/video/relative_video.yaml")
    parser.add_argument("--device", help="Explicit cpu or cuda:N; no automatic fallback")
    parser.add_argument("--precision", choices=["float16", "float32"])
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--timing-repeats", type=int)
    parser.add_argument("--flow-backend", choices=["raft_small", "farneback"])


def build_parser():
    parser = argparse.ArgumentParser(description="Part 1B/2B: shared frames → frozen protocol → relative models → paired evaluation")
    subs = parser.add_subparsers(dest="command", required=True)
    check = subs.add_parser("check", help="Check local resources and actual runtime without inference")
    _config_arguments(check)
    check.add_argument("--output", type=Path)
    prepare = subs.add_parser("prepare", help="Freeze shared RGB frames with exact source timestamps")
    prepare.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/video/relative_video.yaml")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--public-clip", choices=["tokyo_walk", "rollercoaster"])
    source.add_argument("--source", type=Path)
    prepare.add_argument("--clip-id")
    prepare.add_argument("--source-kind", choices=["public", "self_recorded", "mandatory"], default="public")
    prepare.add_argument("--camera-motion", choices=["fixed", "moving", "unknown"], default="unknown")
    prepare.add_argument("--split", choices=["development", "test", "mandatory"], default="development")
    prepare.add_argument("--session-id")
    prepare.add_argument("--source-url")
    prepare.add_argument("--provenance-json", type=Path)
    prepare.add_argument("--target-fps", type=float)
    prepare.add_argument("--max-long-edge", type=int)
    prepare.add_argument("--max-frames", type=int)
    prepare.add_argument("--start-seconds", type=float, default=0.)
    prepare.add_argument("--end-seconds", type=float)
    prepare.add_argument("--background-roi", type=Path)
    prepare.add_argument("--output-dir", type=Path, required=True)
    freeze = subs.add_parser("freeze", help="Snapshot protocol, resources, shared manifest and scientific source")
    _config_arguments(freeze)
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    infer = subs.add_parser("infer", help="Run one frozen relative model, saving all raw float32 outputs")
    infer.add_argument("--model", choices=["dav2", "vda"], required=True)
    infer.add_argument("--protocol", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    flow = subs.add_parser("flow", help="Compute and save shared bidirectional RGB optical flow")
    flow.add_argument("--protocol", type=Path, required=True)
    flow.add_argument("--output-dir", type=Path, required=True)
    evaluate = subs.add_parser("evaluate", help="Paired stability, drift, coverage, figures and fixed-color videos")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--dav2-run", type=Path, required=True)
    evaluate.add_argument("--vda-run", type=Path, required=True)
    evaluate.add_argument("--flow-cache", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    verify = subs.add_parser("verify", help="Reload raw results; no model inference")
    which = verify.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", type=Path)
    which.add_argument("--evaluation", type=Path)
    which.add_argument("--protocol", type=Path)
    verify.add_argument("--output", type=Path)
    aggregate = subs.add_parser("summarize", help="Equal-clip summary; keep public/self-recorded and dev/test separate")
    aggregate.add_argument("--evaluations", nargs="+", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser


def _prepare(args):
    from src.video.frames import prepare_frames
    import yaml
    config = load_config(args.config)
    provenance = read_json(args.provenance_json) if args.provenance_json else {}
    if args.public_clip:
        inventory = yaml.safe_load((PROJECT_ROOT / "configs/video/public_clips.yaml").read_text())
        clip = inventory["clips"][args.public_clip]
        source = PROJECT_ROOT / clip["path"]
        if sha256_file(source) != clip["sha256"]:
            raise ValueError("Public video checksum mismatch; do not silently replace it")
        clip_id = args.clip_id or args.public_clip
        source_kind, camera_motion, split = clip["source_kind"], clip["camera_motion"], clip["split"]
        provenance.update(clip)
    else:
        source, clip_id = args.source, args.clip_id
        if not clip_id:
            raise ValueError("--clip-id is required for a custom source")
        source_kind, camera_motion, split = args.source_kind, args.camera_motion, args.split
        if source_kind == "self_recorded" and not args.session_id:
            raise ValueError("Self-recorded input requires --session-id to audit independent recordings")
        if source_kind == "mandatory" and split != "mandatory":
            raise ValueError("Mandatory input must use --split mandatory and real source provenance")
        if source_kind == "public" and not (args.source_url or provenance.get("source_url")):
            raise ValueError("Custom public input requires --source-url or provenance JSON")
    if args.session_id:
        provenance["recording_session"] = args.session_id
    if args.source_url:
        provenance["source_url"] = args.source_url
    return prepare_frames(
        source, args.output_dir, clip_id=clip_id, source_kind=source_kind,
        camera_motion=camera_motion, split=split,
        target_fps=args.target_fps if args.target_fps is not None else config["input"]["target_fps"],
        max_long_edge=args.max_long_edge if args.max_long_edge is not None else config["input"]["max_long_edge"],
        max_frames=args.max_frames, start_seconds=args.start_seconds, end_seconds=args.end_seconds,
        roi_path=args.background_roi, provenance=provenance,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = _prepare(args)
    elif args.command == "check":
        config = _config(args)
        resources = check_resources(config)
        env = environment_info()
        requires_cuda = config["runtime"]["device"].startswith("cuda")
        ready = not requires_cuda or env["cuda_available"]
        result = {"resources": resources, "environment": env, "ready_for_requested_device": ready,
                  "configuration": config}
        if args.output:
            if args.output.exists():
                raise FileExistsError(args.output)
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if ready else 2
    elif args.command == "freeze":
        result = freeze_protocol(_config(args), args.manifest, args.output, config_path=args.config)
    elif args.command == "infer":
        result = infer_run(args.protocol, args.model, args.output_dir)
    elif args.command == "flow":
        result = cache_flow(args.protocol, args.output_dir)
    elif args.command == "evaluate":
        protocol, _, manifest_path = load_protocol(args.protocol)
        digest = sha256_file(args.protocol)
        for directory, name in ((args.dav2_run, "run.json"), (args.vda_run, "run.json"),
                                (args.flow_cache, "flow_manifest.json")):
            if read_json(directory / name).get("protocol_sha256") != digest:
                raise ValueError("Both models and flow must come from this exact frozen protocol")
        from src.video.evaluation import evaluate_pair
        result = evaluate_pair(manifest_path, args.dav2_run, args.vda_run, args.flow_cache,
                               args.output_dir, config=protocol["configuration"])
    elif args.command == "verify":
        if args.run:
            result = verify_run(args.run)
        elif args.evaluation:
            from src.video.evaluation import verify_evaluation
            result = verify_evaluation(args.evaluation, verify_sources=True)
        else:
            protocol, manifest, _ = load_protocol(args.protocol, verify_resources=True)
            result = {"status": "passed", "frames": len(manifest["frames"]),
                      "protocol_content_sha256": protocol["protocol_content_sha256"]}
        if args.output:
            if args.output.exists():
                raise FileExistsError(args.output)
            write_json(args.output, result)
    elif args.command == "summarize":
        from src.video.collection import summarize_evaluations
        result = summarize_evaluations(args.evaluations, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
