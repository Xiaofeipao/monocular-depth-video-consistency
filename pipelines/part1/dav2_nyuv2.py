#!/usr/bin/env python3
"""DA-V2-S NYUv2 aligned-relative benchmark, no training."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "predict", "evaluate", "verify", "run"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/part1/dav2_nyuv2.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--positions", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", help="显式覆盖设备，CPU工程检查需同时指定float32")
    parser.add_argument("--precision", choices=["float16", "float32"])
    args = parser.parse_args(argv)
    from src.evaluation.relative_nyuv2 import load_config, check_resources, predict, evaluate, verify
    from src.video.common import write_json
    config = load_config(args.config)
    if args.device:
        config["runtime"]["device"] = args.device
    if args.precision:
        config["runtime"]["precision"] = args.precision
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    if args.positions is not None and args.command not in {"predict", "run"}:
        parser.error("--positions only applies to predict/run")
    if args.resume and args.command != "predict":
        parser.error("--resume only applies to predict")
    if args.command == "check":
        try:
            result = {"status": "ready", **check_resources(config)}
        except Exception as exc:
            result = {"status": "not_ready", "reason": str(exc)}
        if args.output:
            write_json(args.output, result)
        print(result)
        return 0 if result["status"] == "ready" else 2
    if args.command in {"predict", "run"}:
        if not args.output_dir:
            parser.error("--output-dir is required")
        result = predict(config, args.output_dir, positions=args.positions, resume=args.resume)
        if args.command == "run":
            evaluate(args.output_dir, args.output_dir / "evaluation")
            result = verify(args.output_dir / "evaluation")
            write_json(args.output_dir / "reload_validation.json", result)
    elif args.command == "evaluate":
        if not args.run_dir or not args.output_dir:
            parser.error("--run-dir and --output-dir are required")
        result = evaluate(args.run_dir, args.output_dir)
    else:
        if not args.evaluation_dir:
            parser.error("--evaluation-dir is required")
        result = verify(args.evaluation_dir)
    if args.output:
        write_json(args.output, result if isinstance(result, dict) else {"path": str(result)})
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
