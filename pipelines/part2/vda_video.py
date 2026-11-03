#!/usr/bin/env python3
"""课程2B：独立VDA离线相对深度，以及单独启动的DA/VDA比较。"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation import temporal_clip as task
from src.video.common import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description='课程2B：VDA-S离线relative；不训练、不运行streaming/metric', allow_abbrev=False)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'predict', 'run'):
        p = sub.add_parser(name, allow_abbrev=False)
        p.add_argument('--config', type=Path, default=task.DEFAULT_CONFIG)
        p.add_argument('--output', type=Path)
        if name != 'check':
            p.add_argument('--cache', type=Path)
    for name in ('analyze', 'verify'):
        p = sub.add_parser(name, allow_abbrev=False)
        p.add_argument('--run', type=Path, required=True)
        p.add_argument('--analysis', type=Path)
        if name == 'verify':
            p.add_argument('--output', type=Path)
    p = sub.add_parser('compare', allow_abbrev=False)
    p.add_argument('--dav2-run', type=Path, required=True)
    p.add_argument('--vda-run', type=Path, required=True)
    p.add_argument('--dav2-analysis', type=Path)
    p.add_argument('--vda-analysis', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('verify-comparison', allow_abbrev=False)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'check':
        result = task.check_resources(task.load_config(args.config))
    elif args.command in ('run', 'predict'):
        directory = task.predict(args.config, output=args.output, cache=args.cache)
        if args.command == 'run':
            task.analyze(directory)
            result = task.verify(directory)
            write_json(directory / 'verification.json', result)
        else:
            result = {'run': str(directory), 'analysis_pending': True}
    elif args.command == 'analyze':
        result = {'analysis': str(task.analyze(args.run, output=args.analysis))}
    elif args.command == 'verify':
        result = task.verify(args.run, analysis=args.analysis)
    elif args.command == 'compare':
        directory = task.compare(args.dav2_run, args.vda_run, output=args.output,
                                 dav2_analysis=args.dav2_analysis, vda_analysis=args.vda_analysis)
        result = task.verify_comparison(directory)
        write_json(directory / 'verification.json', result)
    else:
        result = task.verify_comparison(args.comparison)
    if args.command in ('check', 'verify', 'verify-comparison') and args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
