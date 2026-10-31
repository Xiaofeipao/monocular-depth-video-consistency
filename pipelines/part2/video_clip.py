#!/usr/bin/env python3
"""Part 2B独立入口：只读引用已完成的1B，生成VDA及配对比较。"""
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pipelines.video.clip import build_parser,common_command,emit
from src.video import clip_workflow as work


def main(argv=None):
    args=build_parser('2').parse_args(argv)
    code=common_command(args,'2')
    if code is not None:return code
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',args.clip):raise ValueError('无效视频ID')
    if args.command=='plan':
        result={'clip_id':args.clip,'part':'2','task':args.task,'stages':work.task_plan('2',args.task,dependencies=not args.no_deps),
                'directory':str(work.absolute(args.output_root)/args.clip),
                'required_part1_directory':str(work.absolute(args.part1_root)/args.clip),
                'part1_access':'read_only; must_be_complete; never_auto_run','models_executed':False}
    else:
        result=work.run_comparison(args.clip,task=args.task,config_path=args.config,part1_root=args.part1_root,
                                   output_root=args.output_root,dependencies=not args.no_deps)
    emit(result);return 0


if __name__=='__main__':
    raise SystemExit(main())
