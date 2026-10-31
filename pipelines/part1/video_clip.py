#!/usr/bin/env python3
"""Part 1B独立入口：抽帧、DA-V2、光流与单模型诊断。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pipelines.video.clip import build_parser,common_command,emit
from src.video import clip_workflow as work


def main(argv=None):
    args=build_parser('1').parse_args(argv)
    code=common_command(args,'1')
    if code is not None:return code
    clip=work.resolve_clip(args.clip,registry=args.registry,video=args.video,video_type=args.video_type,
                           session=args.session,output_root=args.output_root)
    if args.command=='plan':
        result={'clip':clip,'part':'1','task':args.task,'stages':work.task_plan('1',args.task,dependencies=not args.no_deps),
                'directory':str(work.absolute(args.output_root)/args.clip),'models_executed':False}
    else:
        result=work.run_clip(clip,task=args.task,config_path=args.config,output_root=args.output_root,
                             cache_root=args.cache_root,dependencies=not args.no_deps)
    emit(result);return 0


if __name__=='__main__':
    raise SystemExit(main())
