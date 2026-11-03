#!/usr/bin/env python3
"""Part 1B课程逐帧基线入口；不调用VDA。"""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.evaluation import framewise_clip as task
from src.video.common import write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description='课程1B：指定片段DA-V2-S逐帧relative基线',allow_abbrev=False)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('check','run'):
        p=sub.add_parser(name,allow_abbrev=False);p.add_argument('--config',type=Path,default=task.DEFAULT_CONFIG)
        p.add_argument('--output',type=Path)
        if name=='run':p.add_argument('--cache',type=Path)
    for name in ('analyze','verify'):
        p=sub.add_parser(name,allow_abbrev=False);p.add_argument('--run',type=Path,required=True)
        p.add_argument('--analysis',type=Path)
        if name=='verify':p.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.command=='check':
        result=task.check_resources(task.load_config(args.config))
        if args.output:
            if args.output.exists():raise FileExistsError(args.output)
            write_json(args.output,result)
    elif args.command=='run':
        directory=task.predict(args.config,output=args.output,cache=args.cache)
        task.analyze(directory);result=task.verify(directory)
        write_json(directory/'verification.json',result)
    elif args.command=='analyze':result={'analysis':str(task.analyze(args.run,output=args.analysis))}
    else:
        result=task.verify(args.run,analysis=args.analysis)
        if args.output:
            if args.output.exists():raise FileExistsError(args.output)
            write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str));return 0


if __name__=='__main__':raise SystemExit(main())
