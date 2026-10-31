"""两独立入口共用的参数定义与只读辅助；本模块不是运行入口。"""
import argparse
import json
from pathlib import Path

from src.video import clip_workflow as work


def build_parser(part):
    if part not in ('1','2'):raise ValueError('必须指定独立部分')
    parser=argparse.ArgumentParser(allow_abbrev=False,description=f'Part {part}B：按单段自录视频运行本部分任务')
    sub=parser.add_subparsers(dest='command',required=True)
    check=sub.add_parser('check',allow_abbrev=False,help='只检查本地资源与CUDA，不预测')
    check.add_argument('--config',type=Path,default=work.DEFAULT_CONFIG)
    check.add_argument('--output',type=Path)
    root=work.OUTPUT_ROOT if part=='1' else work.PART2_OUTPUT_ROOT
    for name in ('plan','run'):
        p=sub.add_parser(name,allow_abbrev=False,help='查看任务依赖，不运行' if name=='plan' else '处理本部分单视频任务')
        p.add_argument('--clip',required=True)
        p.add_argument('--config',type=Path,default=work.DEFAULT_CONFIG)
        p.add_argument('--output-root',type=Path,default=root)
        if part=='1':
            p.add_argument('--video',type=Path)
            p.add_argument('--type',choices=work.TYPES,dest='video_type')
            p.add_argument('--session')
            p.add_argument('--registry',type=Path,default=work.DEFAULT_REGISTRY)
            p.add_argument('--cache-root',type=Path,default=work.CACHE_ROOT)
        else:
            p.add_argument('--part1-root',type=Path,default=work.OUTPUT_ROOT,
                           help='已完成的Part 1B视频根目录；只读引用，不自动运行1B')
        tasks=('all','prepare','dav2','flow','analyze','verify') if part=='1' else ('all','prepare','vda','analyze','verify')
        p.add_argument('--task',choices=tasks,default='all')
        p.add_argument('--no-deps',action='store_true',help='只运行所选任务，本部分依赖须已经存在')
    summary=sub.add_parser('summarize',allow_abbrev=False,help='汇总本部分已完成视频')
    summary.add_argument('--output-root',type=Path,default=root)
    summary.add_argument('--clips',nargs='+')
    summary.add_argument('--output-dir',type=Path,required=True)
    return parser


def emit(result):
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


def common_command(args,part):
    if args.command=='check':
        result=work.check_local(args.config,part=part)
        if args.output:
            path=work.absolute(args.output)
            if path.exists():raise FileExistsError(path)
            work.write_json(path,result)
        emit(result);return 0 if result['ready_for_requested_device'] else 2
    if args.command=='summarize':
        from src.video.clip_analysis import summarize
        emit(summarize(work.absolute(args.output_root),part,work.absolute(args.output_dir),args.clips))
        return 0
    return None
