#!/usr/bin/env python3
"""只替换全新独立克隆的工作树，校验并暂存；不提交、不推送。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

SCOPE = 'part1_part2_B_C_v1_v2_code_docs_only'
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCKED = {'data','checkpoints','outputs','github_repository','Depth-Anything-V2',
           'Video-Depth-Anything','UniDepth','ml-depth-pro','SAM2'}


def replace(package, repository, branch, expected_head):
    package, repository = Path(package).resolve(), Path(repository).resolve()
    for protected in (PROJECT_ROOT,package):
        if repository == protected or repository in protected.parents or protected in repository.parents:
            raise ValueError('克隆目录必须与实验项目和上传包互相独立')
    if not (repository/'.git').is_dir():raise ValueError('必须为全新独立克隆的普通仓库')
    def git(*args):
        return subprocess.check_output(['git','-C',str(repository),*args])
    if Path(git('rev-parse','--show-toplevel').decode().strip()).resolve() != repository:
        raise ValueError('不是Git工作树根目录')
    if git('branch','--show-current').decode().strip() != branch:raise ValueError('目标分支不一致')
    if git('rev-parse','HEAD').decode().strip() != expected_head:raise ValueError('克隆HEAD与保存的提交不一致')
    if git('status','--porcelain','--untracked-files=all').strip():raise ValueError('仓库有未提交内容，停止替换')
    if subprocess.run(['git','-C',str(repository),'clean','-ndx'],capture_output=True,check=True).stdout.strip():
        raise ValueError('仓库有未跟踪或被忽略文件，请重新克隆；不删除个人文件')
    manifest = json.loads((package/'PROJECT_EXPORT.json').read_text(encoding='utf-8'))
    if manifest.get('scope') != SCOPE:raise ValueError('不是本次完整项目上传包')
    files=manifest['files']
    for name,digest in files.items():
        parts=PurePosixPath(name).parts
        if (not parts or PurePosixPath(name).is_absolute() or parts[0] in BLOCKED
                or any(p in ('.','..','.git') or ':' in p or '\\' in p for p in parts)):
            raise ValueError('非法或不应上传的路径：'+name)
        path=package.joinpath(*parts)
        if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError('上传包含非普通文件：'+name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('上传包SHA不符：'+name)
    actual={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    if actual != set(files)|{'PROJECT_EXPORT.json'} or any(p.is_symlink() for p in package.rglob('*')):
        raise ValueError('上传包缺少或混入文件')
    # Only now mutate the independently cloned worktree; preserve .git/history.
    for path in repository.iterdir():
        if path.name == '.git':continue
        if path.is_symlink() or path.is_file():path.unlink()
        else:shutil.rmtree(path)
    for path in package.iterdir():
        if path.is_dir():shutil.copytree(path,repository/path.name)
        else:shutil.copy2(path,repository/path.name)
    git('add','-A')
    for name in files:
        if name.endswith('.sh'):git('update-index','--chmod=+x','--',name)
    indexed=set(git('ls-files','-z').decode().rstrip('\0').split('\0'))
    if indexed != actual:raise ValueError('暂存文件树与上传包不同，不要提交')
    for name in sorted(actual):
        blob=git('show',':'+name)
        if hashlib.sha256(blob).hexdigest() != hashlib.sha256((package/name).read_bytes()).hexdigest():
            raise ValueError('暂存字节与上传包不同，不要提交：'+name)
    return dict(staged_files=len(actual),branch=branch,previous_head=expected_head,committed=False,pushed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--branch',required=True)
    p.add_argument('--expected-head',required=True)
    args=p.parse_args()
    print(json.dumps(replace(args.package,args.repository,args.branch,args.expected_head),ensure_ascii=False,indent=2))


if __name__ == '__main__':main()
