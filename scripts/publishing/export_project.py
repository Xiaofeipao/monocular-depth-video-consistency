#!/usr/bin/env python3
"""只导出Part 1/2、B、C v1/v2的项目自有代码与中文说明。标准库，无实验。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]
SCOPE = 'part1_part2_B_C_v1_v2_code_docs_only'
ROOT_FILES = (
    '.gitignore', 'README.md', 
    'environment.yml', 'environment-unidepth.yml', 'requirements-core.txt',
    'requirements-unidepth.txt', 'requirements-lock.txt', 'pytest.ini',
)
FOLDERS = ('src', 'pipelines', 'scripts', 'configs', 'tests', 'docs')
SUFFIXES = {'.py', '.sh', '.md', '.yaml', '.yml', '.json', '.txt'}
EXCLUDED = {
    'docs/github_upload_part1_part2.md', 'docs/github_upload_part1a_part2a.md',
    'docs/github_upload_part1b_part2b.md', 'configs/video/clips.yaml',
}
REQUIRED = (
    'docs/step0.md', 'docs/step1.md', 'docs/step2.md', 'docs/step3.md', 'docs/step4.md',
    'docs/stepB.md', 'docs/stepC.md', 'docs/stepC_v2.md', 'docs/github_upload_project.md',
    'src/refinement/ORIGIN.json', 'src/refinement/MAIN_LICENSE.txt',
    'src/part3_studies/estimators.py', 'src/part3_studies/classical.py',
    'src/part3_c_v2/refine.py', 'tests/test_part3_c_v2.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(source):
    names = set(ROOT_FILES)
    for folder in FOLDERS:
        for path in (source/folder).rglob('*'):
            relative = path.relative_to(source)
            if any(p.startswith('.') or p == '__pycache__' for p in relative.parts):continue
            if relative.as_posix().startswith('docs/figures/'):continue
            if path.is_file() and path.suffix in SUFFIXES:
                if path.name.lower() in {'credentials.json','secrets.json','tokens.json'}:
                    raise ValueError('疑似凭据文件，不能发布：'+str(relative))
                if path.name.endswith(('.local.yaml','.local.yml')):continue
                names.add(relative.as_posix())
    names -= EXCLUDED
    if not set(REQUIRED) <= names:raise ValueError('缺少正式任务或发布材料')
    for kind in ('src','pipelines','scripts','configs'):
        groups = ('refinement','part3_studies','part3_c_v2') if kind == 'src' else ('part3','part3_studies','part3_c_v2')
        for group in groups:
            if not any(n.startswith(f'{kind}/{group}/') for n in names):
                raise ValueError('缺少B/C v1/v2目录：'+kind+'/'+group)
    for name in sorted(names):
        path = source/name
        if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError('必须为普通文件、不能经符号链接导出：'+name)
        if path.stat().st_size > 5*1024**2:raise ValueError('出现非预期大文件：'+name)
    return sorted(names)


def export(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError('导出目录必须在实验项目之外')
    names = select(source)
    output.mkdir(parents=True, exist_ok=False)
    for name in names:
        target = output/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name,target)
    (output/'.gitattributes').write_text('* -text\n',encoding='utf-8')
    (output/'PUBLISHING.md').write_text('''# 发布范围与复现

本分支包含Part 1/2、Direction B、C v1与v2的项目自有实现、测试、环境和说明。
不包含data、checkpoints、outputs、第三方仓库、队友仓库副本或缓存。
结果数值保留在文档，预测/视频/图像/CSV/冻结参数JSON和原始运行元数据不上传。
文档指向这些本地资源的链接需在恢复相应资源后打开，GitHub文件树中不存在。
先按configs/versions.yaml与docs/step0.md克隆固定上游，再创建环境、准备数据与权重。
各Step给出完整生成/评价/核验命令；只克隆本分支无法直接verify未上传的正式产物。
B/C的part3、part3_studies和part3_c_v2目录都是必需组成，来源及MIT许可完整保留。
不按测试GT选参，不训练，不因发布而改变70/100帧素材、Demo和报告等门禁。
当前使用方法以README和各Step为准。
''', encoding='utf-8')
    files = {p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'PROJECT_EXPORT.json').write_text(json.dumps(dict(scope=SCOPE,files=files),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive = output.with_name(output.name+'.tar.gz')
    if archive.exists() or archive.with_name(archive.name+'.sha256').exists():
        raise FileExistsError('旁边已有同名压缩包或校验文件')
    with tarfile.open(archive,'w:gz') as f:f.add(output,arcname=output.name)
    archive.with_name(archive.name+'.sha256').write_text(f'{sha(archive)}  {archive.name}\n',encoding='utf-8')
    return dict(package=str(output),files=len(files)+1,archive=str(archive),archive_bytes=archive.stat().st_size,
                archive_sha256=sha(archive),scope=SCOPE)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.source,args.output),ensure_ascii=False,indent=2))


if __name__ == '__main__':main()
