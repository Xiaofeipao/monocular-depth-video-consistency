# 将1B/2B代码补充到JiongxuChen分支

> 此页针对先前导出的自录流程代码包。该工作流已清理，不应再用旧包覆盖当前指定片段代码；当前运行与文件结构见[Step 2](step2.md)和[Step 4](step4.md)。

本页从**你已下载到本地的`part1b_part2b.tar.gz`代码包**开始。无需重新准备工作站环境、导出或运行实验，也不再使用之前的`UPLOAD_WORK`等变量或NUL白名单文件。

目标是：保留`JiongxuChen`已有1A/2A等内容，补充1B/2B代码。已有文件不会被整批删除；代码包中存在同名文件时会更新它，提交前查看差异即可。仅推送`JiongxuChen`，不操作`main`。

| 上传 | 不上传 |
|---|---|
| 1B/2B源码、配置、环境依赖、测试、人工复现说明 | 视频、RGB帧、权重、预测、光流、图表、自动报告及实验JSON/CSV |

代码包里的`PART1B_PART2B_EXPORT.json`只是源码文件清单和SHA，不含实验结果。复制程序会拒绝包含`outputs/`、`data/`或`checkpoints/`文件的清单。

所有命令在**本地电脑的Git Bash**执行，需有Git和Python 3.10或更高版本。下面使用`python`；若本机命令是`python3`，替换该名称即可。每一步报错先停下，不继续提交。

## 第1步：解压代码包，克隆目标分支

从你现在保存压缩包的目录开始。新建一个独立目录，旧克隆保留，不删除。如果`part1b_part2b_publish`已经存在，换一个新的目录名，并对应修改后续路径。

```bash
cd /d/26-27fall/AIAA3201L01/github_upload 
mkdir part1b_part2b_publish 
tar -xzf part1b_part2b.tar.gz -C part1b_part2b_publish 
cd part1b_part2b_publish 
git -c core.autocrlf=false clone --single-branch --branch JiongxuChen   https://github.com/Xiaofeipao/monocular-depth-video-consistency.git repo
```

成功后结构为：

```text
part1b_part2b_publish/
  export/    本地代码包
  repo/      从GitHub克隆的JiongxuChen分支，保留其原有内容
```

## 第2步：复制代码，查看改动

复制下面**整个代码块**。这段Python只用来按代码包清单复制文件，并保证没有视频或运行产物混入；它不运行模型。使用相对路径，避免Git Bash与Windows Python对路径变量的解释不同。

```bash
cd /d/26-27fall/AIAA3201L01/github_upload/part1b_part2b_publish &&
(
set -e
test "$(git -C repo branch --show-current)" = 'JiongxuChen'
python - <<'PY'
from pathlib import Path
import hashlib
import json
import shutil

source = Path('export').resolve()
target = Path('repo').resolve()
manifest = source / 'PART1B_PART2B_EXPORT.json'
record = json.loads(manifest.read_text(encoding='utf-8'))
assert record['kind'] == 'incremental_part1b_part2b_code_only_export', '请使用只含代码的新包'
assert record['target_branch'] == 'JiongxuChen'
assert (target / '.git').is_dir(), '请先完成第1步克隆'
files = []
for item in record['files']:
    path = Path(item['path'])
    assert not path.is_absolute() and '..' not in path.parts and '.git' not in path.parts
    assert path.parts[0] not in {'outputs', 'data', 'checkpoints', 'github_repository'}
    assert 'part3' not in path.parts
    assert path.suffix in {'.py', '.sh', '.yaml', '.yml', '.md', '.txt', '.ini'}
    original, copied = source / path, target / path
    assert original.resolve().is_relative_to(source) and not original.is_symlink()
    assert hashlib.sha256(original.read_bytes()).hexdigest() == item['exported_sha256'], path
    assert not copied.is_symlink() and all(not p.is_symlink() for p in copied.parents)
    assert copied.resolve().is_relative_to(target)
    files.append((original, copied))
copied_manifest = target / manifest.name
assert not copied_manifest.is_symlink()
for original, copied in files:
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, copied)
shutil.copy2(manifest, copied_manifest)
print('代码复制完成；没有复制视频、数据、权重或实验产物。')
PY

git -C repo config core.autocrlf false
test "$(git -C repo branch --show-current)" = 'JiongxuChen'
git -C repo add --all
git -C repo update-index --chmod=+x -- \
  scripts/part1/run_dav2_nyuv2.sh scripts/part1/video.sh scripts/part2/video.sh
git -C repo --no-pager diff --cached --stat
git -C repo --no-pager diff --cached --name-status
)
```

这里的`git add --all`只作用于刚刚独立克隆的`repo/`；复制程序只放入代码清单文件，原视频和实验目录从未复制进来。

查看最后的文件列表，应该是1B/2B代码、配置、说明与必要公共依赖的新增或修改。若想看某个同名共享文件的具体改动，例如：

```bash
git -C repo --no-pager diff --cached -- src/models/depth_anything_v2_adapter.py
```

**已有同名文件确实可能更新。** 若其中有需要保留的队友修改，先合并具体文件，再重新暂存；没有出现在清单里的原文件保持原样。本流程不清空分支、不删除旧目录、不强推。

## 第3步：提交并上传

在第2步检查无误后执行：

```bash
cd /d/26-27fall/AIAA3201L01/github_upload/part1b_part2b_publish &&
(
set -e
test "$(git -C repo branch --show-current)" = 'JiongxuChen'
git -C repo commit -m 'Add Part 1B and Part 2B code and reproduction docs'
git -C repo push origin HEAD:refs/heads/JiongxuChen
git -C repo status --short --branch
)
```

到GitHub网页切换到`JiongxuChen`，查看`docs/step2.md`、`docs/step4.md`和新增代码，确认原来的1A/2A还在。这样就是在原分支上增加一个提交，保留既有历史与其他文件。[Git官方add说明](https://git-scm.com/docs/git-add)、[push说明](https://git-scm.com/docs/git-push)。

如果Git要求设置提交者，填写自己的信息后重新执行第3步：

```bash
git -C repo config user.name '你的名字'
git -C repo config user.email '你的提交邮箱'
```

若push提示远端已有新提交，先停止核对，不使用`--force`。本流程不自动清理远端以前就存在的文件；若此前已经上传过视频或1B/2B产物，需要另行确定删除路径。

## 说明

本次只简化本Markdown，没有操作实际GitHub远端，也没有修改本地视频、实验产物、代码、README或implemented。需要上传的是现有代码，因此不要求为了上传再跑推理或测试。

你此前填写的下载地址保留如下，仅在需要重新下载压缩包时使用；已经下载就不需要执行：

```bash
scp 'user@100.124.129.113:/tmp/aiaa3201-part1b-part2b-upload.f6O88T/part1b_part2b.tar.gz' .
scp 'user@100.124.129.113:/tmp/aiaa3201-part1b-part2b-upload.f6O88T/part1b_part2b.tar.gz.sha256' .
```
