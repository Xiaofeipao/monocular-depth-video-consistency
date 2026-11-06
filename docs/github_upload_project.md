# 将完整项目更新到 GitHub 分支（Part 1、Part 2、Direction B/C）

## 本次更新的范围

本流程将目标分支的**最新文件树完整替换为本次上传包**。旧分支上不在上传包里的文件，包括隐藏文件，都会在新提交中删除；提交历史保留，其他分支不变。使用普通提交和普通push，不使用强制推送，也不需要先推送一次空分支。

全过程在独立导出目录和全新克隆目录进行，不清空当前实验项目。本文是操作说明，尚未执行远端修改。

沿用之前的仓库与分支作为示例，运行前核对：

- 仓库：`https://github.com/Xiaofeipao/monocular-depth-video-consistency.git`
- 目标分支：`JiongxuChen`

| 内容 | 是否上传 |
|---|---|
| Part 1A/1B、Part 2A/2B的实现、公共模块、配置、脚本和测试 | 上传 |
| Direction B三种全局尺度估计规则及完整开发/冻结/评估/核验流程 | 上传 |
| Direction C完整滤波、两个消融、两个经典对照及定位/宽度/纹理诊断 | 上传 |
| `part3`和`part3_studies`两组目录、来源记录与MIT许可 | **全部上传，不能只留其中一组** |
| 环境文件、固定上游版本、科学协议、README、PLAN、Step 0–4、Step B/C、执行记录和结果文字 | 上传 |
| 当前课程PDF与`Project3_CN_final.md` | 上传 |
| `data/`：视频、图像、数据集、抽帧及资源清单 | **整个目录不上传** |
| `checkpoints/`：权重、缓存、校验文件 | **整个目录不上传** |
| `outputs/`：原始/处理预测、图视频、CSV、JSON、日志、冻结尺度和维护记录 | **整个目录不上传** |
| 官方第三方仓库、`SAM2/`、`github_repository/`、环境目录、缓存和凭据 | **不上传** |

文档中的结果表会保留，独立结果文件不打包。因此GitHub提供的是完整**代码与说明**，不包含运行产物；B的冻结尺度JSON也属于产物，复现时按Step B从开发集重新拟合。指向本地资源和产物的文档路径在GitHub不会打开，这是未上传资源造成的预期现象。`implemented.md`保留实际执行历史，当前操作以README及各Step为准。70/100真实帧素材门禁、Demo和报告状态不会因上传而改变。

## 1. 在工作站生成完整上传包

以下命令使用系统Python 3标准库，无需切换Conda，不下载、不运行模型、不改科学代码或实验产物。每次更新都重新导出，不能复用旧的Part 1/2包。

从当前项目根目录执行：

```bash
cd /home/user/cjx/3201_Project
export PROJECT_SOURCE="$PWD"
export EXPORT_PARENT="$(mktemp -d /tmp/aiaa3201-project-upload-XXXXXX)"

python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

source = Path(os.environ['PROJECT_SOURCE']).resolve()
parent = Path(os.environ['EXPORT_PARENT']).resolve()
if parent == source or source in parent.parents:
    raise SystemExit('导出位置必须在实验项目之外')
package = parent / 'project_upload'
package.mkdir(parents=True, exist_ok=False)
selected = set()

def add(path):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f'不是普通文件：{path}')
    selected.add(path.relative_to(source).as_posix())

for name in (
    'README.md', 'AGENTS.md', 'PLAN.md', 'implemented.md',
    'Project3_CN_final.md', 'difference_analysis.md', 'plan_1B2B.md',
    'Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf',
    'environment.yml', 'environment-unidepth.yml', 'requirements-core.txt',
    'requirements-unidepth.txt', 'requirements-lock.txt', 'pytest.ini',
):
    add(source / name)
# 仅项目自有目录、指定文本类型；不遍历数据/权重/产物/第三方仓库。
allowed = {'.py', '.sh', '.md', '.yaml', '.yml', '.json', '.txt'}
for directory in ('src', 'pipelines', 'scripts', 'configs', 'tests', 'docs'):
    for path in sorted((source / directory).rglob('*')):
        relative = path.relative_to(source)
        if any(part.startswith('.') or part == '__pycache__' for part in relative.parts):
            continue
        if path.is_file() and path.suffix in allowed:
            if path.name.endswith(('.local.yaml', '.local.yml')):
                continue
            add(path)

required = (
    'src/refinement/ORIGIN.json', 'src/refinement/MAIN_LICENSE.txt',
    'src/part3_studies/estimators.py', 'src/part3_studies/classical.py',
    'src/part3_studies/diagnostics.py', 'docs/stepB.md', 'docs/stepC.md',
)
for directory in ('pipelines', 'scripts', 'configs'):
    for group in ('part3', 'part3_studies'):
        if not any(name.startswith(f'{directory}/{group}/') for name in selected):
            raise RuntimeError(f'缺少完整B/C目录：{directory}/{group}')
if not set(required) <= selected:
    raise RuntimeError('缺少B/C实现、说明或来源许可')
for name in sorted(selected):
    target = package / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / name, target)

(package / '.gitignore').write_text('''# 本仓库只发布项目自有代码、配置、测试和文档。
/data/
/checkpoints/
/outputs/
/github_repository/
/Depth-Anything-V2/
/Video-Depth-Anything/
/UniDepth/
/ml-depth-pro/
/SAM2/
.conda/
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.idea/
.vscode/
.DS_Store
.env
.env.*
*.pem
*.key
*.local.yaml
*.local.yml
*.mp4
*.mov
*.avi
*.mkv
*.npy
*.npz
*.pth
*.pt
*.safetensors
*.log
*.tmp
*.part
''', encoding='utf-8')
# Windows Git也保留原始字节；Python/YAML/Bash文件不做CRLF转换。
(package / '.gitattributes').write_text('* -text\n', encoding='utf-8')
(package / 'PUBLISHING.md').write_text('''# 发布范围

本分支提供Part 1、Part 2和Part 3 Direction B/C的项目自有实现、配置、测试与说明。
数据、视频、权重、所有运行产物及第三方仓库均未随仓库发布。
文档指标来自本地已运行结果；指向data/、checkpoints/、outputs/和github_repository/的路径仅供本地查阅，在本分支中不会打开。
implemented.md保留真实执行历史，当前使用方法以README、Step 0–4、Step B/C为准。
先按configs/versions.yaml和docs/step0.md取得固定上游版本，再创建Conda环境、准备资源。
缺少正式产物时无法直接执行读取已有预测的评价/verify；需按各Step完整流程生成，正式长时间运行由用户启动。
课程100真实帧素材门禁、Demo和最终报告状态仍按PLAN记录。
''', encoding='utf-8')
files = {}
for path in sorted(package.rglob('*')):
    if path.is_file():
        files[path.relative_to(package).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {'scope': 'part1_part2_part3_B_C_code_docs_only', 'files': files}
(package / 'PROJECT_EXPORT.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
archive = parent / 'project_upload.tar.gz'
with tarfile.open(archive, 'w:gz') as bundle:
    bundle.add(package, arcname=package.name)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
(parent / 'project_upload.tar.gz.sha256').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
print(f'文件数：{len(files) + 1}；压缩包：{archive}；大小：{archive.stat().st_size / 1024**2:.2f} MiB')
print(f'SHA256：{digest}')
PY
```

导出只复制项目自有文本和当前课程PDF，不改变原文件；B/C来源记录与许可同时保留。包内生成独立`.gitignore`，屏蔽数据、产物、权重和第三方仓库；`.gitattributes`避免Windows自动转换行尾。仍请检查自己新增的配置，不要放入个人令牌或密码。

查看位置和文件清单：

```bash
printf '%s\n' "$EXPORT_PARENT"
find "$EXPORT_PARENT/project_upload" -type f | sort
```

`PROJECT_EXPORT.json`记录除自身之外每份文件的SHA256。后面的替换步骤会验证上传包及Git暂存字节，避免漏传、混入文件或改变科学源码。

## 2. 选择在哪台电脑推送

工作站能够访问并认证GitHub时，直接使用第1步生成的目录，进入第3步。

如果打算在自己的电脑操作，将这两个文件通过SFTP或文件传输工具复制到同一个新目录：

- `project_upload.tar.gz`
- `project_upload.tar.gz.sha256`

本地电脑只需Git、Python 3和Bash，不需要模型环境。Windows使用Git Bash，不要将Bash命令粘贴到PowerShell。以下假设文件接收到`$HOME/project_publish`，请按实际位置调整：

```bash
cd "$HOME/project_publish"
sha256sum -c project_upload.tar.gz.sha256
# 必须显示project_upload.tar.gz: OK，且校验值与工作站打印值一致。
# 此目录不能已有project_upload；若已有，请换一个新的接收目录。
test ! -e project_upload && tar -xzf project_upload.tar.gz
```

只传本次上传包，不需要复制实验项目、视频、Conda环境或预测。

## 3. 设置仓库、分支和Python

工作站通常为`PYTHON=python3`；Windows Git Bash通常改成`PYTHON=python`，版本必须为Python 3。

```bash
PYTHON=python3
"$PYTHON" --version
git --version
export REPO_URL='https://github.com/Xiaofeipao/monocular-depth-video-consistency.git'
export TARGET_BRANCH='JiongxuChen'
git ls-remote --exit-code --heads "$REPO_URL" "refs/heads/$TARGET_BRANCH"
```

必须看到目标分支的SHA及`refs/heads/JiongxuChen`才继续。报错或没有输出时，先核对地址、分支和权限；本流程面向已有分支。

这里使用HTTPS地址，通过自己的Git凭据完成认证；不要把密码或令牌写进URL或项目文件。此前SSH的`Permission denied (publickey)`说明SSH认证失败，不能据此判断工作站无法联网；也可以把包传到自己的电脑推送。

## 4. 克隆到全新独立目录

不要将`BRANCH_DIR`指向实验项目或已有工作目录。

```bash
export PUBLISH_WORK="$(mktemp -d "$HOME/aiaa3201-github-XXXXXX")"
export BRANCH_DIR="$PUBLISH_WORK/repository"
git clone --single-branch --branch "$TARGET_BRANCH" "$REPO_URL" "$BRANCH_DIR"
git -C "$BRANCH_DIR" rev-parse HEAD > "$PUBLISH_WORK/before-upload-commit.txt"
git -C "$BRANCH_DIR" remote -v
git -C "$BRANCH_DIR" status --short
git -C "$BRANCH_DIR" branch --show-current
```

工作区状态应为空，分支应为目标分支。替换前的提交号保存在仓库外的`before-upload-commit.txt`中，可以据此查看旧版。保留`.git/`。

## 5. 设置上传包路径

如果在工作站沿用第1步的终端：

```bash
export PACKAGE_DIR="$EXPORT_PARENT/project_upload"
```

如果在自己的电脑操作，则使用实际解压路径，例如：

```bash
export PACKAGE_DIR="$HOME/project_publish/project_upload"
```

以上两种只选一种。Windows Git Bash还需转换传给Python的环境变量；Linux/macOS跳过下面这段：

```bash
export PACKAGE_DIR="$(cygpath -am "$PACKAGE_DIR")"
export BRANCH_DIR="$(cygpath -am "$BRANCH_DIR")"
```

核对位置与清单：

```bash
printf '上传包：%s\n克隆目录：%s\n目标分支：%s\n' "$PACKAGE_DIR" "$BRANCH_DIR" "$TARGET_BRANCH"
test -f "$PACKAGE_DIR/PROJECT_EXPORT.json"
test -f "$PACKAGE_DIR/docs/stepB.md"
test -f "$PACKAGE_DIR/docs/stepC.md"
```

## 6. 完整替换并验证暂存内容

**下面会删除`BRANCH_DIR`下除`.git/`以外的全部旧文件，再复制上传包。** 脚本先验证干净状态、目标分支、路径与上传包，只操作上一步的独立克隆，不修改实验项目，不提交或推送。

```bash
"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess

package = Path(os.environ['PACKAGE_DIR']).resolve()
branch = Path(os.environ['BRANCH_DIR']).resolve()
expected = os.environ['TARGET_BRANCH']
def git(*args):
    return subprocess.check_output(['git', '-C', str(branch), *args])
if package == branch or package in branch.parents or branch in package.parents:
    raise SystemExit('上传包和克隆目录必须互相独立')
if not (branch / '.git').is_dir() or git('rev-parse', '--show-toplevel').decode().strip().replace('\\', '/') != branch.as_posix():
    raise SystemExit('必须使用刚刚独立克隆的普通Git仓库')
if git('branch', '--show-current').decode().strip() != expected:
    raise SystemExit('当前分支与TARGET_BRANCH不符')
if git('status', '--porcelain', '--untracked-files=all').strip():
    raise SystemExit('克隆目录存在未提交内容，停止替换')
manifest = json.loads((package / 'PROJECT_EXPORT.json').read_text(encoding='utf-8'))
if manifest['scope'] != 'part1_part2_part3_B_C_code_docs_only':
    raise SystemExit('不是当前完整项目上传包')
files = manifest['files']
for name, digest in files.items():
    parts = PurePosixPath(name).parts
    if not parts or PurePosixPath(name).is_absolute() or any(p in ('.', '..', '.git') or ':' in p or '\\' in p for p in parts):
        raise SystemExit(f'非法路径：{name}')
    path = package.joinpath(*parts)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise SystemExit(f'不是普通文件：{name}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f'上传包内容校验失败：{name}')
actual = {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
if actual != set(files) | {'PROJECT_EXPORT.json'} or any(p.is_symlink() for p in package.rglob('*')):
    raise SystemExit('上传包有多余、缺失文件或符号链接')
# 所有检查通过后，只清空独立克隆中的工作树，保留.git及提交历史。
for path in branch.iterdir():
    if path.name == '.git':
        continue
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
for path in package.iterdir():
    destination = branch / path.name
    if path.is_dir():
        shutil.copytree(path, destination)
    else:
        shutil.copy2(path, destination)
git('add', '-A')
for name in files:
    if name.endswith('.sh'):
        git('update-index', '--chmod=+x', '--', name)
indexed = set(git('ls-files', '-z').decode().rstrip('\0').split('\0'))
if indexed != actual:
    raise SystemExit('暂存文件树与上传包不一致；请勿提交')
for name in sorted(actual):
    if hashlib.sha256(git('show', ':' + name)).hexdigest() != hashlib.sha256((package / name).read_bytes()).hexdigest():
        raise SystemExit(f'暂存字节不一致：{name}；请勿提交')
print(f'替换与暂存验证通过：{len(actual)}个文件；尚未提交或推送。')
PY
```

必须显示“替换与暂存验证通过”才继续。失败时不要提交，可从第4步重新克隆后排查。这里使用`git add -A`，新增、修改、旧文件删除都会进入同一个提交；`.sh`脚本显式设置可执行位。

## 7. 审阅、提交和推送

```bash
git -C "$BRANCH_DIR" --no-pager diff --cached --stat
git -C "$BRANCH_DIR" --no-pager diff --cached --name-status
git -C "$BRANCH_DIR" ls-files data checkpoints outputs github_repository Depth-Anything-V2 Video-Depth-Anything UniDepth ml-depth-pro SAM2
git -C "$BRANCH_DIR" ls-files src/refinement src/part3_studies pipelines/part3 pipelines/part3_studies scripts/part3 scripts/part3_studies configs/part3 configs/part3_studies
```

第一组`ls-files`必须没有输出，第二组必须有B/C两组实现及配置。`D`表示删除旧文件，符合完整替换的目标。检查后执行：

```bash
git -C "$BRANCH_DIR" commit -m 'Publish project Parts 1 and 2 with Directions B and C'
git -C "$BRANCH_DIR" push origin "HEAD:refs/heads/$TARGET_BRANCH"
```

如果没有改动，说明内容已一致，无需空提交。若缺少Git姓名/邮箱，只在此克隆中设置自己的`user.name`和`user.email`再提交。终端出现`(END)`时按`q`退出。

如果push被拒绝为non-fast-forward，说明克隆之后远端又有提交：保留本次包，从第4步重新克隆并审阅队友的新内容，再替换。不要使用`--force`覆盖未查看的新提交。

## 8. 核对远端

```bash
git -C "$BRANCH_DIR" rev-parse HEAD
git -C "$BRANCH_DIR" ls-remote --heads origin "refs/heads/$TARGET_BRANCH"
git -C "$BRANCH_DIR" status --short
```

本地HEAD与远端目标分支SHA应一致，工作区应干净。网页切换到`JiongxuChen`，确认：

- 当前README、Step 1–4、Step B/C和两组B/C实现齐全。
- 旧分支独有文件已从最新文件树删除。
- 没有数据、视频、权重、运行产物或第三方仓库。
- main和其他分支没有被本次操作修改。

这是替换分支**最新文件树**，不是清除历史。以前提交过的大文件仍可能存在于Git历史，本流程不重写历史。

## 新目录中如何复现

先阅读`PUBLISHING.md`与`docs/step0.md`。按`configs/versions.yaml`取得四个固定版本的上游，再创建`aiaa3201`与`aiaa3201-unidepth`；主环境定义包含`-e ./ml-depth-pro`，顺序不能颠倒。另行准备数据和权重，再按各Step生成预测和结果。

B依赖795张开发预测和Part 2A原生基线；C依赖Part 2A原生预测，按Step C先运行联合滤波/消融，再运行经典对照及统一诊断。仅克隆代码不能直接验证未上传的正式产物，也不等于已复现实验。正式训练与长时间运行仍由用户启动。
