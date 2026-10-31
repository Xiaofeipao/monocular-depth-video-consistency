# GitHub上传：Part 1B与Part 2B

目标仓库为`Xiaofeipao/monocular-depth-video-consistency`，目标分支为 **`JiongxuChen`**。本文采用增量上传：在该分支已有Step 0、1A、2A的基础上补充1B/2B，保留原有提交和其他内容。上一次上传说明中“清空分支”的步骤不适用于本次操作。

本次仅上传代码、配置、环境、测试和复现说明，**不上传视频，也不上传任何1B/2B运行产物，包括小型JSON、CSV和生成报告**。

所有Git上传命令由你执行。本次没有连接项目远端、切换分支、提交或推送；项目内仅修改本文，其他文件保持原样。操作在独立导出副本和新克隆中进行，不在实验项目根目录执行`git add .`。

## 1. 本次筛选范围

| 类别 | 上传内容 | 原因 |
|---|---|---|
| 1B：NYUv2 | DA-V2-S的配置、入口、`relative_nyuv2.py`、对齐算子、对应测试 | 完整保存原生relative预测与GT辅助aligned_relative评价的实现 |
| 1B：自录视频 | 独立`part1/video.sh`与`video_clip.py`、三类型配置、逐像素/光流/背景/尺度诊断 | 每段单独运行，产物归1B |
| 2B：自录视频 | 独立`part2/video.sh`与`video_clip.py`、VDA配置和adapter、配对比较 | 只读引用已完成1B，VDA与比较报告另存2B |
| 公共实现 | 全部`src/video/*.py`、`pipelines/video/*.py`及必要数据/指标/画图模块 | `source_fingerprints()`会绑定这些完整目录，不能仅复制新入口和三个clip文件 |
| 环境与资源约定 | 主环境YAML、直接依赖、配置中的上游commit和资源SHA | 保留复现约定，不复制数据、权重或运行资源清单；本次不新增UniDepth环境 |
| 说明 | `docs/step2.md`、`docs/step4.md`、`plan_1B2B.md`、本文；导出时另生成分支概览 | 不用整个项目的README/PLAN覆盖现有分支说明 |
| 测试 | 下方白名单中的11份测试 | 覆盖1B/2B主要实现；公共配置测试依赖分支已有1A/2A配置 |

`src/video/framewise_roi.py`等底层文件和`pipelines/video/cli.py`作为项目自有源码完整保留，是现有依赖/绑定的一部分；它们不作为本次推荐操作入口。正式视频操作仍分别以Step 2、Step 4为准，无人固定用全图背景，不要求画矩形ROI。`src/visualization/stereo.py`因画图包初始化引用而随公共依赖保留，不新增1A实验。

Step 2/4与实施计划属于人工编写的项目说明，可以随代码发布；其中提到的本地结果文件全部排除。导出仅整理说明副本中未发布附件的链接，不改动本地Step 2/4，也不将运行结果复制进上传包。

## 2. 留在本地的内容

| 不上传 | 保留方式 |
|---|---|
| 自录原视频、解码RGB、NYUv2 MAT/split本体、GT | 全部留在本地，连同选帧manifest、视频流检查和源视频清单也不上传；队友自行准备输入 |
| DA-V2、VDA、RAFT及其他模型权重 | 全部留在本地，代码配置保留必要的权重路径与SHA约定 |
| 第三方仓库及其`.git` | 按固定URL/commit另行获取，不嵌套提交 |
| 原始预测NPY/NPZ、aligned数组、flow数组、mask和背景数组 | 全部留在本地，不上传任何预测或中间数组 |
| 实验JSON、CSV、文字报告及运行快照 | 包括summary、run、protocol、verify、逐图/逐对指标、自动analysis.md、provenance等，全部不上传 |
| PNG/JPG、MP4和PDF可视化 | 全部不上传，包括深度图、曲线、差分、案例图和处理后视频 |
| 烟测/维护/清理记录、完整机器日志、临时文件、Conda目录、IDE设置、认证文件 | 不纳入白名单 |
| Part 3、队友克隆目录`github_repository/` | 本次不纳入 |
| 根`README.md`、`PLAN.md`、`implemented.md`、`.gitignore`、旧`EXPORT_MANIFEST.json` | 保留目标分支原文件；本次使用单独的概览和导出清单 |

导出包中不会有`outputs/`、`data/`或`checkpoints/`目录，也不扫描这些目录。它是**代码与复现说明包**；配置可以描述预期输出路径，但不会连带复制那些路径的文件。队友需自己准备数据与权重，在新输出目录运行；没有本地产物时不能直接执行结果复核或续跑。

原生视频始终为`relative / arbitrary / alignment:none`。NYUv2的1B结果为测试GT辅助对齐的`aligned_relative`，不与1A/2A的原生米制主结果混称。

## 3. 在工作站生成独立导出副本

以下步骤只读实验项目，在项目之外准备文件。不启动推理、训练或下载。在同一个Bash终端执行：

```bash
set -euo pipefail
export PROJECT_SOURCE='/home/user/cjx/3201_Project'
export UPLOAD_WORK="$(mktemp -d /tmp/aiaa3201-part1b-part2b-upload.XXXXXX)"
export EXPORT_DIR="$UPLOAD_WORK/export"
printf '导出工作目录：%s\n' "$UPLOAD_WORK"
```

下面的Python块就是本次准确白名单。源码与配置按原字节复制；仅导出副本中的Markdown取消未发布附件的链接，并生成专用概览和源码导出清单。导出清单只含代码文件路径、字节数和SHA，没有模型结果或视频记录。单个文件限制10 MiB，超出时停止，不自动漏掉。

```bash
conda run --no-capture-output -n aiaa3201 python - <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil

source = Path(os.environ['PROJECT_SOURCE']).resolve()
out = Path(os.environ['EXPORT_DIR']).resolve()
assert source.is_dir()
assert not out.is_relative_to(source) and not source.is_relative_to(out)
out.mkdir(parents=True, exist_ok=False)
selected = set()

def add(relative):
    p = source / relative
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(source):
        raise ValueError(f'缺少普通项目文件或路径越界：{relative}')
    if p.stat().st_size > 10 * 1024**2:
        raise ValueError(f'单文件超过本次10 MiB上限：{relative}')
    selected.add(p.relative_to(source).as_posix())

def group(pattern):
    paths = sorted(p for p in source.glob(pattern) if p.is_file())
    if not paths:
        raise FileNotFoundError(pattern)
    for p in paths:
        add(p.relative_to(source))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

for name in (
    'environment.yml', 'requirements-core.txt', 'pytest.ini',
    'configs/versions.yaml', 'configs/evaluation_protocol.yaml',
    'configs/output_schema.yaml', 'configs/part1/dav2_nyuv2.yaml',
    'configs/part2/video_depth_anything.yaml',
    'configs/video/self_recorded.yaml', 'configs/video/clips.yaml',
    'configs/video/relative_video.yaml',
    'scripts/part1/run_dav2_nyuv2.sh', 'scripts/part1/video.sh',
    'scripts/part2/video.sh',
    'pipelines/part1/dav2_nyuv2.py', 'pipelines/part1/video_clip.py',
    'pipelines/part2/video_clip.py',
    'src/__init__.py', 'src/evaluation/__init__.py',
    'src/evaluation/relative_nyuv2.py',
    'src/models/__init__.py', 'src/models/depth_anything_v2_adapter.py',
    'src/models/video_depth_anything_adapter.py',
    'src/data/__init__.py', 'src/data/nyuv2.py', 'src/data/output_schema.py',
    'src/metrics/__init__.py', 'src/metrics/depth.py',
    'src/metrics/boundary.py', 'src/metrics/temporal.py',
    'src/metrics/relative_alignment.py',
    'src/visualization/__init__.py', 'src/visualization/nyuv2.py',
    'src/visualization/stereo.py',
    'docs/step2.md', 'docs/step4.md', 'plan_1B2B.md',
    'docs/github_upload_part1b_part2b.md',
):
    add(name)
group('src/video/*.py')
group('pipelines/video/*.py')
for name in (
    'configs', 'relative_alignment', 'dav2_nyuv2', 'temporal_metrics',
    'video_clip_workflow', 'video_frames', 'video_flow', 'video_runner',
    'video_evaluation', 'video_report', 'video_adapters',
):
    add(f'tests/test_{name}.py')

# 仅允许源码、配置和人工说明；即使今后扩充白名单也不能带入运行产物。
allowed = {'.py', '.sh', '.yaml', '.yml', '.md', '.txt', '.ini'}
blocked_roots = {'outputs', 'data', 'checkpoints', 'github_repository'}
for relative in sorted(selected):
    assert Path(relative).suffix in allowed, relative
    assert Path(relative).parts[0] not in blocked_roots, relative
    assert 'part3' not in Path(relative).parts, relative
    destination = out / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / relative, destination)

# 没有发布的课程原文、原始预测和图片用普通路径注明，避免附件死链接。
link = re.compile(r'\[([^\]\n]+)\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)')
for relative in sorted(r for r in selected if r.endswith('.md')):
    p = out / relative
    def readable(match):
        label, target = match.groups()
        target = target.strip('<>')
        if re.match(r'^(?:https?|mailto|app):', target):
            return match.group(0)
        path = target.split('#', 1)[0]
        destination = (p.parent / path).resolve() if path else p
        if destination.is_relative_to(out) and destination.is_file():
            return match.group(0)
        return f'{label}（本地保留：`{target}`）'
    p.write_text(link.sub(readable, p.read_text(encoding='utf-8')), encoding='utf-8')

(out / 'docs/part1b_part2b_overview.md').write_text(
    '# Part 1B与Part 2B：分支阅读入口\n\n'
    '[Part 1B：协议与运行](step2.md)；'
    '[Part 2B：VDA与逐帧模型比较](step4.md)；'
    '[上传范围和操作](github_upload_part1b_part2b.md)。\n\n'
    '本次仅发布代码、配置、环境、测试和人工复现说明；'
    '不发布视频、RGB、GT、权重、预测、中间数组、运行JSON/CSV、自动报告或其他实验产物。'
    '说明中引用的结果文件仅在本地保留。\n\n'
    '1B：`scripts/part1/video.sh` → `outputs/part1/video/<ID>/`；'
    '2B：`scripts/part2/video.sh` → `outputs/part2/video/<ID>/`。'
    '这些是执行时生成的目录，不在本次Git上传包中。'
    '2B只读引用完整1B，不自动补跑1B。主环境为aiaa3201。\n\n'
    'NYUv2评价为GT辅助aligned_relative；视频为无GT的relative，alignment:none。'
    '首次复现先准备自己的输入与权重，再运行1B，完成后按同视频ID运行2B。'
    '新ID不在clips.yaml时，先用1B的--video/--type登记。\n\n'
    '先按分支已有Step 0恢复上游与主环境；environment.yml依赖本地ml-depth-pro目录。'
    'FFmpeg/ffprobe需要位于PATH。已有资源先校验，缺少时再按已有说明获取。\n', encoding='utf-8')

records = []
for p in sorted(f for f in out.rglob('*') if f.is_file()):
    relative = p.relative_to(out).as_posix()
    original = source / relative if relative in selected else None
    a, b = sha(original) if original else None, sha(p)
    if a != b and original and p.suffix != '.md':
        raise ValueError(f'非说明文件不得改写：{relative}')
    records.append({'path': relative, 'source_sha256': a, 'exported_sha256': b,
                    'reader_text_adjusted': bool(original and a != b),
                    'generated_for_export': original is None, 'bytes': p.stat().st_size})
(out / 'PART1B_PART2B_EXPORT.json').write_text(json.dumps({
    'kind': 'incremental_part1b_part2b_code_only_export', 'target_branch': 'JiongxuChen',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'note': '仅代码、配置、环境、测试与复现说明；无视频、数据、权重或运行产物。',
    'files': records,
}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'完成：{len(records)+1}个文件，{sum(p.stat().st_size for p in out.rglob("*") if p.is_file())/1024**2:.2f} MiB')
print(out)
PY
```

如果白名单文件缺失或SHA不一致，先停止并核对对应文件。导出与视频是否存在、实验是否完成无关，不读取运行状态，不执行模型。此前按含结果版本生成的导出目录或压缩包不要继续使用，请从第3步重新生成。

## 4. 检查内容；需要时传回本地电脑

```bash
find "$EXPORT_DIR" -type f | sort
du -sh "$EXPORT_DIR"
cat "$EXPORT_DIR/docs/part1b_part2b_overview.md"
```

文件树中只能出现代码、配置、说明、测试及`PART1B_PART2B_EXPORT.json`源码清单。不能出现`outputs/`、`data/`、`checkpoints/`、视频、NPY/NPZ或实验JSON/CSV。清单程序和提交前检查都会拒绝这些内容。

**可在工作站直接进行第5步，也可用本地电脑上传。** 如果工作站此前报`Permission denied (publickey)`，那是SSH账户认证失败；改用本地已有GitHub登录的电脑很方便，不必为本次上传建立反向隧道。后半流程需要Git、Python 3.10或更高版本；Windows用Git Bash，若Python命令为`python`，将下面的`python3`替换为`python`。

在工作站打包，输出仍在项目之外：

```bash
tar -czf "$UPLOAD_WORK/part1b_part2b.tar.gz" -C "$UPLOAD_WORK" export
(
  cd "$UPLOAD_WORK"
  sha256sum part1b_part2b.tar.gz > part1b_part2b.tar.gz.sha256
)
printf '请下载：%s\n%s\n' \
  "$UPLOAD_WORK/part1b_part2b.tar.gz" \
  "$UPLOAD_WORK/part1b_part2b.tar.gz.sha256"
```

用VS Code远程文件列表下载这两个文件，或在本地执行`scp`，将工作站地址和下面`/tmp/实际工作目录/`换成刚才打印的位置：

```bash
scp 'user@工作站地址:/tmp/实际工作目录/part1b_part2b.tar.gz' .
scp 'user@工作站地址:/tmp/实际工作目录/part1b_part2b.tar.gz.sha256' .
```

下载后，在本地进入存放压缩包的目录，开Bash终端执行；之后第5–7步也在这个终端完成。选择工作站直接上传时，跳过这个解包块，沿用原来的`UPLOAD_WORK`、`EXPORT_DIR`。

```bash
set -euo pipefail
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c part1b_part2b.tar.gz.sha256
else
  shasum -a 256 -c part1b_part2b.tar.gz.sha256
fi
export UPLOAD_WORK="$(mktemp -d /tmp/aiaa3201-local-part1b-part2b.XXXXXX)"
export EXPORT_DIR="$UPLOAD_WORK/export"
tar -xzf part1b_part2b.tar.gz -C "$UPLOAD_WORK"
test -f "$EXPORT_DIR/PART1B_PART2B_EXPORT.json"
```

## 5. 独立克隆JiongxuChen分支

使用已有的HTTPS凭据登录，需要对该分支有写权限。GitHub登录密码不作为Git的HTTPS密码，使用凭据管理器或自己的访问令牌登录；不要把令牌写进仓库URL。SSH已配置好时也可替换为同仓库SSH地址。[GitHub官方HTTPS说明](https://docs.github.com/en/get-started/git-basics/about-remote-repositories#cloning-with-https-urls)

```bash
export REPO_URL='https://github.com/Xiaofeipao/monocular-depth-video-consistency.git'
export TARGET_BRANCH='JiongxuChen'
export BRANCH_DIR="$UPLOAD_WORK/branch"

test "$TARGET_BRANCH" = 'JiongxuChen'
git ls-remote --exit-code --heads "$REPO_URL" "refs/heads/$TARGET_BRANCH"
git -c core.autocrlf=false clone --single-branch --branch "$TARGET_BRANCH" \
  --no-recurse-submodules -- "$REPO_URL" "$BRANCH_DIR"
git -C "$BRANCH_DIR" config core.autocrlf false
git -C "$BRANCH_DIR" config core.pager cat

test "$(git -C "$BRANCH_DIR" branch --show-current)" = "$TARGET_BRANCH"
test -z "$(git -C "$BRANCH_DIR" status --porcelain)"
git -C "$BRANCH_DIR" rev-parse HEAD > "$UPLOAD_WORK/original-head.txt"
git -C "$BRANCH_DIR" branch "backup-before-1b2b-$(date +%Y%m%d-%H%M%S)"
```

这里新克隆指定分支，备份分支只保存在本地。关闭自动换行转换是为了让导出源码与配置的字节SHA保持一致，避免Windows上传改掉冻结科学源码。Git文件模式会在下一步单独设置。分页器已关闭；其他终端若出现`(END)`，按`q`退出。[Git官方clone说明](https://git-scm.com/docs/git-clone)

## 6. 复制白名单、检查差异并暂存

以下程序只覆盖清单明确列出的路径，不删除其他文件，不覆盖分支根README、LICENSE或`.gitignore`。已有同名共享文件会更新为本地项目版本，输出会列出所有新增/变化路径；提交前重点检查这些共享源码、环境依赖和配置是否包含需要保留的分支修改。

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib
import json
import os
import shutil

out = Path(os.environ['EXPORT_DIR']).resolve()
branch = Path(os.environ['BRANCH_DIR']).resolve()
manifest = json.loads((out / 'PART1B_PART2B_EXPORT.json').read_text(encoding='utf-8'))
assert manifest['kind'] == 'incremental_part1b_part2b_code_only_export', '请重新生成只含代码的新包'
assert manifest['target_branch'] == os.environ['TARGET_BRANCH'] == 'JiongxuChen'
assert (branch / '.git').is_dir() and branch != out
# 本文是增量包；公共配置测试继续使用分支已有的1A/2A配置。
for name in ('configs/part1/middlebury_sgbm.yaml',
             'configs/part2/depth_pro_nyuv2.yaml',
             'configs/part2/unidepth_v2_nyuv2.yaml',
             'configs/part2/key_analysis.yaml'):
    if not (branch / name).is_file():
        raise FileNotFoundError(f'先核对已有1A/2A分支内容，缺少：{name}')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
paths = []
for record in manifest['files']:
    relative = record['path']
    p = Path(relative)
    assert not p.is_absolute() and '..' not in p.parts and '.git' not in p.parts
    assert p.parts[0] not in {'outputs', 'data', 'checkpoints', 'github_repository'}
    assert p.suffix in {'.py', '.sh', '.yaml', '.yml', '.md', '.txt', '.ini'}, relative
    assert 'part3' not in p.parts, relative
    original, target = out / p, branch / p
    assert original.is_file() and not original.is_symlink()
    assert sha(original) == record['exported_sha256'], relative
    # 先检查所有路径，拒绝通过符号链接覆盖清单之外的内容。
    assert not target.is_symlink() and all(not q.is_symlink() for q in target.parents), relative
    assert target.resolve().is_relative_to(branch), relative
    assert not target.exists() or target.is_file(), relative
    if not target.exists():print('A', relative)
    elif sha(target) != sha(original):print('M', relative)
    paths.append(relative)
paths.append('PART1B_PART2B_EXPORT.json')
# 导出清单本身不是其files成员，在副本内原样复制。
target = branch / paths[-1]
assert not target.is_symlink() and (not target.exists() or target.is_file())
for relative in paths:
    target = branch / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / relative, target)
Path(os.environ['UPLOAD_WORK'], 'part1b-part2b-paths.nul').write_bytes(
    b''.join(p.encode('utf-8') + b'\0' for p in paths))
print(f'白名单共{len(paths)}个文件；其余分支文件不删除。')
PY

# 仅暂存白名单中的源码与说明，不强制添加被忽略的文件。
git -C "$BRANCH_DIR" add \
  --pathspec-from-file="$UPLOAD_WORK/part1b-part2b-paths.nul" --pathspec-file-nul
git -C "$BRANCH_DIR" update-index --chmod=+x -- \
  scripts/part1/run_dav2_nyuv2.sh scripts/part1/video.sh scripts/part2/video.sh

git -C "$BRANCH_DIR" --no-pager diff --cached --stat
git -C "$BRANCH_DIR" --no-pager diff --cached --name-status
git -C "$BRANCH_DIR" --no-pager diff --cached -- \
  environment.yml requirements-core.txt configs/part2/video_depth_anything.yaml \
  src/data src/metrics src/models src/visualization
```

如果共享文件中的队友修改需要合并，先停止提交，核对具体差异。不要对整个分支运行`git rm`、`rsync --delete`、`git reset --hard`或强推。本次也不自动删除远端曾有的旧实验文件。若你已经按含结果的版本上传过，后面的检查会发现已有1B/2B数据或产物并停止；需要先在独立克隆中核对具体跟踪路径并单独清理，不能继续把它称为无产物上传。本地完整项目中的视频和产物不删除。

再核对暂存区：所有导出文件内容必须匹配清单，只有三份入口脚本需要可执行位，不应产生子模块或符号链接；实际暂存变化不能超出白名单。[Git官方add与pathspec说明](https://git-scm.com/docs/git-add)

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib
import json
import os
import subprocess

branch = Path(os.environ['BRANCH_DIR'])
out = Path(os.environ['EXPORT_DIR'])
def git(*args):
    return subprocess.check_output(['git', '-C', str(branch), *args])
m = json.loads((out / 'PART1B_PART2B_EXPORT.json').read_text(encoding='utf-8'))
assert m['kind'] == 'incremental_part1b_part2b_code_only_export'
for r in m['files']:
    p = Path(r['path'])
    assert p.parts[0] not in {'outputs', 'data', 'checkpoints', 'github_repository'}
    assert p.suffix in {'.py', '.sh', '.yaml', '.yml', '.md', '.txt', '.ini'}
expected = {r['path']: r['exported_sha256'] for r in m['files']}
expected['PART1B_PART2B_EXPORT.json'] = hashlib.sha256((out / 'PART1B_PART2B_EXPORT.json').read_bytes()).hexdigest()
changed = {p.decode('utf-8') for p in git('diff', '--cached', '--name-only', '-z').split(b'\0') if p}
assert changed <= expected.keys(), changed - expected.keys()
# 旧版包若已经上传，增量复制不会自动删掉它们；发现后先停止。
artifact_roots = ('outputs/part1/video/', 'outputs/part2/video/',
                  'outputs/part1/video_summary/', 'outputs/part2/video_summary/',
                  'outputs/part1/dav2_nyuv2/', 'outputs/part2/video_depth_anything/',
                  'data/video_frames/', 'data/self_recorded/')
existing = [p.decode('utf-8') for p in git('ls-files', '-z').split(b'\0') if p]
unwanted = [p for p in existing if p.startswith(artifact_roots)]
assert not unwanted, '分支已有1B/2B数据或产物，请先单独核对清理这些跟踪路径：' + str(unwanted)
for relative, digest in expected.items():
    line = git('ls-files', '--stage', '--', relative).decode('utf-8').split()
    assert line and line[0] in ('100644', '100755'), relative
    assert hashlib.sha256(git('show', ':' + relative)).hexdigest() == digest, relative
for relative in ('scripts/part1/run_dav2_nyuv2.sh', 'scripts/part1/video.sh', 'scripts/part2/video.sh'):
    assert git('ls-files', '--stage', '--', relative).startswith(b'100755 '), relative
print('暂存区白名单、全部SHA与入口可执行位通过。')
PY
```

可在已有`aiaa3201`环境下，从新克隆根目录运行下面9份不依赖真实数据/权重/上游源码的合成测试。`test_configs.py`还需要已有1A/2A配置；`test_video_adapters.py`需要固定commit的DA-V2/VDA官方源码，准备后再单独运行。不要在本地电脑没有Conda时执行这些环境测试；上传不要求重新跑全片。

```bash
(
  cd "$BRANCH_DIR"
  conda run --no-capture-output -n aiaa3201 python -m pytest -q \
    tests/test_relative_alignment.py tests/test_dav2_nyuv2.py \
    tests/test_temporal_metrics.py tests/test_video_clip_workflow.py \
    tests/test_video_frames.py tests/test_video_flow.py tests/test_video_runner.py \
    tests/test_video_evaluation.py tests/test_video_report.py
)
```

## 7. 提交并仅推送目标分支

确认暂存差异符合预期后执行。Git未设置作者身份时，先只在这个克隆中设置真实姓名与提交邮箱，然后再commit。

```bash
# 仅在Git提示未设置作者时执行，并填写自己的真实信息：
# git -C "$BRANCH_DIR" config user.name '你的名字'
# git -C "$BRANCH_DIR" config user.email '你的提交邮箱'

test "$(git -C "$BRANCH_DIR" branch --show-current)" = 'JiongxuChen'
git -C "$BRANCH_DIR" commit \
  -m 'Add Part 1B and Part 2B code, configs and reproduction docs'
git -C "$BRANCH_DIR" push origin 'HEAD:refs/heads/JiongxuChen'

git -C "$BRANCH_DIR" status --short --branch
git -C "$BRANCH_DIR" rev-parse HEAD
git -C "$BRANCH_DIR" ls-remote origin 'refs/heads/JiongxuChen'
```

远端SHA应与本地HEAD一致。在GitHub网页切到`JiongxuChen`，打开`docs/part1b_part2b_overview.md`、Step 2/4和`PART1B_PART2B_EXPORT.json`源码清单，确认本次提交没有视频或1B/2B运行产物，并确认原来的1A/2A仍在。

这里的push只指定`JiongxuChen`，不更新`main`或其他分支。若远端已有新提交、保护规则拒绝推送或无写权限，保留当前克隆并先处理具体原因，不用`--force`覆盖。Git的普通推送会拒绝不满足要求的非快进更新。[Git官方push说明](https://git-scm.com/docs/git-push)

如需在根README加入入口，可之后在该独立克隆中补一个指向`docs/part1b_part2b_overview.md`的链接；本流程保留根README原文，不整份覆盖。

## 本次范围核验

验证以本页的只含代码版本为准，不沿用含结果包的文件数量和检查记录。项目内仅修改本Markdown，README、implemented、Step 2/4、代码、数据及原始运行元数据不改动；没有实际推送GitHub。

本次实际在项目之外的临时目录导出并检查：**67个文件，约0.50 MiB**。导出包没有`outputs/`、`data/`或`checkpoints/`文件，没有视频、图像、数组或实验JSON/CSV。临时Git仓库的白名单暂存、SHA与入口可执行位检查通过，98个非本次白名单文件保持不变；额外放入旧版运行产物后，提交前检查按预期拒绝继续。未启动模型、训练、下载或实际远端操作。
