# 将当前 Part 1 / Part 2 完整替换到 GitHub 分支

## 目标与上传范围

本流程将目标分支的**最新文件树完整替换**成当前 Part 1/2 的代码与说明：旧分支中未包含在本次上传包内的文件也会删除，包括旧隐藏文件。保留 Git 提交历史，采用普通提交和普通 push，不需要 force push。其他分支不变。

整个操作分为：工作站生成独立白名单包 → 传到能访问 GitHub 的电脑 → 在全新克隆目录替换文件 → 检查并推送。**不要在当前实验项目目录执行清空命令，也不要使用之前的上传包。** 本文提供操作流程，没有替你清空或推送远端。

下文沿用之前的仓库和分支作为示例；运行前确认它们仍是你的目标：

- 仓库：`https://github.com/Xiaofeipao/monocular-depth-video-consistency.git`
- 分支：`JiongxuChen`

| 内容 | 本次处理 |
|---|---|
| Part 1A：Middlebury BM/SGBM、调参、评估和核验实现 | 上传 |
| Part 1B：DA-V2-S 指定片段逐帧基线、诊断、NYUv2 对齐扩展 | 上传 |
| Part 2A：Depth Pro / UniDepth、NYUv2 评估与尺度/边界分析 | 上传 |
| Part 2B：VDA-S 离线 relative、独立分析、DA/VDA 比较及核验 | 上传 |
| 相关公共模块、测试、环境、协议、固定版本、Step 0–4 | 上传 |
| 当前课程 PDF、中文说明 | 上传；它们是完整课程要求，不是 Part 3 实现 |
| 文档中已有的指标表和文字分析 | 保留；不重新运行实验 |
| `data/`，包括指定视频、自录视频、数据集、抽帧缓存 | **全部不上传** |
| `checkpoints/`，包括权重与本地缓存 | **全部不上传** |
| `outputs/`，包括预测、图视频、CSV、JSON、日志和维护记录 | **全部不上传** |
| Part 3 代码、配置、实验计划与开发历史 | **不上传** |
| 四个官方第三方仓库、SAM2、队友仓库副本、环境目录、密钥 | **不上传** |

公共视频组件及其测试会随依赖一起上传，但当前正式视频入口只有 Step 2 和 Step 4 指定的脚本，不需要运行旧自录或公开视频工作流。

**结果状态要如实保留：**当前 1A/2A 已完成；1B/2B 已完成现有素材的 70 帧实验与核验，课程要求的 100 帧素材仍需确认。视频输出是相对深度；NYUv2 的 GT 辅助对齐扩展不能当作 native metric 成绩。

## 一、在工作站导出上传包

从当前项目根目录执行。下面只读取代码与文档，在项目外新建临时目录，不运行模型、不下载、不修改实验产物。需要 Python 3，不安装任何新包。

```bash
cd /home/user/cjx/3201_Project
export PROJECT_SOURCE="$PWD"
export EXPORT_PARENT="$(mktemp -d /tmp/aiaa3201-part12-XXXXXX)"

python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile

source = Path(os.environ['PROJECT_SOURCE']).resolve()
parent = Path(os.environ['EXPORT_PARENT']).resolve()
root = parent / 'part1_part2'
root.mkdir(exist_ok=False)
selected = set()

def select_file(name):
    path = source / name
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f'文件缺失或不是普通文件：{name}')
    selected.add(name)

for name in (
    'environment.yml', 'environment-unidepth.yml',
    'requirements-core.txt', 'requirements-unidepth.txt', 'pytest.ini',
    'Project3_CN_final.md',
    'Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf',
    'configs/versions.yaml', 'configs/evaluation_protocol.yaml',
    'configs/output_schema.yaml', 'configs/video/relative_video.yaml',
    'configs/video/public_clips.yaml', 'src/__init__.py',
    'scripts/prepare_data.sh', 'scripts/download_checkpoints.sh',
    'scripts/validate_data.py', 'scripts/smoke_models.py',
):
    select_file(name)

# 只收取指定目录中的代码/配置，不复制其中的缓存或其他文件。
for directory, patterns in {
    'configs/part1': ('*.yaml',), 'configs/part2': ('*.yaml',),
    'pipelines/part1': ('*.py',), 'pipelines/part2': ('*.py',),
    'pipelines/video': ('*.py',),
    'scripts/part1': ('*.sh',), 'scripts/part2': ('*.sh',),
    'scripts/video': ('*.sh',),
    **{f'src/{name}': ('*.py',) for name in (
        'analysis', 'data', 'evaluation', 'metrics', 'models',
        'stereo', 'video', 'visualization',
    )},
}.items():
    for pattern in patterns:
        for path in sorted((source / directory).rglob(pattern)):
            select_file(path.relative_to(source).as_posix())

# 明确列出 Part 1/2 测试，不把 Part 3 测试一起打包。
tests = '''boundary_metrics configs course_part1b course_part2b data_validation
    dav2_nyuv2 depth_metrics framewise_video geometry middlebury_pipeline
    nyuv2_data output_schema part2_failure_selection part2_key_analysis
    relative_alignment scale_boundary_analysis stereo_io stereo_matching
    stereo_metrics stereo_sensitivity stereo_coverage_study temporal_metrics video_adapters
    video_cli video_collection video_evaluation video_flow video_frames
    video_report video_runner'''.split()
for name in tests:
    select_file(f'tests/test_{name}.py')
for step in range(5):
    select_file(f'docs/step{step}.md')

for name in sorted(selected):
    destination = root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / name, destination)

# 发布包不含 outputs：把其链接改成明确的本地产物路径，避免 GitHub 破图。
# 这里只处理发布副本，源项目文档和自动运行元数据不变。
for path in sorted((root / 'docs').glob('step*.md')):
    text = path.read_text(encoding='utf-8')
    text = re.sub(
        r'!?\[([^\]]*)\]\(\.\./outputs/([^\s)]+)\)',
        lambda m: f'{m[1]}（本地产物：`outputs/{m[2]}`）', text,
    )
    text = text.replace('[当前结果审计](current_results.md)',
                        '当前结果审计（完整记录保存在源项目，本发布包不包含）')
    text = text.replace('工程根目录当前没有可用 Git 历史，不能仅克隆一个上游仓库就恢复项目自有代码。',
                        '应先取得本分支的项目自有代码，再按下列固定版本取得上游依赖。')
    text = text.replace('指定片段的2B比较待后续。',
                        '指定片段当前70帧的2B独立实验与比较已完成，见Step 4。')
    text = text.replace('指定片段的2B比较待后续，旧[Step 4](step4.md)仅保留兼容依赖。',
                        '指定片段当前70帧的VDA独立实验与同帧比较见[Step 4](step4.md)。')
    path.write_text(text, encoding='utf-8')

translation = root / 'Project3_CN_final.md'
translation.write_text(translation.read_text(encoding='utf-8').replace(
    '[difference_analysis.md](difference_analysis.md)',
    '源项目的 difference_analysis.md（未收入本发布包）；本包当前状态见README.md'),
    encoding='utf-8')

(root / 'README.md').write_text('''# AIAA 3201：Part 1 与 Part 2

本分支为经典双目、单目深度与视频时序比较的代码和说明快照。
仅包含 Part 1/2 及相关公共依赖，不包含数据、视频、权重和运行产物。

## 复现前需要准备什么

本分支只提供项目自有代码、配置、测试和说明。克隆后没有 data、checkpoints、outputs，
也没有官方模型仓库；请按下面顺序准备。所有路径均相对于本分支的项目根目录，
也就是包含 configs、scripts 和 src 的目录，后续命令均从这里执行。

### 1. 运行工具与环境

完整复现按 Linux / Bash、Conda、Python 3.10 和 NVIDIA CUDA GPU 准备。
数据下载/解压工具包括 git、wget、unzip；视频读写依赖随项目环境安装。
Windows Git Bash 可用于上传代码，实际实验建议使用 Linux 或具备GPU支持的WSL2；
后者未在本项目单独验证。Middlebury经典匹配和部分统计使用CPU，
但默认单目/视频模型配置使用CUDA和FP16，不能把它们直接当作CPU运行配置。
原实验显存和耗时只是参考，不能当作所有设备的最低资源要求。

先取得下一节的固定版本官方代码，再按 [Step 0：Conda环境](docs/step0.md#conda-环境)
创建 environment.yml 对应的 aiaa3201，以及 environment-unidepth.yml 对应的
 aiaa3201-unidepth。Depth Pro、DA-V2和VDA使用前者；UniDepth与联合尺度/边界分析
使用后者。两个环境分别安装各自依赖，不合并安装到base。
运行模型前在对应环境确认PyTorch能识别CUDA；具体检查见Step 0。

### 2. 官方模型代码

把四个官方仓库单独放到以下位置；它们不包含在本分支里。
固定commit和上游地址见 [configs/versions.yaml](configs/versions.yaml)，
克隆与checkout命令见 [Step 0](docs/step0.md#通用复现顺序)。

| 项目根目录下的位置 | 对应官方仓库 | 使用任务 |
|---|---|---|
| ml-depth-pro/ | apple/ml-depth-pro | 2A Depth Pro及官方边界评价依赖 |
| Depth-Anything-V2/ | DepthAnything/Depth-Anything-V2 | 1B逐帧与NYUv2扩展 |
| Video-Depth-Anything/ | DepthAnything/Video-Depth-Anything | 2B视频深度及指定素材来源 |
| UniDepth/ | lpiccinelli-eth/UniDepth | 2A UniDepth |

先放好 ml-depth-pro 再创建主环境，因为 environment.yml 会安装本地的
 -e ./ml-depth-pro。已有官方仓库先核对版本与自己的修改，不要直接覆盖或reset。

### 3. 数据放在哪里

| 数据 | 放置位置 | 应包含什么、用于什么 |
|---|---|---|
| Middlebury Stereo v3 Quarter | data/middlebury/MiddEval3/trainingQ/ | 15个场景子目录；每个场景含im0.png、im1.png、disp0GT.pfm、calib.txt、mask0nocc.png，用于1A |
| NYU Depth V2 labeled | data/nyuv2/nyu_depth_v2_labeled.mat | 完整1449张labeled MAT；程序读取RGB、填充深度及分析所需标注，不是只下载RGB图片 |
| NYUv2官方split | data/nyuv2/splits.mat | 官方795 train / 654 test划分；用于2A及1B的NYUv2对齐扩展 |
| 指定视频 | data/davis_rollercoaster.mp4 | 1B与2B使用同一文件；不需要另采自录视频 |

Middlebury和NYUv2的下载、解压及检查方法见 [Step 0：数据与权重](docs/step0.md#数据与权重)。
已有这些资源时可按表放好后校验，不必重新下载。不要多套一层目录，也不要换用Middlebury
其他分辨率或NYUv2其他split来匹配本项目的结果。

指定视频来自固定版本 Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4，
将该文件原样复制到 data/davis_rollercoaster.mp4；不要转码、裁剪、重采样或补帧。
预期SHA256为7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8，
对应配置为 [dav2_video.yaml](configs/part1/dav2_video.yaml) 和
[vda_video.yaml](configs/part2/vda_video.yaml)。此版本只有70个真实帧、24 FPS，
课程100帧要求待确认。不能以复制帧凑数或修改预期SHA绕过检查。

### 4. 预训练权重放在哪里

| 模型 | 项目根目录下的权重路径 | 使用任务 |
|---|---|---|
| DA-V2-Small relative | checkpoints/depth_anything_v2/depth_anything_v2_vits.pth | 1B视频与NYUv2扩展 |
| Depth Pro | checkpoints/depth_pro/depth_pro.pt | 2A |
| UniDepthV2-S | checkpoints/unidepth_v2_vits14/model.safetensors 和同目录config.json | 2A；两份文件均需准备 |
| VDA-Small relative | checkpoints/video_depth_anything/video_depth_anything_vits.pth | 2B；不要换成metric视频权重 |

来源、下载脚本与校验步骤见Step 0；固定SHA256和模型版本以 configs/part1、configs/part2
的对应YAML为准。准备脚本可能下载额外兼容资源；当前2B不需要RAFT、metric视频权重、
相机位姿或视频深度GT。1A的BM/SGBM不需要预训练权重。
手工放入已有资源时也需核对固定SHA，而不是仅用文件名或自己新生成的校验清单证明来源。

### 5. 哪些目录由程序生成

- data/video_frames/part1b_davis_v1/：按规定帧号和原PTS生成的RGB缓存，不必自己抽帧；2B会核对并复用或创建兼容缓存。
- data/manifests/：数据准备/验证产生的来源和检查记录，不是另一个需要手工寻找的数据集。
- outputs/part1/：本地生成的Middlebury、DA-V2视频和NYUv2扩展结果。
- outputs/part2/：本地生成的两条米制基线、尺度/边界分析、VDA独立结果与配对比较。

这些目录均不随本分支上传。程序生成后应保留原始预测和真实运行记录，
已有正式目录不要覆盖；复现实验使用步骤文档指定的新目录。
配对比较必须先具备1B和2B两条预测；完整2A尺度/边界分析必须先具备两条654张基线。
如果使用Step 3的 reproduce_v1 等新目录，分析配置中的 baselines 也必须指向新结果，
不能直接使用默认绑定 frozen_v1 的配置。Step 3已说明如何复制配置并指定 --config。

### 6. 开始实验前的检查

按Step 0完成环境、资源版本/SHA、数据结构与相关测试检查；
再按Step 2和Step 4的 check 入口核对视频、权重、官方代码及CUDA。
新克隆仓库没有任何历史预测，不能从 verify 或结果比较命令直接开始。
各任务首次运行、分析与核验的完整指令见下方Step文档，不在README重复维护另一套实验命令。

## 实验顺序与完整指令

1. 阅读 [Step 0](docs/step0.md)：先克隆固定版本的四个官方仓库，再创建环境。
   主环境为 aiaa3201，UniDepth 为 aiaa3201-unidepth，不安装到 base。
2. [Part 1A](docs/step1.md)：Middlebury BM/SGBM、开发集选参和独立评估。
3. [Part 1B](docs/step2.md)：指定片段 DA-V2-S 逐帧 relative 与 NYUv2 对齐扩展。
4. [Part 2A](docs/step3.md)：Depth Pro/UniDepth 原生米制基线与尺度/边界分析。
5. [Part 2B](docs/step4.md)：VDA-S 离线 relative 及 DA/VDA 同帧比较。

## 当前状态与协议

1A/2A 已完成；1B/2B 已完成现有官方素材的70帧实验与原始重算。
课程100帧素材要求仍待确认，不重复或插帧补足。
NYUv2 保持标准654张、冻结crop、有效范围与 alignment:none 的原生米制主轨道；
GT辅助对齐仅作诊断。视频没有深度GT，变化proxy不是深度准确率或真实尺度漂移真值。

Step文档保留已有指标与分析；其中“已在本地”“已经核验”和运行效率描述指源实验项目，
不表示刚克隆本分支就具备数据或通过核验。文档提到的 outputs 路径需在本地运行生成，
历史维护脚本和审计附件没有随本包上传。新复现应自行记录实际环境和核验结果。
第三方仓库按 configs/versions.yaml 固定版本单独获取，第三方许可证随各上游仓库取得。
下载脚本可能准备额外兼容资源，不代表当前2B需要米制视频模式或RAFT。
正式训练与长时预测由用户启动，不要覆盖已有正式输出目录。

## 目录

configs 为配置与协议，scripts 为命令入口，pipelines 为任务编排；
src 为公共实现，tests 为相关回归测试，docs 为完整步骤说明。
课程要求见 Project3_CN_final.md 与根目录课程PDF。
安装好官方依赖和相应环境后，可运行 `conda run -n aiaa3201 python -m pytest`。
本包测试范围小于源项目全部测试，不能沿用全项目测试数量。

发布文件清单与SHA256见 PART1_PART2_EXPORT.json；完成范围见 implemented.md。
''', encoding='utf-8')
(root / 'implemented.md').write_text('''# Part 1/2 实施范围摘要

本文件是发布范围摘要，不是一次新的实验执行记录。

- 1A：两算法完整开发搜索、四组15场景评估、原始输出独立核验。
- 1B：指定片段70帧DA-V2-S逐帧预测、固定色标、尺度/边界/时序诊断与原始重算；保留654张NYUv2对齐扩展。
- 2A：Depth Pro与UniDepthV2-S原生米制654张基线、尺度/物体边界分析与数值重放。
- 2B：指定片段70帧VDA-S独立实验、双方同帧比较及原始重算。

具体协议、结果、局限和命令见 docs/step1.md 至 step4.md。
100帧素材仍待课程确认。完整开发历史、原始预测与自动元数据保留在源项目，未随发布包上传。
''', encoding='utf-8')
(root / '.gitignore').write_text('''/data/
/checkpoints/
/outputs/
/ml-depth-pro/
/Depth-Anything-V2/
/Video-Depth-Anything/
/UniDepth/
/SAM2/
/github_repository/
.conda/
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.vscode/
.idea/
.env
.env.*
*.pem
*.key
*.log
*.tmp
*.part
.DS_Store
''', encoding='utf-8')
# Windows Git也保留导出文件的原始字节，便于比较SHA256。
(root / '.gitattributes').write_text('* -text\n', encoding='utf-8')

files = {}
for path in sorted(root.rglob('*')):
    if path.is_file():
        name = path.relative_to(root).as_posix()
        if 'part3' in path.parts or path.is_symlink():
            raise RuntimeError(f'意外路径：{name}')
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {'scope': 'part1_part2_code_docs_only', 'files': files,
            'note': 'files列出除本清单自身外的全部发布文件；文档副本含发布适配。'}
(root / 'PART1_PART2_EXPORT.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
archive = parent / 'part1_part2.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(root, arcname=root.name)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
(parent / 'part1_part2.tar.gz.sha256').write_text(
    f'{digest}  {archive.name}\n', encoding='utf-8')
print(f'发布文件数（不含清单自身）：{len(files)}')
print(f'上传包：{archive}')
print(f'SHA256：{digest}')
PY

printf '导出目录：%s\n' "$EXPORT_PARENT"
ls -lh "$EXPORT_PARENT/part1_part2.tar.gz"*
```

包内根 README 提供环境、四个官方仓库、数据与权重放置路径、自动生成目录及依赖顺序，具体实验命令链接到 Step 1–4；implemented 是 Part 1/2 发布摘要。**不会复制源项目含 Part 3 的整份 README、PLAN、AGENTS 和开发历史**。Step 0–4 的指标表和实验分析保留；导出副本将未上传的 outputs 链接改为本地产物路径，并修正 Step 0 的旧视频状态。源文件不受影响。

这是一份代码与说明包，不是离线可运行的数据包，也不是包含所有结果附件的课程最终提交 ZIP。

## 二、传到本地电脑并检查

在工作站输出中记下 `/tmp/aiaa3201-part12-xxxxxx` 的实际名称。在本地电脑打开 **Git Bash**（后续命令均使用 Bash，不是 PowerShell）。准备 Git 和 Python 3；先确认：

```bash
git --version
python --version
```

以下变量按实际值填写。工作站账号、地址和导出目录均需替换，不要原样复制占位值。

```bash
mkdir -p ~/aiaa3201-publish
cd ~/aiaa3201-publish
export WORKSTATION='user@100.124.129.113'
export REMOTE_EXPORT='/tmp/aiaa3201-part12-kDjmX9'

scp "$WORKSTATION:$REMOTE_EXPORT/part1_part2.tar.gz" .
scp "$WORKSTATION:$REMOTE_EXPORT/part1_part2.tar.gz.sha256" .
sha256sum -c part1_part2.tar.gz.sha256
```

必须显示 `OK`，再解压。传文件也可使用 IDE/SFTP；两个文件都要传，校验方式不变。工作站本身已能通过 HTTPS 访问仓库且有写权限时，也可以直接在工作站的独立目录进行以下操作；Python 命令改用 `python3` 即可。

```bash
# 确认当前目录下没有旧的同名解压目录，再执行。
test ! -e part1_part2 && tar -xzf part1_part2.tar.gz
export PACKAGE_DIR="$PWD/part1_part2"
if command -v cygpath >/dev/null 2>&1; then
  export PACKAGE_DIR="$(cygpath -am "$PACKAGE_DIR")"
fi
```

## 三、克隆目标分支到全新目录

```bash
export REPO_URL='https://github.com/Xiaofeipao/monocular-depth-video-consistency.git'
export TARGET_BRANCH='JiongxuChen'
export BRANCH_DIR="$PWD/target-branch-part12"
if command -v cygpath >/dev/null 2>&1; then
  export BRANCH_DIR="$(cygpath -am "$BRANCH_DIR")"
fi

# 必须能查到目标分支；失败就停在这里，处理地址、网络或认证。
git ls-remote --exit-code --heads "$REPO_URL" "refs/heads/$TARGET_BRANCH"

# 目录必须不存在；不要复用自己的项目目录或队友的工作目录。
if [ -e "$BRANCH_DIR" ]; then
  echo '请换一个不存在的新目录名，勿继续复用旧目录。'
else
  git clone --single-branch --branch "$TARGET_BRANCH" "$REPO_URL" "$BRANCH_DIR"
fi
```

若上一步提示目录存在，先重新设置 `BRANCH_DIR`，成功克隆后才继续。HTTPS 认证可使用本地 Git 凭据管理器或 Personal Access Token；GitHub 不接受账户密码作为 Git HTTPS 密码。参见 [GitHub 远程仓库与认证说明](https://docs.github.com/en/get-started/git-basics/about-remote-repositories)。先前的 SSH `Permission denied (publickey)` 是 SSH 身份认证失败，不等于工作站不能访问 GitHub。

检查地址、分支和工作区，并把旧提交号记在克隆目录外，方便需要时找回：

```bash
git -C "$BRANCH_DIR" remote -v
git -C "$BRANCH_DIR" branch --show-current
git -C "$BRANCH_DIR" status --short
git -C "$BRANCH_DIR" rev-parse HEAD > target-branch-before-upload.txt
```

确认分支名等于 `TARGET_BRANCH`，`status --short` 没有输出。若 Git 未配置提交身份，设置你自己的名字和 GitHub 邮箱：

```bash
git -C "$BRANCH_DIR" config user.name '你的名字'
git -C "$BRANCH_DIR" config user.email '你的GitHub邮箱或noreply邮箱'
```

## 四、完整替换并自动核对暂存内容

**这一段会删除 `$BRANCH_DIR` 内目标分支的旧文件。** 它保留 `.git` 和历史，并复制刚刚导出的白名单包。只允许对上一步的全新、干净克隆目录执行。

下面先核对包的完整文件集合和 SHA256，再清空克隆工作区并复制；接着核对 Git 暂存区与上传包完全一致。任何报错都先停下来，不要继续提交或推送。

```bash
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

package = Path(os.environ['PACKAGE_DIR']).resolve()
branch = Path(os.environ['BRANCH_DIR']).resolve()
target = os.environ['TARGET_BRANCH']

def git(*args):
    return subprocess.check_output(['git', '-C', str(branch), *args])

if branch == package or branch in package.parents or package in branch.parents:
    raise RuntimeError('上传包与克隆目录必须互相独立')
if not (branch / '.git').is_dir():
    raise RuntimeError('需要全新普通克隆目录，不能使用项目目录或worktree')
if git('branch', '--show-current').decode().strip() != target:
    raise RuntimeError('目标分支不匹配')
if git('status', '--porcelain'):
    raise RuntimeError('克隆目录不干净，请停止')
manifest_name = 'PART1_PART2_EXPORT.json'
manifest = json.loads((package / manifest_name).read_text(encoding='utf-8'))
if manifest['scope'] != 'part1_part2_code_docs_only':
    raise RuntimeError('上传包类型不匹配')
expected = dict(manifest['files'])
expected[manifest_name] = hashlib.sha256((package / manifest_name).read_bytes()).hexdigest()
actual = set()
for path in package.rglob('*'):
    if path.is_symlink():
        raise RuntimeError(f'包内不能包含符号链接：{path}')
    if path.is_file():
        actual.add(path.relative_to(package).as_posix())
if actual != set(expected):
    raise RuntimeError('上传包存在缺失或额外文件')
for name, digest in expected.items():
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts or '.git' in rel.parts:
        raise RuntimeError(f'非法路径：{name}')
    if hashlib.sha256((package / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'文件校验失败：{name}')

# 删除旧文件（含隐藏文件），保留Git数据库；只作用于独立克隆。
for path in branch.iterdir():
    if path.name == '.git':
        continue
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
shutil.copytree(package, branch, dirs_exist_ok=True)
git('add', '-A')
# Windows解压也能确保运行脚本在Git中的可执行权限正确。
for name in expected:
    if name.endswith('.sh'):
        git('update-index', '--chmod=+x', '--', name)
tracked = {name.decode('utf-8') for name in git('ls-files', '-z').split(b'\0') if name}
if tracked != set(expected):
    raise RuntimeError('暂存文件集合不等于发布清单，禁止提交')
for name, digest in expected.items():
    if hashlib.sha256(git('show', f':{name}')).hexdigest() != digest:
        raise RuntimeError(f'暂存文件内容不一致：{name}')
print(f'检查通过：{len(expected)}份暂存文件与上传包一致，旧版额外文件已移除。')
PY

git -C "$BRANCH_DIR" --no-pager diff --cached --stat
git -C "$BRANCH_DIR" --no-pager diff --cached --name-status
```

这里的 `git add -A` 会同时暂存新增、修改和删除，所以旧文件会随同一次提交从分支最新版本消失，而不是只覆盖同名文件。Git 删除与暂存的说明见 [git-rm 官方文档](https://git-scm.com/docs/git-rm)。出现 `D` 的旧文件属于预期；检查清单应仅包含本次 Part 1/2 内容。

本流程不建立“先推送空分支、再推送代码”的中间状态；一次提交即可把旧版完整替换成新版。若查看 Git 输出进入以 `(END)` 结尾的分页器，按 **q** 退出；本文查看命令已加 `--no-pager`。

## 五、提交并推送

检查通过后执行：

```bash
git -C "$BRANCH_DIR" commit -m 'Replace branch contents with current Part 1 and Part 2'
git -C "$BRANCH_DIR" push origin "HEAD:refs/heads/$TARGET_BRANCH"
```

若提示没有变更，说明目标分支已经是相同文件快照，不必制造空提交。push 使用明确的目标分支，不会把内容推到 main；正常 push 的更新规则见 [git-push 官方文档](https://git-scm.com/docs/git-push)。

**若 push 失败，先停下来：**

- 没有写权限或分支保护：处理仓库权限/保护规则后再提交，不改推到别人的分支。
- `non-fast-forward`：说明远端在克隆后出现了新提交；重新获取到新的独立克隆目录，重复“完整替换—核对—提交”，再普通推送。不要直接 force，也不要把旧文件合并回上传包。
- 网络/认证失败：先修复连接或登录，已创建的本地提交仍保留。

## 六、上传后核验

```bash
git -C "$BRANCH_DIR" rev-parse HEAD
git -C "$BRANCH_DIR" ls-remote --exit-code origin "refs/heads/$TARGET_BRANCH"
git -C "$BRANCH_DIR" status --short
git -C "$BRANCH_DIR" ls-tree --name-only HEAD
```

前两条的提交号应一致，工作区应干净。在 GitHub 手动切换到目标分支，确认：

1. 新 README、Step 0–4、Part 1/2 代码和配置存在。
2. 旧分支专有而本包没有的文件已消失。
3. 没有 `data/`、`checkpoints/`、`outputs/`、Part 3 实现和官方仓库副本。
4. Markdown 的结果表可读；图视频只以本地产物路径说明，没有实际上传附件。

GitHub 仍能从历史提交查看旧文件，这是保留提交历史的正常行为。**“全部替换最新内容”不等于“清除历史中的大文件或秘密”。** 本流程不重写历史，也不删除其他分支。当前工作站的完整实验项目和所有原始产物保持原样。
