# GitHub 分支上传：Step 0、Part 1A 与 Part 2A

本文只整理公共基础、Middlebury 经典双目、NYUv2 两模型基线及尺度与边界分析。目标是上传可阅读、可复现的代码、配置和小型结果证据。所有命令由你手动执行；编写本文时没有连接、清空或推送任何远端分支。

采用以下顺序：**准备独立导出副本 → 检查内容 → 克隆目标 branch → 清空并推送该 branch 的文件 → 上传导出副本**。清空表示删除该分支当前版本的全部已跟踪文件，提交历史仍保留。本地完整项目和其他远端分支不参与清理。

## 1. 上传什么

| 类别 | 本次纳入的内容 | 用途 |
|---|---|---|
| 环境 | `environment.yml`、`environment-unidepth.yml`、两份直接依赖 requirements、`pytest.ini` | 主环境与 UniDepth 隔离环境 |
| 基础配置 | `configs/versions.yaml`、`evaluation_protocol.yaml`、`output_schema.yaml`、`paths.example.yaml` | 上游版本、公共协议、格式和相对路径约定 |
| Part 1A | `configs/part1/`、`pipelines/part1/`、`scripts/part1/`、`src/stereo/` | BM/SGBM 调参、评价及几何分析 |
| Part 2A | 两模型、`key_analysis.yaml`与`scale_boundary.yaml`配置、`pipelines/part2/`、`scripts/part2/`、`src/models/` | 原生米制推理与完整尺度/物体边界分析 |
| 公共实现 | `src/data/`、`src/metrics/`、`src/visualization/` 和包初始化文件 | 数据方向、有效域、指标、输出规则和画图 |
| 准备入口 | 数据/权重准备、数据校验及模型 smoke 脚本 | 在资源缺失时恢复本地资源 |
| 公共测试 | 下方明确列出的 13 份测试 | 检查基础协议和 Part 1A/2A 实现 |
| 说明 | 分支专用 README，`docs/step0.md`、`step1.md`、`step3.md`、当前结果审计的相关部分 | 使用方法、实际结果和来源 |
| 资源证据 | 数据来源、数据校验结果、下载 SHA256、权重 SHA256 | 确认资源身份，不包含数据/权重本体 |
| 小型结果 | Step 0 的包清单和 smoke 摘要；Part 1A 调参/正式 JSON、CSV；Part 2A 两基线和四分析的 JSON、CSV、解析配置 | 保留完整精度表格、参数、实际环境和运行来源 |

保留 `src/metrics/temporal.py`及其公共回归；旧自录配置`configs/part2/video_depth_anything.yaml`已清理，不纳入新上传清单。`scripts/smoke_models.py` 检查多个预训练模型，需要 Step 0 中列出的四个上游；这些公共依赖不表示视频实验已完成。

现有根 README、PLAN、implemented 和若干目录索引覆盖整个项目，不直接复制到这个局部分支。下方生成分支专用 README，并只调整**导出副本**中的说明范围与链接；原项目说明及历史记录保持原样。

## 2. 不上传什么

| 不纳入内容 | 处理方式 |
|---|---|
| `data/downloads/`、`data/middlebury/`、`data/nyuv2/`、视频及解码帧 | 留在本地；上传获取说明、版本和 checksum |
| `.pt`、`.pth`、`.safetensors`、其他 checkpoint 本体 | 留在本地；上传下载入口与 checksum |
| `ml-depth-pro/`、`UniDepth/`、`Depth-Anything-V2/`、`Video-Depth-Anything/` 等第三方仓库 | 不复制、不嵌套提交；通过 `configs/versions.yaml` 的 URL/commit 恢复 |
| 所有 Part 3 代码、配置、测试、环境、图和输出 | 本次完全排除 |
| `prediction.npz`、逐图预测/metadata 目录、全量可视化、RGB/GT 导出 | 留在本地，GitHub 仅保存小型结果证据 |
| 早期试跑、重复版本、临时目录、缓存、日志、虚拟环境、IDE 配置 | 不纳入 |
| SSH 私钥、token、`.env`、认证文件、带凭据 URL | 不纳入 |
| `nvidia-smi-q.txt`、完整 Conda explicit 导出 | 默认不纳入；基础环境用 YAML/requirements，实际运行环境用对应 JSON |
| `requirements-lock.txt` | 是早期包清单，本次不将其当成当前环境锁文件上传 |
| 课程 PDF 和完整中文译文 | 默认不纳入；如需共享，按课程允许的范围另行添加 |

GitHub 对普通 Git 文件超过 50 MiB 给出提示，超过 100 MiB 会拒绝。本文的导出步骤更保守，逐文件限制为 10 MiB；这只是本次上传策略，不是 GitHub 的限制。[GitHub 官方文件大小说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

本分支保留的是“代码＋配置＋结果证据”，不是完整的离线实验包。没有原始预测时，不能直接执行 `--mode evaluate` 或 `--mode analyze` 来复评；需在本地补齐对应原始产物，或按说明重新生成预测。

## 3. 设置路径、仓库与 branch

在同一个 Bash 终端依次执行后续步骤。先将仓库地址和 branch 改为你的实际值。示例假定目标 branch 已存在；不存在时，先在 GitHub 创建该 branch，再继续。

示例使用 SSH 地址，需已配置 GitHub SSH 认证，并对目标 branch 有写权限；如使用 HTTPS，只替换 `REPO_URL`，通过 Git 的认证机制登录。

```bash
set -euo pipefail

export PROJECT_SOURCE='/home/user/cjx/3201_Project'
export REPO_URL='git@github.com:Xiaofeipao/monocular-depth-video-consistency.git'
export TARGET_BRANCH='JiongxuChen'

# 独立工作目录，位于完整项目之外
export UPLOAD_WORK="$(mktemp -d /tmp/aiaa3201-github-upload.XXXXXX)"
export EXPORT_DIR="$UPLOAD_WORK/export"
export BRANCH_DIR="$UPLOAD_WORK/branch"

git check-ref-format --branch "$TARGET_BRANCH"
printf '项目来源：%s\n导出位置：%s\n仓库：%s\n目标 branch：%s\n' \
  "$PROJECT_SOURCE" "$EXPORT_DIR" "$REPO_URL" "$TARGET_BRANCH"
```

当前完整项目根目录没有 Git 仓库。本流程不在这里执行 `git init`、清理或切换分支，避免将本地资源误加入版本管理。`/tmp` 工作目录只作临时上传准备，完整项目始终保留；后续上传工作也可采用同样的独立克隆方式。

## 4. 按清单生成导出副本

以下命令只读取项目并复制明确列出的文件。默认只带文本结果；若希望附带少量已生成的结果图，将代码中的 `include_figures = False` 改成 `True`。图中可能含 benchmark RGB，应按数据与课程允许的共享范围选择。

```bash
conda run --no-capture-output -n aiaa3201 python - <<'PY'
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

source = Path(os.environ['PROJECT_SOURCE']).resolve()
destination = Path(os.environ['EXPORT_DIR']).resolve()
include_figures = False
assert source.is_dir() and not destination.is_relative_to(source)
assert not source.is_relative_to(destination)
destination.mkdir(exist_ok=False)

patterns = [
    'environment.yml', 'environment-unidepth.yml',
    'requirements-core.txt', 'requirements-unidepth.txt', 'pytest.ini',
    'configs/versions.yaml', 'configs/evaluation_protocol.yaml',
    'configs/output_schema.yaml', 'configs/paths.example.yaml',
    'configs/part1/*.yaml', 'configs/part1/README.md',
    'configs/part2/*.yaml', 'configs/part2/README.md',
    'src/__init__.py',
    'scripts/prepare_data.sh', 'scripts/download_checkpoints.sh',
    'scripts/validate_data.py', 'scripts/smoke_models.py',
    'docs/step0.md', 'docs/step1.md', 'docs/step3.md',
    'docs/current_results.md',
    'data/manifests/sources.yaml', 'data/manifests/downloads.sha256',
    'data/manifests/validated_datasets.json', 'checkpoints/checksums.sha256',
    'outputs/step0/model_smoke.json', 'outputs/step0/pip-packages.json',
]
for folder in ('analysis', 'data', 'metrics', 'models', 'stereo', 'visualization'):
    patterns.append(f'src/{folder}/**/*.py')
for part in ('part1', 'part2'):
    patterns.extend((f'pipelines/{part}/**/*.py', f'scripts/{part}/*.sh'))
for name in (
    'boundary_metrics', 'configs', 'data_validation', 'depth_metrics',
    'scale_boundary_analysis',
    'geometry', 'nyuv2_data', 'output_schema', 'part2_failure_selection',
    'stereo_io', 'stereo_matching', 'stereo_metrics',
    'stereo_sensitivity', 'middlebury_pipeline', 'temporal_metrics',
):
    patterns.append(f'tests/test_{name}.py')

middlebury = 'outputs/part1/middlebury'
baseline_roots = [f'outputs/part2/{m}/frozen_v1' for m in (
    'depth_pro_nyuv2', 'unidepth_v2_nyuv2',
)]
analysis = 'outputs/part2/key_analysis_v1'
complete_analysis = 'outputs/part2/scale_boundary_v1'
patterns.extend((f'{complete_analysis}/*.json', f'{complete_analysis}/config.yaml',
                 f'{complete_analysis}/report.md', f'{complete_analysis}/summary/*.csv',
                 f'{complete_analysis}/tables/*.csv', f'{complete_analysis}/figures/*.png',
                 f'{complete_analysis}/figures/*.jpg', f'{complete_analysis}/cases/**/*.png'))
for extension in ('json', 'csv'):
    patterns.extend((f'{middlebury}/evaluation/*.{extension}',
                     f'{middlebury}/tuning/*/*.{extension}'))
    patterns.extend(f'{run}/*.{extension}' for run in baseline_roots)
    patterns.extend((f'{analysis}/absolute_scale/*.{extension}',
                     f'{analysis}/boundary/*.{extension}'))
patterns.extend((f'{middlebury}/tuning/summary.json',
                 f'{analysis}/config_resolved.yaml',
                 f'{analysis}/boundary/cases/test_0021_mat_0040_depth_pro_advantage.png',
                 f'{analysis}/boundary/cases/test_0545_mat_1234_unidepth_v2_advantage.png',
                 f'{analysis}/boundary/cases/test_0302_mat_0713_both_low_si_f1.png'))
if include_figures:
    for extension in ('png', 'jpg'):
        patterns.extend((f'{middlebury}/evaluation/*.{extension}',
                         f'{analysis}/absolute_scale/*.{extension}',
                         f'{analysis}/boundary/*.{extension}'))

selected = set()
for pattern in patterns:
    matches = [p for p in source.glob(pattern) if p.is_file()]
    if not matches:
        raise FileNotFoundError(f'清单无匹配，请先核对项目：{pattern}')
    selected.update(p.relative_to(source) for p in matches)
for relative in sorted(selected):
    original = source / relative
    if original.is_symlink() or original.stat().st_size > 10 * 1024**2:
        raise ValueError(f'请人工核对符号链接或大文件：{relative}')
    copied = destination / relative
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, copied)

# 只整理导出说明，不改源码、数值配置、结果或自动运行记录。
path = destination / 'docs/current_results.md'
text = path.read_text()
assert '\n## Part 3' in text
text = text.split('\n## Part 3', 1)[0]
text = text.replace('早期开发与执行历史见 [implemented.md](../implemented.md)。',
                    '本分支只展示 Step 0、Part 1A 与 Part 2A 的来源证据。')
text = text.replace('Part 1B/2B/3、Demo', 'Part 1B/2B、Demo')
path.write_text(text.rstrip() + '\n')

path = destination / 'docs/step0.md'
text = path.read_text()
text = text.replace('工程根目录当前没有可用 Git 历史，不能仅克隆一个上游仓库就恢复项目自有代码。',
                    '本分支保存项目自有代码；第三方实现仍需按固定 commit 单独取得。')
text = text.replace('当前结果为 **43 passed**，覆盖：',
                    '公共基础建立时的记录为 **43 passed**，覆盖：')
text = re.sub(r'## 暂缓的视频输入\n.*?(?=\n## )',
              '## 视频范围\n\n本分支不包含正式视频实验。公共时序指标和模型 smoke 仅作为 Step 0 基础保留，不能视为 Part 1B/2B 完成。\n',
              text, flags=re.S)
text = text.replace('Part 1/2/3 Python 入口', 'Part 1/2 Python 入口')
text = text.replace('Step 0–6 的操作与结果说明', 'Step 0、1、3 的操作与结果说明')
path.write_text(text)

path = destination / 'docs/step3.md'
text = path.read_text().replace('[PLAN.md](../PLAN.md)', '[本分支说明](../README.md)')
path.write_text(text)

(destination / 'README.md').write_text('''# AIAA 3201：Step 0、Part 1A 与 Part 2A

本分支保存公共基础、Middlebury BM/SGBM，以及 NYUv2 上 Depth Pro、UniDepthV2-S 两条原生米制基线和尺度与边界分析。

## 阅读与复现

- [Step 0：环境、上游版本、资源校验和公共测试](step0.md)
- [Part 1A：经典双目调参、评价与结果](step1.md)
- [Part 2A：两模型、尺度与边界分析与结果](step3.md)
- [完整精度结果与来源审计](current_results.md)

从本分支根目录运行命令。先按 Step 0 获取固定 commit 的第三方代码，再创建 `aiaa3201` 与 `aiaa3201-unidepth` 环境，准备数据/权重并验证 checksum。数据、权重、第三方源码和原始预测未随本分支发布，下载入口与固定身份保留。

## 状态和协议

Step 0 公开数据基础、Part 1A 四组和 Part 2A 两基线/四分析已有完成产物。视频、Demo、报告和最终提交不因本分支上传而视为完成。

NYUv2 保持官方 654 test、完整 RGB 输入、评价 crop `[45:471,41:601]`、`0.001 < GT < 10 m`，预测仅评分时 clamp 到 `[0.001,10] m`；主表为 `native metric / meter / alignment:none`。外部 calibrated K/f 与测试 GT 驱动的 oracle alignment 仅用于诊断。接受合理浮点容差，不为匹配旧数字修改预测或 evaluator。

## 结果与资源

`outputs/` 只包含小型 JSON/CSV、解析配置及可选少量结果图，不是完整运行目录。缺少原始预测时不能直接复评或运行只读缓存分析；需在本地补齐输入或重新生成。已有结果目录不要覆盖，复现输出另存新目录。

`data/manifests/` 和 `checkpoints/checksums.sha256` 记录来源与资源身份。自动元数据保留原执行时间、路径和版本，可能引用未发布的大文件；它们不表示本次上传重新执行实验。`EXPORT_MANIFEST.json` 区分源文件与导出副本 checksum，只有面向本分支的说明文字经过范围整理。

正式训练和微调由用户启动。模型推理、完整搜索与评价按 Step 文档手动执行；上传步骤不运行这些实验。
''')

(destination / '.gitignore').write_text('''__pycache__/
*.py[cod]
.pytest_cache/
.conda/
.venv/
.idea/
.vscode/
.DS_Store
.env
.env.*
*.pem
*.key
*.log
*.tmp
*.part
*.npz
*.npy
*.pt
*.pth
*.safetensors
*.mat
*.pfm
*.zip
*.tar*
*.mp4
*.mov
/ml-depth-pro/
/UniDepth/
/Depth-Anything-V2/
/Video-Depth-Anything/
/RAFT/
/SAM2/
/src/refinement/
/pipelines/part3/
/scripts/part3/
/configs/part3/
/outputs/part3/
/data/*
!/data/manifests/
/data/manifests/*
!/data/manifests/sources.yaml
!/data/manifests/downloads.sha256
!/data/manifests/validated_datasets.json
/checkpoints/*
!/checkpoints/checksums.sha256
/outputs/**/predictions/
/outputs/**/metadata/
/outputs/**/visualization/
/outputs/**/visualizations/
''')

digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
records = []
for relative in sorted(selected):
    original, copied = source / relative, destination / relative
    a, b = digest(original), digest(copied)
    if a != b and str(relative) not in (
        'docs/step0.md', 'docs/step3.md', 'docs/current_results.md',
    ):
        raise ValueError(f'不应变化的文件内容不一致：{relative}')
    records.append(dict(path=str(relative), source_sha256=a,
                        exported_sha256=b, reader_text_adjusted=(a != b)))
(destination / 'EXPORT_MANIFEST.json').write_text(json.dumps({
    'kind': 'github_scoped_file_export',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'note': '仅导出文件，不代表重新运行实验；README/.gitignore 为分支专用生成文件。',
    'files': records,
}, ensure_ascii=False, indent=2) + '\n')
files = [p for p in destination.rglob('*') if p.is_file()]
print(f'导出完成：{len(files)} 文件，{sum(p.stat().st_size for p in files) / 1024**2:.2f} MiB')
print(destination)
PY
```

该清单包含 Part 2A 的完整尺度/物体边界逐图表、汇总、来源配置和小型案例图，并保留key_analysis证据，不含原始预测。所有结果、数值配置及自动元数据按原字节复制。来源文件中保留的历史设备/路径说明是原记录，不是新的复现要求。

原项目 `.gitignore` 会忽略 `outputs/`，且没有完整排除第三方仓库；因此导出副本使用上面独立的 `.gitignore`。不要直接在完整项目运行 `git add .`，也不用 `git add -f outputs/`。

Part 1A的小型结果从 `outputs/part1/middlebury/evaluation/` 和 `tuning/` 导出；完整实验尚未运行时先完成Step 1，再执行需要结果文件的导出清单。

## 5. 检查导出内容，再处理远端

```bash
find "$EXPORT_DIR" -type f | sort
du -sh "$EXPORT_DIR"

# 查看分支首页和保留的实际结果来源
cat "$EXPORT_DIR/README.md"
cat "$EXPORT_DIR/outputs/part1/middlebury/evaluation/run_metadata.json"
cat "$EXPORT_DIR/outputs/part2/depth_pro_nyuv2/frozen_v1/run_metadata.json"
cat "$EXPORT_DIR/outputs/part2/unidepth_v2_nyuv2/frozen_v1/run_metadata.json"
```

确认文件范围符合预期，并检查准备公开的 JSON/YAML 中是否有不愿公开的路径或环境信息。需要省略时，只移除导出副本中的对应文件并说明未发布范围，不改写原始运行元数据。默认导出不包含认证文件；不要向仓库地址嵌入 token。

## 第 6 步的连接问题与本地上传方案

### `Permission denied (publickey)` 表示什么

这表示已经连接到 GitHub 的 SSH 服务，但 GitHub 没有接受当前客户端提供的账户密钥，常见原因是没有配置密钥、密钥没有加入 agent，或公钥没有登记到 GitHub 账户。它不能用来判断工作站无法联网。[GitHub 官方排查说明](https://docs.github.com/en/authentication/troubleshooting-ssh/error-permission-denied-publickey)

你看到的 ED25519 主机指纹 `SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU` 与官方公布值一致。接受这个主机指纹只是在确认服务器身份，不等于完成你的账户认证。[GitHub 官方主机指纹](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)

失败的 `git ls-remote` 是只读操作，没有清空远端文件。可以在工作站配置自己的 SSH 公钥后继续，也可以选择下面的本地上传或代理路线。**网络通路和 GitHub 账户认证是两件事；使用 VPN 或隧道不会自动获得仓库权限。**

### 推荐：把导出包传回本地电脑上传

已准备好一份可直接传输的导出包，包含 118 个范围内文件：

- [step0_part1a_part2a_20261028.tar.gz](../outputs/github_upload/step0_part1a_part2a_20261028.tar.gz)，907,496 字节，约 0.87 MiB。
- [对应 SHA256 校验文件](../outputs/github_upload/step0_part1a_part2a_20261028.tar.gz.sha256)。

压缩包中顶层目录为 `export/`。包内源码、数值配置和结果与原文件 checksum 一致；三份说明副本只整理范围和链接。它不包含数据、权重、原始预测或其他 Part 的实验实现，也没有你的 GitHub 凭据。使用这个现成包时，不需要重复第 3–5 步的导出。

**在本地电脑下载文件。** 如果正在使用 VS Code Remote SSH，可以在远端文件列表中找到 `outputs/github_upload/` 中这两个文件，右键下载。也可以在本地终端执行下面的 `scp`，将 `WORKSTATION_SSH` 改为平时连接工作站的 `user@地址` 或 SSH 别名；此处连接的是工作站，不需要工作站通过 GitHub 认证。

```bash
# 在本地电脑执行，两个文件下载到当前目录
export WORKSTATION_SSH='user@WORKSTATION_ADDRESS'
scp "${WORKSTATION_SSH}:/home/user/cjx/3201_Project/outputs/github_upload/step0_part1a_part2a_20261028.tar.gz" .
scp "${WORKSTATION_SSH}:/home/user/cjx/3201_Project/outputs/github_upload/step0_part1a_part2a_20261028.tar.gz.sha256" .
```

**以下都在本地电脑执行。** Windows 使用 Git Bash；macOS/Linux 使用终端。先进入存放这两个下载文件的目录，再执行：

```bash
set -euo pipefail

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c step0_part1a_part2a_20261028.tar.gz.sha256
else
  shasum -a 256 -c step0_part1a_part2a_20261028.tar.gz.sha256
fi

export UPLOAD_WORK="$(mktemp -d /tmp/aiaa3201-local-upload.XXXXXX)"
export EXPORT_DIR="$UPLOAD_WORK/export"
export BRANCH_DIR="$UPLOAD_WORK/branch"
export REPO_URL='https://github.com/Xiaofeipao/monocular-depth-video-consistency.git'
export TARGET_BRANCH='JiongxuChen'

tar -xzf step0_part1a_part2a_20261028.tar.gz -C "$UPLOAD_WORK"
test -f "$EXPORT_DIR/README.md"
test -f "$EXPORT_DIR/EXPORT_MANIFEST.json"
```

本地上传只需要 Git 和解压工具，不需要 Conda、数据集、权重或 GPU。推荐使用 HTTPS 和本地已有的 Git Credential Manager 登录；若本地装有 GitHub CLI，也可以执行：

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
```

登录的账户需要对该仓库及目标 branch 有写权限。公开仓库可读不代表可推送；HTTPS 不使用 GitHub 账户密码作为 Git 密码，不要把 token 拼进 URL。[GitHub 官方 HTTPS 认证说明](https://docs.github.com/en/get-started/git-basics/about-remote-repositories#cloning-with-https-urls)

随后在**同一个本地终端**继续执行下方原第 6、7、8 步：克隆目标 branch → 清空并推送 → 复制 `$EXPORT_DIR` → 提交并推送。这里的所有 Git 删除命令只作用于本地独立克隆，实验工作站上的完整项目保持原位。无需再在工作站执行这些 Git 步骤。

### 可选：工作站通过反向隧道 7900 使用本机代理

可以使用，但本机 VPN/代理软件必须确实提供 HTTP/Mixed 或 SOCKS 监听端口；仅开启 VPN 并不一定存在可供转发的代理端口。下面以**本机 HTTP/Mixed 端口 7890**为例，实际端口以软件设置为准。7900 是工作站上的隧道入口，不必与本机代理端口相同。

在**本地电脑**另开终端建立隧道并保持该终端运行；如已有相同映射，保留现有隧道即可，不重复占用 7900：

```bash
export WORKSTATION_SSH='user@WORKSTATION_ADDRESS'
export LOCAL_PROXY_PORT=7890
ssh -N -T -o ExitOnForwardFailure=yes \
  -R "127.0.0.1:7900:127.0.0.1:${LOCAL_PROXY_PORT}" "$WORKSTATION_SSH"
```

映射为“工作站 `127.0.0.1:7900` → SSH 隧道 → 本地电脑 `127.0.0.1:7890` 代理”。`-R` 在 SSH 远端监听，并将连接转回客户端一侧；若 SSH 服务拒绝转发，应先处理该连接配置，不把失败的隧道当成可用代理。[OpenSSH 官方远端转发说明](https://man.openbsd.org/ssh#R)

然后在**工作站原上传终端**执行：

```bash
export http_proxy='http://127.0.0.1:7900'
export https_proxy="$http_proxy"
export REPO_URL='https://github.com/Xiaofeipao/monocular-depth-video-consistency.git'

# 只检查网络；不会删除或推送文件
curl --connect-timeout 10 --max-time 30 -I https://github.com
git ls-remote --exit-code --heads "$REPO_URL" "refs/heads/$TARGET_BRANCH"
```

上述配置假定转发到 HTTP/Mixed 代理；如果本机端口只支持 SOCKS5，将代理 URL 改为 `socks5h://127.0.0.1:7900`。Git 的 HTTP(S) 传输读取这些代理变量；`git@github.com:...` 使用 SSH，不会因为设置它们而自动经过代理，所以这条路线使用 HTTPS URL。[Git 官方代理配置说明](https://git-scm.com/docs/git-config#Documentation/git-config.txt-httpproxy)

通过网络检查后，仍需完成工作站上的 HTTPS 认证，再继续第 6–8 步。如果已经改为本地电脑上传，就不用建立这个隧道，也不用在工作站配置 GitHub 认证。

本节的代理、隧道和账户登录命令是待执行说明；本次没有启动隧道，也没有实测当前 7900 端口、本机 VPN 设置或实际仓库写权限。

## 6. 克隆目标 branch，并先清空它

这一步开始操作远端。应先确认仓库、branch 和写权限；命令只在新建的 `$BRANCH_DIR` 执行。若某条命令失败，停止并处理失败，不继续复制或推送。

```bash
# 确认指定 branch 存在，并查看默认分支以核对目标
git ls-remote --exit-code --heads "$REPO_URL" "refs/heads/$TARGET_BRANCH"
git ls-remote --symref "$REPO_URL" HEAD

git clone --single-branch --branch "$TARGET_BRANCH" \
  --no-recurse-submodules -- "$REPO_URL" "$BRANCH_DIR"

git -C "$BRANCH_DIR" remote -v
git -C "$BRANCH_DIR" status --short --branch
test "$(git -C "$BRANCH_DIR" branch --show-current)" = "$TARGET_BRANCH"
test -z "$(git -C "$BRANCH_DIR" status --porcelain)"

# 在本地保留清空前的提交位置，不向其他远端分支写入
git -C "$BRANCH_DIR" rev-parse HEAD > "$UPLOAD_WORK/original-head.txt"
git -C "$BRANCH_DIR" branch "backup-before-upload-$(date +%Y%m%d-%H%M%S)"

# 删除目标 branch 的全部已跟踪文件，包括已跟踪的隐藏文件
git -C "$BRANCH_DIR" rm -r --ignore-unmatch -- .
test -z "$(git -C "$BRANCH_DIR" ls-files)"
git -C "$BRANCH_DIR" diff --cached --stat
git -C "$BRANCH_DIR" commit --allow-empty \
  -m 'Clear target branch before scoped project upload'

# 明确只更新这一个 branch
git -C "$BRANCH_DIR" push origin "HEAD:refs/heads/$TARGET_BRANCH"
```

此时 GitHub 上该 branch 的当前文件树已清空，可以在网页切换到该 branch 核对，然后执行下一步。`git rm` 删除被跟踪文件并将删除暂存，提交和推送使远端文件树更新；旧提交仍可查看。[Git 官方 git-rm 说明](https://git-scm.com/docs/git-rm)

若 Git 要求作者身份，只给这个独立克隆设置：

```bash
git -C "$BRANCH_DIR" config user.name '你的名字'
git -C "$BRANCH_DIR" config user.email '你的 GitHub 提交邮箱'
```

然后继续失败的 `commit` 和 `push`。若分支保护要求 PR，按该仓库的 PR 流程先合入文件删除；清空未完成前不继续上传。若推送被拒绝为远端已有新提交，先核对远端修改，不直接强推覆盖。

## 7. 上传已经检查好的副本

```bash
# 复制包含 .gitignore 在内的导出文件，目标克隆自身的 .git 保留
cp -a "$EXPORT_DIR/." "$BRANCH_DIR/"

git -C "$BRANCH_DIR" add --all

# Windows 本地上传时也为 shell 入口保留 Git 可执行位
git -C "$BRANCH_DIR" update-index --chmod=+x -- \
  scripts/prepare_data.sh scripts/download_checkpoints.sh \
  scripts/part1/tune_middlebury.sh scripts/part1/run_middlebury.sh \
  scripts/part2/run_depth_pro_nyuv2.sh scripts/part2/run_unidepth_v2_nyuv2.sh \
  scripts/part2/run_key_analysis.sh

git -C "$BRANCH_DIR" diff --cached --stat
git -C "$BRANCH_DIR" diff --cached --name-status

# 确认不会把第三方目录作为嵌套仓库或 submodule 提交
git -C "$BRANCH_DIR" ls-files --stage
```

上传清单应全部为普通文件；`ls-files --stage` 中不能有模式 `160000`。确认没有数据本体、权重、预测、Part 3 文件及意外新增文件后，再执行：

```bash
test "$(git -C "$BRANCH_DIR" branch --show-current)" = "$TARGET_BRANCH"
git -C "$BRANCH_DIR" commit \
  -m 'Add Step 0, Part 1A and Part 2A code, configs and result evidence'
git -C "$BRANCH_DIR" push origin "HEAD:refs/heads/$TARGET_BRANCH"
```

显式 `HEAD:refs/heads/目标分支` 将当前提交推到指定 branch；不要使用 `--all`、`--mirror` 或强推来扩大此次更新范围。[Git 官方 git-push refspec 说明](https://git-scm.com/docs/git-push)

## 8. 上传后核对

```bash
git -C "$BRANCH_DIR" status --short --branch
git -C "$BRANCH_DIR" log -2 --oneline
git -C "$BRANCH_DIR" rev-parse HEAD
git -C "$BRANCH_DIR" ls-remote origin "refs/heads/$TARGET_BRANCH"
git -C "$BRANCH_DIR" ls-tree -r --name-only HEAD
```

工作区应干净，远端目标 branch 的 SHA 应等于本地 `HEAD`。最近两条提交分别对应上传和清空。在 GitHub 网页切换到目标 branch，核对 README、Step 0/1/3 链接、两环境配置、Middlebury 调参/正式结果、两条 NYUv2 baseline 及尺度与边界分析。

本次上传不需要重新下载资源、运行 smoke、推理、搜索或评价。其他人若要复现，按分支 README 和 Step 文档自行准备本地资源；正式训练仍由用户启动。完整项目、数据、权重及所有本地实验产物保持原位。

## 编写时的检查范围

2026-10-28 已在临时目录实际运行上述导出逻辑：默认 118 个文件，约 2.55 MiB；开启少量图后 131 个文件，约 9.97 MiB。复制文件的 checksum 核对通过，差异仅限三份面向分支的说明副本；源码、数值配置和结果原字节保持一致。

已用临时本地 Git 仓库验证：隐藏的已跟踪文件也能清空；清空提交后再上传全部导出文件；另一个分支的提交不变；未产生嵌套仓库。文档链接、Bash 语法、导出 Python/YAML 解析通过。上述演练没有连接 GitHub，不代表你的实际远端 branch 已清空或上传成功。

同日准备上述本地传输包，SHA256 为 `f6cae64ce27db8a2aab5724cabbaee218e0ea319ff306b8cdad8bbdf15ef065c`。已实际校验、解压并核对包内文件，另在临时 Git 索引检查 118 个文件和七个 shell 入口可执行位；没有对实际远端执行 Git 命令。
