# Step 0：公共基础

## 状态

公开数据所需的环境、数据、权重、输出 schema、公共指标、smoke test 和自动测试已经建立。Part 1B指定官方davis_rollercoaster的70个真实帧已完成DA逐帧预测与核验，100帧素材需课程确认，指定片段的2B比较待后续。CPU工程检查和完整GPU实验分别记录，不视为课程正式视频已完成。

## 通用复现顺序

所有命令从项目根目录执行。已有环境、数据、权重和正式输出时，先做只读检查；下面安装、下载和推理命令用于后续复现，本次整理没有执行。

全新工程需先取得四个固定版本的上游，再创建环境，因为 `environment.yml` 会安装 `-e ./ml-depth-pro`。以下命令仅用于对应目录不存在时；已有仓库先核对版本与用户改动。

```bash
git clone https://github.com/apple/ml-depth-pro.git ml-depth-pro
git -C ml-depth-pro checkout --detach 9e65e4dbe9568d23c546fcec53302b10445e109e

git clone https://github.com/DepthAnything/Depth-Anything-V2.git Depth-Anything-V2
git -C Depth-Anything-V2 checkout --detach a561b849ebae10a6f5ef49e26c83cbbcd36c71bf

git clone https://github.com/DepthAnything/Video-Depth-Anything.git Video-Depth-Anything
git -C Video-Depth-Anything checkout --detach 4f5ae23172ba60fd7bc11ef671cca678842c7072

git clone https://github.com/lpiccinelli-eth/UniDepth.git UniDepth
git -C UniDepth checkout --detach 8d8cfe4c7ee15297099983607febf0d4f32eb3d6

```

用 `git -C <repo> rev-parse HEAD` 对照 `configs/versions.yaml`；保护已有修改，不 reset 或覆盖。工程根目录当前没有可用 Git 历史，不能仅克隆一个上游仓库就恢复项目自有代码。

依次完成：环境 → 数据/权重 checksum → 项目测试与 smoke 门禁 → [Middlebury](step1.md) → [NYUv2 与分析](step3.md)。[逐帧视频步骤](step2.md)使用连续原始帧、固定整段色标与独立冻结协议；指定片段的2B比较待后续，旧[Step 4](step4.md)仅保留兼容依赖。

## Conda 环境

主环境为 `aiaa3201`：

```bash
conda env create -f environment.yml
conda activate aiaa3201
```

已有环境可使用：

```bash
conda run -n aiaa3201 python -m pip install -r requirements-core.txt
conda run -n aiaa3201 python -m pip install --no-deps -e ./ml-depth-pro
```

2026-10-13 通过包元数据读取确认，主环境版本包括 Python 3.10.22、PyTorch 2.1.1+cu121、torchvision 0.16.1+cu121、NumPy 1.24.0、OpenCV 4.11.0、SciPy 1.15.3 和 h5py 3.14.0。完整 Conda 导出与包清单保存在 `outputs/step0/`。

UniDepthV2 的 PyTorch/NumPy 依赖与 Video Depth Anything 冲突，因此使用隔离环境：

```bash
conda env create -f environment-unidepth.yml
```

2026-10-13 通过包元数据读取确认，隔离环境版本为 Python 3.10.21、PyTorch 2.7.1+cu126、torchvision 0.22.1+cu126、NumPy 2.2.6 和 xformers 0.0.31.post1。运行脚本从项目内 `UniDepth/` 与 `ml-depth-pro/src/` 导入代码，不修改第三方仓库。

### 运行库检查

```bash
conda run -n aiaa3201 python -c \
  'import sys, torch, numpy; print(sys.version); print(torch.__version__, torch.version.cuda, numpy.__version__); print(torch.cuda.is_available())'
conda run -n aiaa3201-unidepth python -c \
  'import sys, torch, numpy; print(sys.version); print(torch.__version__, torch.version.cuda, numpy.__version__); print(torch.cuda.is_available())'
```

`nvidia-smi` 显示的 CUDA 是驱动支持上限，项目采用各环境 PyTorch 自带 runtime。现有推理与 2D 评估不需要另行编译 UniDepth KNN 扩展，也不要求单独安装 `nvcc`；将来确需编译扩展时，再在使用该扩展的项目环境内配置与 PyTorch CUDA 相容的编译器，避免安装进 `base`。

`requirements-core.txt` 和 `requirements-unidepth.txt` 是两套直接依赖；`requirements-lock.txt` 保留早期包清单，并非当前完整环境的新导出。当前主环境完整清单见 `outputs/step0/conda-explicit.txt`、`pip-packages.json`；各模型运行的版本以对应 `inference_environment.json` 为准。

## 数据与权重

首次缺少资源时，可用准备脚本下载并建立清单：

```bash
./scripts/prepare_data.sh
./scripts/download_checkpoints.sh
```

当前公开数据包括：

- Middlebury Stereo v3 Quarter Resolution：15/15 个训练场景，含左右图、GT disparity、非遮挡 mask 和标定文件；
- NYU Depth V2 labeled：1,449 对 RGB/depth，标准划分为 795 train / 654 test。

NYUv2 完整 MAT 来自公开 Kaggle 镜像，长度为 2,972,037,809 bytes；其前 5,343,837 bytes 与从官方地址直接获得的部分文件逐字节一致。来源记录在 `data/manifests/sources.yaml`。

验证数据和哈希：

```bash
conda run -n aiaa3201 python scripts/validate_data.py
sha256sum -c data/manifests/downloads.sha256
(cd checkpoints && sha256sum -c checksums.sha256)
```

准备脚本会重建 checksum 清单，`validate_data.py` 会写验证结果；仅校验现有文件时运行 `sha256sum -c`，不要把刚生成的哈希当作独立来源证据。

解析后的结果位于 `data/manifests/validated_datasets.json`。数据集和权重是本地资源，不应提交或重新分发。

## 公共评估协议

公共配置为 `configs/evaluation_protocol.yaml`，模型专用配置位于 `configs/part*/`。核心约束：

- 原生 metric 输出必须为米制，并标记 `alignment: none`；
- aligned relative 结果不得称为 metric depth；
- NYUv2 模型输入使用完整 480×640 RGB，评估时才应用 `[45:471, 41:601]` crop；
- NYUv2 GT 有效条件为 `0.001 < depth < 10 m`，预测计算指标时裁剪到 `[0.001,10] m`；
- 标准 654 张 crop 内最小正 GT 为 0.71330047 m；
- 视频方法比较须共享真实帧/FPS/输出画布、同一palette，每方法整段固定归一化；若附加光流误差，保持同一对应/mask协议；
- 时序误差使用 target-to-source backward flow 与 forward/backward consistency mask。

Depth Pro 和 UniDepthV2-S 的 NYUv2 协议已冻结为 `frozen_v1`。Part 1B的NYUv2 aligned_relative基准已完成654张，独立配置为`configs/part1/dav2_nyuv2.yaml`。视频按各自配置生成绑定输入、资源与源码的`protocol.json`；当前指定片段使用独立`configs/part1/dav2_video.yaml`，保留真实70帧限制，不用自录补齐。

## 测试

```bash
conda run -n aiaa3201 python -m pytest -q
```

当前结果为 **43 passed**，覆盖：

- 七项单图深度指标及有效 mask；
- Middlebury `doffs` disparity-to-depth 几何；
- metric、relative、alignment 元数据规则；
- depth-edge P/R/F1 与官方 `SI_boundary_F1` 包装；
- backward-flow warping、forward/backward consistency、遮挡 mask、时序误差和 median-scale drift；
- PFM、标定文件、NYUv2 HDF5 方向恢复和配置约束。

## 预训练模型 smoke test

CUDA 可用时可手动运行：

```bash
conda run -n aiaa3201 python scripts/smoke_models.py
```

已验证结果位于 `outputs/step0/model_smoke.json`：

- DA-V2-S 与 Depth Pro 各处理一张完整图像；
- VDA-S relative 与 metric 各处理八帧；
- RAFT-Small 处理一对相同图像；
- 所有摘要均为有限 `float32` 输出；VDA 的八帧由 smoke 脚本准备，不代表正式自录视频或 mandatory clips 已完成。

该操作只读取预训练权重，不训练或更新模型；本次文档整理没有执行它。

## 工程结构

```text
configs/       公共协议与各 Part 配置
src/           可复用的数据、几何、模型、指标和可视化实现
pipelines/     Part 1/2/3 Python 入口
scripts/       数据准备、smoke test 与各 Part shell 入口
tests/         项目自有测试
data/          本地数据集与 manifest
checkpoints/   本地预训练权重与 checksum
outputs/       运行输出和验证证据
docs/          Step 0–6 的操作与结果说明
```

## 视频输入与后续补齐

当前课程指定`data/davis_rollercoaster.mp4`，原960×540、24 FPS、70帧。1B按连续原始索引保存640画布与原PTS，不降到10 FPS、不伪装为自录；DA-V2-S原生relative、固定整段TURBO色标，70帧运行及raw重算已完成，见[Step 2](step2.md)。100帧要求待确认是否接受现有官方完整素材或补充版本。

2B以后通过1B的`analysis/handoff_2b.json`复用同一RGB/PTS/画布，本次不运行VDA。旧自录视频及其1B派生文件因为现有2B的冻结绑定而只读保留，不作为新1B任务；本次不下载或删除原数据。

## 独立资源 SHA256

下列固定值用于独立验证原始数据与权重，不生成或改写 manifest。文件缺失时会失败，应先确认缺失原因；不为通过校验改写预期哈希。

```bash
sha256sum -c - <<'SHA256'
a1411f283b523541e0d2e8b3a1dd6618974a0e539aea0122490791a041fffc09  data/downloads/MiddEval3-data-Q.zip
489b6a4015951a3c865aa54c0f37dcb648b5d0efad8019264e22499cef35ca45  data/downloads/MiddEval3-GT0-Q.zip
cf91122bd5d75fb0588e870e5d8f08bee9622ed8ddae408076eb4cc07ce88089  data/downloads/nyu-depth-v2-labeled-mat.zip
2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa  data/nyuv2/nyu_depth_v2_labeled.mat
6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee  data/nyuv2/splits.mat
715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378  checkpoints/depth_anything_v2/depth_anything_v2_vits.pth
3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce  checkpoints/depth_pro/depth_pro.pt
01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27  checkpoints/torchvision/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth
ecc1f898690debe387d10329cca5d9d66a0a447ce83c3a2a7ae3673c7ce43cfe  checkpoints/unidepth_v2_vits14/config.json
93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648  checkpoints/unidepth_v2_vits14/model.safetensors
3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33  checkpoints/video_depth_anything/metric_video_depth_anything_vits.pth
13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609  checkpoints/video_depth_anything/video_depth_anything_vits.pth
SHA256
```

## 当前产物完整性

Middlebury正式评估 `outputs/part1/middlebury/evaluation/` 完成后应有60份 `prediction.npz`（完整运行待用户启动）。两条 NYUv2 baseline 的 prediction、metadata 与 `visualization/test_*.png` 各 654 份。可只读查看：

```bash
conda run -n aiaa3201 python -c \
  'from pathlib import Path; roots=["outputs/part2/depth_pro_nyuv2/frozen_v1", "outputs/part2/unidepth_v2_nyuv2/frozen_v1"]; print([(r, len(list((Path(r)/"predictions").glob("*.npz"))), len(list((Path(r)/"metadata").glob("*.json")))) for r in roots])'
```

数量不能独立证明结果来源。逐图身份、完整精度指标、配置哈希及尺度/边界来源证据见 [当前结果审计](current_results.md)。原始元数据中的实际路径和时间保持真实，不改写为新的执行记录。
