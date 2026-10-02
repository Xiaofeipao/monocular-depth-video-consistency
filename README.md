# AIAA 3201 Project 3

本项目研究单目度量深度估计与视频时序一致性。工程按照 [PLAN.md](PLAN.md) 推进：各 Part 的配置、流水线和输出分别放在 `part1/`、`part2/`、`part3/` 下，数据、几何、模型、光流、指标和可视化等公共实现统一放在 `src/` 下。

## 当前状态

Step 0 的自动化基础工作已经完成；Step 1 / Part 1A 经典双目几何也已完成并通过门禁。环境、公开数据、Small 预训练权重、输出 schema、公共指标、Middlebury 15 场景四组 BM/SGBM 对照、统一调参、敏感性分析及 A800 推理 smoke test 均已就绪。没有启动任何正式训练或微调。

以下事项按用户要求暂缓，不阻塞当前公开数据工作：

- 录制六类自有视频，并加入教师指定的 mandatory clips；
- 在 Part 1/2 的 loader 和可视化工具完成后，人工检查至少 10 组 RGB/depth overlay。

NYUv2 主协议已经决定采用评估代码实际执行的 `0.001--10 m`。标准裁剪后的 654 张测试图中，最小正 GT 深度为 `0.71330047 m`，因此 `0.001 m` 与 `0.1 m` 得到的 GT mask 完全相同；两者只可能因预测深度的下限裁剪而产生差异。主结果使用 `0.001 m` 以复现官方代码，同时报告一次明确标注的 `0.1 m` 敏感性对照。

项目的持久执行规则记录在 [AGENTS.md](AGENTS.md)。用户最新提供的 GPU 资源为 Slurm Job `12882905`；本次 Part 1A 只使用 CPU，没有占用该 GPU。每次执行后续 GPU 命令前仍需确认 Job 状态和实际 GPU 型号。

## 环境

主环境名为 `aiaa3201`：

```bash
conda env create -f environment.yml
conda activate aiaa3201
```

如果环境已经存在，可用以下命令安装或更新依赖：

```bash
conda run -n aiaa3201 python -m pip install -r requirements-core.txt
conda run -n aiaa3201 python -m pip install --no-deps -e ./ml-depth-pro
```

已验证的主要版本为 Python 3.10.21、PyTorch 2.1.1+cu121、torchvision 0.16.1+cu121、NumPy 1.24.0、OpenCV 4.11.0、SciPy 1.15.3 和 h5py 3.14.0。完整 Conda 导出与已安装包清单保存在 `outputs/step0/`。

UniDepth 暂不放入主环境：其 `torch>=2.4, numpy>=2` 要求与 Video Depth Anything 官方使用的 `torch==2.1.1, numpy==1.24.0` 冲突。UniDepth 只作为后续可选扩展，不属于当前基线。

## 数据与权重

所有下载均通过 HPC 网络执行，并支持断点续传：

```bash
./scripts/prepare_data.sh
./scripts/download_checkpoints.sh
```

当前公开数据包括：

- Middlebury Stereo v3 Quarter Resolution：15/15 个训练场景，包含输入图像、GT disparity、非遮挡 mask 和标定文件；
- NYU Depth V2 labeled：1,449 对 RGB/depth，以及互不重叠的标准 795 train / 654 test 划分。

NYUv2 官方服务器在当前集群上限速严重，因此完整 MAT 来自公开 Kaggle 镜像。该文件长度为预期的 2,972,037,809 字节，并且前 5,343,837 字节与从官方地址直接获得的部分文件逐字节一致。下载地址和来源证据记录在 `data/manifests/sources.yaml`。

使用以下命令检查数据结构、几何参数、划分和哈希：

```bash
conda run -n aiaa3201 python scripts/validate_data.py
sha256sum -c data/manifests/downloads.sha256
(cd checkpoints && sha256sum -c checksums.sha256)
```

验证结果保存在 `data/manifests/validated_datasets.json`。数据集和模型权重属于本地文件，不应提交或重新分发。

## 验证方法

运行所有项目自有测试：

```bash
conda run -n aiaa3201 python -m pytest -q
```

当前结果为 **36 passed**，覆盖：

- 七个必需的单图深度指标及有效 mask；
- 包含 Middlebury `doffs` 的 disparity-to-depth 几何；
- 区分 metric、relative 和 alignment 状态的输出元数据规则；
- depth-edge precision、recall 和 F1；
- backward-flow warping、forward/backward flow consistency、遮挡 mask、时序误差及 median-scale drift；
- PFM/标定元数据解析和协议配置约束。

## Step 1 / Part 1A：Middlebury 经典双目

冻结配置为 `configs/part1/middlebury_sgbm.yaml`。实验严格分为四组：`bm_fixed`（固定经验参数 BM）、`bm_tuned`（调参后 BM）、`sgbm_fixed`（固定经验参数 SGBM）和 `sgbm_tuned`（调参后 SGBM）。BM 与 SGBM 使用相同的 5 个 development scenes 选参，其余 10 个 held-out scenes 不参与调参。

### 运行指令

一次性复现 BM 和 SGBM 的单轮调参。BM 的 144 个候选与 SGBM 的 36 个候选分别在一次网格搜索内完成：

```bash
./scripts/part1/tune_middlebury.sh
```

如只想复现其中一个算法的调参：

```bash
./scripts/part1/tune_middlebury.sh --methods bm
./scripts/part1/tune_middlebury.sh --methods sgbm
```

推荐用下面一条命令评估四组实验；它会同时生成逐场景原始预测、指标、可视化、理论敏感性、四组实际距离分桶对比和重载验证：

```bash
./scripts/part1/run_middlebury.sh
```

也可以分别运行四组。为避免四次运行互相覆盖汇总文件，应为每组指定不同输出目录：

```bash
./scripts/part1/run_middlebury.sh --experiments bm_fixed \
  --output-dir outputs/part1/middlebury/single_bm_fixed
./scripts/part1/run_middlebury.sh --experiments bm_tuned \
  --output-dir outputs/part1/middlebury/single_bm_tuned
./scripts/part1/run_middlebury.sh --experiments sgbm_fixed \
  --output-dir outputs/part1/middlebury/single_sgbm_fixed
./scripts/part1/run_middlebury.sh --experiments sgbm_tuned \
  --output-dir outputs/part1/middlebury/single_sgbm_tuned
```

这些都是 OpenCV 经典算法的非学习式参数搜索和评估，不需要 GPU，也不会启动正式训练。

### 参数与结果

左右一致性阈值固定为 1 px，不参与调参。选择目标为 development scenes 上的最低宏平均 Bad-2；并依次用 disparity MAE 和 coverage 打破平局。固定/选择后的关键参数如下：

| 实验组 | `block_size` | `uniqueness_ratio` | `speckle_window_size` | `texture_threshold` |
|---|---:|---:|---:|---:|
| `bm_fixed` | 15 | 10 | 100 | 10 |
| `bm_tuned` | 15 | 0 | 0 | 10 |
| `sgbm_fixed` | 5 | 10 | 100 | 不适用 |
| `sgbm_tuned` | 3 | 0 | 0 | 不适用 |

15 场景逐场景宏平均结果如下。Bad-1/2/4 的分母包含 non-occluded GT 区域内的无效预测，因此不能通过降低 coverage 获得虚假的低 bad-pixel rate；MAE/RMSE 只在有效预测处计算，必须与 coverage 一起阅读。

| 实验组 | Disp. MAE (px) | Bad-1 | Bad-2 | Bad-4 | Coverage | Depth AbsRel | Depth RMSE (m) | 双向匹配时间/场景 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm_fixed` | 0.9790 | 0.5028 | 0.4875 | 0.4793 | 0.5436 | 0.01334 | 0.1420 | 0.0392 |
| `bm_tuned` | 1.7314 | 0.4340 | 0.4069 | 0.3899 | 0.6595 | 0.02385 | 0.2106 | 0.0241 |
| `sgbm_fixed` | 1.0588 | 0.3735 | 0.3474 | 0.3323 | 0.6967 | 0.01226 | 0.1282 | 0.0683 |
| `sgbm_tuned` | 1.3070 | 0.3651 | 0.3373 | 0.3203 | 0.7169 | 0.01536 | 0.1467 | 0.0557 |

BM 调参后 Bad-2 从 48.75% 降至 40.69%，coverage 从 54.36% 提高到 65.95%，但有效预测处的 MAE、Depth AbsRel 和 Depth RMSE 变差；原因是更宽松的过滤保留了更多困难像素。SGBM 调参后 Bad-2 从 34.74% 小幅降至 33.73%，coverage 从 69.67% 增至 71.69%，但同样付出了有效像素误差上升的代价。主报告必须联合展示 Bad-2、coverage 和有效像素误差，不能只挑一个指标宣布“调参更好”。

### 敏感性分析

不应把原有的理论敏感性机械复制成四组。`sensitivity.csv/png` 对 GT disparity 注入 ±0.25/0.5/1/2 px，其结果只由相机标定和公式 `z=fB/(d+doffs)` 决定，与匹配器参数无关，所以全实验共享一份。新增的 `empirical_depth_error_by_distance.csv/png` 才是按四组分别统计实际预测的深度 MAE、disparity MAE、Bad-2 和 coverage 随 GT 距离的变化。

完整正式产物位于 `outputs/part1/middlebury/frozen_v2/`：

- `metrics_per_scene.csv` 与 `metrics_summary.json`：逐场景和宏平均指标；
- 每个 `scene/experiment` 下的 `prediction.npz`、`metadata.json` 和 `visualization.png`；
- 四张 `visualization_contact_sheet_<experiment>.jpg`；
- `failure_zoom_Jadeplant_sgbm_tuned.png`：植物细枝、遮挡边界和低纹理区域的局部失败放大图；
- `sensitivity.csv/png`：共享的理论/数值几何敏感性；
- `empirical_depth_error_by_distance.csv/png`：四组实际误差与 coverage 的距离分桶对比；
- `reload_validation.json`：60 个预测文件、1,200 个指标值的保存后重载复算门禁。

单轮调参证据位于 `outputs/part1/middlebury/tuning_v2/`。旧的 `frozen_v1/`、`tuning_stage1/` 和 `tuning_stage2/` 只作为开发历史保留，不再作为当前正式结果。

该步骤只进行了 OpenCV 经典匹配、非学习式参数选择和评估，没有训练模型。正式训练仍由用户启动。

## 预训练模型 smoke test

在 GPU Job 有效时，可运行纯推理模型门禁：

```bash
srun --jobid=12882905 --overlap \
  conda run -n aiaa3201 python scripts/smoke_models.py
```

已验证的 A800 结果保存在 `outputs/step0/model_smoke.json`：DA-V2-S 和 Depth Pro 分别处理一张全分辨率图像；relative VDA-S 和 metric VDA-S 分别处理八帧；RAFT-Small 处理一对相同图像。所有摘要均为有限的 `float32` 输出。脚本只读取 `checkpoints/` 内的权重，不进行训练。

## 候选评估协议

公共协议位于 `configs/evaluation_protocol.yaml`，模型专用配置位于 `configs/part*/`。必须保持以下约束：

- Depth Pro 使用原生 metric scale，以米为单位，且 `alignment: none`；
- 经过对齐的 DA-V2 结果标记为 `aligned_relative`，不得称为 metric depth；
- NYUv2 模型输入使用未裁剪 RGB，标准 `[45:471, 41:601]` crop 只在评估时应用；
- NYUv2 主结果使用有效代码路径的 `0.001--10 m`、预测裁剪 `[0.001, 10]` 和逐图宏平均；
- 另报 `[0.1, 10]` 预测裁剪敏感性结果，不与主结果混在一起；
- 所有视频方法必须使用相同的解码帧、target FPS、分辨率、插值、mask 和逐片段固定可视化范围；
- 时序误差使用 target-to-source backward flow 和 forward/backward consistency mask。

由于视频片段、帧率和哈希尚未冻结，整个协议仍标记为 `candidate_v1`。

## 工程结构与执行顺序

```text
configs/       公共协议与各 Part 配置
src/           项目公共实现
pipelines/     Part 1/2/3 Python 入口
scripts/       数据准备、smoke test 与各 Part shell 入口
tests/         项目自有单元测试和集成测试
data/          本地数据集；只提交 manifest
checkpoints/   本地预训练权重与 checksum manifest
outputs/       运行输出与 Step 0 验证证据
```

执行顺序为 Step 0 → Part 1 → Part 2 → Part 3 → Demo 与最终交付。当前 Step 1 / Part 1A 已完成；依赖自录视频的 Part 1B 暂缓。公共 evaluator 可以提前实现，但 Part 3 实验必须等待 Part 2 基线和视频协议冻结。详细门禁见 [PLAN.md](PLAN.md)，已经验证的工作记录在 [implemented.md](implemented.md)。

## 自录视频交接

该事项目前按用户要求暂缓。恢复后，录制要求和固定 dev/test 划分见 `data/self_recorded/README.md` 与 `data/manifests/self_recorded_template.csv`。视频复制到项目后，应先记录 SHA256、相机元数据、解码帧数、target FPS 和精确帧索引，再开始任何 Part 3 调参。
