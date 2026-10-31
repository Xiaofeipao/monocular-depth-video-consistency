# Step 2 / Part 1B：相对深度基准与视频时序诊断

## 任务与当前状态

本步骤使用同一个DA-V2-Small相对深度权重，完成两类工作，不训练或微调模型。要求依据课程PDF §5.1、§6.1–6.2（本地保留：`../AIAA3201_Project3_Monocular_Metric_Depth_Video_Consistency_Spring2027 (1).pdf`）及中文说明（本地保留：`../Project3_CN_new.md#2-单目逐帧-baseline`）。

| 部分 | 输入与目标 | 当前状态 |
|---|---|---|
| NYUv2 Scale Alignment | 标准654张test；对齐相对预测后进行定量评估 | **完整运行、重评、原生重算与结果分析已完成** |
| 自录Relative Depth与Temporal Diagnosis | 固定、运动相机视频；独立预测每帧，测背景闪烁与相对输出变化 | 三类型单视频流程、全图逐像素/光流指标及合成验证已具备；正式运行待用户启动 |

**NYUv2分支使用测试图GT对齐，身份为`aligned_relative`；自录视频保持`alignment:none`，不逐帧调整尺度。** 前者检查对齐后的结构质量，后者观察原生输出的时间变化。NYUv2完成不代表自录视频任务完成，已有Part 2A原生米制结果也不被本分支替代。

## NYUv2：DA-V2-S的Scale Alignment

### 基本信息与协议


使用[configs/part1/dav2_nyuv2.yaml](../configs/part1/dav2_nyuv2.yaml)。NYUv2数据、官方split和DA-V2-S权重已在本地，预检核对SHA256与官方仓库commit，不重新下载。

| 项目 | 固定设置 |
|---|---|
| 模型 | 官方DA-V2-Small，24.8M参数，relative权重；不使用NYUv2微调的metric checkpoint |
| 数据划分 | labeled MAT的标准`testNdxs`，654张；MATLAB one-based index保持原顺序 |
| 模型输入 | 完整480×640 RGB；input_size=518，实际网络H×W=518×686；预测恢复到480×640 |
| 保存 | 原生float32、inverse-depth-like、arbitrary、`alignment:none`；合法零值保留 |
| 评价crop | Python半开区间`[45:471, 41:601]`，仅在对齐与评价时使用，不裁剪模型输入 |
| 有效GT | 有限且`0.001 < GT < 10 m`；对齐与评分使用同一组像素 |
| 对齐 | 每张图在逆深度域做非负尺度的scale-and-shift最小二乘拟合 |
| 转换与范围 | 拟合逆深度限制到`[1/10,1/0.001]`后取倒数；保留全部有效GT像素，记录非正拟合值及裁剪数量 |
| 指标 | AbsRel、SqRel、RMSE、RMSE-log、δ1/δ2/δ3，逐图宏平均；10,000次图像级Bootstrap 95% CI，seed=42 |
| 效率 | 实际NVIDIA RTX A6000、CUDA/FP16，预热1次；每图计时3次并保留第一次输出 |

通俗地说，模型先只看RGB给出相对远近，评价器再用GT给这一张预测确定单位和零点，随后检查它的结构与GT是否吻合。DA-V2输出近似逆深度，因此不能把原生数组直接当米制深度缩放：

```text
r = 模型原生相对预测
q = 1 / GT
(a,b) = argmin Σ有效crop像素 (a*r + b - q)²，a≥0
评价深度 = 1 / clip(a*r+b, 1/10, 1/0.001)
```

原生数组不改写。常数预测或无正向相关时，用约束下的常数解`a=0,b=mean(1/GT)`，如实记录原因，并让这张图继续参与汇总；不会丢弃失败图或拟合后非正的像素来改善分数。输入非有限/负值、空有效域等实际错误会明确失败。

这是本项目明确固定的相对深度评价协议：尺度/偏移最小二乘思想参考[官方SSI实现](https://github.com/isl-org/ZoeDepth/blob/main/zoedepth/trainers/loss.py)，在既有NYUv2 split/crop/范围上实施。非负约束和退化处理均在此披露，不声称复现DA-V2论文的官方表格数字。[DA-V2官方说明](https://github.com/DepthAnything/Depth-Anything-V2)区分relative和metric模型；本分支始终使用relative权重。

**这些指标使用了同一测试图GT确定a、b，衡量的是对齐后的结构质量，不是模型直接预测米制尺度的能力。** RMSE按GT尺度以米表示，但方法身份仍为`aligned_relative`；不得与Part 2A的native metric结果混为同一无GT条件下的排名，也不得把对齐参数用于自录视频。

### 运行指令

当前完整产物位于`outputs/part1/dav2_nyuv2/frozen_v1/`，用户已完成预测、评价与重新评价。下面从项目根目录执行；脚本使用`aiaa3201`。**复现请使用新目录，不能覆盖frozen_v1。**

```bash
set -euo pipefail

./scripts/part1/run_dav2_nyuv2.sh check \
  --output outputs/part1/dav2_nyuv2/checks_reproduce_v1/resources.json

# 完整654张：原生预测 → GT辅助对齐评价/图像 → 原生重算核验
./scripts/part1/run_dav2_nyuv2.sh run \
  --output-dir outputs/part1/dav2_nyuv2/reproduce_v1
```

中断恢复和独立重评可分别执行。只有预测阶段中断才需要`predict --resume`；评价不重新加载模型，输出目录须不存在。改变科学源码或协议时使用新版本，不能拼入旧run。

```bash
./scripts/part1/run_dav2_nyuv2.sh predict --resume \
  --output-dir outputs/part1/dav2_nyuv2/reproduce_v1

./scripts/part1/run_dav2_nyuv2.sh evaluate \
  --run-dir outputs/part1/dav2_nyuv2/reproduce_v1 \
  --output-dir outputs/part1/dav2_nyuv2/reproduce_v1/evaluation_recheck_v1

./scripts/part1/run_dav2_nyuv2.sh verify \
  --evaluation-dir outputs/part1/dav2_nyuv2/reproduce_v1/evaluation_recheck_v1 \
  --output outputs/part1/dav2_nyuv2/checks_reproduce_v1/recheck.json
```

仅核验当前正式结果时，运行以下只读命令，无需重新预测或评价，也不覆盖已有核验记录：

```bash
./scripts/part1/run_dav2_nyuv2.sh verify \
  --evaluation-dir outputs/part1/dav2_nyuv2/frozen_v1/evaluation
```

### 654张完整结果

以下从正式`evaluation/metrics_summary.json`读取完整精度，再按8位小数显示；逐图CSV重新聚合一致，`evaluation_recheck_v1`的七指标、CI和CSV也一致。全部654张、215个scene参与统计，每图238,560个有效crop像素，共156,018,240个。

| 指标 | 逐图宏平均 | 图像级Bootstrap 95% CI |
|---|---:|---:|
| AbsRel↓ | 0.06171408 | [0.05854527, 0.06501292] |
| SqRel↓ | 0.03791699 | [0.03294910, 0.04328549] |
| RMSE↓ (m) | 0.29735612 | [0.27833497, 0.31686912] |
| RMSE-log↓ | 0.09352266 | [0.08884892, 0.09842145] |
| δ1↑ | 0.95540346 | [0.94927470, 0.96124863] |
| δ2↑ | 0.98783715 | [0.98518574, 0.99018256] |
| δ3↑ | 0.99578876 | [0.99441998, 0.99694811] |

AbsRel=0.06171408表示对齐后逐图平均相对误差约6.17%；δ1=0.95540346表示逐图平均约95.54%的像素倍率误差小于1.25。**这些数使用了每张测试图的GT来确定尺度与偏移，不能解释为模型直接给出了同等水平的绝对米制深度。** 不能因其低于Part 2A某些native指标，就声称DA-V2的无GT米制能力更好。

AbsRel中位数为0.04970669，均值高于中位数，存在较难样本形成的高误差尾部。描述性统计中，82张AbsRel>0.10、5张>0.25，6张δ1<0.60；这些只是分布说明，不用于删图或重新选参。CI按冻结协议以图像为单位，包含同一scene的相关图像，不代表场景级或跨数据集不确定度。

### 对齐行为与代表性问题

**1．大多数图可通过正尺度和偏移完成对齐，但全局两个参数无法修正局部结构错误。** 653张得到正尺度拟合，没有因为难图而剔除样本。浴室镜面附近、遮挡与远景深度压缩仍有明显误差。拟合最小化的是逆深度平方误差，与直接最小化AbsRel或深度RMSE不是同一个目标。

**2．第266张出现真实的负尺度拟合，常数解必须保留并解释。** `bathroom_0008`（MAT索引650）的无约束尺度为−0.04683307。协议要求保留“数值越大通常越近”的方向，因此采用预先规定的约束解`a=0,b=0.53564013`，对应整图评价深度约1.86692510 m。它的AbsRel为0.28713603，仍参加全部654张平均。平坦结果来自评价器的退化处理，**不表示模型原生预测本身就是常数，也不表示结构恢复成功**。

**3．远端范围裁剪较少，但仍需报告。** 26张图共有249,856个拟合逆深度低于0.1，转换后按最大深度10 m处理，约占全部有效像素的0.160145%。没有拟合逆深度≤0，也没有超过1000的值。这些像素保留在评分中，不能当作缺失或从分母移除；0.160145%不是误差率。没有常数输入触发的退化，唯一退化图是上面的负尺度案例。

| 缓存test位置 / scene | AbsRel | δ1 | 查看与解释 |
|---|---:|---:|---|
| 310 / bathroom_0046 | 0.34122097 | 0.15456070 | 案例图（本地保留：`../outputs/part1/dav2_nyuv2/frozen_v1/evaluation/visualization/test_0310.png`）：镜面及其周围区域的GT和预测差异大；全局对齐无法修复局部不一致 |
| 266 / bathroom_0008 | 0.28713603 | 0.53367287 | 案例图（本地保留：`../outputs/part1/dav2_nyuv2/frozen_v1/evaluation/visualization/test_0266.png`）：约束后的常数评价结果；不能把它作为正常层次恢复展示 |
| 447 / bedroom_0085 | 0.27964143 | 0.44479795 | 案例图（本地保留：`../outputs/part1/dav2_nyuv2/frozen_v1/evaluation/visualization/test_0447.png`）：前景轮廓可辨，但背景/远处层次被压缩，绝对误差仍较大 |

这些观察来自实际查看RGB/GT/预测/误差图，不仅依据分数。没有将所有反光处差异归因于模型，GT在复杂反射场景中的语义也需谨慎解释。完整contact sheet（本地保留：`../outputs/part1/dav2_nyuv2/frozen_v1/evaluation/visualization/contact.jpg`）包含固定位置0/326/653和AbsRel最高6图，共9张，不是全部654张可视化。

按场景类型的补充统计如下，完整16类结果见metrics_by_scene_type.csv（本地保留：`../outputs/part1/dav2_nyuv2/analysis_v1/metrics_by_scene_type.csv`）。这是同一NYUv2内的描述性分组，不是另一个数据域，也不改主结果的逐图权重。

| 场景类型 | 图像数 | AbsRel | RMSE (m) |
|---|---:|---:|---:|
| bedroom | 191 | 0.04848230 | 0.17945597 |
| living_room | 107 | 0.05599062 | 0.31028422 |
| kitchen | 106 | 0.06265542 | 0.31887104 |
| bathroom | 58 | 0.10348581 | 0.38439111 |
| bookstore | 11 | 0.15352100 | 1.09650118 |

卧室、客厅和厨房占较多样本；浴室和书店在当前结果中更难。书店仅11张，不能据此断言对所有书店场景都失效；每类数量不均也意味着全体均值不能代替逐类表现。

### 运行效率

实际环境为`aiaa3201`、Python 3.10.22、PyTorch 2.1.1+cu121、NVIDIA RTX A6000、FP16；网络实际518×686，参数24,785,089，权重99,218,434 bytes。

| 平均耗时 (ms/图) | 中位数 (ms) | P95 (ms) | 模型FPS | 峰值allocated / reserved (MiB) |
|---:|---:|---:|---:|---:|
| 18.64646556 | 18.59909218 | 19.21165398 | 53.62946649 | 314.9014 / 326.0000 |

耗时取每图三次测量的均值，再跨图统计，包含模型输入处理、设备传输和输出恢复；不含数据读取、GT对齐、保存或绘图。全体三次测量累计36.58436543秒，读取样本至原生保存阶段73.50841505秒，模型加载另为0.66774602秒。73.51秒包含预热、三次测量和I/O，不能当作单遍实时系统的耗时；峰值显存是PyTorch分配/保留值，不是整卡占用。

### 正式产物与核验

```text
outputs/part1/dav2_nyuv2/
  frozen_v1/
    run.json, provenance/              原生身份、资源/源码SHA、命令和真实环境
    predictions/*.npz                  654份原生raw_relative，未GT对齐
    evaluation/                        主评价：654份aligned、CSV/汇总、9张案例与contact
    evaluation_recheck_v1/              用户独立重评的完整产物，结果与主评价一致
    reload_validation.json             用户运行时的654张原始重算记录
  checks_user_v1/                       用户资源检查与重评核验记录
  analysis_v1/                          本次只读复核、完整精度分析、分类统计及清理清单
```

本次对主评价及独立重评均重新执行`verify`，654张的raw、资源/源码、对齐数组/mask、CSV、宏平均和CI通过`rtol=1e-6 / atol=1e-8`核验。另对0/326/653、266/310/447及裁剪较多的45共7图，用独立`numpy.linalg.lstsq`和显式七指标公式复核，通过。

详细来源、SHA和全精度统计见results_analysis.json（本地保留：`../outputs/part1/dav2_nyuv2/analysis_v1/results_analysis.json`），实际过程见review.py（本地保留：`../outputs/part1/dav2_nyuv2/analysis_v1/review.py`）。本次没有运行模型、训练或下载，也没有改写正式预测和自动运行元数据。

### 项目架构

| 文件 | 职责 |
|---|---|
| `configs/part1/dav2_nyuv2.yaml` | 固定split、模型、crop、对齐与评价设置 |
| `scripts/part1/run_dav2_nyuv2.sh`、`pipelines/part1/dav2_nyuv2.py` | check/predict/evaluate/verify/run环境入口与编排 |
| `src/evaluation/relative_nyuv2.py` | 资源/源码绑定、原生保存、评价、汇总、案例图与重算 |
| `src/metrics/relative_alignment.py` | 逆深度尺度/偏移拟合、约束与退化处理 |
| 既有NYUv2 loader、DA-V2 adapter与depth metrics | 复用数据读取、官方模型推理和七指标，Part 2A协议不变 |

## 自录视频：Relative Depth与Temporal Diagnosis

### 基本信息与当前状态

每段视频有自己的ID、类型和结果目录。**一条命令完成这一段的抽帧、冻结、DA-V2预测、光流、全部适用诊断、图视频与重算核验**；也可以指定某项任务。新视频不需要改Python，也不要求一次收齐六段。

当前六段原视频已到位，来源/SHA及视频流信息见source_inventory.json（本地保留：`../data/self_recorded/source_inventory.json`）。本流程的实现与合成生命周期验证已通过；**当前正式视频预测、指标和案例等待用户运行**。NYUv2部分已经完成，不需要重跑。本节只运行Part 1B。

入口为`./scripts/part1/video.sh`，主环境`aiaa3201`；科学配置为[self_recorded.yaml](../configs/video/self_recorded.yaml)，视频登记为[clips.yaml](../configs/video/clips.yaml)。命令从项目根目录执行，脚本自行定位根目录并选择环境。

### 三种视频各做什么

| 类型 | 定义 | 主要任务 | 其他保留项 |
|---|---|---|---|
| `fixed_static` | 相机固定、无人且场景基本不动 | 原生relative；全图逐像素MAE/S、RMSE/S；全图背景时间std/S | 全图光流MAE/S、RMSE/S与coverage；中位数曲线、效率、案例 |
| `fixed_dynamic` | 相机固定、有人经过/遮挡，或场景中存在真实运动 | 原生relative；光流对应后的MAE/S、RMSE/S；人物/边缘变化案例 | 逐像素MAE/S、RMSE/S与中位数作补充；背景std为N/A |
| `moving` | 相机移动，视差和可见内容随时间改变 | 原生relative；光流对应后的MAE/S、RMSE/S；中位数比值/变化曲线与尺度proxy解释 | 逐像素MAE/S、RMSE/S作补充；背景std为N/A |

共同的RGB、模型和光流只生成一次，不为不同指标重复预测。三类都保留光流项，以满足课程的共同评价；差别在于背景任务是否适用，以及每项指标能支持什么结论。

**无人固定的背景就是整幅画面，不需要画矩形ROI。** 全图统计能覆盖墙、家具、地面和边缘。前提是相机和场景确实基本静止；轻微抖动、曝光与光照变化仍可能影响结果，不把变化全部归因于模型。如果有真实运动，应明确登记类型，不根据评分事后挑一块安静区域。

### 固定科学协议

| 项目 | 设置 |
|---|---|
| 原视频 | 保留手机原文件、源SHA和真实PTS，不覆盖原片 |
| 选帧/缓存 | 10 FPS，按真实PTS选帧，不复制帧；长边≤960，保持宽高比 |
| 模型 | 官方DA-V2-Small relative，CUDA/FP16，`input_size=518`；16:9实际网络H×W=518×924、输出540×960 |
| 光流 | RAFT-Small、双向、FP32、12次更新；backward warp，FB残差≤1 px，排除越界/非有限插值支持 |
| 固定单位 | 各模型第0帧全图`S=P95−P5`，整段不变；不是逐帧对齐 |
| 逐像素域 | 整幅图像相同坐标，不warp、不使用光流mask；正式预测要求完整有限float32 |
| 光流域 | 全图中的有效对应像素；所有非空帧对等权，coverage<5%只标记、不筛除 |
| 背景域 | 仅fixed_static，全画面时间std；其余两类N/A，不补零 |
| 显示 | TURBO，原生[0,50]、固定参考副图[0,1]、两种差分[0,0.1]；不逐帧调色标 |
| 计时 | 每片预热1次、完整计时3次，保存第一轮原生结果；记录实际尺寸、精度、设备及显存 |
| 数据身份 | 六段均为固定协议的test评价片；不选参、不称盲测；未知session保持null |

原生值为`relative / inverse_depth_like / arbitrary / alignment:none`，数值大表示相对更近；不取倒数，不增加平滑，不读取视频深度GT。合法零值保留；NaN/Inf明确失败，不靠删难像素掩盖问题。参考跨度为零时归一化项N/A，仍保留原生变化及原因。

### 指标怎样读

记`D_t(x,y)`为第t帧原生预测。表中fixed_dynamic和moving的“逐像素变化作补充”，具体就是下面不经光流对齐的逐像素MAE/S、RMSE/S；它们仍计算并保存，只是解读时不能将真实运动或视差都算作模型闪烁。

- **逐像素MAE/S**：对整图`|D_t−D_(t−1)|`求平均，再除以S；**逐像素RMSE/S**先对同一差值求均方根，再除以S。完全不用光流，衡量相同屏幕位置的变化。
- **光流MAE/S、RMSE/S**：先把上一帧搬到当前帧，再在共同有效对应域上计算。更适合真实运动下检查输出连续性，但仍受光流误差和遮挡筛选影响。
- **背景std/S**：fixed_static的每个像素先对整段计算时间标准差，再对全图取平均，`ddof=0`。它同时反映短时跳动和缓慢变化，不是纯高频闪烁指标。
- **尺度proxy**：保留每帧原生中位数`m_t`、`m_t/m_0`、`(m_t−m_0)/S`及其曲线；摘要为平均绝对log比值与平均绝对归一化变化。原生比值受任意固定加性偏移影响；真实距离、视野与人物占比也会改变中位数。

各项误差先对每个帧对计算，再等权平均；不是把整段像素混在一起计算一个RMSE。逐像素域是全图，光流域是有效对应区域，两项有效域不同；应各自比较，而不是简单把两者的大小差当作算法优劣。**固定有人和移动视频的逐像素项混有真实运动，不能单独叫作模型flicker；没有GT，中位数变化也不是已测得的真实米制尺度漂移。**

### 当前六段的使用安排

| ID（省略.mp4） | 类型 | 按10 FPS预计帧数 | 背景任务 |
|---|---|---:|---|
| `fixed_scenario1` | fixed_static | 122 | 全图 |
| `fixed_scenario2` | fixed_static | 130 | 全图 |
| `fixed_scenario1_people_moving` | fixed_dynamic | 139 | N/A |
| `fixed_scenario2_people_moving` | fixed_dynamic | 135 | N/A |
| `moving_left_to_right` | moving | 141 | N/A |
| `moving_rear_to_front` | moving | 151 | N/A |

预计818帧/812对是根据当前源PTS和选帧规则得到的安排，不是本流程已运行的结果。最终以每片新manifest、run和summary为准；不将六段名称、帧数或尺寸写死在调度器里。

### 用户运行：六段自录完整实验

本页命令只运行Part 1B，结果保存在`outputs/part1/video/<ID>/`。它不会启动VDA或生成Part 2B报告。Part 2B由[Step 4](step4.md)的独立入口运行并另存；比较时只读引用这里的RGB、DA-V2和光流。

先检查一次本地资源与实际CUDA；不推理、不下载。若指定输出文件已存在，换一个检查文件名，不覆盖记录。

```bash
./scripts/part1/video.sh check --output outputs/part1/video_checks/resources.json
```

要求`ready_for_requested_device: true`。下面每行独立完成一个视频的全部Part 1B任务，主要耗时在DA-V2与双向RAFT，由用户启动。不必按功能把六段来回分批处理。

```bash
./scripts/part1/video.sh run --clip fixed_scenario1
./scripts/part1/video.sh run --clip fixed_scenario2
./scripts/part1/video.sh run --clip fixed_scenario1_people_moving
./scripts/part1/video.sh run --clip fixed_scenario2_people_moving
./scripts/part1/video.sh run --clip moving_left_to_right
./scripts/part1/video.sh run --clip moving_rear_to_front
```

每段自动执行`prepare → dav2 → flow → analyze1 → verify1`。结束应有`outputs/part1/video/<ID>/verify1.json`的`status: verified`；命令失败会停止并保留实际失败记录，不把后续阶段记成成功。

六段都完成后可汇总；单段视频的完整产物不依赖汇总。

```bash
./scripts/part1/video.sh summarize --output-dir outputs/part1/video_summary/all_v1
```

按三种类型分别等视频汇总，N/A不补零。新视频完成后可再次生成到新的汇总目录，如`outputs/part1/video_summary/with_new_clip_v1`；这只是汇总快照，每段视频仍只有自己的同一套完整产物。

### 只做某项任务与续跑

`plan`只显示会做哪些依赖，不解码、不预测、不生成正式结果。

```bash
./scripts/part1/video.sh plan --clip fixed_scenario1
./scripts/part1/video.sh run --clip fixed_scenario1 --task prepare
./scripts/part1/video.sh run --clip fixed_scenario1 --task dav2
./scripts/part1/video.sh run --clip fixed_scenario1 --task flow
./scripts/part1/video.sh run --clip fixed_scenario1 --task analyze --no-deps
./scripts/part1/video.sh run --clip fixed_scenario1 --task verify --no-deps
```

`--task`可取`all/prepare/dav2/flow/analyze/verify`，默认all。默认补齐所选任务缺少的依赖；`--no-deps`严格只执行该任务，缺依赖就报错。`analyze`会生成所有适用的逐像素、光流、背景和尺度诊断；`verify`从raw/flow重算全部指标，不执行模型。

再次运行同一条`run --clip ID`即可续跑：已完成阶段须通过来源/完整性核验才复用，原预测和元数据不覆盖。未完成或失败目录不会被当作成功跳过；查看`prepare_failure.json`、对应run/status及`commands/`后处理。若确实需要重试失败阶段，可把失败目录原样移到维护位置，再恢复原ID；改变原视频、类型、配置或代码后应使用新ID或新的输出和缓存根目录，不绕过绑定。

### 新增视频

把原片放到`data/self_recorded/`，用尚未使用的ID，明确类型。下面是假定已经放入`my_new_video.mp4`后的实际命令格式：

```bash
./scripts/part1/video.sh run --clip my_new_video --video data/self_recorded/my_new_video.mp4 --type fixed_static
```

第一次调用会冻结路径、类型、真实源SHA、PTS和配置；之后直接`run --clip my_new_video`即可从保存的登记恢复，不要求修改六段清单。也可以先在`configs/video/clips.yaml`增加一项`video/type/session/description`，再仅按ID运行。session知道时加`--session 实际录制批次`；不知道就保留null，不根据文件名虚构独立性。

### 项目架构与产物

| 文件 | 职责 |
|---|---|
| `scripts/part1/video.sh`、`pipelines/part1/video_clip.py` | Part 1B独立入口，只调度prepare/DA-V2/光流/单模型诊断与核验 |
| `pipelines/video/clip.py` | 两入口共用参数定义和预检/汇总辅助；不提供可执行入口 |
| `src/video/clip_workflow.py` | 按单视频解析登记、处理依赖、保存命令/失败、核验后复用 |
| `src/video/clip_analysis.py` | 全图逐像素指标、光流指标、类型限定背景、尺度proxy、原生重算与汇总 |
| `src/video/clip_report.py` | 单/双模型共用的固定色标深度、两种差分视频、曲线与案例 |
| `src/video/frames.py`、`runner.py`、`flow.py`及官方adapter | 真实PTS选帧、冻结、官方推理与RAFT；共用数值基础 |

```text
data/self_recorded/<ID>.mp4                 原视频，不改写
data/video_frames/self_recorded/<ID>/       只准备一次的RGB、PTS和manifest
outputs/part1/video/<ID>/
  clip.json、protocol.json                  视频类型、源/配置/源码冻结
  dav2/{run.json,raw/}                      逐帧原生relative
  flow/{flow_manifest.json,pairs/}          共享双向RAFT
  part1/
    summary.json、status.json
    pair_metrics.csv、frame_metrics.csv
    dav2/                                  光流mask、适用的全图背景数组
    report/
      analysis.md、report.json、curves.png/pdf
      depth_raw.mp4、depth_normalized.mp4
      pixel_difference.mp4、flow_difference.mp4
      pixel_cases.png、flow_cases.png、contact_raw.png、contact_normalized.png
      background_std.png                   仅fixed_static
  verify1.json                             原生重算核验
  commands/                                每次调用及实际执行/复用/失败
```

### 结果与分析状态

正式结果待用户运行后填写；当前不填入其他协议的背景std或任何预测结果。每片先看`part1/report/analysis.md`的类型说明与各项指标，再看两种差分、曲线、coverage、饱和比例和案例。固定无人看短时闪烁和整段缓变；固定有人看人物进入/离开、遮挡及背景受影响；移动看对应位置变化和中位数趋势，并明确真实运动限制。

当前验证使用真实临时MP4/PNG/NPY和合成预测，检查全画面、平移、零值/N/A、类型分工、来源篡改、续跑、失败保留、CSV/图视频与原始重算；不加载深度或RAFT权重，也不代表真实视频已取得新结果。

### 自录视频怎样拍

下面是便于得到清楚、可解释结果的**本项目建议**，不是课程规定的固定时长、段数或距离。

#### 手机设置与场景布置

| 项目 | 建议与原因 |
|---|---|
| 时长 | 每段10–15秒；10 FPS分析约100–150帧，足以画变化曲线。另留约1秒用于放稳相机，不必拍几分钟 |
| 原片尺寸 | 优先1920×1080；1280×720也可。横屏16:9，30 FPS即可，通常无需4K/60 FPS |
| 镜头 | 后置普通1×镜头；全片不变焦、不切镜头，不用人像虚化、电影景深或滤镜 |
| 清晰度 | 擦镜头，避免极暗、严重运动模糊和近距离失焦；保留手机原文件，避免聊天软件转发压缩 |
| 近处 | 约0.7–2米有桌沿、椅子、箱子或植物，不让一个近物占满画面 |
| 中处 | 约2–5米有家具、门框或走廊结构，提供清楚的遮挡与轮廓 |
| 远处 | 约5–10米有墙面、门或走廊尽头；房间较小也可，只要近中远层次可辨 |
| 画面内容 | 同时有较平整表面、纹理、几条明确物体边界及少量细结构，如椅腿；不要只拍一面白墙或一整片天空 |
| 光线 | 均匀、稳定、不过曝；优先白天室内柔和光或稳定照明。主实验避免正对强光窗、开关灯和明显闪烁照明 |
| 自动设置 | 若手机支持，可锁定曝光、对焦和白平衡；无法锁定时保留自动设置并记录，不必另买设备 |
| 防抖 | 固定镜头靠支架；移动时平稳小步。记录是否开防抖，别在片中途切换；不要靠后期防抖改写原片 |

上述距离只是布置场景的参考，**不是GT、不是标定，也不是模型被限制在0.7–10米预测**。不要求逐点测距、RGB-D传感器或相机位姿。普通有层次的房间、办公室、门厅和走廊都合适。

镜面、玻璃、强反光、暗处、快速运动可以另拍挑战片，用于失败分析；先保证主片清楚且条件可解释，不把所有困难都堆进一段。

#### 固定相机怎样拍

1. 手机放三脚架、桌面支架或稳定支撑，放稳后不再碰，避免“手持但尽量不动”。
2. 画面保留近中远层次。先拍场景基本不动的10–15秒，检查背景预测是否跳动。
3. 另一段可让人缓慢经过画面一侧，观察上下文变化带来的影响；留出不被人遮挡的墙面/家具作为稳定背景。
4. 无人固定主片尽量保持整幅画面静止，避免窗外树叶、播放中的屏幕和明显反光变化；有这些真实运动时登记为fixed_dynamic，不按预测分数临时裁画面。

固定相机、静态内容下分数很小甚至接近零也是有效结果，不必故意制造闪烁。

#### 移动相机怎样拍

1. 选择静止场景，手机大致保持朝向和高度，10–15秒缓慢侧移约0.5–1.5米；或者缓慢前进约0.5–1米。
2. 保持近中远物体大部分仍在视野内，避免快速甩动、整圈旋转、突然遮住镜头或连续变焦。
3. 侧移有利于观察对应位置变化；前进会真实改变物体距离，分析时必须承认这一点，不能把所有变化称为预测漂移。
4. 可在另一段加入动态物体或轻微遮挡用于失败分析，不需要相机轨迹GT。


## Part 1B完成门禁

| 项目 | 当前状态 |
|---|---|
| NYUv2标准654张relative、GT辅助对齐、七指标、效率/案例与原始重算 | 已完成，原产物保持不变 |
| 三类型单视频入口、全图逐像素与光流指标、背景/尺度诊断、报告与核验 | 实现和合成验证已通过 |
| 六段正式全片预测、光流、适用指标和原始重算 | 待用户运行本节单视频命令 |
| 实际flicker/scale proxy、效率、收益/失败解释 | 待正式产物后分析，不把无GT proxy当米制真值 |
| mandatory clips来源确认、最终Demo、报告及视频打包 | 单列待完成 |
