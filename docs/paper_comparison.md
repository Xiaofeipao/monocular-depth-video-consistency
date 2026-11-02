# Part 3 相对 Part 2 及本地论文的证据审查

## 可以支持的结论

当前有证据支持的是：**在本项目固定协议下改善已复现模型的局部指标**，
没有证据支持“整体明显超过这些论文”或“SOTA”。

- **C 的边界收益最可靠**：NYUv2 全部 654 图上，Depth Pro 与 UniDepthV2-S 的
  SI Boundary F1 分别相对提高 7.43% 和 8.45%。这是在同模型、同帧、同评分代码下
  相对于 Part 2 的真实增益；不是与论文异数据表格直接比较的结果。
- 距离精度提升很小：两模型 AbsRel 相对下降约 0.106% 和 0.100%；RMSE 均值分别
  只降低约 0.677 mm 和 0.786 mm。这里是 **RMSE 指标的减少量**，不是毫米级测距精度。
- A/E 在自录静态片段上降低波动，E_half 相比 A 有后处理效率优势；E 不比原始 Part 2
  模型更快，它仍增加后处理开销。C_cuda 的约 22–24 倍也仅是相对 CPU C 的后处理速度。
- D 在自录短片段中的 warp 分数最低，但无真值；既有 TUM GT 测试中 δ1 损失超标，
  不能因为这个 proxy 更低就认定深度更准确或整体优于 Part 2。

## C 的配对统计与实际幅度

在冻结的逐图 CSV 上补做 10,000 次配对场景 bootstrap，seed=42。
654 张图来自 215 个 scene；每次抽取整场景并保留该场景所有图，
用抽样后总和除以图数保持图像加权估计，避免把同场景图简单视为独立样本。

- Depth Pro：F1 **0.159269 → 0.171107**；绝对增益 **0.011838**，
  场景 bootstrap 95% 区间 **[0.010160, 0.013552]**；约 **72.8%** 图的 F1 改善。
- UniDepthV2-S：F1 **0.113010 → 0.122561**；绝对增益 **0.009551**，
  95% 区间 **[0.008045, 0.011163]**；约 **67.3%** 图改善。
- 两模型 AbsRel 的配对区间也在有利方向，但量级只有约 **0.0001**。
  不能把统计可检测性等同于明显的实际精度提升。

这些为条件于本划分/协议的探索性区间，没有多重比较修正，不能证明跨域最优。
脚本：`scripts/audit_refinement_statistics.py`；数值：
`evidence/paper_comparison/refinement_scene_bootstrap.json`。

## 比较前必须处理的差异

必须匹配数据及划分、模型和权重、输入分辨率、训练/微调情况、GT 内参、
尺度/偏移对齐、裁剪、有效深度与预测裁剪、边界容差、聚合和完整计时范围。
本项目 NYUv2 使用 654 图、crop `[45:471,41:601]`、`0.001 < GT < 10 m`，
原生米制、不做 GT 尺度拟合；先对完整 RGB 细化再评分。

**运动段的 warp 差不是纯闪烁。** VDA 论文 §3.2 明确解释：相机前进时，同一物体
在相邻帧的真实深度也会改变，光流对应不意味着深度不变。它的补充材料使用 GT
对整段逆深度作 scale/shift 对齐，并以相机变换进行双向投影计算 TAE。
我们的 Farneback/RAFT log-depth warp 差没有用相机位姿更新深度 z，既包含预测
变化，也包含真实几何变化；RAFT 只提供另一种光流，不会将该 proxy 变为 GT 指标。
因此移动和人物段的“降低 8%–17%”只能描述这个固定 proxy，不能等同真实闪烁、
米制尺度漂移或论文 TAE 改善。静态相机且场景静止的段更适合支持闪烁结论。

## 逐篇审查

已遍历项目全部 PDF：12 篇研究论文及 1 份作业要求。以下页码均为本地 PDF 的
物理页序，包含补充材料，不使用印刷页码。可检索文本提取存在部分旋转表格/字体缺失，
关键表格另行渲染核对；TUM 本地 PDF 正文为图像，结合相同论文的作者官方 PDF 阅读。
论文身份、页数与 SHA256 见 `evidence/paper_comparison/paper_inventory.json`。

### 1. SGM：Stereo Processing by Semiglobal Matching and Mutual Information

本地第 9 页 Table 1：旧 Middlebury 的 Tsukuba、Venus、Teddy、Cones，
非遮挡区 bad-1 / bad-0.5；例如 SGM 在 Teddy 的 bad-1 为 **6.02%**。
正文还报告 Teddy 上 SGM / C-SGM 分别约 **1.8 / 2.7 秒**，为论文当时的设备。

我们使用 Middlebury v3 Q 的 15 对和 OpenCV BM/SGBM，bad2、覆盖率、深度误差
不能与该表直接排名。OpenCV 官方说明 SGBM 是修改版，未实现论文的 mutual-information
代价；不能把调用 OpenCV 等同于完整重现 MI-SGM。也不能拿双目输入相对单目论文宣称公平优势。
结论：方法来源与经典基线，**没有证明超过原论文**。

来源：[本地 PDF](<papers/Stereo_Processing_by_Semiglobal_Matching_and_Mutual_Information.pdf#page=9>)；
[作者机构记录](https://elib.dlr.de/55367/)；
[OpenCV 差异说明](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html)。

### 2. DPT：Vision Transformers for Dense Prediction

第 5 页 Tables 1–3 区分相对深度跨域和微调米制深度。
NYUv2 的 DPT-Hybrid 为 **AbsRel 0.110、RMSE 0.357 m、δ1 0.904**。
我们的 Depth Pro+C 为 **0.100367、0.391543 m、0.915454**：
仅按数字看，AbsRel/δ1 更好而 RMSE 更差，并不全面占优。

基础模型、训练数据、权重与实现不同；主要差距也来自 Depth Pro，C 仅增加很小的
AbsRel 收益。结论：可作有条件的历史参考，**不能作为 Part 3 超越 DPT 的受控证据**。
来源：[本地 PDF](<papers/Vision Transformers for Dense Prediction.pdf#page=5>)；
[论文](https://arxiv.org/abs/2103.13413)。

### 3. Depth Anything V1

第 5 页 Table 2：ViT-L 相对深度 NYUv2 **AbsRel 0.043、δ1 0.981**；
第 6 页 Table 3：在 NYUv2 微调后的米制模型 **0.056、RMSE 0.206 m、δ1 0.984**。
第 9 页补充说明相对深度评测使用 GT scale/shift 对齐。

我们的原生米制结果不能与对齐后的 0.043 比；与微调表的数值也没有明显优势，
且训练条件不同。未运行 V1 同帧、同权重规模的对照。结论：**没有超过论文的证据**。
来源：[本地 PDF](<papers/Depth Anything Unleashing the Power of LargeScale Unlabeled Data.pdf#page=6>)；
[论文](https://arxiv.org/abs/2401.10891)。

### 4. Depth Anything V2

第 8 页 Table 2 的相对深度 NYUv2：ViT-S **0.053 / 0.973**，
ViT-L **0.045 / 0.979**（AbsRel / δ1）。第 9 页 Table 4 的 NYUv2 微调米制结果：
ViT-S **AbsRel 0.073、RMSE 0.261 m、δ1 0.961**；
ViT-L **0.056、0.206 m、0.984**。

相对、微调、模型规模与原生米制不能混比。正文 §7.3 还区分 NYU/KITTI 微调表
与公开的 Hypersim/Virtual KITTI 米制模型，不能默认下载权重就对应 Table 4。
我们也未评 DA-2K 的 2K 深度排序点对，不能声称细结构/透明材质全面超过 V2。
结论：**没有明显优于论文的结果**。
来源：[本地 PDF](<papers/Depth Anything V2.pdf#page=9>)；
[论文](https://arxiv.org/abs/2406.09414)。

### 5. Video Depth Anything

第 6 页 Table 1：VDA-S 的 NYUv2 视频 **AbsRel 0.077、δ1 0.959**，
ScanNet 170 帧 **TAE 0.703**；VDA-L 对应 **0.062 / 0.971**、**TAE 0.570**。
第 7 页 Table 3 报告 VDA-S 在 GPU、518×518 下约 **9.1 ms/帧**。
第 5、13 页明确 GT 对齐、数据划分和位姿投影协议。

我们是同一室内环境的四段 16 帧、input-size 252 的 metric Small 短片段，没有 GT/pose；
用光流 log-error 不能对比 TAE，用 CPU 后处理 FPS 不能对比 GPU 全模型 latency。
新增 A/C/D/E 是对已发布模型的推理后处理，不是新训练的视频基础模型。
结论：**有本项目基线增益，没有论文基准胜出的证据**；移动段解释须遵守上面的几何限制。
来源：[本地 PDF](<papers/Video Depth Anything Consistent Depth Estimation for Super-Long.pdf#page=6>)；
[论文全文与指标说明](https://arxiv.org/html/2501.12375v3)。

### 6. Depth Pro

第 8 页 Table 2 的 SI boundary F1：**Sintel 0.409、Spring 0.079、iBims 0.176**。
我们的 **NYUv2 0.171107** 不在同一数据集，数值接近不意味着接近或超过论文成绩。
第 33 页 Table 18 的 Middlebury 为原高分辨率 15 图，不是我们 Q 分辨率双目评分。

第 22 页 Table 5 的完整模型约 **341.3 ms/图**，含预处理/推理；不能拿 C_cuda
后处理的 91 FPS 与它相比。第 26 页 Table 9 另有只按 Stage 1 训练的受控实验，
Depth Pro NYUv2 δ1 **96.1%**；那也不是与我们同协议、同最终 checkpoint 的直接复现目标。
结论：**C 对本地同权重基线的边界改善成立，但尚未证明在论文 iBims/Sintel/Spring 上优于原文**。
来源：[本地 PDF](<papers/Depth Pro Sharp Monocular Metric Depth in Less Than a Second.pdf#page=8>)；
[论文全文](https://arxiv.org/html/2410.02073v2)。

### 7. UniDepth V1

第 6 页 Table 2，UniDepth-V 的 NYUv2 为 **AbsRel 0.0578、RMSE 0.201 m、δ1 0.984**，
属于论文描述的 zero-shot 配置。论文同时区分 GT 与预测相机设置，评测行必须匹配。

我们评的是 **UniDepthV2-Small**，不能把 V1 的 ViT-L 成绩当作同一模型基线。
我们的最佳 Depth Pro+C 数值也没有全面胜出；没有同配置重跑 V1。
结论：**不能证明超过 UniDepth V1**。
来源：[本地 PDF](<papers/UniDepth Universal Monocular Metric Depth Estimation.pdf#page=6>)；
[论文](https://arxiv.org/abs/2403.18913)。

### 8. UniDepthV2

第 9 页 Table V，NYUv2 **微调 Large** 为 **A.Rel 4.68%=0.0468、RMSE 0.180 m、δ1 98.8%**。
我们是公开通用 Small，细化后为 **0.114427 / 0.398035 m / 89.260%**，
既非相同容量也非 NYUv2 微调，不能把“超过自有 Small 基线”写成超过论文 Large。

第 8 页 Table IV 的边界表在 ETH3D/iBims/Sintel，不是 NYUv2。
第 9 页 Table VII 用 AUSE/nAUSE/Spearman 评估学习的不确定性；我们的 D 是局部残差
启发式，没有这些标定指标，不能声称实现或优于论文的不确定性机制。
第 10 页效率表也包含输入大小与精度约束。结论：**C 的局部改善成立，没有论文级胜出证据**。
来源：[本地 PDF](<papers/UniDepthV2 Universal Monocular Metric Depth Estimation Made.pdf#page=9>)；
[论文全文](https://arxiv.org/html/2502.20110v2)。

### 9. RAFT

第 11 页 Table 1 报告光流精度，包括完整 RAFT 的 KITTI F1-all **5.10%**、
Sintel final warm-start **约 2.86 px EPE**。F1-all 在这里是光流错误率，不是深度边界 F1。
该表还单列 Small 和不同训练组合。

我们使用 torchvision RAFT-Small 作为 CPU 评分工具，没有光流 GT，使用默认
12 次更新，与论文标准完整模型推理条件也不同。因此 RAFT 评分改善仅支持与
Farneback 不完全相同的交叉检查，**不代表我们的模型或光流超过 RAFT**。
来源：[本地 PDF](<papers/RAFT Recurrent All-Pairs Field Transforms for Optical Flow.pdf#page=11>)；
[论文](https://arxiv.org/abs/2003.12039)。

### 10. NYUv2 数据论文：Indoor Segmentation and Support Inference from RGBD Images

论文包含 **1,449 张 RGB-D、464 个场景**；第 11 页 Table 1 评层次分割区域重叠，
例如 RGBD+Support+Structure 加权分数 **63.9**。第 12 页 Table 2 的 LP 支持关系
预测在 GT 区域上的 type-aware 准确率 **72.6%**，不是单目深度精度。

我们的 NYUv2 654 测试图用于深度评测，没有重现其分割/支持关系任务。
结论：**这是数据与应用背景来源，没有相同任务可宣称胜出**。
来源：[本地 PDF](<papers/Indoor Segmentation and Support Inference from RGBD Images.pdf#page=11>)；
[作者机构页面](https://www.microsoft.com/en-us/research/publication/indoor-segmentation-support-inference-rgbd-images/)。

### 11. KITTI 数据论文：Are we ready for Autonomous Driving?

第 6 页 Table 2：KITTI 2012 的 OCV-SGBM 为 **7.64% 非遮挡 / 9.13% 全部 bad-3**，
密度 **86.50%**；表中对无效预测进行插值，数据、阈值和覆盖处理均有规定。

我们不是该 KITTI 2012 测试服务器提交；Middlebury Q 的 bad2 或四张 KITTI smoke
结果不能与它直接比较。更不能把后来 KITTI Eigen depth split 与该旧 stereo 排名混用。
结论：**不能证明超过该论文的算法或数据基准**。
来源：[本地 PDF](<papers/Are_we_ready_for_autonomous_driving_The_KITTI_vision_benchmark_suite.pdf#page=6>)；
[官方基准](https://www.cvlibs.net/datasets/kitti/eval_stereo.php)。

### 12. TUM 数据论文：A Benchmark for the Evaluation of RGB-D SLAM Systems

本地第 6–7 页 §VII 定义 **RPE 与 ATE**，评的是估计相机轨迹与运动捕捉 GT
之间的相对/绝对位姿误差；数据包含同步 RGB、深度和相机位姿。

我们的短 TUM 实验计算深度误差和使用 GT 位姿的深度重投影残差，没有输出并评价
SLAM 估计轨迹；`pose_temporal_log_error` 不是 ATE 或 RPE。
结论：**可引用其数据/位姿协议，不能宣称击败论文的 SLAM 系统**。
来源：[本地 PDF](<papers/A_benchmark_for_the_evaluation_of_RGB-D_SLAM_systems.pdf#page=6>)；
[作者官方 PDF](https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf)。

## 项目中应采用的贡献表述

“在冻结的 NYUv2 654 图测试协议下，固定的无训练 RGB/深度联合细化提高两条
已复现米制基线的 SI Boundary F1；按场景配对分析支持本划分内的边界增益。
在自录短片段上，轻量时序处理降低了固定光流 warp proxy，半分辨率光流减少
相对完整光流融合的 CPU 后处理开销。精度、边界、速度和时序的收益分别报告。”

目前不应写“全面超过基础模型论文”“达到 SOTA”“移动段真实尺度漂移下降 17%”
或“CPU 后处理 FPS 比论文 GPU 模型快”。

## 扩展验证与反例

已扩展到四段自录视频各 64 帧及同一 TUM 序列两个不重叠片段（64/54 帧），
并在公开 MoGe 处理版 iBims 100 图上固定参数验证两条米制图像基线。
完整结果、逐对评分和限制见 [扩展验证](extended_validation.md)。

- iBims / Depth Pro：C 的 masked ratio SI F1 **0.171601 → 0.155297**，
  相对下降约 **9.50%**；AbsRel **0.160301 → 0.160264**，仅有很小改善。
- iBims / UniDepthV2-S：同指标 **0.091521 → 0.093874**，相对提高约 **2.57%**；
  AbsRel **0.136771 → 0.136708**。不能把 NYUv2 的收益推广到所有模型/数据。
- iBims 是 JPEG/log-quantized 深度的处理版，评分要求相邻 GT 像素有效，
  不能与 Depth Pro 原始 MAT/unmasked 表格直接排名。
- 252 输入的两个扩展 TUM 片段中，D 均通过相对 RAW 的精度/边界约束，
  pose-log-error 最低；这与此前短片段的失败共同表明结果依赖片段。
  原生 VDA 的 AbsRel 仍约 0.43–0.45，后处理没有修复主要米制偏差。

这些是同任务可配对的补充实验，并非复现所有论文原任务：SGM/KITTI 的旧版立体基准、
NYU 的分割/支撑推断、RAFT 的光流 GT 及 TUM 的 SLAM 轨迹仍没有对应排名实验。
不能以新增深度实验代替上述任务，也不能据此宣称超过每篇论文。

518 同帧视频检查见 [分辨率比较](video_resolution_comparison.md)。
518 组两段 TUM 的原始 AbsRel 为 0.129460 / 0.214268，显著低于 252 组；
设备和分辨率同时改变，不能单独归因。D 在两段仍通过相对精度/边界约束，
pose-log-error 降低约 2.50% / 6.30%，但未进行跨序列统计显著性检验。
C 的 log-edge F1 两段均下降超过 0.01，未通过边界退化约束。
因此高分辨率下，不能继续把 C 当作视频边界增强的默认选择。

[D 多序列研究](d_research.md)进一步用独立的开发/测试序列划分检查了机制：
完整 D 在三条新序列均通过预设精度/边界约束，但固定权重消融的时序分数更低，
其中一段以边界 F1 退化超标为代价。开发组优化配置未改善原 D 的测试预测。
这些结果支持“固定后处理的局部折中”，不支持比 VDA 的原始 TAE、UniDepthV2 的
学习式不确定性或其他论文整体更好的结论。失败和无增益结果均保留。
若采用论文相对深度/TAE 协议，需要单独标为 GT 对齐的独立实验，不能替换原生米制结果。
