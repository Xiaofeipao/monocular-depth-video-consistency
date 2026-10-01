# AIAA 3201 Project 3 高分执行计划

> 项目：Monocular Metric Depth Estimation & Video Consistency  
> 计划版本：2026-10-01    
> 总策略：先保证三个 Part、评估协议与全部交付物完整，再围绕一个可验证的核心贡献做深入实验和展示。

---

## 0. 最终目标与完成标准

最终系统应形成一条可复现的完整链路：

```text
RGB / 双目 / 视频
  -> 静态与视频深度模型
  -> 原始浮点深度（米或明确标注的相对深度）
  -> 时序改进模块
  -> 定量评估 + 可视化
  -> 点云与点击测距 Demo
```

项目只有同时满足以下条件才算完成：

- [ ] Part 1：Middlebury Stereo v3 上完成 StereoBM/StereoSGBM、左右一致性、视差到深度、误差敏感性分析。
- [ ] Part 1：完成逐帧 Depth Anything V2-Small 相对深度基线，测量静态闪烁和运动相机尺度漂移。
- [ ] Part 2：Depth Pro 在 NYUv2 标准测试划分上完成**不做测试时尺度对齐**的零样本度量深度评估。
- [ ] Part 2：Video Depth Anything Small 完成离线视频推理，并与逐帧模型在完全相同的帧、分辨率和显示范围下比较。
- [ ] Part 3：实现一个轻量级时序改进，至少给出完整方法、受控消融、失败案例；即使主指标未提升也不得省略。
- [ ] 报告全部必需指标：AbsRel、SqRel、RMSE、RMSE-log、δ1/δ2/δ3、边界指标、时序指标、速度与显存。
- [ ] Demo 同屏展示 RGB、深度、点云/点击测距结果，并对测距误差做定量验证。
- [ ] 公开 GitHub 仓库可从全新环境复现主要表格和示例；不提交数据集或受限权重。
- [ ] 6–8 页 CVPR camera-ready 报告已上传 arXiv；摘要末尾有公开 GitHub 链接，Canvas 版本首页有 arXiv ID。
- [ ] `depth_results.zip` 含所有指定视频、原始结果说明和可运行或录制好的 Demo。

高分的核心不是堆模型，而是形成一条可信论证：

> 原始逐帧/视频模型在遮挡、快速运动和局部不确定区域会出现时序误差；我们的方法只在可靠对应区域传播历史深度，并用原始度量预测持续锚定尺度，因此在降低时序误差和漂移的同时，尽量不损害单帧精度与边界。

---

## 1. 技术路线与优先级

### 1.1 主线方案

| 模块 | 主方法 | 对照/备选 | 输出性质 |
|---|---|---|---|
| 经典双目 | OpenCV StereoSGBM | StereoBM | 度量深度，使用标定参数 |
| 逐帧相对深度 | Depth Anything V2-Small | DPT（仅时间允许时） | 相对深度；benchmark 对齐后仍不得称为 metric |
| 单图度量深度 | **Depth Pro** | UniDepthV2-Small（扩展对照） | 原生米制深度，不做 GT 对齐 |
| 视频深度 | **Video Depth Anything Small**，离线 FP16 | streaming Small；Base 仅在资源允许时 | 分开评估 relative 与 metric checkpoint |
| 光流 | RAFT-Small | Farnebäck 作为低成本对照/回退 | 前向流、后向流、遮挡 mask |
| Part 3 | 可靠性门控的逆深度时序融合 + 尺度锚定 | 朴素 EMA、仅 flow warp | 保持输入分支的尺度定义 |
| 应用 | 点云 + 点击测距 | 碰撞预警 | 可视化并可定量测量 |

选择 Depth Pro 作为单图度量主模型的理由：直接输出米制深度，同时预测/使用焦距，边界质量好，并自带 scale-invariant boundary F1 实现。UniDepthV2 作为扩展实验而非关键路径，以免环境和模型数量挤占 Part 3、分析和写作时间。

### 1.2 实验必须分成两条互不混淆的轨道

1. **Metric track（主轨）**
   - Depth Pro、Metric Video Depth Anything、Part 3 metric 输出。
   - NYUv2 上不允许用测试 GT 做 median 或 scale-and-shift alignment。
   - 单位必须是米，保存为 `float32 .npy/.npz`。

2. **Relative track（公平时序复现轨）**
   - 逐帧 Depth Anything V2-Small 对比 Relative Video Depth Anything-Small。
   - 只有官方协议明确要求时才对 GT 做对齐，并在表头写 `aligned`。
   - 自录视频无 GT 时，用同一初始帧/片段级映射固定尺度，不可逐帧对齐后再声称消除了漂移。

绝不把“对每帧单独对齐的相对深度”与“原生 metric depth”放在同一栏中暗示二者等价。

### 1.3 工作优先级

- **P0：** 数据协议、三个 Part、必需指标、原始浮点输出、全部提交物。
- **P1：** Part 3 完整消融、统计可信度、失败案例、固定范围视频。
- **P2：** 高质量 pipeline 图、局部放大图、交互 Demo、完整 README。
- **P3：** UniDepthV2、Base 模型、ONNX/量化等扩展；只有 P0–P2 稳定后再做。

不计划从头训练 foundation model，也不把 Large 模型作为关键路径。

---

## 2. 官方仓库

官方资料：

- [Depth Pro](https://github.com/apple/ml-depth-pro)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)

---

## 3. 建议的新工程结构

不建议把项目拆成三个彼此独立的 `part1/`、`part2/`、`part3/` 大工程。三个 Part 会共同使用数据读取、模型 adapter、指标、可视化和结果格式；完全按 Part 复制这些代码，后期很容易出现同一指标有三个版本、修复不同步、比较不公平的问题。

建议采用**混合结构**：

- `pipelines/part1|part2|part3`、`scripts/part1|part2|part3`、`configs/part1|part2|part3` 和 `outputs/part1|part2|part3` 按作业 Part 组织，让执行入口和结果归属一眼可见。
- `src/` 中只保留可复用实现，按功能组织；每个 pipeline 只负责编排，不重复实现指标或模型。
- Part 3 直接读取 Part 2 冻结的原始预测，避免为了做改进而悄悄改变 baseline。

```text
.
├── PLAN.md
├── README.md
├── LICENSE
├── environment.yml
├── requirements-lock.txt
├── configs/
│   ├── versions.yaml
│   ├── paths.example.yaml
│   ├── part1/
│   │   ├── middlebury_sgbm.yaml
│   │   └── dav2_framewise.yaml
│   ├── part2/
│   │   ├── depth_pro_nyuv2.yaml
│   │   └── video_depth_anything.yaml
│   └── part3/
│       ├── temporal_fusion.yaml
│       └── ablations.yaml
├── src/
│   ├── data/                 # 数据转换、split、manifest
│   ├── stereo/               # BM/SGBM、L-R check、深度转换
│   ├── models/               # 第三方模型 adapter，统一输出格式
│   ├── flow/                 # RAFT/Farneback 与遮挡检测
│   ├── temporal/             # Part 3 方法
│   ├── metrics/              # image/boundary/temporal/efficiency
│   ├── visualization/        # 固定色标、局部放大、视频排版
│   └── demo/                 # 点云、点击测距 UI
├── pipelines/                # 按作业 Part 暴露清晰的 Python 入口
│   ├── part1/
│   │   ├── run_stereo.py
│   │   └── run_framewise_depth.py
│   ├── part2/
│   │   ├── run_metric_depth.py
│   │   └── run_video_depth.py
│   └── part3/
│       ├── run_temporal_fusion.py
│       └── run_ablation.py
├── scripts/
│   ├── prepare_data.sh
│   ├── part1/{run_all.sh,eval_all.sh}
│   ├── part2/{run_all.sh,eval_all.sh}
│   ├── part3/{run_all.sh,eval_all.sh}
│   ├── make_figures.sh
│   └── package_submission.sh
├── tests/
│   ├── test_geometry.py
│   ├── test_warp.py
│   ├── test_metrics.py
│   └── test_output_schema.py
├── data/                     # gitignored；只留 README/manifest
├── checkpoints/              # gitignored；只留下载脚本/checksum
├── outputs/                  # gitignored；先按 Part，再按 run_id 保存
│   ├── part1/
│   ├── part2/
│   └── part3/
├── assets/                   # 小型、可公开的代表性结果
├── report/                   # CVPR LaTeX、BibTeX、图表
└── third_party/              # 固定版本，不混放自己的实现
```

依赖方向保持单向：

```text
shared src + data/evaluation contracts
       ├── Part 1 pipelines（建立几何、逐帧与指标基线）
       ├── Part 2 pipelines（产生冻结的 metric/video baseline）
       └── Part 3 pipelines（读取 Part 2 结果并做改进，不反向改 baseline）
```

README 顶层按 Part 1、Part 2、Part 3 给出三组命令；读者不需要理解内部模块就能依次复现实验。开发时仍只有一份 `metrics`、一份 `visualization` 和一套输出 schema。

统一预测文件规范：

```text
outputs/<part>/<run_id>/
├── config_resolved.yaml      # 完整参数与版本
├── environment.txt           # GPU、CUDA、PyTorch、依赖版本
├── manifest.json             # 输入帧及哈希
├── depth/*.npy               # H×W float32；米或相对值
├── valid/*.png               # 有效像素 mask
├── flow/*.npz                # 可选
├── metrics_per_sample.csv
├── metrics_summary.json
├── timing.json
└── visualization/*
```

每个深度结果必须带 `depth_type: metric|relative`、`unit: meter|arbitrary`、`alignment: none|median|scale_shift`，从文件层面杜绝结果混淆。

---

## 4. Phase 0：环境、数据与协议锁定

### 4.1 环境

- [ ] 记录实际 GPU 型号、显存、驱动、CUDA、PyTorch；当前终端无法连接 NVIDIA driver，正式实验前先解决计算节点/GPU 分配问题。
- [ ] 根据兼容性分环境：`depth-core`（本项目、DA/VDA/RAFT）与 `depth-pro`；若依赖可统一再合并。
- [ ] 固定随机种子；开启 `torch.inference_mode()`；正式速度测试前固定 cudnn 与精度设置。
- [ ] 下载权重后计算 SHA256；严禁把 checkpoint 直接提交公开 GitHub。
- [ ] 写 smoke test：每个模型对 1 张图/8 帧视频产生有限、正值、尺寸正确的浮点深度。

### 4.2 数据

1. **Middlebury Stereo v3 Quarter Resolution**
   - 输入：[MiddEval3-data-Q.zip](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip)
   - GT：[MiddEval3-GT0-Q.zip](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip)
   - 15 个 training scenes 全部处理；解析每个场景的 `calib.txt` 与 PFM GT。

2. **NYU Depth V2 labeled**
   - 数据：[nyu_depth_v2_labeled.mat](https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)
   - 只下载约 2.77 GB labeled 文件，不下载 428 GB raw collection。
   - 固化标准 654 张 test split 的样本索引和来源；对 split 文件计算哈希。

3. **自录视频**
   - 至少 6 类，每段 15–30 秒；保留原视频、EXIF/相机信息、录制说明。
   - `V1-static`：固定相机、场景基本静止，测纯 flicker。
   - `V2-dynamic`：固定相机、人物横穿，测动态物体与遮挡。
   - `V3-pan`：平滑横移/旋转，测运动相机尺度漂移。
   - `V4-forward`：向前移动，含近远景，测尺度变化与点云稳定性。
   - `V5-hard-motion`：快速运动、运动模糊和大遮挡。
   - `V6-hard-material`：镜子/玻璃/反光表面、细杆或栅栏、低纹理区域。
   - 2 段作为开发集调 Part 3 参数，剩余 4 段冻结为测试集；最终再加入教师指定 mandatory clips。
   - 对测距 Demo 额外录制带卷尺实测距离/已知尺寸目标的片段，至少 20 个测点。

### 4.3 在运行大实验前冻结协议

写入配置并在报告中逐项说明：

- NYUv2 split、crop、预测 resize 方式、有效深度范围（预期为 0.1–10 m，以最终采用协议为准）。
- 是“逐图平均”还是“所有像素汇总”；主表采用官方定义，另一种只能作补充。
- 相对深度的对齐是在 disparity 还是 depth 域、按图还是按数据集。
- 所有模型的输入长边/短边、保持比例方式、padding、输出插值方式。
- 视频 target FPS、抽帧索引、片段长度、离线窗口/重叠设置。
- 可视化使用整个 clip 的固定 2%–98% 分位或固定米制范围，所有方法共享；禁止逐帧归一化。
- 超参数只在开发集调；冻结测试集只用于最终一次主结果。

先用 3 张合成小图为指标写单元测试：完全相同预测应为零误差/δ=1，固定倍数预测应得到可手算结果，含 NaN/0/越界值时 mask 必须正确。

---

## 5. Part 1：几何与逐帧基线

### 5.1 Middlebury：StereoBM/StereoSGBM

实现步骤：

- [ ] 读取 `im0.png`、`im1.png`、`disp0GT.pfm`、`mask0nocc.png` 和 `calib.txt`。
- [ ] 正确处理 OpenCV 视差的 1/16 定点缩放、`min_disp`、`ndisp` 和无效值。
- [ ] 分别从左到右和从右到左估计视差，将右图视差 warp 回左图。
- [ ] 左右一致性 mask：`|d_L(x) + d_R(x-d_L)| < τ_lr`；同时剔除出界、非正视差和遮挡。
- [ ] 使用 SGBM 的 `P1/P2`、block size、uniqueness ratio、speckle filter 等参数建立可复现配置。
- [ ] 固定 5 个覆盖不同困难因素的开发场景选参数，其余场景不调参；同时报告 15 场景逐项结果与宏平均。

Middlebury 标定含主点偏移时，深度应使用：

\[
z = \frac{fB}{d + d_{offs}}
\]

若某场景 `doffs=0`，退化为作业中的 `z=fB/d`。必须核对 baseline 的单位，并统一输出米。GT 深度用同一标定从 GT disparity 得到，不在无效/遮挡区评分。

报告内容：

- StereoBM 与 StereoSGBM 的 disparity MAE/RMSE、Bad-1/Bad-2/Bad-4、有效覆盖率。
- 度量深度 AbsRel/RMSE，并按 GT 深度近/中/远三个区间分桶。
- 每场景展示 RGB、GT disparity、预测、绝对误差、无效区、L-R inconsistency。
- 至少一个局部放大图说明薄结构和遮挡边缘为何失败。

深度敏感性理论与实验必须对应：

\[
\left|\delta z\right| \approx \frac{fB}{d^2}\left|\delta d\right|
= \frac{z^2}{fB}\left|\delta d\right|.
\]

对 GT disparity 人为加入 ±0.25、±0.5、±1、±2 px 误差，绘制深度误差随距离增长曲线，并与上式的一阶近似对照。这会成为报告中连接经典几何与现代单目模型的重要分析图。

### 5.2 逐帧 Depth Anything V2-Small

- [ ] 每帧独立推理；保存未着色的原始 `float32` 相对深度/视差。
- [ ] 对静态图像指标仅按明确协议做 median 或 scale-and-shift alignment；表格明确写 `aligned relative`。
- [ ] 在所有自录视频上运行，保持与 Video Depth Anything 相同的输入尺寸、帧率、帧编号和插值方式。
- [ ] 静态相机 clip 报告像素时序标准差、flow-aligned temporal error。
- [ ] 运动相机 clip 报告相邻帧稳健尺度比及累计 log-scale drift。
- [ ] 输出固定色标视频和逐帧中位深度/时序误差曲线。

验收：随机抽取三帧从保存的原始深度重新生成图，必须与推理时可视化一致；改变可视化色标不得改变任何指标。

---

## 6. Part 2：SOTA 度量深度与视频深度复现

### 6.1 Depth Pro：NYUv2 零样本度量深度

- [ ] 使用官方预训练权重与 transform，记录模型预测焦距 `focallength_px`。
- [ ] 将未裁剪的原始 RGB 输入模型；预测恢复到原图尺寸后，评估阶段才应用 NYUv2 的统一 crop，避免焦距预测与裁剪视场不一致。
- [ ] 主结果不传入 NYUv2 GT 焦距，不做任何 test-time scale/shift alignment。
- [ ] 扩展消融可比较“模型估计焦距”与“可获得的真实/标定焦距”，但两者必须分栏，后者标记为额外相机信息。
- [ ] 将预测 resize 回 GT 尺寸后再按冻结 crop 和 valid mask 评估；深度只按协议 clamp，不为提高分数私自改范围。
- [ ] 保存每张图的全部指标，报告均值、bootstrap 95% CI 和场景类别失败样例。
- [ ] 使用 Depth Pro 官方 `SI_boundary_F1`，并把阈值、有效区域和 GT 边缘生成方式写入配置。

NYUv2 主表至少含：

| Method | Scale | Extra camera info | AbsRel↓ | SqRel↓ | RMSE↓ | RMSE-log↓ | δ1↑ | δ2↑ | δ3↑ | Boundary F1↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DA-V2-S aligned | relative/aligned | GT alignment | | | | | | | | |
| Depth Pro | native metric | none | | | | | | | | |
| UniDepthV2-S（可选） | native metric | none | | | | | | | | |

### 6.2 Video Depth Anything

先跑 smoke test，再跑所有固定片段：

```bash
# 参数以正式 adapter 为准；这里记录官方调用意图
python run.py --input_video <clip> --output_dir <out> \
  --encoder vits --input_size 518 --target_fps <fixed_fps> --save_npz

python run.py --input_video <clip> --output_dir <out_metric> \
  --encoder vits --input_size 518 --target_fps <fixed_fps> --metric --save_npz
```

实验组：

- Relative DA-V2-S frame-wise。
- Relative Video-Depth-Anything-S offline。
- Metric Depth Pro frame-wise。
- Metric Video-Depth-Anything-S offline。
- Video-Depth-Anything-S streaming（扩展，不作为主线；官方说明 streaming 可能降精度）。

公平性规则：

- relative 方法互比、metric 方法互比；不能只选对自己有利的配对。
- 所有方法使用同一实际输入帧；输入尺寸差异若受模型硬约束，必须同时报告原生设置和“共同分辨率”公平设置。
- 固定 clip 级可视化范围，同一视频横向排版 RGB / frame-wise / video / ours / error。
- 离线和 streaming 分开报告 latency、吞吐、峰值显存以及端到端延迟。
- Small 为主模型；Base 只做 1–2 个代表性片段的规模分析，显存不足就不做。

验收：至少一个静态片段和一个运动片段中，Video Depth Anything 的时序指标和逐帧基线完成成对比较；若没有提升，检查相对/metric 权重是否匹配、深度/逆深度是否读反、抽帧与归一化是否一致。

---

## 7. Part 3：可靠性门控的时序深度融合

### 7.1 明确研究问题

**问题：** 直接逐帧预测会闪烁；视频模型在快速运动、遮挡、反光和薄结构附近仍可能不稳定。朴素 EMA 虽能平滑，却会产生运动拖影、复制错误边缘并造成尺度漂移。

**假设：** 只在光流前后向一致、外观一致、非边缘且深度预测相容的像素传播历史逆深度，并用当前原生 metric 预测持续锚定全局尺度，可以降低时序误差而不明显损害单帧准确度和边界。

### 7.2 算法

令当前原始深度为 `D_t`，上一帧最终深度为 `D*_{t-1}`，在逆深度域 `q=1/(D+ε)` 融合：

1. 用 RAFT-Small 计算 `F_{t-1→t}` 和 `F_{t→t-1}`。
2. 用当前到前一帧的 flow 做 backward sampling：`W(q_{t-1},F_{t→t-1})(x)=q_{t-1}(x+F_{t→t-1}(x))`，将上一帧 RGB、逆深度和有效 mask warp 到当前帧；另一方向的 flow 用于前后向一致性检测。
3. 计算可靠性：
   - 前后向流一致性 `e_fb`；
   - RGB photometric residual `e_rgb`；
   - 当前与 warp 深度的 log disagreement `e_d`；
   - 当前 RGB/depth edge mask，保护深度不连续边界；
   - 出界、遮挡、极小深度、NaN 直接判无效。
4. 在可靠静态像素上估计当前与历史的稳健全局 log-scale 偏差；用截尾中位数/Huber 估计，且限制每帧最大修正幅度。
5. 尺度锚定：平滑的历史尺度不得无限累计，而是以当前原生 metric 预测为锚。建议优化：

\[
\hat{s}_t = \arg\min_s \sum_{x\in M_t}\rho\big(\log D_t(x)-\log(s\,\widetilde D_{t-1}(x))\big),
\]

再将 `log(s)` 与 0 做带权收缩，防止用历史误差覆盖当前米制尺度。若 `s` 作用在历史深度上，则对应历史逆深度必须除以 `s` 后才能参与融合。

6. 生成置信权重：

\[
w_t = M_t\exp(-e_{fb}/\tau_{fb})\exp(-e_{rgb}/\tau_{rgb})
      \exp(-e_d/\tau_d)(1-E_t),
\]

并限制 `0 ≤ w_t ≤ α_max`。

7. 在逆深度域融合：

\[
q_t^*=(1-w_t)q_t+w_t\,\widetilde q_{t-1}, \qquad
D_t^*=1/\max(q_t^*,\epsilon).
\]

8. 每次 scene cut、长时间遮挡或可靠像素比例过低时重置历史；记录重置次数。

选择逆深度域是因为近处几何和遮挡边缘对应用更重要，也能避免远距离深度的大数值主导平均；报告中要通过消融验证，而不是只凭直觉陈述。

### 7.3 实现顺序

- [ ] 用合成平移方块测试 flow warp 方向、坐标归一化和 `align_corners`。
- [ ] 实现 `no_flow EMA`，作为必需弱基线。
- [ ] 实现 `flow warp + constant alpha`。
- [ ] 加入前后向一致性与出界/遮挡 mask。
- [ ] 加入 photometric、depth disagreement 与边缘保护。
- [ ] 加入 robust scale anchoring 与 scene-cut reset。
- [ ] 所有中间 mask、权重和误差图可保存，便于解释失败原因。
- [ ] 先接 Depth Pro frame-wise；接口稳定后再接 Metric VDA-S，判断改进对两个基线是否都有效。

### 7.4 预注册消融矩阵

主消融只改变一个因素，使用完全相同的输入和 flow：

| ID | 方法 | Flow | FB/occlusion | RGB/depth gate | Edge protect | Scale anchor |
|---|---|---:|---:|---:|---:|---:|
| A0 | 原始模型 | ✗ | ✗ | ✗ | ✗ | ✗ |
| A1 | 朴素 EMA | ✗ | ✗ | ✗ | ✗ | ✗ |
| A2 | Flow + 固定权重 | ✓ | ✗ | ✗ | ✗ | ✗ |
| A3 | Flow + occlusion | ✓ | ✓ | ✗ | ✗ | ✗ |
| A4 | 可靠性门控 | ✓ | ✓ | ✓ | ✗ | ✗ |
| A5 | + edge protection | ✓ | ✓ | ✓ | ✓ | ✗ |
| A6 | 完整方法 | ✓ | ✓ | ✓ | ✓ | ✓ |

小规模超参数敏感性只在开发片段上进行：

- `α_max ∈ {0.2, 0.4, 0.6}`
- `τ_fb ∈ {0.5, 1.0, 2.0}` px（或按流幅归一化的等价定义）
- 尺度锚收缩强度 `λ_anchor ∈ {0.5, 1, 2}`
- depth 域 vs inverse-depth 域
- RAFT-Small vs Farnebäck（质量/速度权衡）

选定参数后锁定测试片段。不要在最终测试 clip 上逐个调参。

### 7.5 成功标准与负结果处理

预先定义“成功”，避免只挑有利结果：

- 主测试集 flow-aligned temporal error 相对 A0 下降 **≥10%**；
- 累计 log-scale drift 或静态区域 flicker 下降 **≥15%**；
- NYUv2/有 GT 帧上的 AbsRel 绝对退化 **≤0.01**，δ1 绝对退化 **≤0.5 个百分点**；
- Boundary F1 不低于 A0 超过 **1 个百分点**；
- 若 RAFT 后处理很慢，仍须报告端到端 FPS，并给出 Farnebäck/降分辨率 flow 的速度版本。

这些是工程目标，不伪装为已取得结果。如果未达标：

- 完整报告各消融与置信区间，不删除 Part 3。
- 分类解释：flow 失败、动态非刚体、反射、scene cut、深度边界泄漏、尺度锚错误。
- 给出“何时启用融合”的适用域，例如只在静态/慢运动且可靠 mask 比例足够时启用。
- 将有价值的负结果写成结论：时序平滑与瞬时精度/边界之间存在什么权衡。

---

## 8. 评估协议与统计

### 8.1 单帧度量深度

在有效像素集合 `V` 上实现并测试：

- `AbsRel = mean(|D-G|/G)`
- `SqRel = mean((D-G)^2/G)`
- `RMSE = sqrt(mean((D-G)^2))`
- `RMSE-log = sqrt(mean((log D-log G)^2))`
- `δ_i = mean(max(D/G,G/D) < 1.25^i), i=1,2,3`

主表给出图像级宏平均和 bootstrap 95% CI；补充材料可给逐图分布。所有方法共享完全相同的 valid mask、crop 和 resize 后尺寸。

### 8.2 边界质量

- 主指标：Depth Pro 的 scale-invariant Boundary F1。
- 补充：depth-edge precision/recall，允许 1–3 px tolerance；阈值和形态学操作必须固定。
- 选至少 20 个包含桌边、椅腿、人体轮廓、薄物体的 NYUv2/自录帧做局部图。
- 不只报平均值：分别报告高纹理平面、弱纹理边缘、细结构三类。

### 8.3 时序一致性

在前后向一致且非遮挡 mask `M_t` 上：

1. **尺度敏感 temporal error**

\[
E_{temp}=\operatorname{mean}_{x\in M_t}
|\log D_t(x)-\log W(D_{t-1})(x)|.
\]

2. **局部 flicker（scale-invariant）**：从上式的 log difference 中减去该帧中位数，再取绝对均值，区分全局尺度变化与局部闪烁。

3. **相邻尺度比**：

\[
r_t=\operatorname{median}_{x\in M_t}\frac{D_t(x)}{W(D_{t-1})(x)};
\]

绘制累计 `Σ log r_t` 作为漂移曲线。固定相机 clip 额外报告固定背景 ROI 的 median depth CV/标准差。

4. 若视频有真实相机位姿/深度，再报告 pose-warped TAE；没有 GT 时不得把光流 proxy 描述为真实几何误差。

防止评价泄漏：评分用的 flow 默认与融合 flow 固定为同一算法时，额外用另一种 flow 或手工稳定 ROI 做交叉检查，避免方法仅“迎合”某个 flow estimator。

### 8.4 效率

- batch size 1；同一 GPU、相同功耗/precision 设置。
- 10 次 warm-up，100 帧计时，前后 `torch.cuda.synchronize()`。
- 分别报告纯模型 latency 和含 decode/resize/flow/write 的端到端 latency。
- 速度运行 3 次，给均值±标准差；报告 FPS、P50/P95 latency。
- 用 `torch.cuda.reset_peak_memory_stats()` 和 `max_memory_allocated()` 记录峰值显存。
- 同时记录参数量、checkpoint 大小、输入分辨率、FP16/FP32、离线窗口长度。

### 8.5 统计与结果管理

- 所有汇总表由 `metrics_per_sample.csv` 自动生成，不手抄数字。
- 使用 paired bootstrap（按图或按 clip 重采样）计算 ours vs baseline 的差值 95% CI。
- 每个表格单元能追溯到 run ID、config 和 git SHA。
- 失败样例按预定义类别抽取，既展示最好也展示最差，避免只 cherry-pick。

---

## 9. Demo：点云与点击测距

建议做一个 Gradio/Streamlit 应用，一次完成两个展示需求：

- 左：RGB/视频输入。
- 中：固定米制范围的深度及置信度/可靠性 mask。
- 右：可旋转点云；点击 RGB 可显示轴向深度和相机到点的欧氏距离。
- 视频模式可并排切换 Raw 与 Ours，显示当前时序误差、FPS 和尺度曲线。

点云反投影：

\[
X=(u-c_x)Z/f_x,\quad Y=(v-c_y)Z/f_y,\quad Z=D(u,v).
\]

内参优先级：真实相机标定 > 视频元数据可靠焦距 > Depth Pro 预测焦距；主点缺失时假设图像中心并明确标注。点击值使用 5×5 或 7×7 有效像素中位数，避免单像素噪声。两次点击可计算 3D 欧氏距离。

Demo 定量验证：

- 对 0.5、1、2、3、5 m 等至少 20 个实测点报告 MAE、median AE、相对误差。
- 对已知物体尺寸做 10 组两点 3D 测量，报告误差。
- 比较 Raw Depth Pro、Metric VDA、Ours；说明时序融合对读数抖动的改善。
- 明确声明这不是安全级测距系统，镜面/透明/超出训练域时可能严重失败。

---

## 10. 完整实验矩阵

### 10.1 必做主表

1. **Table 1 — Middlebury geometry**：BM vs SGBM；disparity、metric depth、coverage。
2. **Table 2 — NYUv2**：aligned relative DA-V2 与 native metric Depth Pro；全部 7 个深度指标 + Boundary F1。
3. **Table 3 — Video temporal**：frame-wise vs Video DA vs ours；`E_temp`、flicker、drift、Boundary F1、FPS/VRAM。
4. **Table 4 — Part 3 ablation**：A0–A6；主指标、单帧精度、边界、效率。
5. **Table 5 — Demo measurement**：不同距离/材质/方法的测距误差与读数抖动。

### 10.2 必做图

- Fig. 1：整套 pipeline，清楚区分原始模型、flow/reliability、fusion、应用。
- Fig. 2：Middlebury disparity→depth 与误差随距离二次增长图。
- Fig. 3：NYUv2 RGB/GT/DA aligned/Depth Pro/error/boundary zoom。
- Fig. 4：视频 x–t slice 或固定 scanline 时空图，直观看 flicker/拖影。
- Fig. 5：相同 clip 的逐帧尺度漂移与 temporal error 曲线。
- Fig. 6：Part 3 权重、遮挡 mask、Raw/Ours 局部放大。
- Fig. 7：分类失败案例：镜面、玻璃、天空/远景、运动模糊、薄结构。
- Fig. 8：点云与点击测距 Demo 截图。

所有可视化要求：字体/色条统一、同场景固定范围、标明米/相对值、局部放大框对齐、色盲友好；深度图旁必须有 colorbar。

### 10.3 推荐扩展（仅在主线完成后）

- UniDepthV2-S 作为第二个原生 metric 方法，分析焦距/内参处理差异。
- VDA-S vs VDA-B 的 accuracy–latency–VRAM trade-off。
- 真实标定焦距 vs Depth Pro 预测焦距。
- 离线 vs streaming VDA；强调 streaming 精度损失和实时延迟的权衡。
- 低分辨率 RAFT 或 Farnebäck 快速版，形成 accuracy–speed Pareto 图。

---

## 11. 测试与质量门禁

每阶段未通过门禁就不进入大规模实验：

### 数据门禁

- [ ] 下载文件 SHA256、解压场景数、NYUv2 split 数量正确。
- [ ] RGB/depth 方向和尺寸匹配；人工检查 10 个 overlay。
- [ ] 单位测试：Middlebury baseline、焦距、视差偏移和输出米制值数量级合理。

### 模型门禁

- [ ] 每个模型在 1 图/8 帧无 NaN/Inf，深度为正，帧顺序不变。
- [ ] 权重类型与命令匹配：relative/metric、vits/vitb/vitl 不混用。
- [ ] 原始数组保存后重新加载，指标 bitwise/容差内一致。

### 评价门禁

- [ ] synthetic metric 单元测试全部通过。
- [ ] GT mask、crop、resize 在所有方法间完全一致。
- [ ] 随机手算 3 个样本，与自动结果一致。
- [ ] 对齐开关能在输出 metadata 和表头中被明确识别。

### 时序门禁

- [ ] 合成 warp 测试通过；forward/backward 方向无误。
- [ ] scene cut 会重置，遮挡区不会复制上一帧物体。
- [ ] fixed-range 视频与 raw depth 使用同一数据源。
- [ ] 评价 flow 做至少一种独立交叉检查。

### 发布门禁

- [ ] 在干净环境从 README 复现至少一个主表子集和 Demo。
- [ ] `git status` 无数据、checkpoint、个人路径、密钥或大文件。
- [ ] 所有图表由脚本生成；所有引用、代码与模型 license 已核对。
- [ ] ZIP 在另一台机器解压，视频可播放，README/manifest 齐全。

---

## 12. 任务执行顺序与阶段门禁

任务总体上按 **准备工作 → Part 1 → Part 2 → Part 3 → Demo 与最终交付** 推进，但不是简单地把某个 Part 全部做完才允许碰下一个 Part。正确顺序由依赖关系决定：公共数据协议和指标必须最先完成；Part 3 必须等待 Part 2 baseline 冻结；README、实验记录和图表则从产生第一批结果时就持续更新。

### Step 0：建立公共基础

1. 保护并冻结第三方仓库版本，建立混合工程目录。
2. 建立环境，确认 GPU、CUDA 和各模型 smoke test。
3. 下载并校验 Middlebury、NYUv2、模型权重；建立 data manifest。
4. 冻结 split、crop、valid mask、resize、对齐和输出 schema。
5. 完成 geometry、image metric、boundary metric、temporal metric 的单元测试。
6. 录制并冻结自有视频的开发/测试划分。

**进入 Part 1 的门禁：** 数据数量和方向正确；输出 schema 可用；合成指标测试通过；至少一个模型能产生可加载的原始浮点深度。

### Step 1：完成 Part 1A——经典双目几何

1. 实现 Middlebury PFM/calibration loader。
2. 先跑 StereoBM，确认 disparity 定义、1/16 缩放和无效值。
3. 实现 StereoSGBM 与左右一致性检查。
4. 完成 `d → z`、`doffs`、baseline 单位处理。
5. 冻结参数后跑 15 个场景，生成 disparity/depth 指标和可视化。
6. 做 ±0.25/0.5/1/2 px 扰动实验，完成深度误差随距离的分析。

**门禁：** 一个命令可复现全部 15 场景结果；GT/prediction overlay 通过人工检查；理论敏感性曲线与数值实验趋势一致。

### Step 2：完成 Part 1B——逐帧相对深度基线

1. 实现统一 DA-V2-S adapter，保存未经着色的原始相对深度。
2. 在固定图像协议上验证允许的 median/scale-shift alignment。
3. 对全部冻结视频逐帧推理。
4. 实现固定范围可视化、静态 flicker、flow-aligned error 和 scale drift 曲线。
5. 冻结 Part 1 的表格、关键图与失败案例。

**进入 Part 2 的门禁：** relative/aligned/metric 标签不会混淆；同一输入可由统一 evaluator 评价；视频不使用逐帧颜色归一化。

### Step 3：完成 Part 2A——单图度量深度复现

1. 接入 Depth Pro，先跑样例并核对输出单位、尺寸和预测焦距。
2. 在 NYUv2 小子集跑通无 GT alignment 的完整 evaluator。
3. 人工核对 crop、valid mask、resize 和三个样本的手算指标。
4. 冻结配置后跑 654 张测试图。
5. 生成全部 metric depth 指标、Boundary F1、置信区间、效率和分类失败案例。
6. 主线稳定后，才决定是否增加 UniDepthV2 扩展对照。

### Step 4：完成 Part 2B——视频深度复现

1. 接入 relative VDA-S，并与 DA-V2-S 逐帧基线做严格同设置比较。
2. 接入 metric VDA-S，并与 frame-wise Depth Pro 做 metric track 比较。
3. 跑全部冻结视频，生成 temporal error、local flicker、scale drift、FPS 和 VRAM。
4. 对照检查原始浮点结果、固定色标视频和指标使用同一批帧。
5. streaming/Base 仅在主比较完整后作为扩展。

**进入 Part 3 的门禁：** A0 baseline 的配置、原始预测、指标与运行环境全部冻结；Part 3 只能读取这些结果，不能为改善分数改动 baseline 的预处理或评估 mask。

### Step 5：完成 Part 3——从最小改进逐步增加模块

1. 先实现 A1 朴素 EMA，建立“降低闪烁但产生拖影”的弱基线。
2. 用合成数据验证 flow warp 后实现 A2。
3. 依次加入 occlusion、可靠性门控、edge protection、scale anchor，得到 A3–A6。
4. 只在开发视频选择超参数，然后锁定并一次性评估测试视频。
5. 检查 temporal improvement、单帧 metric accuracy、boundary quality 和效率四类权衡。
6. 用独立 flow/固定 ROI 交叉检查评价，整理成功与失败类别。
7. 即使 A6 未胜过 A0，也保留完整消融并解释原因，不回头更改协议。

**Part 3 完成门禁：** A0–A6 表格可自动生成；每个模块有对应可视化证据；主张与置信区间一致；失败案例不是事后随意挑选。

### Step 6：应用、复现与提交

1. 用冻结的最佳模型接入点云和点击测距 Demo。
2. 完成至少 20 个距离测点与 10 组尺寸测量。
3. 从自动结果生成最终表格、pipeline 图、曲线和失败案例图。
4. 在干净环境执行 README 的 Part 1→Part 2→Part 3 最小复现路径。
5. 检查 licenses、引用、公开仓库、原始浮点输出和所有 mandatory clips。
6. 完成 CVPR 报告、arXiv、`depth_results.zip` 和最终 checksum。

这套顺序中，Part 1 提供几何理解和逐帧诊断工具，Part 2 提供可信且冻结的强 baseline，Part 3 才在这些 baseline 上做受控改进。因此最终展示是 Part 1→2→3，核心实验也大体按此顺序；只有公共基础、记录、绘图和写作是贯穿全程的。

---

## 13. 报告写作计划（CVPR 6–8 页）

建议正文控制在 8 页：

1. **Abstract（约 0.25 页）**：任务、方法、数据、最重要的量化发现；末句公开 GitHub URL。
2. **Introduction（约 0.75 页）**：metric scale 与 temporal consistency 的矛盾；列 3 条实际贡献。
3. **Related Work（约 0.8 页）**：必须覆盖作业列出的 10 篇论文，按 stereo / monocular / metric / video 分类，不写成论文摘要堆砌。
4. **Method（约 1.5 页）**：baseline、可靠性 mask、逆深度融合、尺度锚定、复杂度；放总 pipeline。
5. **Experiments（约 3 页）**：协议、Part 1/2、主比较、消融、效率、Demo；表和图优先。
6. **Failure Cases & Limitations（约 0.6 页）**：分类失败、指标局限、flow 依赖、metric 泛化。
7. **Conclusion（约 0.3 页）**：可验证结论和具体未来工作。

贡献表述只能基于最终结果，例如：

- 建立统一、可复现的经典 stereo—单图 metric—视频 depth 评估系统；
- 提出无需训练的可靠性门控时序融合与尺度锚定；
- 通过受控消融分析时序平滑、边界保持、metric accuracy 与效率的权衡；
- 用定量点击测距 Demo 验证实际价值。

若实验没有支持某一点，就删除/弱化该 claim，不用视觉效果代替量化证据。

---

## 14. 交付物清单

### 14.1 GitHub

- [ ] 公共仓库 URL 出现在摘要末尾和 README。
- [ ] README 含安装、数据、权重、推理、评估、消融、Demo 命令。
- [ ] 提供预期目录树、下载链接、checkpoint hash、常见错误和资源需求。
- [ ] 至少放 1 张 pipeline、1 个结果表、1 个短 GIF/视频链接。
- [ ] 引用所有数据集、论文、模型、借用代码；说明各 checkpoint license。
- [ ] 不重新发布 NYUv2/Middlebury 数据、第三方权重或无权发布的素材。

### 14.2 `depth_results.zip`

建议结构：

```text
depth_results/
├── README.txt
├── manifest.csv              # clip、fps、分辨率、method、depth type
├── mandatory_clips/
│   └── <clip>/{rgb_depth_app.mp4, raw_depth.npz, metadata.json}
├── self_recorded/
├── demo/
│   ├── demo_video.mp4
│   ├── launch_instructions.txt
│   └── measurement_results.csv
└── checksums.sha256
```

- [ ] 每个视频 H.264/通用格式可播放，含方法名、固定 colorbar 和单位。
- [ ] mandatory clips 一个不少；教师后续发布新 clip 时更新 manifest。
- [ ] 原始浮点深度若体积过大，可压缩 NPZ 并在 README 说明；不要只交伪彩色视频。

### 14.3 PDF/arXiv/Canvas

- [ ] CVPR camera-ready，6–8 页正文，参考文献不计页数。
- [ ] 所有推荐阅读均正确引用，数字与公开代码输出一致。
- [ ] arXiv 源文件不依赖本地绝对路径，图和 BibTeX 完整。
- [ ] arXiv ID 写在 Canvas 报告首页；摘要末尾 GitHub 链接可访问。
- [ ] 最终 PDF、GitHub tag、ZIP checksum 记录在 release notes。

---

## 15. 风险与回退方案

| 风险 | 预警信号 | 处理/回退 |
|---|---|---|
| 当前节点无 GPU | `nvidia-smi` 失败 | 先完成数据/指标/CPU stereo；切换 GPU 计算节点后跑模型 |
| 第三方依赖冲突 | Depth Pro/UniDepth import 失败 | 分 conda 环境，通过 NPZ schema 连接；UniDepth 降为扩展 |
| 显存不足 | VDA OOM | vits + FP16、降低 max resolution/clip window；不使用 Large |
| NYUv2 协议混乱 | 不同实现得分差异大 | 固化 split/crop/mask，合成单测，逐图对照公开实现 |
| Part 3 拖影 | 遮挡处复制旧物体 | FB consistency、depth/RGB gate、edge mask、scene reset |
| 融合导致尺度漂移 | 累计 log-scale 单调偏离 | 每帧锚定 raw metric、限制修正、低可靠率时禁用 |
| RAFT 过慢 | FPS 远低于 baseline | 半分辨率 flow、隔帧 flow、Farnebäck 速度版本并画 Pareto |
| 主指标不升 | temporal 降但 AbsRel/边界恶化 | 降低 `α_max`、只对不确定区融合；完整报告负结果和适用域 |
| 自录视频无 GT | 无法证明 metric accuracy | 用卷尺/已知尺寸测点；只把 flow 指标称 proxy，不冒充 GT |
| 主线完成前实验仍扩张 | 表格/图反复变化 | 先冻结主模型与协议；优先完整性、消融和复现，停止 P3 扩展 |

---