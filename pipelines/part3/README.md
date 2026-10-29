# Part 3：时序改进、消融与应用

本部分回答：可靠性门控能否在保持尺度和边界的同时改善时序稳定性？深度能否用于实际测距？

## 运行入口

- `run_temporal_fusion.py`：读取 Part 2 冻结米制预测，输出时序改进结果。
- `run_ablation.py`：自动运行 A0–A6 消融，并输出时序评分。
- `run_demo.py`：RGB/深度/测距展示、点云导出、可选交互点击测距。
- `compare_directions.py`：统一比较方向 A–F，保留每个方法的原始输出、指标和视频。

从仓库根目录运行：

```powershell
python -m pipelines.part3.run_temporal_fusion --run outputs/part2/metric_static --output outputs/part3/static_A6
python -m pipelines.part3.run_ablation --run outputs/part2/metric_static --output outputs/part3/static_ablations --static-roi 20 80 20 120
python -m pipelines.part3.run_demo --run outputs/part3/static_A6 --intrinsics configs/intrinsics.example.json --output outputs/part3/demo --interactive
```

内参和 ROI 必须按自己的数据修改。测距内参对应保存后 RGB 的分辨率。

## 算法实现在哪

- [src/temporal.py](../../src/temporal.py)：核心融合算法、可靠性权重、边缘保护、尺度锚定和历史重置。
- [src/flow.py](../../src/flow.py)：Farneback/RAFT、反向采样和前后向一致性。
- [src/refine.py](../../src/refine.py)：逐帧融合编排、诊断图保存。
- [src/ablate.py](../../src/ablate.py)：消融矩阵和光流缓存。
- [src/demo.py](../../src/demo.py)：点云/点击测距应用。
- [src/geometry.py](../../src/geometry.py)：应用共用的反投影、PLY 和测距几何。

消融依次为：A0 原始结果、A1 朴素 EMA、A2 光流固定权重、A3 遮挡检测、
A4 可靠性门控、A5 边缘保护、A6 尺度锚定及重置。

只读取 Part 2 的原始米制视频预测；相对逆深度结果不能直接输入当前融合入口。
本部分输出放 `outputs/part3/`。应用必须用真实测点评估，合成示例不作为真实效果结论。

## 已保留的其他方向

- **A 光流融合**：`src/temporal.py`、`src/flow.py`；原有 A0–A6 代码保留。
- **B 外部尺度标定**：`src/exploration/calibration.py`。用独立标定帧的至少三个已测轴向距离，
  稳健拟合一个固定尺度；不接受测试帧 ID，不用测试 GT 做逐帧对齐。
- **C 边缘保留滤波**：`src/exploration/edge_aware.py`。RGB 与深度联合引导的逆深度双边滤波，
  保留无效区，避免跨颜色/深度边界平滑；不保证能恢复模型已经丢失的薄结构。
- **D 不确定性融合**：`src/exploration/uncertainty.py`。根据当前/历史 log-depth 方差加权，
  使用光流、遮挡与外观检查；默认方差来自局部残差，是启发式 proxy，不是校准过的模型置信度。
  类接口也接受正值 log-depth 方差图，尚未接入实际 UniDepthV2 不确定性输出。
- **E 后处理加速**：`src/exploration/efficient.py`。半分辨率光流、根据帧间变化切换全/半分辨率，
  正确缩放 flow 向量，保持全分辨率深度。只测后处理加速；没有声称已做 ONNX/量化或主模型加速。
- **F 深度应用**：`src/exploration/application.py`。ROI 近距离分位数 + 滞回/连续帧确认预警，
  对照无滞回预警，报告 precision/recall/F1/accuracy。F 不改变深度图；需检查预警延迟和漏报。
- **B+C+D 组合**：独立标定后做边缘滤波和不确定性融合，与单方向分别报告。

所有方向都通过 `src/exploration/compare.py` 编排。未提供独立测点时，B 和含 B 的组合明确跳过，
不会借用评估 GT 补标定数据。示例配置见 `configs/calibration.example.json`，其中数值必须换成实测值。

```powershell
# 已有真实 Part 2 米制视频结果时：
python -m pipelines.part3.compare_directions --run outputs/part2/metric_static --manifest data/video_test_manifest.json --calibration configs/calibration.example.json --output outputs/part3/directions_real

# 没有外部标定时：B 和 B+C+D 自动跳过，其余方向正常比较
python -m pipelines.part3.compare_directions --run outputs/part2/metric_static --output outputs/part3/directions_no_calibration

# 不需要模型/GPU的合成诊断，输出路径必须是新目录：
python -m scripts.compare_part3_synthetic --output outputs/part3/directions_synthetic
```

合成实验固定五种情况：静态像素噪声、整体尺度偏差、运动遮挡、阈值附近抖动、切镜头；
标定帧与测试帧分开。每种场景使用同一批原始深度、RGB、GT、有效 mask 和评分光流。
`comparison.csv/json` 分别列出精度、边界、时序、后处理速度与预警指标，给出各指标最优者，
以及满足精度/边界退化约束的时序最优者，不宣布一个跨任务通用的“最好”。

计时是一轮预热后多轮完整 CPU 后处理（含所需光流及预警计算），排除主模型、解码、磁盘写入和评分。
F 的额外确认帧可能提高稳定性但推迟报警；不能只看 F1 而忽略 recall。
预警评分要求 ROI 每帧都有完整有效 GT；对稀疏 KITTI 或包含 crop 外区域的 ROI，
保留预警输出但明确跳过应用评分，避免用不同像素集合的分位数作比较。
合成结果只能验证算法行为。后续已完成真实 CUDA 推理和短 TUM RGB-D 对照，
见 [GPU 记录](../../docs/part3_results.md)；图像全量基准已完成，主探索选择 C；自录视频仍需补齐。

## 保留原版的优化实验

显式传入 `--methods` 可比较 `A_conservative`（历史权重上限 0.2）、
`C_fast`（3×3 邻域）、`D_guarded`（权重上限 0.25、log-depth 更新上限 0.01、
深度边缘保护）、`E_local`（8×8 分块变化触发全分辨率光流）、
`F_coverage`（ROI 有效比例至少 50%，缺测清空连续确认计数）。
这些预设是工程探索，不是独立验证集调出的最优参数；原版默认配置继续保留。
支持位姿的 RGB-D manifest 现在同时报告 GT 定义可见像素上的位姿时序误差，
避免只依赖同一种光流做融合和评分。B 不允许重复地标充当多个独立测点。

`C_cuda` 保留原 C 的 5×5 FP32 权重公式，在 CUDA 上并行计算，保持原分辨率
和无效区。与 `C_fast` 的缩小邻域不同；必须显式选择且需要 CUDA PyTorch。
CPU 原版仍为默认。串行固定线程复现入口为 `scripts/benchmark_part3_cuda.py`，
它先验证孔洞、边界和薄结构的数值一致性，再计入数据传输测量处理速度。

当前数值、方向选择与负结果见
[Part 3 结果](../../docs/part3_results.md)。
