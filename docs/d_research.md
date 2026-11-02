# D 多序列研究与消融

## 主要结论

- D_temporal：通过预设精度/边界约束 2/3。
- D：通过预设精度/边界约束 3/3。
- E_half：通过预设精度/边界约束 3/3。
- C：通过预设精度/边界约束 1/3。
- 冻结优化版与原 D 在 3/3 条新测试序列预测完全相同，本轮未获得新增优化收益。
- 没有一种方法在每段的时序、边界和速度上同时最好；不据此宣称超过基础模型论文。

新增四条 TUM 序列：desk 为开发组，rpy、sitting_static、walking_xyz 为冻结测试组，各 64 帧。
另检查四段自录视频与已有 xyz 两段参考片段。既有参考不是新测试集，不参与参数选择。
这些是少量序列，fr1/fr3 内也可能共享物理场景；不报告帧级显著性或跨域最优。

VDA-Small / 518 / FP32 / RGB 长边 448，固定 5 FPS 抽样。原生米制，不拟合测试 GT 尺度。
相机使用 TUM 建议的注册 PNG/ROS 默认约定，深度除以 5000，不重复应用深度校正。
来源：[TUM 下载与序列描述](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)、
[TUM 格式和相机约定](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)。

## 消融定义与参数冻结

- D_temporal：固定历史权重，关闭置信度与全局重置；保留相同的光流、光度和创新门限。
- D_confidence：加入局部 log-depth 残差方差权重，仍关闭全局重置。
- D：原始完整方法，加入全局场景切换重置；默认数值已与旧版逐值对照。
- D_tuned：仅在 desk 开发组选择 alpha 上限、边缘保护与低覆盖重置；随后冻结。
- A、E_half、C 和 RAW 均保留；C 用于检查边界反例。

开发组选择：`D_candidate_2`，参数 `{"alpha_max": 0.7, "minimum_coverage": 0.02}`。
选择约束：AbsRel ≤ RAW+0.01、δ1 ≥ RAW−0.005、边界 F1 ≥ RAW−0.01；合格项中最小 pose-log-error。
selection.json 在任何新测试组推理前保存；未根据测试结果重新选择参数。

## 新测试组结果

### freiburg1_rpy

RAW：AbsRel 0.142064，δ1 85.269%，edge F1 0.298595，pose-log-error 0.058633。

- A：pose 变化 +0.78%（正值为改善），AbsRel 0.142059，edge F1 0.298826；约束通过。
- D_temporal：pose 变化 +1.56%（正值为改善），AbsRel 0.142213，edge F1 0.300108；约束通过。
- D_confidence：pose 变化 +0.94%（正值为改善），AbsRel 0.142079，edge F1 0.298550；约束通过。
- D：pose 变化 +0.95%（正值为改善），AbsRel 0.142092，edge F1 0.298462；约束通过。
- E_half：pose 变化 +0.98%（正值为改善），AbsRel 0.142090，edge F1 0.298789；约束通过。
- C：pose 变化 +0.51%（正值为改善），AbsRel 0.141852，edge F1 0.292487；约束通过。
- D_tuned：pose 变化 +0.95%（正值为改善），AbsRel 0.142092，edge F1 0.298462；约束通过。

### freiburg3_sitting_static

RAW：AbsRel 0.112931，δ1 90.267%，edge F1 0.580940，pose-log-error 0.020325。

- A：pose 变化 +11.11%（正值为改善），AbsRel 0.112916，edge F1 0.581850；约束通过。
- D_temporal：pose 变化 +23.12%（正值为改善），AbsRel 0.112831，edge F1 0.569460；约束未通过。
- D_confidence：pose 变化 +14.27%（正值为改善），AbsRel 0.112751，edge F1 0.573030；约束通过。
- D：pose 变化 +14.27%（正值为改善），AbsRel 0.112751，edge F1 0.573030；约束通过。
- E_half：pose 变化 +11.96%（正值为改善），AbsRel 0.112950，edge F1 0.581901；约束通过。
- C：pose 变化 +3.40%（正值为改善），AbsRel 0.112669，edge F1 0.553568；约束未通过。
- D_tuned：pose 变化 +14.27%（正值为改善），AbsRel 0.112751，edge F1 0.573030；约束通过。

### freiburg3_walking_xyz

RAW：AbsRel 0.180269，δ1 82.418%，edge F1 0.397945，pose-log-error 0.047730。

- A：pose 变化 +3.78%（正值为改善），AbsRel 0.180106，edge F1 0.398451；约束通过。
- D_temporal：pose 变化 +8.70%（正值为改善），AbsRel 0.179575，edge F1 0.395512；约束通过。
- D_confidence：pose 变化 +5.19%（正值为改善），AbsRel 0.179867，edge F1 0.396432；约束通过。
- D：pose 变化 +5.19%（正值为改善），AbsRel 0.179867，edge F1 0.396432；约束通过。
- E_half：pose 变化 +5.66%（正值为改善），AbsRel 0.180101，edge F1 0.398829；约束通过。
- C：pose 变化 +1.00%（正值为改善），AbsRel 0.179942，edge F1 0.378736；约束未通过。
- D_tuned：pose 变化 +5.19%（正值为改善），AbsRel 0.179867，edge F1 0.396432；约束通过。

## 机制、边界与遮挡诊断

- D_temporal：测试组通过约束 2/3；重置次数 0。
- D_confidence：测试组通过约束 3/3；重置次数 0。
- D：测试组通过约束 3/3；重置次数 6。
- D_tuned：测试组通过约束 3/3；重置次数 6。
- E_half：测试组通过约束 3/3；重置次数 18。
- C：测试组通过约束 1/3；重置次数 0。

D 与 D_confidence 预测完全相同的片段：freiburg3_sitting_static, freiburg3_walking_xyz, landscape_static, moving, portrait_static, static_with_people_moving, tum_xyz_early, tum_xyz_late。
固定权重基线为 0.4，而置信度版产生可变权重；比较也包含实际平滑强度差异，不能把全部变化归因于方差排序质量。
若自然片段没有触发重置，不能以分数相同证明重置有效；场景切换与低覆盖的作用另由机制测试验证。
regions.csv 报告 GT 边界两像素带、内部和未被刚性 GT 重投影匹配区域的 AbsRel。
最后一种区域混合了新显露表面、动态物体、视野变化和传感器误差，不是精确运动分割。
另保存 fr3 官方 RGB 内参的探索性评分敏感性检查（camera_sensitivity.csv）；只改变评测投影，不给模型额外相机信息，也不用于参数选择。
光度不匹配区报告相对 RAW 的改动量，自录视频的 RAW 边界保留量也只是结构诊断，不能当作 GT 精度或无拖影证明。

## 自录视频与观感检查

- landscape_static，RAFT proxy 降幅：A 10.66%；D 16.91%；D_tuned 16.91%；E_half 10.65%。[动态区域连续帧](../outputs/d_research/landscape_static/motion_review.jpg)，[对比视频](../outputs/d_research/landscape_static/comparison.mp4)。
- moving，RAFT proxy 降幅：A 7.75%；D 12.44%；D_tuned 12.44%；E_half 8.06%。[动态区域连续帧](../outputs/d_research/moving/motion_review.jpg)，[对比视频](../outputs/d_research/moving/comparison.mp4)。
- portrait_static，RAFT proxy 降幅：A 14.22%；D 16.39%；D_tuned 16.39%；E_half 14.21%。[动态区域连续帧](../outputs/d_research/portrait_static/motion_review.jpg)，[对比视频](../outputs/d_research/portrait_static/comparison.mp4)。
- static_with_people_moving，RAFT proxy 降幅：A 8.33%；D 10.98%；D_tuned 10.98%；E_half 8.13%。[动态区域连续帧](../outputs/d_research/static_with_people_moving/motion_review.jpg)，[对比视频](../outputs/d_research/static_with_people_moving/comparison.mp4)。

移动相机/人物会改变真实 z；光流 proxy 不是纯闪烁或论文 TAE。

## 计时范围

summary.json 中 fusion_seconds 不含光流；postprocess_seconds 加入重新计算光流的时间。
二者均不含 VDA、评测和 IO，使用 1 次预热与 3 次重复。

另外直接测量“读取保存的 RGB → 离线 VDA → CPU 光流/融合 → 写压缩深度”的暖进程管线：

- freiburg3_walking_xyz / RAW：中位吞吐 10.46 FPS。
- freiburg3_walking_xyz / A：中位吞吐 6.09 FPS。
- freiburg3_walking_xyz / D：中位吞吐 5.57 FPS。
- freiburg3_walking_xyz / E_half：中位吞吐 7.06 FPS。
- freiburg3_walking_xyz / D_tuned：中位吞吐 5.32 FPS。
- moving / RAW：中位吞吐 7.96 FPS。
- moving / A：中位吞吐 5.44 FPS。
- moving / D：中位吞吐 5.73 FPS。
- moving / E_half：中位吞吐 6.81 FPS。
- moving / D_tuned：中位吞吐 5.71 FPS。
freiburg3_walking_xyz：E_half 的文件管线吞吐比 A 高 15.8%，仍比 RAW 低 32.6%。
moving：E_half 的文件管线吞吐比 A 高 25.2%，仍比 RAW 低 14.4%。

混合宽高比的初次暖进程计时出现显存占满，手机 RAW 约 0.79 FPS；
释放无用 CUDA 缓存并重测后为 7.96 FPS。异常观察另存，不混入主要方法速度比较。
主计时在片段/方法边界、计时区间之外清理无用缓存。此处是运行内存管理，不是 D/E 的算法加速。

该管线不含初次模型加载、原视频抽帧、显示、告警和评分；VDA 为整段离线推理，不能把吞吐叫作流式实时帧率。

## 复现和证据

`scripts/prepare_d_research_data.py` → `scripts/research_direction_d.py` →
`scripts/benchmark_d_pipeline.py` → `scripts/analyze_d_research.py`。
先运行项目缓存脚本，使用项目既有兼容环境；模型/数据前提同 518 扩展验证。
所有数值、逐帧/逐对表、开发选择与校验见 `docs/evidence/d_research/`。
压缩 NPY 数组无损保存；prediction_locations 明确指向 NPZ 键或经 SHA256 验证的既有冻结运行，避免重复占用空间。
