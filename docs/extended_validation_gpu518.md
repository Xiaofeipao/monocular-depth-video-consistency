# 扩展验证：自录视频与有真值 sample

四段自录视频各 64 帧（约 12.6 秒），TUM fr1/xyz 两段不重叠 sample 为 64/54 帧。
所有视频用同一 metric VDA-Small、cuda FP32、input-size 518、RGB 长边 448、5 FPS 名义播放率，
固定 RAW/A/C/D/E_half 参数；TUM 评分使用实际同步时间戳和 GT 位姿。没有训练或测试集调参。

64 帧覆盖多个官方推理窗口，但仍非完整视频或超长视频基准。四段手机视频是同一物理场景，
两段 TUM 也来自同一序列，不应视为六个独立场景。所有颜色固定为 0.1–10 m。

## 自录视频：独立 RAFT 光流代理评分

以下为相对于原始 metric VDA 的 log-warp proxy 降幅，顺序 A / C / D / E_half。
移动相机/人物会改变真实 z，不能把这些降幅等同纯闪烁或真实尺度漂移。

- landscape_static：10.66% / 4.38% / 16.91% / 10.65%。[对比视频](../outputs/extended_video_validation_gpu518/landscape_static/comparison.mp4)。
- moving：7.75% / 3.49% / 12.44% / 8.06%。[对比视频](../outputs/extended_video_validation_gpu518/moving/comparison.mp4)。
- portrait_static：14.22% / 4.87% / 16.39% / 14.21%。[对比视频](../outputs/extended_video_validation_gpu518/portrait_static/comparison.mp4)。
- static_with_people_moving：8.33% / 2.18% / 10.98% / 8.13%。[对比视频](../outputs/extended_video_validation_gpu518/static_with_people_moving/comparison.mp4)。

## TUM sample：深度真值与相机运动补偿

以下 pose-log-error 在共同的 GT 可见表面上补偿相机旋转/平移与 z 变化，不是光流 proxy，
也不是原始 TUM 论文的轨迹 ATE/RPE。边界指标为 log-depth edge F1，不是 Depth Pro SI F1。

### tum_xyz_early，64 帧

原始 VDA：AbsRel 0.129460，δ1 86.589%，pose-log-error 0.033332。

- A：AbsRel 0.129382；δ1 86.620%；pose-log-error 0.032724（降低 1.83%）；edge F1 0.206674；约束通过。
- C：AbsRel 0.129228；δ1 86.618%；pose-log-error 0.033046（降低 0.86%）；edge F1 0.195938；约束未通过。
- D：AbsRel 0.129363；δ1 86.633%；pose-log-error 0.032499（降低 2.50%）；edge F1 0.202633；约束通过。
- E_half：AbsRel 0.129376；δ1 86.616%；pose-log-error 0.032515（降低 2.45%）；edge F1 0.206627；约束通过。

通过精度/边界约束的 pose 指标最优方法：**D**。

### tum_xyz_late，54 帧

原始 VDA：AbsRel 0.214268，δ1 69.226%，pose-log-error 0.030448。

- A：AbsRel 0.213998；δ1 69.392%；pose-log-error 0.029219（降低 4.03%）；edge F1 0.324002；约束通过。
- C：AbsRel 0.214008；δ1 69.206%；pose-log-error 0.030071（降低 1.24%）；edge F1 0.311231；约束未通过。
- D：AbsRel 0.213647；δ1 69.564%；pose-log-error 0.028528（降低 6.30%）；edge F1 0.315415；约束通过。
- E_half：AbsRel 0.213895；δ1 69.501%；pose-log-error 0.028694（降低 5.76%）；edge F1 0.324039；约束通过。

通过精度/边界约束的 pose 指标最优方法：**D**。

原生 VDA 在这两段上仍有米制误差，后处理的距离精度改善幅度很小。
通过相对基线的退化约束不代表绝对精度已经良好；D 在本轮通过，但此前短片段的负结果仍需保留。

## iBims 跨数据集检查

- depth-pro，100 图：AbsRel 0.160301 → 0.160264；masked ratio SI F1 0.171601 → 0.155297；精度 `fp32`。
- unidepth-v2，100 图：AbsRel 0.136771 → 0.136708；masked ratio SI F1 0.091521 → 0.093874；精度 `fp32`。

iBims 为研究作者公开的 MoGe 处理版：JPEG RGB、log-quantized depth，缺原始完整掩码。
边界评分明确扩展为“相邻两 GT 像素均有效”的 masked ratio SI F1；全有效时已验证与官方公式一致，
但不能直接与论文的原始 MAT/unmasked 表格排名。Depth Pro 使用 GPU FP32，
UniDepthV2 使用 CPU FP32；仅在各自模型内部做 RAW/C 配对，不把精度/硬件差异当作方法优势。

## 复现与限制

运行入口：`scripts/run_extended_video_validation.py`、`scripts/download_ibims_core.py --processed-mirror`、
`scripts/run_ibims_refinement_cpu.py --model depth-pro --device cuda --output outputs/ibims_refinement_gpu` 与 CPU `--model unidepth-v2`；
环境和下载/临时缓存先通过 `. ./scripts/project_cache.ps1` 限制在项目目录。
GPU 完整入口：`./scripts/run_extended_gpu_validation.ps1 -Python python`；它按顺序执行，避免多个 GPU 作业争用。
Depth Pro 需使用兼容其官方 NumPy 约束的依赖组合；本地 `.runtime_depthpro` 在 `.runtime` 前。
数值副本、逐对表与校验在 `docs/evidence/extended_validation_gpu518/`；原始 RGB/NPY/valid/flow/视频在 outputs 对应目录。
论文逐项协议与仍缺少的直接对照见 [论文审查](paper_comparison.md)。这些实验不构成 12 篇论文所有原任务的完整复现。
