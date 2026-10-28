# Part 2 配置

`depth_pro_nyuv2.yaml` 已冻结为 `frozen_v1`，用于 Depth Pro 在 NYUv2 654 张标准测试图上的原生米制度量深度评估。该协议明确禁止测试时尺度/偏移对齐，也不向模型提供 NYUv2 GT 焦距。

直接运行命令与已有输出保护说明见 [Step 3](../../docs/step3.md)。流水线入口为：

```bash
./scripts/part2/run_depth_pro_nyuv2.sh
```

正式输出位于 `outputs/part2/depth_pro_nyuv2/frozen_v1/`。它包含未裁剪的 `float32` 原始深度、逐图 metadata、七项 metric-depth 指标、Depth Pro 官方 `SI_boundary_F1`、10,000 次图像级 bootstrap 95% CI、场景类别统计、失败案例和可视化。

`unidepth_v2_nyuv2.yaml` 使用与 Depth Pro 完全相同的 NYUv2 split、crop、有效深度范围、指标、bootstrap、失败阈值和五联图布局，冻结 UniDepthV2-S 的第二条原生米制基线。它不接收 NYUv2/GT 内参，而是保存模型预测的 3×3 `intrinsics_px`；运行时使用独立环境 `aiaa3201-unidepth`。正式输出位于 `outputs/part2/unidepth_v2_nyuv2/frozen_v1/`。

`key_analysis.yaml` 冻结两条 baseline 的四项联合分析：相机焦距/内参诊断、按作业 §6.2 的边界质量、无 GT 对齐的绝对尺度分解，以及单一 NYUv2 目标域范围内的迁移/场景稳健性。外部标定相机参数和 oracle median-scale 只允许作为诊断值，主结果仍为无额外相机输入、`alignment:none` 的原生米制输出。本次明确不增加第二目标数据域，因此配置禁止把域内分类统计表述成多目标域泛化结论。入口为：

```bash
./scripts/part2/run_key_analysis.sh --mode all
./scripts/part2/run_key_analysis.sh --mode analyze
```

正式分析输出位于 `outputs/part2/key_analysis_v1/`。

`video_depth_anything.yaml` 仍为候选配置。Part 2B 依赖自录视频和 mandatory clips，按用户要求暂缓。

当前指标与分析来源核验见 [当前结果审计](../../docs/current_results.md)。`all` 会运行外部 K 推理，`analyze` 会重写分析产物；当前结果已存在，重新运行时先指定新的输出目录。
