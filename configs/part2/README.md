# Part 2 配置

`depth_pro_nyuv2.yaml` 已冻结为 `frozen_v1`，用于 Depth Pro 在 NYUv2 654 张标准测试图上的原生米制度量深度评估。该协议明确禁止测试时尺度/偏移对齐，也不向模型提供 NYUv2 GT 焦距。

直接运行命令与已有输出保护说明见 [Step 3](../../docs/step3.md)。流水线入口为：

```bash
./scripts/part2/run_depth_pro_nyuv2.sh
```

正式输出位于 `outputs/part2/depth_pro_nyuv2/frozen_v1/`。它包含未裁剪的 `float32` 原始深度、逐图 metadata、七项 metric-depth 指标、Depth Pro 官方 `SI_boundary_F1`、10,000 次图像级 bootstrap 95% CI、失败案例和可视化。

`unidepth_v2_nyuv2.yaml` 使用与 Depth Pro 完全相同的 NYUv2 split、crop、有效深度范围、指标、bootstrap、失败阈值和五联图布局，冻结 UniDepthV2-S 的第二条原生米制基线。它不接收 NYUv2/GT 内参，而是保存模型预测的 3×3 `intrinsics_px`；运行时使用独立环境 `aiaa3201-unidepth`。正式输出位于 `outputs/part2/unidepth_v2_nyuv2/frozen_v1/`。

`key_analysis.yaml`只绑定两条完整原生基线，分析绝对尺度与物体边界：逐图尺度比、距离分桶、测试GT辅助的oracle尺度诊断、SI Boundary F1、1/2/3 px深度边缘P/R/F1、边缘/非边缘AbsRel与实际案例。主表保持`alignment:none`；oracle使用测试GT，只能诊断。无需模型、额外预测或权重依赖。

```bash
./scripts/part2/run_key_analysis.sh \
  --config configs/part2/key_analysis.yaml \
  --output-dir outputs/part2/key_analysis
```

分析在CPU上执行，使用`aiaa3201-unidepth`。入口核验数据、基线配置、654张身份、七项native指标与输入SHA，拒绝已有输出目录，并如实记录running/complete/failed。此入口的已有记录保持真实。当前结果来源为`outputs/part2/key_analysis_v1/absolute_scale/`与`boundary/`，解释见[Step 3](../../docs/step3.md)。

`scale_boundary.yaml`配置独立的完整静态误差分析，输出`outputs/part2/scale_boundary_v1/`。与`key_analysis.yaml`互不覆盖，保留两条原生米制核心工作。它从现有预测计算raw/clipped尺度、超范围长尾、距离分桶、log误差分解、GT辅助诊断、三阈值/三容差边缘、物体标注支持召回、边缘定位、区域贡献、GT共同剖面与局部图像；不加载模型。

```bash
./scripts/part2/analyze_scale_boundary.sh --config configs/part2/scale_boundary.yaml \
  --output-dir outputs/part2/scale_boundary_v1
./scripts/part2/analyze_scale_boundary.sh --config configs/part2/scale_boundary.yaml \
  --output-dir outputs/part2/scale_boundary_v1 --mode verify
```

本轮已完成完整654张和独立数值重放。运行时使用`aiaa3201-unidepth`、CPU；run拒绝已有目录，独立复现应选新目录。verify按合理浮点容差重放，身份/整数覆盖严格一致。区间为scene成簇bootstrap且点估计仍为逐图宏平均；三种诊断轨道、GT剖面选择和N/A规则均固定在配置中。具体结果与限制见[Step 3](../../docs/step3.md)。

`vda_video.yaml`是当前课程2B配置：指定davis_rollercoaster、Small relative、offline、FP16、原FPS/前100真实帧上限、640画布和整段固定P1/P99 TURBO。`scripts/part2/run_vda_video.sh run`独立运行VDA及同坐标/光流/边缘等诊断，不依赖DA预测；另一个compare命令只读1B/2B原生结果，核对同帧/PTS/尺寸/精度/规则并另存比较。当前70帧VDA独立及配对比较均完成并通过raw核验，保留收益/退化和实际成本；完整命令见[Step 4](../../docs/step4.md)。

旧自录配置`video_depth_anything.yaml`及其专用工作流已清理；当前视频配置仅使用上述`vda_video.yaml`。公共配置回归核对新版1B/2B的同源、原FPS/前100帧上限、640画布、FP16与相同诊断/显示规则。

当前指标与分析来源核验见[当前结果审计](../../docs/current_results.md)。已有原生预测与自动生成元数据保持真实，分析复现使用新的输出目录。
