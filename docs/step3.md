# Step 3 / Part 2A：Depth Pro 与 UniDepthV2-S

## 状态与统一协议

Step 3 已有完成门禁记录。当前两条 baseline 和四项分析的逐图来源已于 2026-10-13 只读核对，见 [当前结果审计](current_results.md)。本次整理没有重新推理或全量评估。所有表格从当前 JSON/CSV 的完整精度数值生成，再按八位小数显示。

- 使用 NYUv2 labeled MAT 的标准 654 test（`testNdxs`，MATLAB one-based index）；
- 模型输入完整 480×640 RGB；预测按 `bilinear_align_corners_false` 恢复到 GT 尺寸后，评估时才使用 `[45:471, 41:601]` crop；
- GT 有效条件 `0.001 < depth < 10 m`，计算指标时预测裁剪到 `[0.001,10] m`；
- 主结果 `depth_type:metric`、`unit:meter`、`alignment:none`，不输入标定焦距/K，不使用测试 GT 做尺度/偏移对齐；
- 逐图宏平均，固定 seed 的 10,000 次图像级 Bootstrap 95% CI；
- 官方 `SI_boundary_F1(t_min=1.05,t_max=1.25,N=10)`；
- 保存 crop 与指标裁剪前的完整 `float32` 米制深度。

crop 是参考 evaluator 的固定边界区域，不能按当前分数调整。微小跨设备浮点差异可接受，核验以协议和合理数值容差为准，不为追平旧数字改变预测或 evaluator。

## 直接运行命令

以下是后续复现说明，本次未执行。命令从项目根目录运行，主脚本使用 `aiaa3201`，UniDepthV2 与联合分析脚本使用 `aiaa3201-unidepth`。当前正式输出已存在，示例使用新的目录。

三个固定审计样本：

```bash
./scripts/part2/run_depth_pro_nyuv2.sh --positions 0 326 653 \
  --output-dir outputs/part2/depth_pro_nyuv2/audit3_reproduce_v1
./scripts/part2/run_unidepth_v2_nyuv2.sh --positions 0 326 653 \
  --output-dir outputs/part2/unidepth_v2_nyuv2/audit3_reproduce_v1
```

完整 654 张：

```bash
./scripts/part2/run_depth_pro_nyuv2.sh \
  --output-dir outputs/part2/depth_pro_nyuv2/reproduce_v1
./scripts/part2/run_unidepth_v2_nyuv2.sh \
  --output-dir outputs/part2/unidepth_v2_nyuv2/reproduce_v1
```

中断后可对相同配置和目录加 `--resume`；不能把已有完整目录当作一次新推理。`--mode evaluate` 只读取已有预测，不加载模型，但会重写评估产物和运行记录；保留原记录时先复制整个 run 到新的目录，再评估该副本。

## 两条正式 baseline


### Depth Pro

| 指标 | 宏平均 | Bootstrap 95% CI |
|---|---:|---:|
| AbsRel↓ | 0.10047157 | [0.09606078, 0.10510840] |
| SqRel↓ | 0.06400258 | [0.05631914, 0.07288968] |
| RMSE↓ (m) | 0.39221681 | [0.37050663, 0.41521821] |
| RMSE-log↓ | 0.13533882 | [0.12874200, 0.14215905] |
| δ1↑ | 0.91517394 | [0.90420593, 0.92580502] |
| δ2↑ | 0.97985983 | [0.97534362, 0.98377750] |
| δ3↑ | 0.99187774 | [0.98949774, 0.99402079] |
| 官方 SI Boundary F1↑ | 0.15927408 | [0.15182152, 0.16669272] |

### UniDepthV2-S

| 指标 | 宏平均 | Bootstrap 95% CI |
|---|---:|---:|
| AbsRel↓ | 0.11454373 | [0.10960142, 0.11970636] |
| SqRel↓ | 0.06895209 | [0.06200653, 0.07647232] |
| RMSE↓ (m) | 0.39884996 | [0.37819465, 0.42053339] |
| RMSE-log↓ | 0.14121805 | [0.13538792, 0.14733374] |
| δ1↑ | 0.89217973 | [0.87977248, 0.90408387] |
| δ2↑ | 0.98073107 | [0.97734413, 0.98402083] |
| δ3↑ | 0.99333280 | [0.99139743, 0.99506872] |
| 官方 SI Boundary F1↑ | 0.11300740 | [0.10723487, 0.11884602] |

Depth Pro 使用模型预测的 `focallength_px`，均值 519.36266404 px，范围 505.33520508–536.53881836 px。

UniDepthV2-S 使用 `lpiccinelli/unidepth-v2-vits14`，revision 固定为 `038c238f06c87b6c2f5b3749fd51fbf442b1f218`；正式轨道保存预测的完整 3×3 K。均值为 `fx=544.76085098`、`fy=545.84481749`、`cx=322.28608503`、`cy=246.69364533 px`。

当前主表中 Depth Pro 的 AbsRel、SqRel、RMSE、RMSE-log、δ1 和 SI Boundary F1 更好，UniDepthV2-S 的 δ2、δ3 略高；这些平均值的方向不等于每个差异都有统计支持，见下文配对分析。

### 当前运行效率

数值来自各 baseline 的 `metrics_summary.json.efficiency` 与 `inference_summary.json`；模型加载时间单独记录。两条运行记录中的 GPU 都为 NVIDIA RTX A6000，PyTorch/精度模式分别为 `2.1.1+cu121/float16` 与 `2.7.1+cu126/official_float16_autocast`。内部输入处理和环境有差异，此处报告各自实际耗时，不称为统一设置下的公平速度排名。


| 模型 | 平均 (s/图) | 中位数 (s) | P95 (s) | 模型 FPS | 峰值 GPU bytes | 端到端 FPS | 模型加载 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Depth Pro | 0.29722507 | 0.29860653 | 0.30497521 | 3.36445373 | 3973226496 | 3.04906982 | 9.99864572 |
| UniDepthV2-S | 0.01561590 | 0.01557973 | 0.01582386 | 64.03728774 | 508515328 | 18.00893720 | 0.39910521 |

### 失败阈值与 baseline 产物

失败条件为满足任一项：`AbsRel>0.25`、`δ1<0.60`、`SI Boundary F1<0.01`。当前 Depth Pro 有 49 张失败（17/30/18 张分别触发三个阈值），UniDepthV2-S 有 77 张（26/36/39）；允许重叠。

正式目录：

```text
outputs/part2/depth_pro_nyuv2/frozen_v1/
outputs/part2/unidepth_v2_nyuv2/frozen_v1/
```

每条 baseline 包含 654 份 `predictions/*.npz`、654 份 `metadata/*.json`、654 张 `visualization/test_*.png`、纯失败案例 contact sheet、`metrics_per_image.csv`、`metrics_summary.json`、`metrics_by_scene_type.csv`、`failure_cases.json` 和 `three_sample_metric_audit.json`。固定 positions 0、326、653 正常参与正式指标；独立 NumPy 公式审计并不排除这些样本。

## 四项关键分析

配置 `configs/part2/key_analysis.yaml` 指向当前两条 `frozen_v1` baseline，输出 `outputs/part2/key_analysis_v1/`。`native` 为正式轨道；`calibrated` 与 `oracle_scale_aligned` 为诊断轨道。

### 复现与输入依赖

在两条完整 baseline 已存在的前提下，从新的分析目录运行：

```bash
./scripts/part2/run_key_analysis.sh --mode all \
  --output-dir outputs/part2/key_analysis_reproduce_v1
```

`all` 生成外部 K 预测并完成分析，不重新生成 baseline。若要分析前面新生成的 `reproduce_v1` baseline，复制分析配置到新文件并修改 `baseline_outputs` 指向它们，通过 `--config` 显式选择；保留冻结配置与现有元数据。`--output-dir` 控制实际写入位置。

只有同一分析目录的外部 K 预测完整时才可用 `--mode analyze`；它会重写统计、图表与分析调用记录。外部 K 中断时，先对原配置和同一目录运行 `--mode infer-external --resume`，再运行 `--mode analyze`。当前产物实际记载 `mode:all`、新推理 654 张、`resumed:0`；早期的分阶段执行属于历史记录。

### 1. 相机焦距/内参作用

NYUv2 参考 K 来自 `UniDepth/unidepth/datasets/nyuv2.py`，约为 `fx=518.8579`、`fy=519.4696`、`cx=325.5824`、`cy=253.7362 px`；计算时使用配置中的完整精度，经过实现的 float32 转换。Depth Pro 标量参考焦距为 `sqrt(fx×fy)`。


| 模型与相机轨道 | AbsRel↓ | RMSE↓ (m) | δ1↑ |
|---|---:|---:|---:|
| Depth Pro predicted f（正式） | 0.10047157 | 0.39221681 | 0.91517394 |
| Depth Pro calibrated f（诊断） | 0.10119966 | 0.39388455 | 0.91316288 |
| UniDepthV2-S predicted K（正式） | 0.11454373 | 0.39884996 | 0.89217973 |
| UniDepthV2-S calibrated K（诊断） | 0.09219157 | 0.35787938 | 0.92634091 |

Depth Pro 预测焦距平均绝对相对误差为 0.679020%；标定焦距的 AbsRel 差值为 0.00072809，95% CI [0.00024646,0.00120441]，没有改善。

UniDepthV2-S 几何平均焦距误差为 14.261786%，主点误差 7.80019919 px。标定 K 的 AbsRel 差值为 -0.02235216，95% CI [-0.02536869,-0.01938668]。相机输入对误差有明显影响；该受控变化同时改变完整 K，不能只归因于焦距。

Depth Pro calibrated f 由官方推理式精确重缩放得到：

```text
depth_calibrated = depth_native × focal_calibrated / focal_predicted
```

当前相机分析记录 raw depth 未命中官方内部 clamp 边界，闭式关系未因饱和失效。UniDepth calibrated K 使用额外相机信息重新推理；本次只读核对其新生成记录，未重新运行。外部相机轨道不能替代正式主结果。

### 2. 深度边界质量

官方 SI Boundary F1 为主指标；补充使用深度比阈值 1.05（`log_threshold=log(1.05)`）的 edge P/R/F1，1/2/3 px Chebyshev 容差。来源为 `boundary/boundary_summary.csv`：


| 方法 | 容差 (px) | Precision↑ | Recall↑ | F1↑ |
|---|---:|---:|---:|---:|
| depth_pro | 1 | 0.69124720 | 0.47709903 | 0.54959906 |
| depth_pro | 2 | 0.78242165 | 0.55577171 | 0.63408085 |
| depth_pro | 3 | 0.82929986 | 0.60590332 | 0.68434805 |
| unidepth_v2 | 1 | 0.73855053 | 0.39286283 | 0.49884949 |
| unidepth_v2 | 2 | 0.84204579 | 0.46111234 | 0.58000211 |
| unidepth_v2 | 3 | 0.88411893 | 0.50387307 | 0.62506820 |

| 模型 | GT 边缘 3 px 邻域 AbsRel | 非边缘 AbsRel |
|---|---:|---:|
| depth_pro | 0.13383808 | 0.09581914 |
| unidepth_v2 | 0.15495697 | 0.10904332 |

UniDepthV2-S precision 更高而 recall 更低；Depth Pro 在三个容差下 F1 都更高，配对 F1 差值 CI 均不跨 0。20 张八联案例由指标自动选择：7 个 Depth Pro 优势、7 个 UniDepthV2-S 优势、6 个两者都差；本次没有重新做人工图像验收。

### 3. 绝对尺度准确性

实现先按评估范围裁剪 native prediction，在 valid 像素上计算每图 `s_i=median(clipped_prediction_i/GT_i)`。尺度分布来源 `absolute_scale/summary.json`：


| 模型 | 尺度比中位数 | P05–P95 | 低于 0.9 | 0.9–1.1 | 高于 1.1 | 平均有符号偏差 |
|---|---:|---:|---:|---:|---:|---:|
| depth_pro | 0.95805952 | 0.82894664–1.10450978 | 22.1713% | 72.1713% | 5.6575% | -0.03510133 |
| unidepth_v2 | 1.04742581 | 0.86824602–1.20817842 | 7.4924% | 67.4312% | 25.0765% | 0.04953966 |

Depth Pro 整体略低估，UniDepthV2-S 略高估。分桶 AbsRel 为逐图有该桶有效像素的样本宏平均，不是把全部桶像素直接汇总。来源 `absolute_scale/depth_bin_summary.csv`：


| 方法 | GT 距离桶 (m) | 图像数 | AbsRel↓ |
|---|---:|---:|---:|
| depth_pro | 0.001–1.0 | 254 | 0.17263034 |
| depth_pro | 1.0–2.0 | 641 | 0.11151014 |
| depth_pro | 2.0–4.0 | 645 | 0.10467632 |
| depth_pro | 4.0–6.0 | 415 | 0.13599288 |
| depth_pro | 6.0–10.0 | 220 | 0.18437289 |
| unidepth_v2 | 0.001–1.0 | 254 | 0.32732494 |
| unidepth_v2 | 1.0–2.0 | 641 | 0.15664731 |
| unidepth_v2 | 2.0–4.0 | 645 | 0.10266586 |
| unidepth_v2 | 4.0–6.0 | 415 | 0.13110974 |
| unidepth_v2 | 6.0–10.0 | 220 | 0.18922909 |

oracle 诊断用每图各自的 `s_i` 将原始预测除以它，再按相同范围裁剪并重算指标：

```text
s_i = median(clipped_prediction_i / GT_i)  # valid pixels
prediction_i_oracle = raw_prediction_i / s_i
```

它只调整一个乘法尺度，但直接使用同一测试图 GT，属于事后误差归因；不得进入正式 benchmark 或被称为模型改进。

| 模型 | native AbsRel | oracle-scale AbsRel（诊断） |
|---|---:|---:|
| depth_pro | 0.10047157 | 0.05573348 |
| unidepth_v2 | 0.11454373 | 0.06038854 |

### 4. 单目标域范围与场景稳健性

只有 NYUv2 一个目标域，当前只能报告向 NYUv2 的零样本迁移和域内场景稳健性，不能宣称验证多目标域泛化。两模型均未在本项目中于 NYUv2 微调；UniDepth checkpoint 配置列出 ETH3D、Waymo 训练与 IBims 验证，Depth Pro 完整训练组成未被本次核实，不能据此断言彻底排除了 NYUv2 训练重叠。

逐图原生未对齐 AbsRel 比较：Depth Pro 在 373 张更低，UniDepthV2-S 在 281 张更低，无相等；计数不编码胜出幅度。以下为当前 `cross_domain/summary.json` 中 Depth Pro − UniDepthV2-S 的配对均值与 CI：


| 指标 | 配对均值差 | 95% CI |
|---|---:|---:|
| abs_rel | -0.01407216 | [-0.02008910,-0.00793192] |
| sq_rel | -0.00494951 | [-0.01249870,0.00335237] |
| rmse | -0.00663314 | [-0.02241372,0.00961915] |
| rmse_log | -0.00587923 | [-0.01162648,0.00003411] |
| delta1 | 0.02299421 | [0.00925377,0.03716898] |
| delta2 | -0.00087124 | [-0.00507303,0.00272140] |
| delta3 | -0.00145506 | [-0.00344845,0.00039129] |
| si_boundary_f1 | 0.04626668 | [0.04210028,0.05051707] |

AbsRel、δ1、SI Boundary F1 的 CI 支持 Depth Pro 优势；SqRel、RMSE、RMSE-log、δ2、δ3 的 CI 跨 0。类别排名只使用 `n>=5`，`bookstore`、`bathroom` 对两模型都较困难。

## 关键分析产物

- 根目录：`config_resolved.yaml` 保存协议；`run_metadata.json` 与 `invocation_metadata.json` 区分语义、本次调用和累计完整性；`per_image_analysis.csv` 为 654 张联合逐图表。
- `camera_ablation/`：四条相机轨道指标和图表；`unidepth_external_intrinsics/` 含 654 份外部 K 诊断预测、metadata 与本次推理摘要。
- `absolute_scale/`：尺度分布、4,350 行距离分桶记录、分桶汇总和 oracle 诊断。
- `boundary/`：3,924 条逐图/模型/容差记录、汇总、20 张案例与 contact sheet。
- `cross_domain/`：16 类场景×2 模型统计、配对比较和单目标域范围声明。

正式主表只使用 `native`。门禁与尚缺的视频、Demo、报告和提交工作见 [本分支说明](../README.md)。

