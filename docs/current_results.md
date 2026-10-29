# 当前结果与来源审计

> 本页保留同伴历史结果；2026-10-29 已完成另外保存的本机全量 GPU 复跑，
> 当前下载状态与新增细化结果见 [完整基准记录](full_benchmark_20261029.md)。

审计日期：2026-10-13（UTC+8）。本页读取当前 `outputs/` 的完整精度 JSON/CSV，正文表格再格式化显示；未修改预测、评估协议或运行元数据。本分支只展示 Step 0、Part 1A 与 Part 2A 的来源证据。

## NYUv2 正式原生米制结果

来源：`outputs/part2/depth_pro_nyuv2/frozen_v1/metrics_summary.json` 和 `outputs/part2/unidepth_v2_nyuv2/frozen_v1/metrics_summary.json` 的 `macro_mean`。两者均为 654 test、完整 RGB 输入、固定 crop、`0.001–10 m`、`alignment:none`、无外部相机信息。

| 指标 | Depth Pro（完整精度） | UniDepthV2-S（完整精度） |
|---|---:|---:|
| AbsRel↓ | 0.10047157333839377 | 0.11454372957333221 |
| SqRel↓ | 0.06400258360254008 | 0.06895209489957993 |
| RMSE↓ (m) | 0.3922168142032252 | 0.39884995782725563 |
| RMSE-log↓ | 0.13533881856042604 | 0.14121805082915398 |
| δ1↑ | 0.915173937355017 | 0.8921797284727733 |
| δ2↑ | 0.9798598292097128 | 0.9807310734949964 |
| δ3↑ | 0.9918777381413865 | 0.9933327987804502 |
| 官方 SI Boundary F1↑ | 0.15927408060093043 | 0.11300739651910406 |

## Middlebury 四组结果

来源：`outputs/part1/middlebury/frozen_v2/metrics_summary.json` 的 `all`；15 场景逐场景宏平均。

| 实验组 | MAE (px) | Bad-2 | Coverage | Depth AbsRel | Depth RMSE (m) | 双向时间/场景 (s) |
|---|---:|---:|---:|---:|---:|---:|
| bm_fixed | 0.9790182810919639 | 0.48746251528646606 | 0.543560039101519 | 0.013339634814117755 | 0.1420388864510264 | 0.015550243419905503 |
| bm_tuned | 1.7314005389377825 | 0.40691314701337505 | 0.6594842901060117 | 0.0238476467201213 | 0.21063200280676336 | 0.009473679183671872 |
| sgbm_fixed | 1.0587535544461757 | 0.347418625607543 | 0.6966939250439883 | 0.012259898983278608 | 0.12824877441986693 | 0.024953576813762385 |
| sgbm_tuned | 1.307015561428136 | 0.3373200307999628 | 0.7168935922221646 | 0.015361069375918145 | 0.14668839580267548 | 0.020397917724524935 |

## 四项分析的来源核验

联合总表 `outputs/part2/key_analysis_v1/per_image_analysis.csv` 有 654 行。按 sample ID、test position 和 MATLAB one-based index 对照当前 split 与两条 baseline 逐图 CSV；每模型 5,232 个值（七个深度指标加 SI Boundary F1），合计 10,464 个值的最大绝对差为 0。这里的一致性描述的是当前文件之间的关系，不是跨设备逐位一致要求。

两条 baseline 的 config SHA256 与 `run_metadata.json` 一致；分析配置与 `invocation_metadata.json` 的 SHA256 一致，且 `config_resolved.yaml` 语义相同。相机轨道均值、尺度中位数与 oracle AbsRel、边缘区域 AbsRel、六组边缘 P/R/F1 和逐图胜负已与逐图表核对。

当前分析使用当前 UniDepthV2-S baseline，不能继续引用较早的 UniDepthV2-S 汇总值。边界表 3,924 行、深度分桶表 4,350 行、场景类型表 32 行，符合当前分析数量。

## 外部标定 K 轨道

当前 `invocation_metadata.json` 记录 UTC `2026-10-13T04:50:08.613642+00:00`（UTC+8 为 2026-10-13 12:50:08），`mode: all`，本次推理与分析均执行，累计输出完整。`camera_ablation/unidepth_external_intrinsics/inference_summary.json` 明确记录 `inferred_this_invocation:654`、`resumed:0`，环境与当前 UniDepth baseline 相同（模型加载耗时独立记录）。这些证据支持外部 K 轨道为当前环境新生成。

654 份 prediction 与 metadata 的 stem 对应当前 split；所有 metadata 都是 `metric/meter/alignment:none`、`external_intrinsics_provided:true`，有效 K 与固定标定矩阵一致。抽查 test positions 0、326、653 的外部 K NPZ，数组为有限正值 `float32 [480,640]`，保存 K 与 metadata 一致。正式 native 主表仍不接收标定 K。

来源：`camera_ablation/summary.json` 的 `track_metrics`，以下 `calibrated` 仅作诊断：

| 相机轨道 | AbsRel（完整精度） | RMSE (m) | δ1 |
|---|---:|---:|---:|
| dp_native | 0.10047157333839377 | 0.3922168142032252 | 0.915173937355017 |
| dp_calibrated | 0.10119966390109128 | 0.39388455019743207 | 0.9131628776225139 |
| ud_native | 0.11454372957333221 | 0.39884995782725563 | 0.8921797284727733 |
| ud_calibrated | 0.09219157011884233 | 0.357879384310006 | 0.9263409073195545 |

## 核验范围与限制

本次进行了读取与小规模格式抽查，没有运行模型、训练、下载、经典参数搜索或全量原始预测复评。Middlebury 的 `reload_validation.json` 原有门禁为 60 份预测、1,200 个指标值、`rtol=1e-6, atol=1e-8`，本次读取其记录，未重新执行该门禁。NYUv2 三样本独立公式审计的原有 `all_passed:true` 记录也未改写。

跨设备微小浮点差异允许存在；应核对数据 checksum、split/crop/range/对齐定义、模型版本、逐图身份与合理容差，不通过修改预测或 evaluator 追平旧数字。计时来自当前产物，虽在相同 GPU 上运行，两模型的精度模式、输入内部处理与独立环境仍不同，不能把模型推理 FPS 当作统一受控效率实验。

视频输入仍暂缓，Part 1B/2B、Demo 与提交未完成。仅有 NYUv2 一个目标域，不能宣称已验证多目标域泛化；Depth Pro 的完整训练数据组成也没有被本次核实。

## 来源 SHA256

以下哈希由本次只读审计计算，用于识别所引用的具体文件；不会替换任何原始运行记录。

| 文件 | SHA256 |
|---|---:|
| `outputs/part2/depth_pro_nyuv2/frozen_v1/metrics_summary.json` | `0755a6501654f10079f4bb90bc251b15bd7c8139974655424145936bb6f705e4` |
| `outputs/part2/unidepth_v2_nyuv2/frozen_v1/metrics_summary.json` | `22a1793a1d9123729830b9bcab699092815551560fe4c39426ccdbc6af6c0f0e` |
| `outputs/part2/key_analysis_v1/camera_ablation/summary.json` | `aa88cc25008ec89276db0e2793b42f67060a5ccaf9c4e787fcb2d3134be04cef` |
| `outputs/part2/key_analysis_v1/boundary/summary.json` | `f5a3eca80fe122cede39b8fa4b509fe8f92f1194edbf565bdadbe0fa58bd6c09` |
| `outputs/part2/key_analysis_v1/absolute_scale/summary.json` | `923544f0f0e765696b446c4173439c903ad19c48548bc3d7d500f3342a744f9f` |
| `outputs/part2/key_analysis_v1/cross_domain/summary.json` | `3faadecc71e2d7784b1b10807c410f471236fb174fb055bbef10b7d25804438f` |
| `outputs/part2/key_analysis_v1/per_image_analysis.csv` | `e3091ca2c3329caa4e55b9b28f2e4ffff159d241214fe3240a60b56b2bc54c44` |
| `outputs/part2/key_analysis_v1/invocation_metadata.json` | `41c325df12455f849d9d5f03c0da5e22ba14701693d2eb82049ddd51523e8718` |
| `outputs/part2/key_analysis_v1/camera_ablation/unidepth_external_intrinsics/inference_summary.json` | `09694de5c20f5c0af6e32b99e2408f1074b3f4a92e3b994b9edddf8a260f35fb` |
| `outputs/part1/middlebury/frozen_v2/metrics_summary.json` | `0e266b026fb6843486f3be6c3636f7597e47da2d1fabb3f070c19c50249b1199` |

配置 SHA256：

| 运行 | 配置 SHA256 |
|---|---:|
| middlebury | `eafdc83ec9f47b8333cf1a9f46cea5b460aac6075f576bc7150c11080d6bca0f` |
| depth_pro | `8fa799bf80d6f70051a664b69c4733666c911cf4535d1213ffba7a20f72cc595` |
| unidepth_v2 | `03cb323ef00f2ba88ef9661f285c4bce660f7fcdf2fae9851ddd81e80c26ee38` |
| external_K | `64a7f57dc9816db6a45981755b303c61bbbb00711eb6bcc3b1f336b78266f94b` |
