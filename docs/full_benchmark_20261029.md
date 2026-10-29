# 完整数据下载、GPU 复现与细化优化

先执行 `git pull --ff-only origin main`：没有新提交，基线为 `66ec02f`。
本轮按 Spring 2027 作业下载完整数据和官方权重，运行预训练推理与评估。
没有训练或微调基础模型；这与作业“不从零训练基础模型”的要求一致。

## 数据和官方模型

- Middlebury 两份 Q 分辨率 ZIP、15 对训练数据及标定/GT 已下载、解压、校验。
- NYUv2 labeled MAT 完整下载，2,972,037,809 字节，SHA256 与冻结记录一致；
  官方 `testNdxs` 654 图已全部用于本机评测。未下载 raw collection。
- 四个官方源码版本已固定在 `third_party/`；DAV2 Small、VDA Small、metric VDA
  Small、UniDepthV2 Small、Depth Pro 和 RAFT Small 的权重均已齐全并核对哈希。
- 来源、版本、字节数和哈希见 `data/manifests/full_model_assets.json` 和
  `data/manifests/spring2027_download_validation.json`。大文件留在本地。

## 完整复跑结果

Middlebury 15 场景 × 四组 BM/SGBM 固定/调参实验已完成；保存了 60 份预测，
1,200 项重载指标校验通过，准确率与同伴历史结果一致。经典双目运行于 CPU。

本机 GPU 记作 `CUDA_GPU_A`；两条米制基线均不做测试集尺度对齐，不提供 GT 内参：

- Depth Pro：654 图，AbsRel **0.100473264**，RMSE **0.392219569 m**，
  δ1 **0.915185154**，官方 SI Boundary F1 **0.159268547**。
- UniDepthV2-S：654 图，AbsRel **0.114541365**，RMSE **0.398820607 m**，
  δ1 **0.892364681**，官方 SI Boundary F1 **0.113009609**。

模型运行时来源为本次保存预测的逐图元数据：Depth Pro 约 0.6301s/图，峰值分配
4,000,329,728 字节；UniDepthV2 约 0.05420s/图，峰值分配 535,040,000 字节。
这些是各自官方输入处理、精度和实现下的实测值，不构成统一输入架构的效率排名。
没有用差异化评测或修改预测来追平历史结果；跨运行时有微小浮点差异。

## 根据结果采取的优化

保留上轮已经冻结的 C_cuda 5×5 FP32 参数，对两条基线的全部 654 图作一次对照，
未在测试集搜索参数。完整 RGB 上细化后，在相同 crop/mask/.001–10m 设置评分。

- Depth Pro + C_cuda：AbsRel **0.100366985**、RMSE **0.391542824 m**、
  δ1 **0.915454328**、SI Boundary F1 **0.171106615**。
- UniDepthV2-S + C_cuda：AbsRel **0.114427372**、RMSE **0.398034681 m**、
  δ1 **0.892600532**、SI Boundary F1 **0.122560834**。

两者的 AbsRel 均小幅改善，官方边界 F1 分别相对提高约 7.4% 和 8.5%；
没有据此声称统计显著或跨域最优。图像评分不支持时序改善结论。
建议把 C_cuda 作为可选图像细化支线；正式原生米制基线保留，禁止把细化结果混入旧主表。
SI Boundary F1 与早期视频 log-depth edge F1 不同，独立记录约束和结果。

复跑还修复了两处实际问题：

1. 官方 SI boundary 评测现在可直接定位源码文件，避免 UniDepth 的完整评分因
   缺少 Depth Pro 包导入路径而中断；评测公式没有改动，CPU 评分无需加载模型。
2. 可用 `visualization.max_panels=32` 导出审计位置和最差 AbsRel 代表图，避免
   每次绘制 654 张大面板。七项指标、边界指标和失败分类仍覆盖全部图像。
   冻结配置默认行为保留，仅本次运行配置和新运行器启用数量上限。

## 复现和保存位置

```powershell
python scripts/download_nyuv2_parallel.py
./scripts/run_full_image_benchmarks.ps1 -Python python -Output outputs/new_full_benchmark
```

运行器要求先取得来源清单中的源码/权重并配置各模型依赖；官方 Depth Pro 的 NumPy
约束与 UniDepth 不同，可使用隔离环境或项目内 `.runtime_depthpro` 与 `.runtime`。
脚本不会改写已有 Conda 环境；它支持复用已存在的原始预测。
`scripts/evaluate_nyuv2_refinement.py --save-predictions` 可额外保存细化深度 NPZ。

本次完整本地输出：`outputs/full_benchmark_20261029/`，包括原始预测、逐图指标、
配置、计时/环境、可视化与原始/细化对照。公开的数值证据位于
`docs/evidence/full_benchmark_20261029/`，环境身份按既定规则泛化。

回归测试：101 passed。源码、完整数据和权重齐全不代表完成自录视频、正式报告、
arXiv 或最终 `depth_results.zip`；这些仍是最终交付缺项。

## 同伴要求的视频复现包

已另行补齐 [relative_video_demo](../artifacts/relative_video_demo/README.md)：
两份完整浮点预测运行目录、保存 RGB/valid、配置/环境、计时/视频、可执行评分脚本、
31 对评分表、共同 mask、flow 与参考帧设置，以及 ZIP/分片 SHA256。
重新评分精确复现原有两个均值。中性名称与个人地址无关，上游视频原名仅用于来源核对。
