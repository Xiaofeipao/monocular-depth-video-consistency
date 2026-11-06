# 公共实现

| 目录 | 职责 |
|---|---|
| `data/` | NYUv2 HDF5/split读取、稳定样本身份与输出schema |
| `models/` | 官方DA-V2、VDA、Depth Pro、UniDepth适配器，保持原生输出 |
| `stereo/` | Middlebury读取、毫米转米、BM/SGBM、LR、全GT评价与选参绑定 |
| `metrics/` | 七项深度指标、SI/其他边缘指标与时序诊断 |
| `analysis/` | 静态尺度/边界与经典双目误差—覆盖率分析 |
| `evaluation/` | 当前课程1B/2B共同输入、独立和同帧分析/重算 |
| `video/` | 帧提取、官方预处理、光流、运行与报告公共依赖 |
| `visualization/` | 经典双目、米制图像、尺度/边界与视频展示 |
| `refinement/` | Part 3 B/C，见下文 |
| `part3_studies/` | 三种全局尺度估计、两个固定经典对照和边界/纹理诊断；与refinement算子按职责配合 |

`refinement/global_scale.py`只从完整开发图的log比例拟合一个全局尺度，随后对测试预测应用；不逐测试图对齐。

`refinement/edge_aware.py`是main固定5×5 RGB/log-depth联合引导逆深度滤波的CPU/CUDA移植，保留无效位置；`ablation.py`提供同设置的CPU组件移除与统一后端选择。它们不读取GT、相机或另一模型深度，不训练。

来源commit、原始及LF源码SHA见`refinement/ORIGIN.json`，MIT许可见`refinement/MAIN_LICENSE.txt`。B/C三规则与六方法正式结果、核验报告及分析已生成；测试/同公式滤波重放的范围仍分别说明。方法、命令和状态见[Step B](../docs/stepB.md)、[Step C](../docs/stepC.md)。
