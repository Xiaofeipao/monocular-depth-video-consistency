# Project 3：作业要求

AIAA 3201 — Introduction to Computer Vision，Spring 2027。

## 核心任务

从单张 RGB 或普通单目视频估计稠密深度，恢复有意义的米制尺度，保持视频预测稳定，
并展示点云、测距、深度效果或障碍预警等应用。三部分均需完成。

## 数据与模型

- Middlebury Stereo v3 Q 分辨率输入和左视差 GT ZIP：使用 15 对训练图，读取标定，
  运行 BM/SGBM、视差转深度和误差分析。
- NYUv2 labeled MATLAB 文件与标准 654 图测试划分：遵循官方米制工具的 crop、
  mask、深度上限和对齐规则；无需下载 raw collection。
- 官方 DAV2、VDA、UniDepthV2 或 Depth Pro 源码与预训练权重；优先 Small、FP16。
- 使用自录视频检验静态闪烁、运动尺度漂移和应用效果，无需额外公开视频数据集。

## 实验

Part 1 包含双目几何和逐帧单目基线；Part 2 复现米制模型并比较 DAV2/VDA 视频。
Part 3 选择一个可测量问题作轻量改进，提供受控消融与失败分析。
相对深度的评测对齐必须注明，不能称为原生米制深度。

必报 AbsRel、SqRel、RMSE、RMSE-log、δ1/δ2/δ3、边界质量、有效非遮挡像素的
光流时序误差、逐帧中位尺度漂移，以及 FPS、延迟、峰值显存、模型大小、输入分辨率。
比较使用相同帧集合、数据划分、预处理与可视化固定色标，保存浮点原始预测。

## 提交

CVPR camera-ready 格式 PDF，正文 6–8 页，参考文献不计；摘要末尾给出公共
GitHub 链接，覆盖阅读清单，包含方法图、定量/定性结果、消融和分类失败案例。
按课程要求上传 arXiv，并在 Canvas 首页注明 arXiv ID。
公共仓库提供依赖、数据/权重取得步骤、运行与评测命令、代表结果。
提交所有要求片段的处理视频及可运行或录制的应用 Demo，汇总为 depth_results.zip。

当前完成状态见 [项目状态](project_status.md)，数值见 [基准结果](benchmark_results.md)，
探索选择见 [Part 3 结果](part3_results.md)。原始作业 PDF 和阅读论文保留供核对。
