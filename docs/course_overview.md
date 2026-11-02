# Project 3：作业要求

AIAA 3201 — Introduction to Computer Vision，Spring 2027。
以 [当前五页作业文件](../Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf)
为准；逐项完成度见 [新版要求核对](new_brief_compliance.md)。

## 核心任务

在 Middlebury 学习双目几何，在 NYUv2 评价一个预训练米制模型，在指定视频上比较
逐帧与视频相对深度，再实现一个轻量改进与简单应用。三部分均需完成。

## 数据与模型

- Middlebury Stereo v3 Q 分辨率输入和左视差 GT ZIP：使用 15 对训练图，读取标定，
  运行 BM/SGBM、视差转深度和误差分析。
- NYUv2 labeled MATLAB 文件与 splits.mat：795 图训练/开发、654 图测试。
  读取 filled `depths`，按课程协议评分，不采用 rawDepths 或测试集尺度对齐。
- 官方 DAV2、VDA、UniDepthV2 或 Depth Pro 源码与预训练权重；优先 Small、FP16。
- 指定 `davis_rollercoaster.mp4` 原始帧率前 100 帧：DAV2-Small 逐帧与 VDA-Small
  relative、离线 FP16，保留原始浮点输出。当前官方直链实测只有 70 帧，须确认课程版本。
- 自录视频、TUM、iBims、KITTI 数据和 metric-video 模式均为可选扩展。

## 实验

Part 1 包含双目几何和逐帧单目基线；Part 2 复现米制模型并比较 DAV2/VDA 视频。
Part 3 选择一个可测量问题作轻量改进，提供受控消融与失败分析。
相对视频不能称为米制深度；各方法整段固定色阶、相同色板，改进沿用其基线归一化。

NYU 必报 AbsRel、RMSE、δ1：480×640、crop `[45:471,41:601]`、有效 GT
0.001–10 m、预测裁剪到该范围、654 图等权平均、不做尺度对齐。
双目报告全部有限正值 GT 范围的视差 MAE 和有效预测比例，并说明无效规则。
视频必做定性比较；时序误差、尺度漂移、光流评分和边界精确率/召回率均为可选。
效率报告推理耗时、峰值显存、硬件、模型变体、精度与输入大小。
同一方法及其改进使用相同测试输入、mask、输出尺寸和指标，保留浮点预测。

## 提交

CVPR 模板 PDF，正文 6–8 页，参考文献不计；摘要末尾给出公共
GitHub 链接，覆盖阅读清单，包含方法图、定量/定性结果、消融和分类失败案例。
arXiv 为可选；若上传，改为 camera-ready，并在 Canvas 首页注明 arXiv ID。
公共仓库提供依赖、数据/权重取得步骤、运行与评测命令、代表结果。
提供指定视频并排对照、一个提供图像/clip 的应用输出和三个实际困难案例，
汇总演示成果为 `depth_results.zip`。不向 Canvas 直接提交散装源代码。

当前完成状态见 [项目状态](project_status.md)，数值见 [基准结果](benchmark_results.md)，
探索选择见 [Part 3 结果](part3_results.md)。原始作业 PDF 和阅读论文保留供核对。
