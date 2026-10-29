# 当前项目状态

## 数据与交付

必需公开数据和权重已齐全：Middlebury Q 分辨率的 15 对训练图、标定与视差 GT；
完整 NYUv2 labeled MAT 和标准 654 图测试划分；四套官方源码及所需预训练权重。
来源、版本和校验值见 `data/manifests/`，无需再补 KITTI/TUM 才能满足必需数据范围。

主要还缺自录视频：建议包含静态段和相机运动段，覆盖近远物体、薄结构、遮挡、
易失败材质，用于检验闪烁、尺度漂移和应用 Demo。若选择 B，还需独立实测标定点；
若选择 F，还需应用事件/距离标注。以 C 为主方向时，这两类数据不是前提。

最终还需完成 CVPR 格式 6–8 页报告、arXiv 上传与 Canvas ID，以及包含全部要求
视频和应用 Demo 的 `depth_results.zip`。

## 主要结论

Part 1 的 Middlebury 15 场景、四组 BM/SGBM 实验已完成；60 份预测的 1,200 项
重载校验通过。Part 2 的两条米制模型均完成 NYUv2 654 图 GPU 推理与评分。
Depth Pro 的精度和边界表现优于 UniDepthV2-S，与团队图像基准的结论一致。
完整结果与命令见 [基准结果](benchmark_results.md)。

在同帧、同分辨率的公开示例视频上，DAV2/VDA 的固定参考尺度相对 warp error
分别为 0.006370 / 0.004228，VDA 约低 33.6%。它支持时序稳定性优势，
没有深度 GT，不能证明米制准确率更高。完整运行与评分包见
`artifacts/relative_video_demo/`。

Part 3 选择 C 为主要探索：完整测试中改善边界 F1，精度也有小幅改善。
C_cuda 是保持公式的加速实现，A 作为视频时序补充。
证据与各方向限制见 [Part 3 结果](part3_results.md)。

回归测试：101 passed。公开说明使用通用依赖和相对路径。
