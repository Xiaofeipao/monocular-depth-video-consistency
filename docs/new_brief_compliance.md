# 新版要求完成度

依据本地五页 `Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf`，SHA256 `ea65076917e87febb0c3c67773b9fc6edf304e2821a74266a03f22ef33abc226`。

目前非报告交付见 [成果入口与复现](course_delivery.md)。

- Middlebury：15 对、几何转换、无效区域、四组结果齐全；已补全部有限正值 GT 的 MAE/coverage，非遮挡评分单独保留。
- NYUv2：654 图 filled depths、标准 split、课程 crop、预测裁剪、等权平均与无尺度对齐；两条米制基线及 C 对照可用。
- 指定视频：DAV2/VDA relative Small FP16 已完成同帧输出，原始 24 FPS，固定整段色阶；保存 NPY/RGB/valid/配置/环境/计时/视频。
- Part 3：保留一个可用于正式主线的 C 及完整 654 图组件消融；D/E 等方向与负结果保留为扩展。
- 应用：提供的 NYUv2 图已生成 RAW/C 背景虚化视频、截图与点云；另有四段自录应用对照。
- 三个困难案例：来自提供的 NYUv2 图像，有浮点 GT/预测、完整图、zoom、指标和失败说明。
- 打包：`depth_results.zip` 有逐成员 CRC/SHA256、外部 SHA256 与验证记录；独立 CPU 回放已验证。旧角色打包器已扩展为新版角色。
- 依赖和运行命令、原始预测、公开代表图、小型证据均已整理。公共仓库不包含私人视频或大 ZIP。

尚不能声称严格完成题面的 100 帧：重新下载官方直链仍为 70 帧、24 FPS，SHA256 `7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8`。本次保存所有真实帧，没有补造。该素材矛盾及课程接受规则仍需确认。

学术报告按用户最新要求暂停；arXiv 为可选；Canvas 实际提交不由本地文件自动完成。自录/TUM/iBims/KITTI 数据、光流时序数字与边界指标是可保留扩展，不是新版全部必做项。
