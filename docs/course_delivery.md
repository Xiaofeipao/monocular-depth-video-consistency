# 新版 PDF：非报告交付

报告按要求暂停；现有数据能够完成的实验、媒体和打包已准备。官方素材只有 70 帧，与题面 100 帧的差异仍需课程确认。

## 成果入口

- 本地交付包：[depth_results.zip](../depth_results.zip)，[SHA256](../depth_results.zip.sha256)，[校验记录](../depth_results.zip.validation.json)。包不含学术报告。
- [指定视频 RGB / DAV2 / VDA 对比](../outputs/course_submission/rollercoaster_comparison.mp4)：70 个真实连续帧，原始 24 FPS，FP16 Small，输出 640×360，输入设置 518。每个方法整段固定 P2–P98，颜色不是共同米制尺度。
- [提供图像应用演示](../outputs/course_submission/nyu_application/application_demo.mp4)，[应用截图](figures/course_delivery/nyu_application.png)：同一 NYUv2 图的 RAW/C 深度背景虚化，采用完全相同的对焦深度与模糊强度。另附预测相机参数下的点云文件，不声称独立测距准确性。
- 三个来自 NYUv2 的真实困难案例：[大范围米制误差](figures/course_delivery/difficult_case_1.png)、[边界恢复困难](figures/course_delivery/difficult_case_2.png)、[C 边界退化](figures/course_delivery/difficult_case_3.png)。原始浮点 RAW/C/GT、zoom、指标与解释在包内。
- 四段 self-recorded 的额外应用对照在 `outputs/course_submission/self_recorded/`，各 64 个抽样帧、5 FPS；并非完整原始帧率推理，也没有深度 GT。

## Part 1

15 场景、四组匹配均使用已保存视差重新评分，没有重跑匹配。全部有限正值 GT 下的逐场景宏平均为：

- BM fixed：MAE 1.30384 px，覆盖率 49.34%。
- BM tuned：MAE 2.25491 px，覆盖率 60.56%。
- SGBM fixed：MAE 1.38256 px，覆盖率 63.37%。
- SGBM tuned：MAE 1.68687 px，覆盖率 65.25%。

MAE 仅计有效预测，覆盖率分母为全部有效 GT；保留原匹配/LR 无效规则。不能忽略覆盖率来排名。non-occluded 指标单独保留，参数仍是原固定/开发组设置，没有按新评分重新调参。

## Part 2 / Part 3

NYUv2 654 图原生米制基线与 C 对照保留原课程 crop/mask/预测裁剪和等权平均协议，没有基础模型训练或测试集尺度拟合。指定视频使用独立的 relative Small 权重。

本次 C 组件消融覆盖同样 654 图：完整 C 的 SI F1 为 0.171107；去 RGB 引导为 0.137025，去深度门控为 0.169701；RAW 为 0.159269。完整 C 不是每个指标、每张图都最好，三个失败案例包含实际退化。

完整数值在 [证据目录](evidence/course_submission/)，组件表为 [NYU_C_comparison.csv](evidence/course_submission/NYU_C_comparison.csv)。计时、峰值分配显存、输入、精度与匿名硬件规格均有记录；离线吞吐不是流式实时速度。

## 复现

已有完整基线、权重与数据时：

```powershell
./scripts/run_course_delivery.ps1 -Python python -Archive outputs/my_new_depth_results.zip
```

新环境先按 [基准复现](benchmark_results.md) 生成 `outputs/full_benchmark_20261029/` 所需 NYU/Middlebury 结果。可选自录输出存在时才加入；公开重现不要求取得私人视频。源 ZIP 有独立的仅 numpy/OpenCV 回放脚本，已经实际验证 70 帧 / 24 FPS。

公开仓库同步代码、小型证据和 NYUv2 示例图；自录视频与本地大 ZIP 不随代码公开上传。包内有原始数组、RGB、valid、manifest、配置、环境、计时、视频和逐成员校验。GPU 可用时 113 项测试通过；最终 CPU 复核 112 项通过，1 项 CUDA 专用测试跳过。

## 保留事项

- 不用补帧或改帧率冒充 100 帧。源长度说明随包附上，需要确认教师接受规则或取得正确长版本。
- 学术报告按用户要求暂不制作；arXiv 可选。Canvas 实际提交仍需由课程提交者操作。
