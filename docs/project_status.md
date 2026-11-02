# 当前项目状态

最新非报告交付见 [成果入口](course_delivery.md)，要求与例外见 [新版要求核对](new_brief_compliance.md)。

## 必需实验与交付

- Middlebury Q 15 对和原始预测已齐全；60 份匹配输出保留，已补全部有限正值 GT 的视差 MAE/coverage，旧 non-occluded 结果单列。
- NYUv2 labeled MAT、标准划分与官方权重在本地；Depth Pro/UniDepthV2 两条 654 图米制基线及 C 对照已完成。课程 crop、filled GT、预测裁剪与无尺度拟合协议一致。
- 指定 rollercoaster 的 DAV2/VDA relative Small FP16 同帧结果已完成，原始 24 FPS，固定整段归一化。官方文件实测仅 70 帧，题面 100 帧差异已在包内说明，仍需课程确认。
- NYUv2 应用演示、三个真实困难案例及四段可选自录应用预览已生成；原始数组、配置、计时和证据保留。
- `depth_results.zip` 已生成并逐成员核验，附 SHA256；独立 CPU 回放已验证。学术报告按用户要求暂停，不包含在包内；arXiv 可选，Canvas 提交为外部步骤。

代码与小型证据用于公共 GitHub；私人视频和大交付 ZIP 留在本地。依赖/软件版本与匿名硬件规格可复现，省略个人路径和 GPU 型号。GPU 可用时 113 passed；最终 CPU 复核 112 passed、1 CUDA 专用测试跳过。

## 研究结论与限制

C 在完整 NYUv2 上提高两条基线的边界 F1，距离精度只小幅变化；iBims/Depth Pro 和若干视频中存在退化，不作为普遍跨域保证。新增组件消融支持 RGB 引导对边界的重要性，但并非所有指标都因完整 C 最优。

D/E 的开发/冻结测试、四段 self-recorded 和 TUM 参考仍保留：D 与 E_half 均通过三条新序列的精度/边界约束；固定权重消融有更低时序分数，但一段边界退化超标；冻结 D 优化未取得新预测收益。视频折中可考虑 E_half，图像主线保留 C。

- [图像与双目基准](benchmark_results.md)
- [Part 3 结果与各方向](part3_results.md)
- [D 多序列消融](d_research.md)与[观感复核](d_research_review.md)
- [论文逐项协议比较](paper_comparison.md)
- [自录检查](self_recorded_review.md)与[518 同帧验证](extended_validation_gpu518.md)

自录/TUM/iBims 作为扩展证据，不替代指定视频；未宣称超过基础模型论文或跨场景最优。
