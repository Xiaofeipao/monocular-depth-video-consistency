# AIAA 3201：Step 0、Part 1A 与 Part 2A

本分支保存公共基础、Middlebury BM/SGBM，以及 NYUv2 上 Depth Pro、UniDepthV2-S 两条原生米制基线和四项关键分析。

## 阅读与复现

- [Step 0：环境、上游版本、资源校验和公共测试](docs/step0.md)
- [Part 1A：经典双目调参、评价与结果](docs/step1.md)
- [Part 2A：两模型、四项分析与结果](docs/step3.md)
- [完整精度结果与来源审计](docs/current_results.md)

从本分支根目录运行命令。先按 Step 0 获取固定 commit 的第三方代码，再创建 `aiaa3201` 与 `aiaa3201-unidepth` 环境，准备数据/权重并验证 checksum。数据、权重、第三方源码和原始预测未随本分支发布，下载入口与固定身份保留。

## 状态和协议

Step 0 公开数据基础、Part 1A 四组和 Part 2A 两基线/四分析已有完成产物。视频、Demo、报告和最终提交不因本分支上传而视为完成。

NYUv2 保持官方 654 test、完整 RGB 输入、评价 crop `[45:471,41:601]`、`0.001 < GT < 10 m`，预测仅评分时 clamp 到 `[0.001,10] m`；主表为 `native metric / meter / alignment:none`。外部 calibrated K/f 与测试 GT 驱动的 oracle alignment 仅用于诊断。接受合理浮点容差，不为匹配旧数字修改预测或 evaluator。

## 结果与资源

`outputs/` 只包含小型 JSON/CSV、解析配置及可选少量结果图，不是完整运行目录。缺少原始预测时不能直接复评或运行只读缓存分析；需在本地补齐输入或重新生成。已有结果目录不要覆盖，复现输出另存新目录。

`data/manifests/` 和 `checkpoints/checksums.sha256` 记录来源与资源身份。自动元数据保留原执行时间、路径和版本，可能引用未发布的大文件；它们不表示本次上传重新执行实验。`EXPORT_MANIFEST.json` 区分源文件与导出副本 checksum，只有面向本分支的说明文字经过范围整理。

正式训练和微调由用户启动。模型推理、完整搜索与评价按 Step 文档手动执行；上传步骤不运行这些实验。
