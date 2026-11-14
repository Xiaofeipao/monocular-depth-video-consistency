# 项目操作索引

命令从项目根目录执行。科学协议和门禁见[PLAN](../PLAN.md)，实际完成见[implemented](../implemented.md)，现有结果见[current_results](current_results.md)。

| 内容 | 状态 | 文档 |
|---|---|---|
| 环境、资源与协议 | 已具备 | [Step 0](step0.md) |
| Part 1A：经典双目 | 两组fixed及误差—覆盖率研究完成 | [Step 1](step1.md) |
| Part 1B：逐帧relative | 当前70帧运行/分析/重算完成 | [Step 2](step2.md) |
| Part 2A：原生米制模型 | 两模型654图与尺度/物体边界分析完成 | [Step 3](step3.md) |
| Part 2B：VDA与同帧比较 | 当前70帧完成；100帧素材仍待确认 | [Step 4](step4.md) |
| Part 3 B：开发集全局尺度 | 三规则完整结果/核验/分析具备，保留米制目标取舍 | [Step B](stepB.md) |
| Part 3 C：固定边缘感知滤波 | 六方法完整诊断/核验/分析具备，保留指标、宽度与纹理代价 | [Step C](stepC.md) |
| C v2：受控边界收窄 | 完整开发/冻结/654图七方法及四份重放报告通过；比RAW更锐的目标未达成 | [Step C v2](stepC_v2.md) |
| 应用、报告与提交 | 待完成 | [Step 6](step6.md) |

B和C独立，不执行B+C。正式训练与长时间运行由用户启动；不能把小型检查或队友独立分数写成本地全量结果。完整项目（含B、C v1/v2）发布见[工作站7897上传指南](github_upload_project.md)，不上传视频、数据、权重或outputs。

发布流程在工作站完成：白名单导出、独立克隆、完整替换和SHA校验，再通过7897推送，无需搬到本地电脑。

当前清理范围与保留理由见[项目清理记录](project_cleanup.md)。
