# 项目操作索引

命令从项目根目录执行。科学协议和门禁见[PLAN](../PLAN.md)，实际完成见[implemented](../implemented.md)，现有结果见[current_results](current_results.md)。

| 内容 | 状态 | 文档 |
|---|---|---|
| 环境、资源与协议 | 已具备 | [Step 0](step0.md) |
| Part 1A：经典双目 | 主四组及误差—覆盖率研究完成 | [Step 1](step1.md) |
| Part 1B：逐帧relative | 当前70帧运行/分析/重算完成 | [Step 2](step2.md) |
| Part 2A：原生米制模型 | 两模型654图与尺度/物体边界分析完成 | [Step 3](step3.md) |
| Part 2B：VDA与同帧比较 | 当前70帧完成；100帧素材仍待确认 | [Step 4](step4.md) |
| Part 3 B：开发集全局尺度 | 三规则完整结果/核验/分析具备，保留米制目标取舍 | [Step B](stepB.md) |
| Part 3 C：固定边缘感知滤波 | 六方法完整诊断/核验/分析具备，保留指标、宽度与纹理代价 | [Step C](stepC.md) |
| 应用、报告与提交 | 待完成 | [Step 6](step6.md) |

B和C独立，不执行B+C。正式训练与长时间运行由用户启动；不能把小型检查或队友独立分数写成本地全量结果。Part 1/2上传GitHub见[分支替换指南](github_upload_part1_part2.md)，不上传数据、视频、权重或outputs。

完整项目（含Direction B/C）更新GitHub分支见[完整上传指南](github_upload_project.md)，包含白名单打包、转移、完整替换、校验与推送命令；不上传视频、数据、权重或运行产物。

动态片段的目标/背景/边界检查见[动态实验](dynamic_clips.md)；与 NYU 主表分别报告。
