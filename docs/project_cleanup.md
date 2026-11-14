# 2026-11-14项目清理与发布范围

旧自录输入和结果已退出当前课程流水线。本次检查正式入口、配置、测试及正式运行的父运行/输入声明后，清理了以下无当前依赖的内容：

| 路径 | 文件数 | 原大小 (MiB) |
|---|---:|---:|
| `outputs/part1/video` | 3959 | 10033.79 |
| `data/self_recorded` | 9 | 115.64 |
| `data/video_frames/self_recorded` | 959 | 484.70 |
| `outputs/part1/video_checks` | 1 | 0.00 |
| `outputs/github_upload` | 2 | 0.87 |
| `configs/video/clips.yaml` | 1 | 0.00 |
| `data/manifests/self_recorded_template.csv` | 1 | 0.00 |

另清理项目自有Python缓存、pytest缓存，以及三份过期上传指南。资源/缓存共5081份、11,153,119,610字节（10.39 GiB）；三份旧说明另有63,743字节。旧说明、原视频登记与预检小文件已在维护审计中保存副本。

以下内容保留：

- `data/davis_rollercoaster.mp4`及`data/video_frames/part1b_davis_v1`，当前1B/2B依赖它们。
- Middlebury、NYUv2、对应下载压缩包与来源/权重checksum；已有资源不重复下载。
- Part 1A fixed与coverage study、1B NYUv2及指定片段、Part 2A全部基线/分析、2B VDA/比较。
- Direction B开发/冻结/三规则评价、C v1和C v2全部正式结果、核验及工程历史。
- 第三方仓库、队友副本、旧课程原文/译文和维护历史；不机械删除用户资料或许可来源。
- 有实际导入/测试依赖的公共视频模块与`relative_video.yaml/public_clips.yaml`；它们不是要求继续自录视频的任务入口。

清理前后的27,074份受保护资源/产物文件大小和修改时间一致，正式运行声明没有依赖删除目标。没有改写预测或自动运行元数据。旧执行记录中仍可能保留当时的路径，这是实际历史，不代表路径现在仍存在。

删除清单、逐文件SHA、来源/旧说明快照和保留检查在[维护审计](../outputs/maintenance/project_cleanup_20261114/)。这类删除清单只保存身份和校验，不是大文件备份，不能据此恢复已删除的视频/数组。

当前上传操作统一见[工作站7897上传指南](github_upload_project.md)。范围包括Part 1/2、B、C v1/v2的项目自有代码、配置、测试、环境与中文说明；视频、data、checkpoints、outputs和第三方仓库不上传。旧的局部上传指南已归档，避免混用不含C v2的旧清单。

本轮全项目测试365项通过，14条既有matplotlib/pyparsing警告；发布流程只在临时本地仓库测试，尚未对实际远端提交或推送。
