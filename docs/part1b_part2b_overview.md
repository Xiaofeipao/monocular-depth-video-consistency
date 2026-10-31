# Part 1B与Part 2B：分支阅读入口

[Part 1B：协议与运行](step2.md)；[Part 2B：VDA与逐帧模型比较](step4.md)；[上传范围和操作](github_upload_part1b_part2b.md)。

本次仅发布代码、配置、环境、测试和人工复现说明；不发布视频、RGB、GT、权重、预测、中间数组、运行JSON/CSV、自动报告或其他实验产物。说明中引用的结果文件仅在本地保留。

1B：`scripts/part1/video.sh` → `outputs/part1/video/<ID>/`；2B：`scripts/part2/video.sh` → `outputs/part2/video/<ID>/`。这些是执行时生成的目录，不在本次Git上传包中。2B只读引用完整1B，不自动补跑1B。主环境为aiaa3201。

NYUv2评价为GT辅助aligned_relative；视频为无GT的relative，alignment:none。首次复现先准备自己的输入与权重，再运行1B，完成后按同视频ID运行2B。新ID不在clips.yaml时，先用1B的--video/--type登记。

先按分支已有Step 0恢复上游与主环境；environment.yml依赖本地ml-depth-pro目录。FFmpeg/ffprobe需要位于PATH。已有资源先校验，缺少时再按已有说明获取。
