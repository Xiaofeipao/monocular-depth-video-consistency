# Part 1B与Part 2B：指定片段实施计划

## 共同任务

课程依据[新版PDF §5.1、§5.2、§6.1–6.2](Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf)。指定`data/davis_rollercoaster.mp4`、前100帧、原始帧率；视频仅作定性比较，无需相机位姿或时序GT。原片实际70帧、24 FPS，不能重复/插帧补到100，需课程确认素材长度。

1B与2B分别运行和保存。当前1B在已有70帧上的逐帧基线及尺度/边界/时间分析完成；2B指定片段70帧独立VDA已由用户运行、真实分析和重算通过，同帧配对比较及双方raw重算也已完成，不能用旧自录实验替代指定片段要求。

## Part 1B：当前主线

`configs/part1/dav2_video.yaml` → `scripts/part1/run_dav2_video.sh` → `pipelines/part1/dav2_video.py`。只调用DA-V2-S，独立逐帧、FP16、官方预处理。输入画布640×360，实际网络518×924，原生恢复360×640。

连续原帧缓存于`data/video_frames/part1b_davis_v1/`；输出`outputs/part1/dav2_video/rollercoaster_v1/`保存run/protocol、70份raw NPY、固定色标视频、曲线/CSV、三个连续局部窗口、原始重算与2B交接。relative/arbitrary/alignment:none，无视频GT，无时间平滑；整段P1/P99固定TURBO显示。

数值时序/边缘proxy是可选案例定位，不是GT准确率。解读时区分相机/物体运动、遮挡与输出变化；没有证据不强行声称全画面严重闪烁。现有70帧真实运行及重算通过，100帧素材门禁未释放。完整命令/结果见[Step 2](docs/step2.md)。

## 保留的NYUv2扩展

DA-V2-NYUv2完整654张对齐诊断及原配置、入口、预测、评价保持不变。测试GT仅在评价时拟合逆深度scale/shift，必须标aligned_relative；不当作视频相对深度的米制标定，更不替代Part 2A的native metric主表。

## Part 2B：独立VDA实验与另行比较

`configs/part2/vda_video.yaml`与`scripts/part2/run_vda_video.sh`独立运行官方VDA-Small relative、offline、FP16，不做streaming/metric。RGB缓存存在时校验后只读复用，不存在时创建连续原始帧缓存；不要求已有DA深度结果。VDA原生序列保存为整段NPZ和逐帧NPY，输出`outputs/part2/vda_video/rollercoaster_v1/`。

VDA分析调用与1B相同的measure及指标函数，保存同坐标/对应后MAE/S、RMSE/S、中位数/范围/边缘比例、边缘变化与光流覆盖；同样提供固定色标RGB/relative/红色边缘视频、四子图、自动高变化候选和三个相同连续局部窗口。尺度约定只作无GT诊断，固定裁剪框只控制展示。官方窗口衔接及内部尾部补齐保留，不把内部补齐算成课程100帧。

VDA运行和核验完成后，单独compare读取1B的`analysis/handoff_2b.json`、双方raw及分析，核对同一RGB/源帧号/PTS/画布/网络尺寸/精度/规则；另存`outputs/part2/video_comparison/rollercoaster_v1/`，不自动运行或改写1B/2B。比较有共同帧对表、差值/降低率、效率、同帧视频及相同局部窗口；保留负结果、N/A和逐项共同数量。

同一TURBO，每模型允许自己的整段固定归一化；不逐帧归一化，不把颜色当共同米制距离。主要比较时间平滑、边缘稳定与运动伪影；flow/数值漂移可选，不设为必做前置。VDA是否改善及其代价必须等真实运行后判断。

[Step 4](docs/step4.md)分为VDA独立任务与正式1B/2B比较两部分，提供完整运行/重分析/复核命令。当前VDA独立70帧已完成真实运行、局部观察和raw核验：主体结构连贯，遮挡与细结构仍有局限；窗口衔接相关的中位数变化不等于真实尺度漂移。第二部分已核对69个共同帧对和正负案例：对应后MAE/S下降18.85%、RMSE/S下降2.96%，边缘均值近乎持平略差；耗时和显存成本增加，不称全面改善或真实米制漂移降低。旧自录2B及专用工作流已清理，`outputs/part1/video/`与自录源视频只作归档，不再提供自录实验入口。当前70帧指定片段比较完成；教师素材确认、应用Demo、报告和ZIP继续单列门禁。
