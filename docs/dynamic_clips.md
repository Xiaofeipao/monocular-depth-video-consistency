# 动态片段补充：目标、背景与边界的独立检查

## 任务与当前状态

本节补充同伴 Part 2B 同帧视频对照和 Part 3 C 的适用范围，不替换指定 rollercoaster
或 NYUv2 主表。课程共享的 bmx-trees 80 帧、tennis 70 帧及对应二值目标 mask 均已完成
独立 GPU 推理、分区评分与无模型回放核验。结果来自本次本地运行，不是同伴机器上的新运行。
未改同伴主线参数，未使用 mask 指导预测或细化，没有根据本片段成绩选择参数。

## 输入与冻结协议

- 原始顺序全部 150 帧，432×240，不抽样；RGB/mask 逐一对应，标签 0/255。
- mask 表示被标注的目标，不是逐像素运动真值或深度 GT。BMX 目标仅占 0.36%–3.61%，
  Tennis 为 7.35%–14.02%，全图平均不能代替目标上的表现。
- DAV2-Small、VDA-Small 使用 relative 权重及 FP16，同 RGB、同保存尺寸。
  请求 input_size=518，官方宽高比调整后的短边为 504；VDA 离线处理。
- Depth Pro 为 RGB-only FP16 原生米制；C/C_no_RGB/C_no_depth_gate 用同一份原始预测，
  固定 5×5、sigma_rgb=.10、sigma_log=.18、sigma_space=1.5、FP32。
- 原始 FPS 未提供。VDA API 传入占位值 1，该上游函数仅原样返回该值，不依赖它采样或预测。
  预览固定 10 FPS，仅为展示速度，不是原始 FPS，也不换算每秒闪烁。
- 相对模型各自固定整段 P99−P1 用于评分；C 各版本共用未修改 Depth Pro 的范围。
  彩色视频固定整段 P2–P98、TURBO；C 与 RAW 共用色阶。相对模型颜色不代表共同米制尺度。

## 评分、区域与核验

分别用 Farneback 与独立 RAFT-Small 做反向采样，前后向一致性阈值 1 px，排除越界。
所有方法使用共同有限预测域，不按某方法的分数选帧。每个区域至少 8 像素才评分，
空项保留缺失，不填 0。当前数据所有 79/69 帧对均可评价。

评分为对应后的绝对预测差 / 固定整段跨度，按帧对等权平均。四个区域分别是全图、
当前与回采上一帧 mask 都为前景的目标、双方都为背景的区域，以及当前 mask ±2 px 带。
目标/背景要求回采标签近乎纯净（>.999/<.001），边界带不加这种标签一致限制，避免把边缘全部删除。
保存双向 flow、FB 残差和共同域，允许读取原始数组重新评分。

轮廓支持 proxy 为 mask 轮廓附近是否存在预测梯度：跨度归一化、sigma=.8 Gaussian、
Sobel 梯度阈值 .02、2 px 匹配容差。它不是深度边界准确率；mask 轮廓不一定都有深度跳变，
投影、阴影及标注粒度也会影响该值，不能与 NYU 官方 SI F1 混用。

## Part 2B 可补充的观察

以 RAFT 对应后的归一化 MAE 为例，正百分比表示相对 DAV2 的降低：

- BMX：VDA 全图降低 32.06%，目标 25.77%，背景 33.21%，边界带 4.66%。
- Tennis：全图降低 53.43%，目标 15.21%，背景 64.66%，边界带 31.51%。

Farneback 下两段目标降低 30.78%/8.28%，全图 32.77%/48.01%，方向一致但幅度不同。
这说明本次 VDA 的较低整体变化主要不能直接外推为同等幅度的目标改善。
它仍是相对变化 proxy；整段动态深度真实变化、遮挡和光流误差都可能参与残差，
不能据此声称 VDA 距离更准确或恢复了真实尺度。

## Part 3 C 可补充的观察

相对同一 Depth Pro RAW，RAFT 下 C 的全图 proxy 仅降低 BMX 0.17%、Tennis 0.18%；
目标分别降低 0.26%/0.63%，边界带 0.89%/0.14%，没有大的时序收益。
Farneback 下 BMX 目标反而升高约 0.033%，显示很小的收益会随后端改变符号，不能声称稳定改善。

轮廓支持 proxy：BMX RAW 0.49460 → C 0.43697，下降约 5.76 个百分点；
Tennis 0.23461 → 0.24239，上升约 0.78 个百分点。两个组件移除版本在某些变化指标上
更低，但不是完整 C 在所有目标上最优的证据，也不能据此修改已冻结参数。

抽查首/中/末帧：BMX 中骑车者与自行车在 RAW 已有较弱的深度分离，C 未恢复已丢失的细结构；
Tennis 中人物主体可辨，但 RAW/C 的差异较小。此观察与 proxy 一起报告，不凭静态截图证明时序改善。
该补充支持同伴现有结论：SI 指标改善不等于全面锐化；轻量滤波不能保证动态目标边界收益。

两段只有两个序列，帧对相互相关，不做帧级显著性声明。它们与 NYU 的输入分辨率不同，
不能把跨实验差异全部归因于动态场景。没有视频深度 GT，所以这也不是有真值的跨域精度验证。

## 运行指令与产物

已完成目录是 `outputs/course_dynamic_peer_v1/`；复现请使用新目录，从仓库根目录执行。
模型依赖与权重先按项目说明准备，缓存放在项目内。

```powershell
python -m pipelines.part3.dynamic_masks run --data data/course_provided/dynamic_clips --output outputs/dynamic_reproduce_v1
python -m pipelines.part3.dynamic_masks verify --output outputs/dynamic_reproduce_v1 --data data/course_provided/dynamic_clips
```

`check` 检查输入与本地权重路径，不加载模型、不创建输出；`predict` 仅预测；`run` 为预测+分析。
`analyze` 只允许分析尚未存在 summary.json 的预测目录，拒绝重写完成结果；`verify` 只读重算。
配置来自 configs/part3/dynamic_masks.yaml，冻结科学参数须保持一致；--data/--assets-root 只覆盖本地位置。
核验脚本只读，不重跑深度模型或光流：复核输入与 1216 份运行文件 SHA，
重采样并重算全部 7104 条分区结果，已通过；汇总的 96 组均值也独立核对通过。

运行目录包括 900 份 NPY、双后端 flow/mask、输入哈希、模型与权重身份、配置、计时、
逐对 CSV、轮廓 CSV、固定色阶视频、首/中/末帧 RGB/RAW/C 对照。C 为后处理时间，
不能与含基础模型的延迟混称系统加速；一轮预热后单次测量，不声称严格速度排名。

小型证据在 `docs/evidence/dynamic_masks/`。1216 文件清单的路径相对于完整运行目录，
不是指全部文件都位于公开小证据目录。原始数组与 flow 位于本地运行目录。

## 与主实验的关系

适合追加到 Step 4 的“动态目标分区检查”，并从 Step C 链接为独立视频适用范围补充。
保留其现有 NYU 六方法主表，不用本次无 GT 视频 proxy 替换 SI/米制结果。
本次只做 Depth Pro 的 C/两个消融，不冒称覆盖同伴两模型六方法的所有组合。
运行入口已接入本分支原有 RGB 模型 API、src/refinement 和 RAFT 评价类；桥接层显式完成 BGR→RGB。
Farneback 保留本实验 21 窗口/3 次迭代，不改通用评价类的 15 窗口设置。
本节补充已按同伴分支结构接入，完成全量重新推理/分析与只读核验；不替换原有 NYU 主表。

## 分支适配验证

基于 JiongxuChen 的 c0a81ac 代码快照，没有更改其原有 Part 1/2/B/C 算法。
新增桥接的颜色、米制返回值、反向采样、FB 方向与双向 flow 六项测试全部通过。
两次运行中 VDA/Depth Pro/C 的 750 份数组逐值一致；DAV2 的 150 份存在差异。
原入口先用 float64 归一化，分支入口显式 float32，已检查到输入张量差异；不将结果强行逐位对齐。
本节百分比来自分支适配器重新运行后的分数，不混用旧 DAV2 分数。
原始预测差异记录和预处理检查随小证据保存；光流重算也独立保存，不混用两版 flow。

权重和官方仓库需自行准备。--assets-root 可指向已有资源根目录，支持 third_party/下的官方仓库，
也兼容同伴 README 的根目录仓库布局；四个检查点包括 torchvision RAFT-Small 本地权重。
无需为了复现复制其他课程项目数据；不默认联网下载。可在根目录运行：

```bash
bash scripts/part3/dynamic_masks.sh check --output outputs/check_unused
bash scripts/part3/dynamic_masks.sh predict --output outputs/dynamic_new
bash scripts/part3/dynamic_masks.sh analyze --output outputs/dynamic_new
bash scripts/part3/dynamic_masks.sh verify --output outputs/dynamic_new
```

本地已有结果只读验证无需 GPU。新预测/RAFT/C_cuda 需要配置好的 CUDA 环境。

实际推理阶段源码归档于小证据目录的 executed_pipeline.py，其 SHA 与 protocol.json 对应。
当前命令在此基础上增加了冻结 YAML 校验及分阶段入口，未改变预测和评分公式。
