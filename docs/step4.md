# Step 4 / Part 2B：按视频比较DA-V2与Video Depth Anything

## 任务与当前状态

依据课程PDF §5.2、§6.2（本地保留：`../AIAA3201_Project3_Monocular_Metric_Depth_Video_Consistency_Spring2027 (1).pdf`），使用官方**VDA-Small、relative、离线batch、FP16**处理自录视频，并与逐帧DA-V2在相同输入分辨率、可视化范围和评估帧下比较。不训练或微调，不额外要求公开视频、Base/Large、metric与streaming都做。

**以视频为单位：每段一条命令，自动完成该类型适用的全部比较与核验。** 本部分使用独立入口和`outputs/part2/video/<ID>/`目录。运行前必须先完成该视频的Step 2；只读引用`outputs/part1/video/<ID>/`的DA-V2、RAFT、诊断及其RGB清单，不写入1B目录。缺少完整1B输入时直接报错，由用户先运行Step 2。

当前实现与合成验证通过；六段原片、三个权重和官方代码已有。**正式全片与新协议结果待用户运行**，本页没有预填模型改善结论。NYUv2和静态研究无需重跑。主环境为`aiaa3201`。

## 三类视频的比较重点

| 类型 | 主要比较 | 补充与限制 |
|---|---|---|
| `fixed_static`：固定无人、场景静止 | 全图逐像素MAE/S、RMSE/S；全图背景时间std/S；短时跳动与缓慢变化分开 | 同光流MAE/RMSE与中位数；不画矩形ROI |
| `fixed_dynamic`：固定有人或其他真实运动 | 同光流MAE/S、RMSE/S；人物/遮挡和背景上下文变化 | 逐像素MAE/S、RMSE/S作补充，混入真实运动；背景std N/A |
| `moving`：移动相机 | 同光流MAE/S、RMSE/S；中位数比值和固定跨度变化曲线 | 逐像素MAE/S、RMSE/S作补充，含视差；中位数也受真实距离和视野组成影响；背景std N/A |

两模型都计算不经光流的逐像素指标；它使用同一屏幕坐标、全画面、无FB mask。光流项使用有效对应区域。两类有效域不同，应分别在同项中比较DA/VDA，不能把某项更小直接解释成更准确。指标公式与无人固定全图背景前提见[Step 2](step2.md#指标怎样读)。

## 共同协议与科学边界

- 同一个manifest、每张RGB、真实源PTS和帧顺序；10 FPS、长边≤960，16:9缓存960×540；两模型相同官方预处理，网络518×924、输出540×960、Small/FP16。
- 同一双向RAFT、12次更新、FB≤1 px、全图有效对应；两模型光流mask逐像素相同。逐像素项使用整幅画面，不套光流mask。
- 原生`relative / inverse_depth_like / arbitrary / alignment:none`；合法零值保留，无GT，不取倒数或额外平滑；非有限预测明确失败。
- 每模型固定首帧全图`S=P95−P5`；同一评价规则，原生尺度可不同，参考跨度只是整片固定的单位；零跨度N/A。所有非空帧对等权，低coverage仅标记。
- 同一TURBO与原生[0,50]；固定首帧参考副图[0,1]；pixel/flow差分均[0,0.1]。范围全片固定，记录饱和比例，显示截断不改变指标。
- 逐帧中位数`m_t/m_0`与`(m_t−m_0)/S`均保存曲线及摘要。比值受任意固定零点影响，运动/视野/遮挡也影响两项；它们是尺度proxy，不等于真实米制漂移。
- 两部分分别保存自己的`clip.json`、`protocol.json`和调用记录。2B的`clip.json.baseline`绑定1B目录及其clip、协议、DA run、flow清单、分析和verify的SHA256，VDA协议引用同一RGB manifest。两份协议的文件哈希不要求相同，但科学配置和输入须一致，2B重算的DA数值须与1B一致。不混用不同批次的帧或预测。

官方VDA使用32帧窗口、10帧重叠、22帧步长、关键帧`[0,12,24,25,26,27,28,29,30,31]`及8帧混合；尾部重复补齐后裁回实际T帧。窗口之间的预测scale/shift及非负截断是官方算法的一部分，不读取GT，也没有额外逐帧校准。曲线窗口起点仅作诊断参考，不等于精确输出接缝。

离线VDA使用未来帧，不称因果在线；官方支持streaming并不表示必须同时做streaming。模型整片耗时/T是均摊成本，不是在线首帧延迟。DA/VDA的FPS、显存、参数量、权重大小、实际尺寸与环境分别如实记录，模型FPS不含RAFT、解码、保存和绘图，不称同时受控负载实验。

## 用户运行：六段完整比较

所有命令从项目根目录执行。先确认该视频的Step 2已完成且`verify1.json`为`verified`，然后本入口只运行VDA、比较与原始重算。无需再次抽另一套帧或重跑NYUv2。

可先只读查看一段的计划和资源，不执行模型：

```bash
./scripts/part2/video.sh check
./scripts/part2/video.sh plan --clip fixed_scenario1
```

随后每条命令完成一段视频的全部适用任务，正式模型推理和主要耗时由用户启动：

```bash
./scripts/part2/video.sh run --clip fixed_scenario1
./scripts/part2/video.sh run --clip fixed_scenario2
./scripts/part2/video.sh run --clip fixed_scenario1_people_moving
./scripts/part2/video.sh run --clip fixed_scenario2_people_moving
./scripts/part2/video.sh run --clip moving_left_to_right
./scripts/part2/video.sh run --clip moving_rear_to_front
```

每段自动完成`prepare → vda → analyze2 → verify2`。这里prepare只绑定已有1B并冻结2B协议，不抽帧、不运行DA或RAFT。每段结束应有`outputs/part2/video/<ID>/verify2.json`的`status: verified`，以及`part2/status.json`的`status: complete`。不要求先把所有视频分别完成抽帧、再分别完成预测。

六段完成后汇总：

```bash
./scripts/part2/video.sh summarize --output-dir outputs/part2/video_summary/all_v1
```

按三种类型分别等视频汇总，保留单片、N/A与负结果。新增视频后输出到新的汇总快照目录，如`outputs/part2/video_summary/with_new_clip_v1`；汇总不运行模型，视频数量不限六段。

## 单项任务、续跑与新视频

只补VDA、只做比较或只重算核验：

```bash
./scripts/part2/video.sh run --clip fixed_scenario1 --task vda
./scripts/part2/video.sh run --clip fixed_scenario1 --task analyze --no-deps
./scripts/part2/video.sh run --clip fixed_scenario1 --task verify --no-deps
```

`--task`支持`all/prepare/vda/analyze/verify`；不提供dav2/flow任务。默认补齐本部分缺少的依赖；`--no-deps`只做所选任务，依赖缺失直接报错。`analyze`生成本类型全部适用指标与报告；`verify`从双方raw/共享flow完整重算，不推理。

再次运行同一视频ID就是续跑。已完成阶段须通过来源和完整性核验，失败/不完整阶段不能自动跳过或覆盖。保留失败run/status和`commands/`中的调用记录；需要重试失败阶段时，将失败目录原样移到维护位置再运行。原视频/类型/配置/源码改变时使用新ID或两部分各自新的输出根及1B新的缓存根目录，不改写已冻结协议。

新增视频先在Step 2明确源路径、类型和实际session，完成1B后，再按同一ID运行2B；不需要修改Python或在2B重复登记来源：

```bash
./scripts/part1/video.sh run --clip my_new_video --video data/self_recorded/my_new_video.mp4 --type moving
./scripts/part2/video.sh run --clip my_new_video
```

如果该视频的1B已完成并保存在自定义根`outputs/part1/my_video_run/`，用`--part1-root`明确指定输入；`--output-root`只控制2B自己的产物。汇总时也要指定同一个2B输出根：

```bash
./scripts/part2/video.sh run --clip my_new_video \
  --part1-root outputs/part1/my_video_run \
  --output-root outputs/part2/my_video_run
./scripts/part2/video.sh summarize \
  --output-root outputs/part2/my_video_run \
  --output-dir outputs/part2/video_summary/custom_v1
```

两部分输出目录必须分开，不能互相包含。2B不会自动启动1B，`--no-deps`也不会绕过完整1B输入检查。

## 架构与完整产物

| 文件 | 职责 |
|---|---|
| `scripts/part2/video.sh`、`pipelines/part2/video_clip.py` | Part 2B入口，默认完成同一视频全部比较 |
| `pipelines/video/clip.py`、`src/video/clip_workflow.py` | 共用参数/辅助；独立的1B、2B调度函数，分别冻结与记录实际执行/复用/失败 |
| `src/video/clip_analysis.py` | 两模型同协议加载、逐像素/光流/背景/尺度指标、基线桥接及重算 |
| `src/video/clip_report.py` | 同步固定色标、两种差分、曲线、背景图、收益/恶化案例及自动中文说明 |
| `configs/video/self_recorded.yaml`、`configs/video/clips.yaml` | 共同科学设置与可扩展视频登记；模型配置不写死片段数量 |

```text
outputs/part1/video/<ID>/                 1B独立保存，只读引用
  clip.json、protocol.json、dav2/、flow/、part1/、verify1.json、commands/
outputs/part2/video/<ID>/
  clip.json、protocol.json、commands/       2B身份、baseline输入引用、自身协议与调用
  vda/{run.json,raw/}                      官方离线relative预测
  part2/
    summary.json、status.json
    pair_metrics.csv                      两模型pixel/flow逐对指标、coverage与绝对收益
    frame_metrics.csv                     两模型逐帧原生中位数、ratio和固定跨度变化
    dav2/、vda/                           共同mask和适用的全图背景数组
    report/
      analysis.md、report.json、curves.png/pdf
      depth_raw.mp4、depth_normalized.mp4  RGB/DA/VDA，同帧同色标
      pixel_difference.mp4                相同坐标变化，无flow/FB mask
      flow_difference.mp4                 对应位置变化与共同mask
      pixel_cases.png、flow_cases.png     高变化、最大/最小改善（可能恶化）
      contact_raw.png、contact_normalized.png
      background_std.png                  仅fixed_static，两模型全画面
  verify2.json                            双方raw和共享flow完整重算
```

RGB与manifest位于`data/video_frames/self_recorded/<ID>/`，原视频位于`data/self_recorded/`。一个视频使用相同ID，在两部分各有自己的产物目录；DA原始预测和RAFT只保存在1B，VDA原始预测与双模型比较保存在2B。2B报告中的dav2/为比较所需的派生mask/背景数组，不是另一套DA预测。2B的完整复现需要一并保留它引用的1B目录和RGB缓存。共同科学配置与代码在各自prepare时冻结，不能在同一run里修改设置。

## 结果与分析状态

正式结果待用户运行后补充，不预设VDA一定更好。报告同时列出逐像素、光流、适用背景std和两类中位数摘要；降低率`(DA−VDA)/DA`只在DA>0且两值可评价时计算，负数保留，N/A不补零。

分析时分别回答：无人固定是否减少短时闪烁、整段缓变是否也减少；人物进入/离开与遮挡处是否出现局部恶化；运动镜头的对应位置与中位数趋势是否改善。检查差分、coverage、原生色标、参考副图饱和、细结构与实际成本。没有边界/深度GT，稳定不能替代准确性；没有拍到的镜面/玻璃/天空/强模糊也不能称已验证。

## 验证与完成门禁

当前合成文件生命周期覆盖两步独立入口/目录、2B只读复用1B、缺少1B阻止执行、三类任务分工、全图背景、逐像素平移对照、零跨度/零值、完整有限值、精度/尺寸/官方窗口不匹配、原数组/CSV/mask篡改、失败保留、任意视频数量汇总、固定色标图视频及完整raw重算。合成预测不代表正式模型效果。

| 门禁 | 状态 |
|---|---|
| 单视频入口、共同科学协议、三类指标和报告 | 已实现并通过合成验证 |
| 实际CUDA、本地官方代码/权重与六段源身份 | 资源预检记录另存维护目录；不等同全片结果 |
| 六段正式VDA、同帧/尺寸/FP16、共同mask、DA桥接 | 待用户运行 |
| 双方raw重算、真实收益/失败、效率与限制解释 | 待正式产物后核验与分析 |
| mandatory来源、课程报告、应用Demo和最终视频打包 | 单列待完成 |
