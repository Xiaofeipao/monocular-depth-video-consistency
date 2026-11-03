# Step 4 / Part 2B：Video Depth Anything离线相对深度

课程依据为[新版项目PDF §5.2、§6.1–6.2](../Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf)和[完整中文说明](../Project3_CN_final.md)。本页分为VDA独立实验和DA/VDA同帧比较两部分，分别启动、分别保存。只做Small、离线、relative；streaming与米制视频模式暂不开展，不训练或微调。

## 第一部分：VDA独立实验

### 任务与当前状态

使用官方**Video Depth Anything-Small相对深度权重**，处理指定`data/davis_rollercoaster.mp4`的前100个真实原始帧，保留原始帧率。保存原生浮点预测，制作RGB/相对深度/预测边缘同步视频，分析尺度约定、边界稳定性、时序变化和运动伪影，并记录实际推理成本。

**当前70个真实帧的VDA独立预测、图视频、诊断和完整raw重算已完成，独立结果与分析见本部分后半段。同帧比较也已完成，结果单列在第二部分。** 实际使用RTX A6000、FP16；现有源视频、官方仓库和relative权重已核验。运行前资源检查见resources.json（本地产物：`outputs/maintenance/part2b_course_20261103/resources.json`），独立复核见recheck.json（本地产物：`outputs/maintenance/part2b_standalone_review_20261103/recheck.json`）。不需要重新下载，也不依赖RAFT、相机位姿、深度GT或其他深度模型的运行产物。

当前文件实际只有**70帧、24 FPS、约2.917秒**。程序处理全部70个真实帧，输出索引0–69；课程“前100帧”素材数量仍待教师确认70帧是否接受，或提供更长版本。不会重复、循环或插帧来凑100，官方网络内部的尾部补齐也不算新增真实帧。

### 输入与固定协议

| 项目 | 设置 |
|---|---|
| 输入 | `data/davis_rollercoaster.mp4`；960×540、原始24 FPS |
| 源SHA256 | `7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8` |
| 选帧 | 连续前`min(100,源帧数)`帧，保留原始PTS，不按新FPS网格重采样 |
| RGB画布 | 长边≤640、保持比例；当前640×360 |
| 模型 | 官方VDA-Small / ViT-S，relative权重；commit `4f5ae23172ba60fd7bc11ef671cca678842c7072` |
| 权重SHA256 | `13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609` |
| 推理 | offline、FP16、CUDA；`input_size=518`，当前画布对应网络H×W为518×924，恢复360×640 |
| 原生身份 | `relative / inverse_depth_like / arbitrary / alignment:none`；不读取GT，合法零值保留 |
| 保存 | 完整非负有限float32；整段`depths.npz`及逐帧`raw/*.npy`，两种保存逐值核验 |
| 显示 | TURBO；全段P1/P99确定一次上下限，整段固定，不逐帧归一化 |
| 辅助诊断 | 同坐标MAE/S、RMSE/S；Farneback双向光流、FB≤1 px；预测边缘变化、2 px匹配容差 |
| 局部观察 | 固定连续帧和固定画面坐标；仅作展示，不限制全图指标评价区域 |
| 环境 | `aiaa3201`；脚本自动从自身位置确定项目根目录，不安装到base |

项目入口调用官方`infer_video_depth`，保持其窗口和衔接算法；选Small、前100帧上限、长边640、保留NPZ，对应课程起始选项`--encoder vits --max_len 100 --max_res 640 --save_npz`，没有直接沿用官方脚本可能逐帧变化的显示设置。

默认RGB缓存为`data/video_frames/part1b_davis_v1/`。这里读取的只是RGB输入和manifest，不要求存在其他模型的深度结果。缓存存在时先核对源SHA、逐张RGB SHA、原始帧号/PTS、尺寸和FPS，再只读复用；缓存不存在时按连续原始帧规则创建，已有但不匹配时直接报错，不覆盖。

### 官方时序处理与效率口径

VDA一次接收整段RGB，使用32帧窗口、10帧重叠、22帧步长，关键帧为`[0,12,24,25,26,27,28,29,30,31]`，相邻预测进行8帧混合。跨窗口的预测间scale/shift及负值截断属于官方算法，不使用GT，不是额外给每张图做米制对齐，也不是人为修改原始输出以改善指标。

按官方规则，70帧对应4个窗口，起点为0/22/44/66，尾部内部补齐28帧，最后只保留70份真实对应输出；这些是窗口计算规则，不是100帧素材已满足的证据。实际窗口数、输入网格和输出帧数会在每次运行中检查。

离线模型使用未来帧，不是因果streaming。计时先完整预热1遍，再完整测量3遍，保存第1次测量的预测。平均整段时间除以真实输出帧数得到均摊ms/帧，倒数为FPS；包含预处理、传输、窗口前向、衔接和输出恢复，不含解码、保存、光流分析和绘图。峰值allocated/reserved显存、实际硬件/版本、参数和权重字节另存。中位/P95也是“三次整段耗时的统计再除以帧数”，不能称为逐帧延迟分布或在线首帧响应时间。

### 用户运行：完整VDA实验

下面从项目根目录执行。**用户已完成当前默认目录的首次完整运行，无需再次执行`run`或`predict`。** 以下保留完整复现流程；检查现有结果只需第3条verify，重新预测须按后文指定新输出目录。不需要重跑静态NYUv2或训练模型。

```bash
# 1. 只读检查源、官方代码/权重及实际CUDA，不执行模型
./scripts/part2/run_vda_video.sh check

# 2. RGB准备/复用 → VDA离线预测 → 全部分析/图视频 → 原始重算核验
./scripts/part2/run_vda_video.sh run

# 3. 独立复核已有结果；不运行模型，不改写已有verification.json
./scripts/part2/run_vda_video.sh verify \
  --run outputs/part2/vda_video/rollercoaster_v1
```

第2步结束应有`run.json.status: complete`、`analysis/status.json.status: complete`，以及`verification.json.status: verified`。`required_100_frames_satisfied:false`在当前70帧素材下是如实记录的课程限制，不能手动改为true。

首次从空目录分开执行预测和分析时，用下面的分步流程**代替上面的第2步**，不要在当前已有完整目录再执行`predict`：

```bash
./scripts/part2/run_vda_video.sh predict
./scripts/part2/run_vda_video.sh analyze \
  --run outputs/part2/vda_video/rollercoaster_v1
./scripts/part2/run_vda_video.sh verify \
  --run outputs/part2/vda_video/rollercoaster_v1
```

重新分析不需要预测，必须写入新的分析目录：

```bash
./scripts/part2/run_vda_video.sh analyze \
  --run outputs/part2/vda_video/rollercoaster_v1 \
  --analysis outputs/part2/vda_video/rollercoaster_v1/analysis_replot_v1
./scripts/part2/run_vda_video.sh verify \
  --run outputs/part2/vda_video/rollercoaster_v1 \
  --analysis outputs/part2/vda_video/rollercoaster_v1/analysis_replot_v1
```

需要独立复现预测时使用新目录；可以继续只读复用已通过检查的RGB：

```bash
./scripts/part2/run_vda_video.sh run \
  --output outputs/part2/vda_video/reproduce_v1
```

也可以显式指定新的缓存，例如`--cache data/video_frames/part2b_davis_reproduce_v1`。输入视频、尺寸或科学参数改变时，复制`configs/part2/vda_video.yaml`形成新配置，真实更新源路径/SHA及相应观察窗口，再使用`--config`和新输出目录。不要修改已经被run绑定的配置。若显存不足，可在新配置降低`input_size`并报告实际设置；后续同条件比较仍需双方网络尺寸一致。

已有输出一律不覆盖。预测失败会保留`run.json`及真实错误；失败预测不自动当作可续跑结果。预测已完整、只有分析失败时，可以用新`--analysis`目录重新分析并指定该目录核验；不要再次预测或覆盖失败分析记录。

### 完整产物与架构

```text
outputs/part2/vda_video/rollercoaster_v1/
  protocol.json、run.json               冻结协议、源码归档、真实执行/窗口/效率
  depths.npz                           整段float32，字段depths，T×H×W
  raw/frame_*.npy                      每个真实帧的原生相对预测
  verification.json                    run命令生成的原始重算记录
  analysis/
    status.json、summary.json           阶段状态、完整精度汇总、附件SHA
    analysis.md                        自动中文读图与数值说明；实际观察后补充结论
    frame_diagnostics.csv              每帧范围/中位数/预测边缘比例
    pair_diagnostics.csv               每对相邻帧的同坐标/光流/边缘/覆盖指标
    visualization/
      rgb_relative_edges.mp4           RGB / VDA relative / 红色预测深度边缘
      contact.png                      全段均匀取帧总览
      diagnostics.png、diagnostics.pdf  四张诊断子图
      fixed_colorbar.png、display.json  全段固定色标及显示饱和计数
      cases/*.png                      自动定位的高变化连续窗口
    review/
      thin_boundary.png                65–69帧：细支架/轨道及深度交界
      occlusion_train.png              52–56帧：运动列车与支架交叉遮挡
      moving_edge.png                  57–61帧：运动轮廓及显露区域
      review_sources.json              原始运行引用、固定色标、帧号和裁剪框
```

每一份NPY应对应一个真实输入帧；NPZ与所有NPY完全一致。`verify`重新检查源/权重/官方仓库/科学源码、缓存、帧序/PTS、窗口、原生值、CSV、完整精度汇总及图视频SHA，再从raw计算全部辅助指标；不加载深度模型。核验中的`models_executed:false`只表示核验未执行模型，不否定前面的真实预测。浮点使用`rtol=1e-6、atol=1e-8`，不要求跨设备逐位相同。

| 项目文件 | 职责 |
|---|---|
| `configs/part2/vda_video.yaml` | 指定素材、Small/FP16/offline、画布、固定显示和三个局部窗口 |
| `scripts/part2/run_vda_video.sh`、`pipelines/part2/vda_video.py` | 独立check/predict/run/analyze/verify入口 |
| `src/evaluation/temporal_clip.py` | VDA运行、原生保存、协议/源码绑定、真实计时和完整重算 |
| `src/models/video_depth_anything_adapter.py` | 已有官方API适配器，保留窗口与相对深度分支，不修改上游 |
| `src/evaluation/course_video_inputs.py` | 连续原始帧/PTS及只读缓存校验所用的输入规则 |
| `src/metrics/video_diagnostics.py` | 无GT的固定跨度、同坐标/光流/边缘诊断 |
| `src/visualization/temporal_clip.py` | 独立视频、固定色标、四子图、连续案例和局部放大 |

### 指标、诊断图和观察重点

整段预测的`S=P99−P1`是固定相对跨度，没有米制单位；本次实际`S=5.251484932899474`，详细指标见后面的独立结果表。

| 指标/子图 | 含义与读法 |
|---|---|
| 原生中位数；诊断图① | 每帧全部原生值的中位数；反映分布，受大面积背景和任意零点影响，不能读成相机距离或真实尺度漂移 |
| 同坐标MAE/S、RMSE/S；②的蓝线为RMSE/S | 全画面同一屏幕位置的预测变化；MAE为绝对差均值，RMSE对大变化更敏感；包含真实相机/物体运动 |
| 光流MAE/S、RMSE/S；②的橙线为RMSE/S | 用RGB Farneback近似对应后，在FB有效域计算变化，不拟合深度scale/shift；仍含遮挡和光流误差 |
| 边缘变化；③ | 对应后、2 px邻域内双向未匹配预测边缘比例；越低表示本次有效域内边缘较一致，不是GT边界准确率 |
| 光流有效覆盖；④ | 通过越界/FB检查的像素占全图比例；不是深度准确率，也不保证对应一定正确 |
| 每帧最小/最大值和边缘比例；CSV | 补充观察原生输出范围及梯度区域占比，不把合法零值删除 |

横坐标为原视频PTS；①每点是一帧，②–④每点是一对相邻帧，放在后一帧时间；图中只画RMSE，不画MAE，但CSV保存两者。所有非空帧对等权；不可评价项保留null/N/A，不填成0，不按分数删除难帧。S为0时，归一化指标不可评价，不把常数输出认定为完美稳定。图中数值只表示以固定S衡量的变化，不是“多少比例像素预测错了”。

红色区域由原生预测/S、sigma=0.8像素的Gaussian平滑、3×3 Sobel/8及梯度阈值0.02提取。它表示预测深度变化较明显的位置，没有人工GT支持；红带宽度受深度差、过渡、插值和平滑影响，不是物体厚度或置信度。三个局部框只用于看细节，不参与评价mask。

本次按下面四个问题检查实际产物，具体观察见后文：

1. 天空、轨道、支架与列车的远近层次是否合理？relative本身没有米制校准，不能报告视频AbsRel或“错了多少米”。
2. 相同物体的预测轮廓是否随真实运动连续移动？有无额外粗细变化、断裂、黏连或细结构消失？
3. 同一物体是否出现无法由RGB真实变化解释的颜色跳动？是否有平滑后的拖影、滞后或过度合并？
4. 列车与杆件交叉、快速轮廓位移和显露区域是否困难？只分析实际出现的内容，不虚构镜面、行人等场景。

### 70帧独立实验的实际结果

来源为当前`outputs/part2/vda_video/rollercoaster_v1/`，从完整精度JSON/CSV读取后按8位小数展示。独立复核阶段检查源身份、70份原始NPY与整段NPZ一致性、70行逐帧/69行帧对统计、官方窗口、报告附件SHA，并重新执行独立verify；全部通过。该阶段没有重新预测、训练或改写原分析及运行元数据；随后完成的配对比较见第二部分。

- RGB / VDA相对深度 / 红色预测边缘视频（本地产物：`outputs/part2/vda_video/rollercoaster_v1/analysis/visualization/rgb_relative_edges.mp4`）：实际解码70帧、24 FPS。
- 全段总览（本地产物：`outputs/part2/vda_video/rollercoaster_v1/analysis/visualization/contact.png`）、诊断图（本地产物：`outputs/part2/vda_video/rollercoaster_v1/analysis/visualization/diagnostics.png`）、固定色标（本地产物：`outputs/part2/vda_video/rollercoaster_v1/analysis/visualization/fixed_colorbar.png`）。
- 完整精度summary（本地产物：`outputs/part2/vda_video/rollercoaster_v1/analysis/summary.json`）、实际run与效率（本地产物：`outputs/part2/vda_video/rollercoaster_v1/run.json`）、用户运行的核验记录（本地产物：`outputs/part2/vda_video/rollercoaster_v1/verification.json`）。

| 独立诊断量：69个相邻帧对等权 | 实际值 | 可以说明什么 |
|---|---:|---|
| 同坐标MAE/S | 0.09746971 | 同一屏幕位置平均变化，包含真实运动 |
| 同坐标RMSE/S | 0.24548739 | 对较大同坐标变化更敏感 |
| 对应后MAE/S | 0.01050272 | 近似光流对应后仍存在的平均变化 |
| 对应后RMSE/S | 0.03950941 | 对应后的较大残余变化 |
| 光流有效覆盖 | 0.91111526 | 平均91.1115%像素通过对应检查，不是准确率 |
| 边缘变化proxy | 0.00340540 | 有效域、2 px邻域下，双向未匹配预测边缘平均约0.3405% |

本次六项量均有69个有效帧对，没有因分数高而删除帧对。同坐标项和对应项的有效域不同，不能用两者相减来计算“消除了多少闪烁”。例如对应后MAE/S约0.01050，表示平均变化约为固定跨度S的1.05%，不是1.05%的真实深度误差。

实际原生值范围为`[0,7.399656295776367]`，整段固定显示范围为`[0,5.251484932899474]`。16,128,000个像素中有161,280个仅在显示上界饱和，恰为1%；原生预测不裁剪，指标仍使用完整原值。近处区域同为色标顶端颜色，不表示它们原始值完全相同。

| 实际效率与规模 | 结果 |
|---|---:|
| 平均整段推理时间（70个真实输出帧） | 2.05250862 s |
| 平均 / 中位 / P95均摊时间（ms/帧） | 29.32155170 / 29.39120173 / 29.42362062 |
| 模型均摊FPS | 34.10460709 |
| 峰值allocated / reserved（MiB） | 2689.2197 / 3138.0000 |
| 参数 / 权重字节 | 29,035,137 / 116,440,756 |
| 模型加载 | 0.71959114 s |
| 预测阶段：含加载、预热、三次测量及保存 | 9.90872587 s |

实际为`aiaa3201`、Python 3.10.22、PyTorch 2.1.1+cu121、RTX A6000/FP16；四个窗口的实际网络输入均为518×924，原生输出360×640。每遍处理128个窗口内帧位置，内部补齐后裁回70个真实帧。34.10 FPS只代表本次离线序列的均摊模型成本；9.91秒包含多遍执行和保存，不是单遍推理时间，也不能由此认定在线实时性能。

### 分析一：相对结构与中位数的分段上升

总览中，天空与主体轨道、支架、列车形成可辨的相对层次，近远支架也有不同色带；但它没有米制校准。色标上的5不表示5米，没有视频深度GT，不能报告视频AbsRel、米制RMSE或真实距离漂移。`2D+5`及相应色标变换的辅助演示最大归一化差约`7.54e-08`，只说明相对数值单位不能由显示确定，没有写成新预测。

诊断图①的中位数不是一条常数线，而是“平台—缓升—平台”的形状。实际帧段如下：

| 原始帧号（0-based） | 原生中位数变化 | 与官方输出衔接的关系 |
|---|---|---|
| 0–24 | 0 | 初始低值平台 |
| 25–31 | 0.00148015逐步升至0.01036105 | 位于24–31帧的8帧混合区间 |
| 32–46 | 0.01036105 | 下一段低值平台 |
| 47–53 | 0.01157848逐步升至0.01888306 | 位于46–53帧的8帧混合区间 |
| 54–68 | 0.01888306 | 平台持续到下一次混合开始 |
| 69 | 0.01923549 | 最后混合区间68–75中，仅68–69留在真实输出内 |

从全部raw核对，**70帧的中位数都恰好等于各自最小值**，每帧约67.63%–76.05%的像素落在该低值平台。结合大面积天空，这说明中位数主要反映背景平台，而不是列车或支架的距离变化。上升区间又与官方预测间scale/shift衔接和8帧线性混合一致，因此更适合解释为低值背景在窗口衔接中的数值偏移线索，不能直接判为物理尺度漂移。运行没有单独保存每窗拟合的scale/shift系数，不能从最终raw恢复或声称量出了它们的具体数值。

首末中位数差只相当于固定S的约0.3663%，**这也不是“真实尺度漂移0.3663%”**。不要把曲线变平或上升全部解释为时序模型成功或失败；场景内容、任意数值零点和官方衔接都参与其中。

### 分析二：边界、运动列车与细结构

本次查看了全段总览、三个固定连续局部图，以及自动候选附近的连续帧。总体上，主体轨道与支架轮廓在已查看的连续窗口内保持连贯；局部细节仍明显比RGB粗，稳定的轮廓并不等于准确的轮廓。

| 实际连续窗口 | 看到的表现 | 解释与局限 |
|---|---|---|
| 65–69：细结构与深度交界（本地产物：`outputs/part2/vda_video/rollercoaster_v1/analysis/review/thin_boundary.png`） | 橙色弯轨与支架外形连续，列车/座椅附近呈较粗的青绿色块；远处旗杆在相对图上只是较弱的细线，未被完整、清楚地分离 | 大结构可持续跟踪，但小目标、空隙与薄结构仍不足；不能把较宽红带当成物体变厚或确定的定位错误 |
| 52–56：列车与杆件交叉（本地产物：`outputs/part2/vda_video/rollercoaster_v1/analysis/review/occlusion_train.png`） | 车体、乘客及附近结构在局部相对图上合并，交叉处的色带/轮廓较粗；RGB同时发生真实遮挡和位移 | 时序建模没有自动解决所有动态细节；局部合并值得记录，但没有GT，不能测出遮挡深度错误或延迟帧数 |
| 57–61：移动轮廓与显露区域（本地产物：`outputs/part2/vda_video/rollercoaster_v1/analysis/review/moving_edge.png`） | 轨道和支架随画面移动，列车下缘与细杆附近仍有形状/宽度变化，旗杆等低对比结构较弱 | 需要同时看RGB，不能把真实运动和显露变化全算成闪烁；相同屏幕坐标也不代表相同物体点 |

边缘变化proxy平均约0.3405%，最高点为第27→28帧、`1.166667 s`，约0.7173%；26–30帧候选图（本地产物：`outputs/part2/vda_video/rollercoaster_v1/analysis/visualization/cases/case_01_boundary_change_0028.png`）中，列车靠近中央立柱、轮廓发生交叠，值得检查局部边缘对应。但低值或高值都不是GT边界准确率；2像素以内的抖动、持续存在的错误轮廓和少量细杆问题可能被容差与整体平均掩盖。

三个固定窗口来自同一短片，不是三个独立场景；没有因此声称已覆盖反光、行人或其他未出现的困难类型。

### 分析三：时序变化与运动伪影

诊断图②中，同坐标RMSE/S约为0.1540–0.3132，在片段后段较高；对应后RMSE/S约为0.0334–0.0485，波动相对集中。前者包含列车和相机运动，后者尽量比较相同画面内容，但仍受视角、遮挡及Farneback误差影响。第66→67帧的对应后变化最高，为`0.04847950`，可结合65–69帧窗口查看；它是变化较大的位置，不自动等于严重闪烁。

在已查看的连续窗口中，没有足够证据把结果描述成全画面剧烈随机闪烁；更直接的局限是运动列车附近的粗色块、交叉遮挡处细节合并，以及细结构轮廓/色带的局部变化。没有相机位姿或时序GT，当前也不量化声称存在或消除了多少拖影、在线滞后。

诊断图④的有效覆盖范围约87.3681%–95.4887%，平均91.1115%；剩余区域没有进入对应后指标。即使通过一致性检查，弱纹理天空上的光流也未必准确。又因为大面积低值平台主导全图，较低的对应后均值不能证明乘客、旗杆或交叉细杆已经稳定准确恢复。

独立实验支持的结论是：**VDA恢复了连贯的主体相对结构，仍有动态遮挡与细结构局限；原始输出和辅助诊断完整可复核。** 本部分只讨论VDA自身，模型之间的收益、退化与成本见第二部分。

### 独立复核来源与当前范围

独立`verify`从原始预测重算全部辅助量，通过`rtol=1e-6、atol=1e-8`；六项CSV宏平均与summary的完整精度一致，NPZ与70份NPY逐值一致。原视频、预测、自动分析和运行元数据均未改写。补充描述统计与来源SHA见results_review.json（本地产物：`outputs/maintenance/part2b_standalone_review_20261103/results_review.json`），复核过程见review_stats.py（本地产物：`outputs/maintenance/part2b_standalone_review_20261103/review_stats.py`）。第一部分当前70帧工作已完成，配对比较单列在第二部分；100帧素材数量及报告/应用Demo/ZIP仍单列。

## 第二部分：与Part 1B逐帧基线正式比较

**用户已完成当前70帧的同条件配对比较，双方raw独立重算通过。** VDA的对应后平均变化有所下降，但边缘变化均值几乎持平、略差，且计算成本上升；不能据此宣称全面改善。实际指标、连续案例及限制见本部分后半段。

### 比较条件与当前基线

这里才引入[Step 2](step2.md)的DA-V2-Small。比较程序只读双方已经保存的预测，不自动启动1B，不重新预测DA，也不写入任何一条原始运行或RGB缓存。比较输出独立于两条预测目录。

默认VDA复用1B当前RGB manifest，两侧都是同一批70个源帧、同一PTS、640×360画布、518×924实际网络输入和FP16。比较会核对逐张RGB SHA、源SHA、帧号/PTS、尺寸/精度、诊断设置及显示规则，核验1B的`handoff_2b.json`；独立复现若使用另一目录的缓存，也必须有相同真实帧和相同RGB校验值，不能靠帧数相等就通过。

1B与2B调用**同一个`framewise_clip.measure`和`video_diagnostics`实现**，不另写一套近似指标。每模型各自用整段P1/P99固定跨度S，使用同一TURBO；不能强迫任意单位不同的两个模型共享原生数字范围，也不能把相同颜色解释成同样的米数。光流仅由相同RGB确定，两侧有效像素数和覆盖一致；不根据预测值大小删除零值或困难像素。

### 用户运行：同帧比较与独立核验

当前默认比较目录已由用户运行完成，无需再次执行第2条`compare`；以下保留完整流程。日常复核只需第3条`verify-comparison`，重新比较时使用新的`--output`，不覆盖现有结果。两条原生预测均已存在，以下命令不重新推理：

```bash
# 1. 核验已有1B输入
./scripts/part1/run_dav2_video.sh verify \
  --run outputs/part1/dav2_video/rollercoaster_v1

# 2. 同帧对比 → 双模型表/曲线/视频/案例 → 双方raw独立重算
./scripts/part2/run_vda_video.sh compare \
  --dav2-run outputs/part1/dav2_video/rollercoaster_v1 \
  --vda-run outputs/part2/vda_video/rollercoaster_v1 \
  --output outputs/part2/video_comparison/rollercoaster_v1

# 3. 独立核验比较产物；不运行任何深度模型
./scripts/part2/run_vda_video.sh verify-comparison \
  --comparison outputs/part2/video_comparison/rollercoaster_v1
```

比较入口先完整核验两条输入，再创建新的比较目录；输出路径不得位于任一原始预测目录内。第2步自动生成`verification.json`，应为`status: verified、both_raw_recomputed:true`。已有比较目录不会覆盖；重新比较时改用新的`--output`。

如果第一部分使用了新的分析目录，在`compare`增加`--vda-analysis`；1B也支持显式`--dav2-analysis`。两者都只指定分析来源，不改变原始预测：

```bash
./scripts/part2/run_vda_video.sh compare \
  --dav2-run outputs/part1/dav2_video/rollercoaster_v1 \
  --vda-run outputs/part2/vda_video/rollercoaster_v1 \
  --vda-analysis outputs/part2/vda_video/rollercoaster_v1/analysis_replot_v1 \
  --output outputs/part2/video_comparison/rollercoaster_replot_v1
```

### 比较产物与表格

```text
outputs/part2/video_comparison/rollercoaster_v1/
  status.json、summary.json             双方run/analysis身份、完整精度、逐项差值
  verification.json                    双方原始预测和全部比较数值重算
  frame_comparison.csv                 两模型逐帧范围/中位数/边缘比例
  pair_comparison.csv                  同帧对的双方五项指标、DA−VDA、共同覆盖
  analysis.md                         中文数值表、效率表及实际观察提示
  visualization/
    rgb_dav2_vda.mp4                   RGB / DA relative / VDA relative
    rgb_dav2_vda_edges.mp4             RGB / DA预测边缘 / VDA预测边缘
    contact.png                       相同帧的三栏总览
    diagnostics.png、diagnostics.pdf   双模型四子图，原始PTS
    display.json                      每模型固定参考及显示饱和数
    cases/*.png                       最大/最小DA−VDA的连续帧候选
  review/
    thin_boundary.png                 与1B相同65–69帧及同坐标
    occlusion_train.png               与1B相同52–56帧及同坐标
    moving_edge.png                   与1B相同57–61帧及同坐标
    review_sources.json               双方来源与固定展示窗口
```

实际来源为`outputs/part2/video_comparison/rollercoaster_v1/summary.json`及逐帧/帧对CSV，从完整精度读取后展示；双方原始预测均重新核验，不沿用其他机器或旧自录结果。五项指标各有69个双方共同可评价帧对，没有删去负差、难帧或零值像素。双方对应后数值与各自独立汇总一致，逐帧范围/中位数/边缘比例也与独立CSV一致。

- RGB / DA-V2 / VDA同步视频（本地产物：`outputs/part2/video_comparison/rollercoaster_v1/visualization/rgb_dav2_vda.mp4`）、双方红色预测边缘视频（本地产物：`outputs/part2/video_comparison/rollercoaster_v1/visualization/rgb_dav2_vda_edges.mp4`）：实际均解码70帧、24 FPS。
- 双方诊断曲线（本地产物：`outputs/part2/video_comparison/rollercoaster_v1/visualization/diagnostics.png`）、完整精度比较汇总（本地产物：`outputs/part2/video_comparison/rollercoaster_v1/summary.json`）、用户运行的重算核验（本地产物：`outputs/part2/video_comparison/rollercoaster_v1/verification.json`）。

| 69个共同帧对，逐对等权 | DA-V2-S | VDA-S | DA−VDA | 相对降低率 |
|---|---:|---:|---:|---:|
| 同坐标MAE/S | 0.09631664 | 0.09746971 | −0.00115307 | −1.1972% |
| 同坐标RMSE/S | 0.23702812 | 0.24548739 | −0.00845927 | −3.5689% |
| 对应后MAE/S | 0.01294299 | 0.01050272 | +0.00244028 | +18.8540% |
| 对应后RMSE/S | 0.04071483 | 0.03950941 | +0.00120542 | +2.9606% |
| 边缘变化proxy | 0.00339027 | 0.00340540 | −0.00001514 | −0.4465% |
| 光流有效覆盖 | 0.91111526 | 0.91111526 | 0 | 不作为模型收益 |

前五项同时保存绝对差`DA−VDA`和降低率`(DA−VDA)/DA×100%`：正值只表示本项VDA变化较小，负值如实保留，不把失败替换成DA或删去难帧。降低率仅在DA>0且双方可评价时计算。每项按双方共同非空帧对等权汇总，并保存共同数量；零跨度/N/A不补零。

### 分析一：运动对应后的平均变化降低，但收益不均匀

对应后MAE/S从0.01294299降到0.01050272，降低18.85%，且**69个帧对的VDA MAE/S全部较低**，因此不是仅靠少数有利帧拉低平均。对应后RMSE/S只降低2.96%，49个帧对较低、20个较高；说明对较大残余变化的改善较有限，不能说所有位置或所有帧对都更稳定。

MAE平等累加每个像素的变化，RMSE先平方，对大变化更敏感。两者的差异与“较常见变化减小，但较大残余仍存在”的解释相符；当前指标没有把残余逐像素区分成模型闪烁、真实深度变化或光流错误，不能进一步断言18.85%的纯闪烁被消除了。结论限定为：在相同RGB对应检查和各自固定跨度下，VDA的对应后平均变化较低。

| 辅助量 | VDA较低的帧对 | VDA较高的帧对 | 完全相同 |
|---|---:|---:|---:|
| 同坐标MAE/S | 32 | 37 | 0 |
| 同坐标RMSE/S | 19 | 50 | 0 |
| 对应后MAE/S | 69 | 0 | 0 |
| 对应后RMSE/S | 49 | 20 | 0 |
| 边缘变化proxy | 37 | 32 | 0 |

同坐标MAE/S和RMSE/S分别升高1.20%和3.57%，并不直接推翻上面的对应后结果。同一屏幕位置可能从天空变成列车或支架，预测轮廓、深度对比和真实位移都会影响这项变化；它与对应后指标的评价域也不同。因此，不能用同坐标项单独认定VDA闪烁更严重，也不能只选对应后MAE便宣称全面胜出。

这些69个帧对来自同一约2.917秒短片，时间上相互相关。这里的数量和降低率是描述性结果，不当作69个独立场景的统计显著性证据。

### 分析二：总体边缘变化几乎持平，局部有改善也有退化

边缘变化proxy从0.00339027变为0.00340540，约从0.3390%变为0.3405%。VDA平均相对升高0.4465%，但绝对只增加约**0.001514个百分点**；应描述为本短片上总体几乎持平、略差，不能说总体边界已改善，也没有依据把这么小的均值差称为显著退化。

虽然37个帧对的VDA值较低，多于32个较高的帧对，但各对变化幅度不同，平均值仍可略高。这一指标只检查近似光流后、2像素邻域内的预测边缘能否匹配，**不评价它是否落在真实物体轮廓上**；粗而一致或一直画错的轮廓，也可能很稳定。

下面保留同一指标下的正、负案例。表中分数是对应帧对的全图有效域分数，不是对列车、旗杆或某根支架单独评分，不能把全部变化归因到图中某一物体。

| 实际帧对与观看窗口 | 指标 | DA-V2-S → VDA-S | 实际解读 |
|---|---|---:|---|
| 28→29，查看27–31帧（本地产物：`outputs/part2/video_comparison/rollercoaster_v1/visualization/cases/01_flow_rmse_normalized_largest_dav2_minus_vda_0029.png`） | 对应后RMSE/S，最大正差 | 0.04225216 → 0.03578321 | 列车经过中央立柱附近，两模型主体轮廓接近、局部色带不同；该帧对的变化减小，不等于列车距离更准确 |
| 2→3，查看1–5帧（本地产物：`outputs/part2/video_comparison/rollercoaster_v1/visualization/cases/02_flow_rmse_normalized_smallest_dav2_minus_vda_0003.png`） | 对应后RMSE/S，最大负差 | 0.03766652 → 0.04179529 | 早段两者均可辨列车和轨道，但VDA这一项更高，说明改善并非逐帧对成立 |
| 66→67，查看65–69帧（本地产物：`outputs/part2/video_comparison/rollercoaster_v1/visualization/cases/03_edge_instability_largest_dav2_minus_vda_0067.png`） | 边缘变化，最大正差 | 0.00827897 → 0.00514779 | 弯轨、车体外沿及显露远景处，VDA该帧对的预测边缘更易匹配；两侧细杆仍弱，不能据此认定恢复了真实细节 |
| 36→37，查看35–39帧（本地产物：`outputs/part2/video_comparison/rollercoaster_v1/visualization/cases/04_edge_instability_smallest_dav2_minus_vda_0037.png`） | 边缘变化，最大负差 | 0.00446338 → 0.00662387 | 列车与交叉支架附近仍有轮廓/色带变化，VDA此对未匹配比例更高；不能只展示有利的66→67帧 |

固定三个局部窗口也全部保留：

- 52–56帧交叉遮挡（本地产物：`outputs/part2/video_comparison/rollercoaster_v1/review/occlusion_train.png`）：两者的车体/乘客及附近杆件都存在色块合并；没有清楚证据表明VDA已解决遮挡处的细节问题。
- 57–61帧移动轮廓（本地产物：`outputs/part2/video_comparison/rollercoaster_v1/review/moving_edge.png`）：整体形状相似，局部色带与粗细有差异；真实位移、遮挡显露与RGB模糊仍参与变化，当前不能量化声称减少了多少拖影或滞后。
- 65–69帧细结构（本地产物：`outputs/part2/video_comparison/rollercoaster_v1/review/thin_boundary.png`）：轨道主体连续，但两者都未把远处旗杆、乘客与细小空隙稳定清楚地分离。局部更平滑不等于更准确。

已查看的同步帧与连续窗口中，视觉差异总体较细微，最明确的收益证据来自对应后平均变化；不将本短片描述成由剧烈闪烁转为完全稳定，也不以红带粗细判定GT边界准确性。

### 分析三：没有测出真实米制尺度漂移的改善

诊断图①中，DA中位数始终为0，VDA则分段上升。DA零线受大面积低值天空主导；VDA全部中位数等于对应帧最小值，上升区间与官方窗口间scale/shift及混合一致，详细见第一部分。这不能说明DA的真实尺度更稳定，也不能说明VDA物理距离越估越远。

两模型分别用`S_DA=2.685546875`、`S_VDA=5.251484932899474`固定全段单位，数值和颜色没有共享米制标定。同一TURBO下单帧颜色不同，不直接表示谁距离估错；这里比较的是相同输入下、各自固定相对跨度中的变化。没有视频深度GT、已知米制参考或位姿支持，本次无法定量证明真实scale drift改善。

### 运行成本与本片结论

两条实际记录均为RTX A6000、FP16、518×924网络输入与360×640输出，DA在11月2日、VDA在11月3日执行，不是同一时刻的受控GPU负载测试。

| 实际成本 | DA-V2-S | VDA-S |
|---|---:|---:|
| 平均ms/真实输出帧 | 21.43496807 | 29.32155170 |
| 模型FPS | 46.65274036 | 34.10460709 |
| 峰值allocated（MiB） | 329.4043 | 2689.2197 |
| 峰值reserved（MiB） | 404.0000 | 3138.0000 |
| 参数量 | 24,785,089 | 29,035,137 |
| 权重字节 | 99,218,434 | 116,440,756 |

VDA均摊时间增加约7.8866 ms/帧，即36.79%；FPS降低26.90%；峰值allocated约为DA的8.16倍。其时序窗口同时处理多帧，不能只按参数量增幅推断显存成本。DA为独立单帧调用，VDA为完整离线序列时间除以真实帧数；两者成本都不含光流分析或绘图，VDA也不代表因果在线时延，不将双方P95视为同一种延迟分布。

**现有短片支持“VDA降低了运动对应后的平均相对变化”，但不支持“全面提升边缘、米制尺度和所有运动细节”。** 收益主要体现在对应后MAE，较大残余变化的改善有限；总体边缘近乎持平，动态遮挡与细结构问题仍在，同时付出额外时间和显存。课程期望是检验时序建模是否有帮助，不要求把每个指标写成正结果。

### 配对复核与科学边界

本次再次运行`verify-comparison`，对双方来源、原生预测、独立分析、相同RGB/帧序/PTS/尺寸/精度、共享光流覆盖及全部比较数值重算，通过`rtol=1e-6、atol=1e-8`；五项均为69个共同帧对，无N/A或额外排除。两段正式比较视频均完整解码为70帧/24 FPS。222份现有原生结果、报告和科学源码的保护SHA一致，没有重新预测、训练或修改自动元数据。

补充逐对较低/较高计数、来源SHA与成本比值见results_review.json（本地产物：`outputs/maintenance/part2b_comparison_review_20261103/results_review.json`），实际只读复核脚本见review_stats.py（本地产物：`outputs/maintenance/part2b_comparison_review_20261103/review_stats.py`），本次重算记录见recheck.json（本地产物：`outputs/maintenance/part2b_comparison_review_20261103/recheck.json`）。当前70帧的独立实验和同帧比较均已完成；素材仅70帧这一课程门禁、应用Demo、报告与最终打包仍未因本次分析而完成。

### 逐项比较方法与解释边界

| 比较问题 | 看哪些相同产物 | 需要同时排除的解释 |
|---|---|---|
| 时序颜色跳动是否减少 | 同帧固定色标视频、对应后MAE/S与RMSE/S、相同时间窗口 | 真正的物体/相机运动、遮挡和光流误差；较低变化也可能来自细节丢失 |
| 细边界是否更稳定 | 红色边缘同步视频、65–69帧放大及边缘变化proxy | 2像素容差会容忍小抖动；一直画错或一直模糊也可能很稳定 |
| 运动/遮挡伪影是否改善 | 52–56、57–61帧相同坐标RGB/两模型深度 | 过度平滑、残影、滞后、列车/背景合并，以及RGB本身的模糊 |
| 整体数值分布是否变化 | 双方逐帧中位数与范围、各自固定S、显示饱和比例 | 两模型单位/零点任意、视野组成变化；不能当作真实米制尺度漂移 |
| 实际成本增加多少 | 双方run中的ms/帧、FPS、allocated/reserved、硬件/尺寸/精度 | DA单帧调用与VDA离线序列均摊是不同推理方式，不比较P95为同一分布，也不声称同时受控负载测试 |

比较图①的两条原生中位数使用各自任意单位，数值高低不能排名；②为两模型各自的同坐标/光流RMSE/S，共四条曲线；③为两模型边缘变化；④为共同光流覆盖。光流覆盖下降时，较低的橙线/对应项也可能来自可评价区域减少，不能单独判为改善。

自动候选使用“最大DA−VDA”和“最小DA−VDA”的名称：如果所有差值都为负，所谓最大差仍然是恶化；如果都为正，最小差也不应叫失败。固定三个连续窗口仍全部展示，避免只挑有利画面。现有70帧短片不足以支持跨场景泛化或真实米制准确率结论。

### 文件整理与完成门禁

当前课程入口为`run_dav2_video.sh`和`run_vda_video.sh`；1B、2B、比较三套产物分开。旧自录2B结果、专用自录入口、工作流、配置和对应回归已按用户要求移除；本页不再依赖它们。原始自录视频和旧1B原生结果仅作归档，指定片段、NYUv2和共享模型/指标组件保持不变。删除清单与源码指纹保存在`outputs/maintenance/self_recorded_part2b_cleanup_20261103/`，不改写旧执行历史。

实现验证使用合成预测和临时小文件，覆盖独立VDA无需DA、复用RGB不写缓存、官方窗口/NPZ、固定色标图视频、两条原生输出只读、同协议比较、负降低率/N/A、篡改拒绝和完整raw重算。合成测试通过只证明流程可用，不是正式VDA效果。

| 门禁 | 当前状态 |
|---|---|
| VDA独立运行/分析/核验，与1B同实现的指标和展示 | 当前70帧真实运行、分析与核验完成 |
| 官方仓库/relative权重/源SHA、实际CUDA | 预检通过，无下载 |
| 当前真实70帧VDA预测、效率、完整raw重算 | 已完成；本次独立复核通过 |
| 相同RGB/帧序/尺寸/精度下的DA/VDA正式比较 | 当前70帧已完成；双方raw再次重算通过 |
| 实际困难窗口与限制分析 | 独立及配对分析已完成，保留正负案例和成本 |
| 课程100真实帧素材 | 仅70帧，待教师确认或补充素材 |
| streaming/米制视频模式 | 按当前范围暂不开展 |
| 应用Demo、课程报告和最终提交 | 留后续整合 |
