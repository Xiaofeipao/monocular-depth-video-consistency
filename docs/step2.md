# Step 2 / Part 1B：指定片段的单目逐帧基线

## 任务与当前状态

依据[新版课程PDF §5.1、§6.1–6.2](../Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf)和[中文说明](../Project3_CN_final.md)，使用官方**Depth Anything V2-Small**，独立处理`data/davis_rollercoaster.mp4`的连续原始帧。输出标为**relative depth**，不估计米、不读取视频深度GT，不做逐帧尺度拟合或时间平滑。

本页主线是：准备指定输入 → DA-V2逐帧预测 → 保存浮点结果与固定色标视频 → 分析尺度约定、边界和时间变化。**本次不运行Video Depth Anything，也不作DA/VDA比较**；相同帧上的比较交给Part 2B。原DA-V2-NYUv2尺度对齐分支作为扩展诊断保留在本页后半部分。

当前指定文件的**全部70个真实帧**已经完成预测、可视化和原始重算核验。课程要求前100帧，但文件实际仅70帧、24 FPS、约2.917秒；没有重复、循环、插帧或换片凑数。因此“当前素材的逐帧基线”已具备，**课程100帧要求仍待教师确认70帧是否接受，或提供至少100帧的版本**。这项限制不会由`--max_len 100`自动解决。

## 输入与固定协议

| 项目 | 本次设置 |
|---|---|
| 原片 | `data/davis_rollercoaster.mp4`，官方Video Depth Anything示例副本；960×540、24 FPS |
| SHA256 | `7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8` |
| 选帧 | 按解码索引连续读取前`min(100,实际帧数)`；当前0–69，保留原始PTS，不使用FPS网格重采样 |
| 缓存画布 | 长边≤640，保持比例，当前640×360；供后续2B只读复用 |
| 模型 | 官方DA-V2-Small / ViT-S；本地固定commit与权重SHA，主环境`aiaa3201` |
| 网络输入与输出 | 官方预处理，`input_size=518`；实际H×W为518×924，恢复输出360×640 |
| 精度与设备 | CUDA / FP16，NVIDIA RTX A6000；逐帧独立前向，骨干不训练 |
| 原生预测 | 完整有限非负float32 NPY；arbitrary、近处数值通常较大，alignment:none、gt_used:false |
| 显示 | TURBO；从全部原生预测的P1/P99确定一次范围，整段固定；不逐帧归一化 |
| 保存 | 原始预测不裁剪；仅显示时截断固定范围外的颜色，并记录饱和比例 |
| 效率 | 首帧预热1次，每帧测3次，保存首次预测；时间包含模型预处理、传输、前向与输出恢复，不含解码/保存/绘图 |
| 辅助诊断 | CPU Farneback双向对应、FB≤1 px；不修改预测，不需要RAFT权重，也不是课程必做数值评价 |

原始PTS按视频提供的小数保存，显示帧率使用流中的精确24 FPS。连续索引选帧避免把0.083333这样的PTS舍入值放到24 FPS浮点网格上比较而误跳帧。不同视频不能靠修改路径或既有运行JSON拼成同一实验。

## 运行指令

命令从项目根目录执行；shell入口会自行定位根目录并选择`aiaa3201`。当前结果已经存在，阅读/检查它不需要再次预测。该分支只使用DA-V2，不要求先准备VDA或RAFT权重。

### 首次从空目录完整运行

```bash
# 只读检查指定片段、DA代码/权重、实际CUDA与原始帧数
./scripts/part1/run_dav2_video.sh check

# 抽连续原帧 → DA逐帧预测 → 图视频与诊断 → 完整raw重算
./scripts/part1/run_dav2_video.sh run
```

默认配置为`configs/part1/dav2_video.yaml`。已有目录不会覆盖，失败时保留run/status及已写出的原始产物，不自动把残缺阶段当作成功续跑。独立复现或教师提供新素材时，用新配置、源SHA和空目录，不把70帧记录改写成100帧：

```bash
./scripts/part1/run_dav2_video.sh run   --output outputs/part1/dav2_video/reproduce_v1   --cache data/video_frames/part1b_davis_reproduce_v1
```

仅更换输出位置可直接使用以上参数；更换视频或科学参数时，先复制配置并如实修改源路径/SHA，再通过`--config`指定新配置。

### 核验当前结果

```bash
./scripts/part1/run_dav2_video.sh verify   --run outputs/part1/dav2_video/rollercoaster_v1
```

`verify`不运行深度模型；它重新核对源/权重/源码、RGB、原始预测、帧序/PTS、整段归一化、CSV、报告SHA和2B交接记录，并从raw重新计算辅助指标。容差为`rtol=1e-6、atol=1e-8`。

### 只重新生成分析和图视频

```bash
./scripts/part1/run_dav2_video.sh analyze   --run outputs/part1/dav2_video/rollercoaster_v1   --analysis outputs/part1/dav2_video/rollercoaster_v1/analysis_replot_v1

./scripts/part1/run_dav2_video.sh verify   --run outputs/part1/dav2_video/rollercoaster_v1   --analysis outputs/part1/dav2_video/rollercoaster_v1/analysis_replot_v1
```

新分析目录必须不存在；它不重推理、不更改原始预测和原分析。长时间训练仍由用户启动，本任务不需要训练或微调。

## 结果与观看入口

- RGB / 相对深度 / 预测边缘同步视频（本地产物：`outputs/part1/dav2_video/rollercoaster_v1/analysis/visualization/rgb_relative_edges.mp4`）：70帧、24 FPS，同一固定色标；颜色不表示米。
- 全段总览（本地产物：`outputs/part1/dav2_video/rollercoaster_v1/analysis/visualization/contact.png`）、诊断曲线（本地产物：`outputs/part1/dav2_video/rollercoaster_v1/analysis/visualization/diagnostics.png`）、固定色标（本地产物：`outputs/part1/dav2_video/rollercoaster_v1/analysis/visualization/fixed_colorbar.png`）。红色是预测深度梯度，不是人工GT边界。
- 原生运行与效率（本地产物：`outputs/part1/dav2_video/rollercoaster_v1/run.json`）、完整精度诊断（本地产物：`outputs/part1/dav2_video/rollercoaster_v1/analysis/summary.json`）、原始重算核验（本地产物：`outputs/part1/dav2_video/rollercoaster_v1/verification.json`）。

实际恢复70份360×640浮点预测、70个源帧索引和69个相邻帧对。整段固定显示范围为`[0.00000000, 2.68554688]`；超过上界的160,584个像素仅在显示时饱和，约占全段0.995685%，原生数值未修改。

| 效率项目 | 实际结果 |
|---|---:|
| 平均 / 中位 / P95耗时（ms/帧） | 21.43496807 / 21.28723877 / 22.37827087 |
| 模型FPS | 46.65274036 |
| 峰值allocated / reserved（MiB） | 329.4043 / 404.0000 |
| 参数 / 权重字节 | 24,785,089 / 99,218,434 |
| 模型加载（秒） | 0.74890405 |
| 准备输入至原生保存阶段（秒） | 6.59967969 |

FPS是上述模型计时的倒数；端到端6.60秒另含准备输入、加载、预热、三次测量及I/O，不是“单遍实时播放速度”。报告与CPU诊断不计入模型FPS，显存是PyTorch实际分配/保留峰值，不是整卡占用。

## 分析一：为什么它不是正确的米制尺度

画面中天空、过山车轨道、支架和车体的前后层次总体可辨；这说明模型恢复了有用的相对结构，不能说明它知道物体距离相机几米。原生值的方向是“通常越大越近”，并不是“数值越大，米数越远”。因此不能把色标上的2.0标成2米，也不应在没有视频GT时给出“错了多少米”的误差表。

辅助尺度演示对同一原生输出仅作`2D+5`的数学变换，同时相应变换整段固定显示范围；其归一化显示与原图最大差约`5.95e-08`。这说明同一层次/颜色可以对应不同数字单位；演示没有作为新预测保存，更没有修改原生输出。视频里没有已知米制参考、深度GT或相机位姿，无法据此估计绝对尺度。后面的NYUv2对齐诊断用GT求评价映射，必须与这里无GT的relative身份分开。

## 分析二：边界稳定性与三个实际困难案例

先由全段诊断分数定位候选，再人工检查候选前后的连续RGB/深度；完整候选图与局部放大来源均保留。下面是本片确实存在的三个观察窗口，不要求虚构镜面、行人或其他未出现的场景。

| 观察窗口（0-based帧号） | 实际内容与观察 | 能支持的结论 |
|---|---|---|
| 65–69：薄结构与深度交界（本地产物：`outputs/part1/dav2_video/rollercoaster_v1/review/thin_boundary.png`） | 中央支架、轨道及车体外沿对着天空；大轮廓连续，细杆和车体边缘的预测厚度、局部色带有变化，远处旗杆较弱 | 细结构和遮挡轮廓仍是难点；局部厚度变化不等于测得了GT定位误差 |
| 52–56：运动列车与支架交叉遮挡（本地产物：`outputs/part1/dav2_video/rollercoaster_v1/review/occlusion_train.png`） | 列车从交叉杆件后经过，乘客与车体在相对图上合并成粗轮廓，局部颜色/锯齿变化；RGB里车体本身也在运动 | 逐帧输出缺少显式跨帧约束，动态小结构难以持续分开；不能把真实位移、运动模糊和显露变化全算成模型伪影 |
| 57–61：快速轮廓位移与显露区域（本地产物：`outputs/part1/dav2_video/rollercoaster_v1/review/moving_edge.png`） | 列车沿弯轨下降，画面随相机移动；稀细轮廓及下方新显露的远景变化，深度有局部形状/颜色波动 | 应结合RGB观看时序，而不能只看固定像素差就判定闪烁；本窗口也展示光流近似与遮挡的局限 |

三例都是同一短片内的窗口，不构成三个独立场景或覆盖所有困难类型。放大图使用同一画面坐标和同一整段色标，没有为某个案例单独调色，也没有把显示裁剪区域用作数值评价mask。来源、帧号、PTS、放大框和SHA见review_sources.json（本地产物：`outputs/part1/dav2_video/rollercoaster_v1/review/review_sources.json`）。

## 分析三：时序闪烁与运动伪影

这个片段中，主体杆架大轮廓总体连贯，天空大区域的输出也较平坦；**没有足够证据把它描述成全画面剧烈随机闪烁**。能直接看到的是列车/细杆附近局部颜色与轮廓的变化。它们值得保留给后续2B比较，但当前不能预先断言VDA会改善多少。

为帮助找变化位置，另外保存两组可选数值。记`S=P99−P1=2.68554688`，它是整段一次确定的相对输出跨度，不是米，也不是逐帧对齐参数。

| 辅助量（69个帧对等权） | 实际值 | 解释 |
|---|---:|---|
| 全图同坐标MAE/S | 0.09631664 | 不使用光流，包含真实物体/相机运动 |
| 全图同坐标RMSE/S | 0.23702812 | 对大变化更敏感，不能单独叫作纯flicker |
| Farneback对应MAE/S | 0.01294299 | FB有效域上比较，不做深度scale/shift拟合 |
| Farneback对应RMSE/S | 0.04071483 | 含近似光流与遮挡残差，不是时序GT误差 |
| 光流有效像素比例 | 0.91111526 | 越界与FB不一致位置不参加对应项 |
| 边缘变化proxy | 0.00339027 | 对应后边缘在2 px容差内的双向未匹配比例；不是GT边界F1 |

同坐标项较高，而近似对应后的项较低，与画面确有相机/列车运动相符；由于两者有效域不同，不能将两者之差当作被“消除的闪烁”。大量天空及平坦相对输出会压低全图均值，低边缘变化proxy也不证明乘客或细杆细节已经准确恢复。辅助数值只定位案例、补充解释，课程的视频主结果仍是同帧、固定显示下的定性观察。

### 怎样看诊断图 `diagnostics.png`

诊断图（本地产物：`outputs/part1/dav2_video/rollercoaster_v1/analysis/visualization/diagnostics.png`）从上到下有四张子图；`diagnostics.pdf`是同一图的导出版本。横坐标均为原视频时间戳PTS，单位是秒。第一张每个点对应一帧，共70个点；下面三张每个点对应一对相邻帧，共69个点，放在后一帧的时间位置。例如横坐标`2.791667 s`对应第66→67帧，而不是“从开头累计到此刻的平均误差”。帧号从0开始，曲线用直线连接实际采样点，没有增加帧。

| 从上到下 | 图上名称与数据列 | 这条曲线在问什么 |
|---|---|---|
| ① 原生相对值中位数 | `Raw relative median`；`frame_diagnostics.csv`的`median` | 每帧所有像素的预测值排序后，中间那个数是多少？用于观察整体数值分布，但不是相机距离或米制尺度。 |
| ② 相邻帧变化 | `Pair RMSE / fixed S`；蓝线`pixel_rmse_normalized`，橙线`flow_rmse_normalized` | 不寻找对应位置、以及先近似寻找对应位置时，两帧预测分别变化了多少？ |
| ③ 边缘变化 | `Edge change proxy`；`pair_diagnostics.csv`的`edge_instability` | 经过近似运动对应后，有多少预测边缘找不到邻近的对应边缘？ |
| ④ 光流有效覆盖 | `Flow valid fraction`；`pair_diagnostics.csv`的`flow_coverage` | 本帧对有多少像素可以参加光流对应后的深度变化计算？ |

**① 为什么第一张是一条零线？** 当前CSV中70帧的中位数确实全部为0，不是绘图漏画。完整非负相对预测中的零值合法；这一统计量被占较大面积的零值输出主导，结合画面中的大面积天空理解即可。纵轴上的负刻度只是绘图库对常数零线自动留白，不表示原生预测有负值。中位数只看数值排序的中间位置，即使列车、支架等局部预测变化很明显，中位数也可能不动。因此，零线不表示“所有像素深度都是0”，不表示“距离相机0米”，也不能作为“没有尺度漂移或闪烁”的证据。这里没有视频深度GT，不能从这条线测得真实尺度漂移。

**② 蓝线和橙线有什么区别？** 蓝线`Same pixel`直接比较两帧同一屏幕坐标；如果列车驶过某个像素，该像素从天空换成车体，合理的深度变化也会被计入。橙线`Farneback + FB`先从RGB估计运动，尽量找到两帧同一画面内容的对应位置，再比较预测；FB是前向、后向运动是否一致的检查，本次阈值为1像素，越界或不一致位置不参加橙线计算。

两条线都使用RMSE：先计算两帧预测差，平方后求平均，再开平方，最后除以整段固定跨度`S=2.685546875`。MAE则是对差值取绝对值后直接求平均；较大的局部变化在平方后影响更大，因此RMSE比MAE更敏感。蓝线对全图计算，橙线只在光流有效域计算；这张图**没有画MAE曲线**，MAE/S仍可在CSV和上表中查看。计算使用原生预测，不把显示时的颜色截断用于误差计算，也不逐帧拟合尺度。

当前蓝线平均`0.23702812`、橙线平均`0.04071483`。例如橙线为`0.04`，表示这一帧对的对应后变化RMSE约为固定跨度S的4%，不是4%的像素预测错误，也不是4%的真实深度误差。当前蓝线在视频后段明显上升，而橙线大体保持较低水平，与画面运动影响相符；不是两个不同模型的比较，也不表示光流改善了原始预测。橙线的波动仍混合了预测变化、实际视角变化、遮挡和光流误差，没有一个现成阈值可以将它直接判为“严重闪烁”。

**③ 边缘变化曲线的高低怎么读？** 先把上一帧的预测边缘按光流对应到当前帧，再在有效区域内双向查找：一侧的边缘能否在另一侧的2像素邻域中找到对应边缘？两侧未匹配比例取平均，得到该点的值。0表示在本次有效域和容差下全部能匹配；越高表示不匹配的比例越高。没有有效边缘时记为不可用，不强行记0。

当前平均为`0.00339027`，即约0.3390%；最高点在第66→67帧、`2.791667 s`，为`0.00827897`，即约0.8279%，可以回看65–69帧的放大图。这些百分比的分母是参与比较的预测边缘，不是全图像素，更不是真实物体边界。曲线低只能说明这些预测边缘较一致，不能证明它们位置准确；2像素以内的抖动、两帧一致的错误边缘，以及少量细结构问题都可能被掩盖。

**④ 最下面是不是准确率？** 不是。它是通过光流越界和前后向一致性检查的像素数除以全图像素数。当前平均为91.1115%，范围约87.3681%–95.4887%。数值下降表示可用于对应比较的位置减少，可能与遮挡、出画或光流不可靠有关，不能直接认定深度预测变差；天空等弱纹理区域通过检查，也不保证光流一定正确。读橙线时应同时看这张图：如果橙线下降、有效域也减少，就不能仅凭橙线认为时序变好了。边缘指标还会额外收缩有效域，避免2像素匹配窗口借用无效区域。

实际阅读时，先在②和③找到变化较大的时间，再看④确认对应覆盖情况，最后回到相同时间的RGB、深度视频及局部图。曲线用于告诉我们“这里值得看”，是否为模型闪烁、真实运动或遮挡，需要结合画面判断。

### 红色预测边缘的来源与粗细

三栏视频的第三栏是RGB上叠加红色的**预测深度变化明显区域**。DA-V2只输出浮点相对深度图，没有单独预测一张物体边界图。这里由项目分析程序从深度图提取边缘：

```text
原生相对深度 / 整段固定跨度S
  → Gaussian轻微平滑（sigma=0.8像素）
  → Sobel计算横向、纵向变化，幅值组合为梯度强度
  → 梯度强度达到固定阈值0.02的像素标红
```

Sobel使用3×3核并除以8，阈值作用于这一定义下的梯度；它不是“相邻两点深度恰好相差0.02米”。平滑和梯度计算只用于诊断，不修改原生预测。叠加图直接标记满足阈值的像素，没有再人为加粗；边缘匹配指标中的2像素容差，也不是这张红色图的绘制线宽。

因此可以把红色理解为“模型预测的深度分界线索”，但不能把它当作模型明确识别的物体类别边界或GT。例如车体与天空之间通常有较大深度差，容易出现红色；同一深度平面上的花纹未必应该有红线，模型错误地把纹理当深度变化时却可能出现红线。某处没有红线，也可能只是预测深度差较小、过渡梯度未达到阈值，不能直接说模型没识别出物体。

红色有时是一条窄线，有时是一条较宽的带，表示达到梯度阈值的像素区域有宽有窄。它同时受深度变化幅度、过渡形状、相邻边缘叠加、输出插值和平滑影响；**不是物体实际厚度、置信度或错误大小，越粗也不必然越模糊**。例如较宽的深度过渡可能形成红带，深度跳变量增大也可能让固定阈值覆盖更多像素；很细的杆件两侧边缘靠近，还可能合成一条较粗的带。

判断不稳定时，应跟着RGB里的同一个物体看连续帧：实际轮廓没有相应变化，预测深度过渡却突然变宽、缩窄、偏移、断裂或消失，才是值得讨论的局部现象。同时检查彩色深度图；单看红线变粗，无法区分深度跳变幅度、过渡宽度和阈值效应。物体或相机运动、遮挡和RGB模糊造成的正常变化仍需单独考虑。

## 架构、产物与后续2B输入

| 文件 | 职责 |
|---|---|
| `configs/part1/dav2_video.yaml` | 指定素材、DA-S、前100真实帧上限、原FPS、640画布和固定显示 |
| `scripts/part1/run_dav2_video.sh`、`pipelines/part1/dav2_video.py` | check/run/analyze/verify；只调用DA-V2 |
| `src/evaluation/course_video_inputs.py` | 连续原始索引和PTS、来源SHA、只准备一套RGB |
| `src/evaluation/framewise_clip.py` | 模型运行、来源/源码快照、原生保存、计时、分析与核验 |
| `src/metrics/video_diagnostics.py` | 固定整段单位、可选Farneback、边缘变化与候选定位 |
| `src/visualization/framewise_clip.py` | RGB/relative/边缘视频、连续案例、固定色标和曲线 |

```text
data/davis_rollercoaster.mp4
data/video_frames/part1b_davis_v1/
  manifest.json、rgb/                    连续原始帧/PTS/SHA与640画布
outputs/part1/dav2_video/rollercoaster_v1/
  protocol.json、run.json                实际输入/配置/源码归档、模型/效率
  raw/*.npy                             70份未对齐原生relative
  verification.json                    完整原始重算
  analysis/
    summary.json、status.json、analysis.md
    frame_diagnostics.csv、pair_diagnostics.csv
    handoff_2b.json                      后续2B只读引用的RGB/DA身份与显示约定
    visualization/                      同步视频、总览、曲线、色标和候选图
  review/                               实际查看的连续局部图、来源及复绘脚本
```

handoff_2b.json（本地产物：`outputs/part1/dav2_video/rollercoaster_v1/analysis/handoff_2b.json`）明确`vda_executed:false`。后续2B应使用同一份RGB manifest、真实帧号/PTS和输出画布，保留DA原生预测；VDA可有自己的整段固定归一化，但同TURBO颜色不代表共享米制尺度。1B与2B运行入口及产物继续分开。

旧自录2B产物及其专用自录入口/工作流/配置已按用户要求清理；当前指定片段与NYUv2分支不受影响。原始自录视频、旧1B原生结果和历史执行记录仅作归档，不再属于当前运行主线；实际删除清单见`outputs/maintenance/self_recorded_part2b_cleanup_20261103/`。

## 完成门禁

| 项目 | 状态 |
|---|---|
| DA-V2-S独立逐帧、无视频GT/米制对齐、原生保存 | 当前70帧已完成并核验 |
| 原FPS/连续帧/固定色标/真实效率/三个实际案例 | 当前70帧已完成 |
| 尺度约定、边界和局部时序现象的诚实分析 | 本页已补充；不强行声称全局严重闪烁 |
| 课程要求100个真实原始帧 | 素材仅70帧，待教师确认或补充版本 |
| 与VDA同帧比较 | 当前70帧已在Part 2B完成并重算，配对分析见[Step 4第二部分](step4.md) |
| 最终应用Demo、报告和ZIP交付 | 留给后续整合 |

---

## NYUv2：DA-V2-S的Scale Alignment

本节是保留的扩展诊断，不是新版课程的逐帧视频主线，也不是Part 2A的原生米制结果。以下结果与核验描述沿用该分支已有审计；本轮只读保留，未重新运行NYUv2模型、对齐或评价，原始预测、协议和来源均保持原样。

### 基本信息与协议


使用[configs/part1/dav2_nyuv2.yaml](../configs/part1/dav2_nyuv2.yaml)。NYUv2数据、官方split和DA-V2-S权重已在本地，预检核对SHA256与官方仓库commit，不重新下载。

| 项目 | 固定设置 |
|---|---|
| 模型 | 官方DA-V2-Small，24.8M参数，relative权重；不使用NYUv2微调的metric checkpoint |
| 数据划分 | labeled MAT的标准`testNdxs`，654张；MATLAB one-based index保持原顺序 |
| 模型输入 | 完整480×640 RGB；input_size=518，实际网络H×W=518×686；预测恢复到480×640 |
| 保存 | 原生float32、inverse-depth-like、arbitrary、`alignment:none`；合法零值保留 |
| 评价crop | Python半开区间`[45:471, 41:601]`，仅在对齐与评价时使用，不裁剪模型输入 |
| 有效GT | 有限且`0.001 < GT < 10 m`；对齐与评分使用同一组像素 |
| 对齐 | 每张图在逆深度域做非负尺度的scale-and-shift最小二乘拟合 |
| 转换与范围 | 拟合逆深度限制到`[1/10,1/0.001]`后取倒数；保留全部有效GT像素，记录非正拟合值及裁剪数量 |
| 指标 | AbsRel、SqRel、RMSE、RMSE-log、δ1/δ2/δ3，逐图宏平均；10,000次图像级Bootstrap 95% CI，seed=42 |
| 效率 | 实际NVIDIA RTX A6000、CUDA/FP16，预热1次；每图计时3次并保留第一次输出 |

通俗地说，模型先只看RGB给出相对远近，评价器再用GT给这一张预测确定单位和零点，随后检查它的结构与GT是否吻合。DA-V2输出近似逆深度，因此不能把原生数组直接当米制深度缩放：

```text
r = 模型原生相对预测
q = 1 / GT
(a,b) = argmin Σ有效crop像素 (a*r + b - q)²，a≥0
评价深度 = 1 / clip(a*r+b, 1/10, 1/0.001)
```

原生数组不改写。常数预测或无正向相关时，用约束下的常数解`a=0,b=mean(1/GT)`，如实记录原因，并让这张图继续参与汇总；不会丢弃失败图或拟合后非正的像素来改善分数。输入非有限/负值、空有效域等实际错误会明确失败。

这是本项目明确固定的相对深度评价协议：尺度/偏移最小二乘思想参考[官方SSI实现](https://github.com/isl-org/ZoeDepth/blob/main/zoedepth/trainers/loss.py)，在既有NYUv2 split/crop/范围上实施。非负约束和退化处理均在此披露，不声称复现DA-V2论文的官方表格数字。[DA-V2官方说明](https://github.com/DepthAnything/Depth-Anything-V2)区分relative和metric模型；本分支始终使用relative权重。

**这些指标使用了同一测试图GT确定a、b，衡量的是对齐后的结构质量，不是模型直接预测米制尺度的能力。** RMSE按GT尺度以米表示，但方法身份仍为`aligned_relative`；不得与Part 2A的native metric结果混为同一无GT条件下的排名，也不得把对齐参数用于自录视频。

### 运行指令

当前完整产物位于`outputs/part1/dav2_nyuv2/frozen_v1/`，用户已完成预测、评价与重新评价。下面从项目根目录执行；脚本使用`aiaa3201`。**复现请使用新目录，不能覆盖frozen_v1。**

```bash
set -euo pipefail

./scripts/part1/run_dav2_nyuv2.sh check \
  --output outputs/part1/dav2_nyuv2/checks_reproduce_v1/resources.json

# 完整654张：原生预测 → GT辅助对齐评价/图像 → 原生重算核验
./scripts/part1/run_dav2_nyuv2.sh run \
  --output-dir outputs/part1/dav2_nyuv2/reproduce_v1
```

中断恢复和独立重评可分别执行。只有预测阶段中断才需要`predict --resume`；评价不重新加载模型，输出目录须不存在。改变科学源码或协议时使用新版本，不能拼入旧run。

```bash
./scripts/part1/run_dav2_nyuv2.sh predict --resume \
  --output-dir outputs/part1/dav2_nyuv2/reproduce_v1

./scripts/part1/run_dav2_nyuv2.sh evaluate \
  --run-dir outputs/part1/dav2_nyuv2/reproduce_v1 \
  --output-dir outputs/part1/dav2_nyuv2/reproduce_v1/evaluation_recheck_v1

./scripts/part1/run_dav2_nyuv2.sh verify \
  --evaluation-dir outputs/part1/dav2_nyuv2/reproduce_v1/evaluation_recheck_v1 \
  --output outputs/part1/dav2_nyuv2/checks_reproduce_v1/recheck.json
```

仅核验当前正式结果时，运行以下只读命令，无需重新预测或评价，也不覆盖已有核验记录：

```bash
./scripts/part1/run_dav2_nyuv2.sh verify \
  --evaluation-dir outputs/part1/dav2_nyuv2/frozen_v1/evaluation
```

### 654张完整结果

以下从正式`evaluation/metrics_summary.json`读取完整精度，再按8位小数显示；逐图CSV重新聚合一致，`evaluation_recheck_v1`的七指标、CI和CSV也一致。全部654张、215个scene参与统计，每图238,560个有效crop像素，共156,018,240个。

| 指标 | 逐图宏平均 | 图像级Bootstrap 95% CI |
|---|---:|---:|
| AbsRel↓ | 0.06171408 | [0.05854527, 0.06501292] |
| SqRel↓ | 0.03791699 | [0.03294910, 0.04328549] |
| RMSE↓ (m) | 0.29735612 | [0.27833497, 0.31686912] |
| RMSE-log↓ | 0.09352266 | [0.08884892, 0.09842145] |
| δ1↑ | 0.95540346 | [0.94927470, 0.96124863] |
| δ2↑ | 0.98783715 | [0.98518574, 0.99018256] |
| δ3↑ | 0.99578876 | [0.99441998, 0.99694811] |

AbsRel=0.06171408表示对齐后逐图平均相对误差约6.17%；δ1=0.95540346表示逐图平均约95.54%的像素倍率误差小于1.25。**这些数使用了每张测试图的GT来确定尺度与偏移，不能解释为模型直接给出了同等水平的绝对米制深度。** 不能因其低于Part 2A某些native指标，就声称DA-V2的无GT米制能力更好。

AbsRel中位数为0.04970669，均值高于中位数，存在较难样本形成的高误差尾部。描述性统计中，82张AbsRel>0.10、5张>0.25，6张δ1<0.60；这些只是分布说明，不用于删图或重新选参。CI按冻结协议以图像为单位，包含同一scene的相关图像，不代表场景级或跨数据集不确定度。

### 对齐行为与代表性问题

**1．大多数图可通过正尺度和偏移完成对齐，但全局两个参数无法修正局部结构错误。** 653张得到正尺度拟合，没有因为难图而剔除样本。浴室镜面附近、遮挡与远景深度压缩仍有明显误差。拟合最小化的是逆深度平方误差，与直接最小化AbsRel或深度RMSE不是同一个目标。

**2．第266张出现真实的负尺度拟合，常数解必须保留并解释。** `bathroom_0008`（MAT索引650）的无约束尺度为−0.04683307。协议要求保留“数值越大通常越近”的方向，因此采用预先规定的约束解`a=0,b=0.53564013`，对应整图评价深度约1.86692510 m。它的AbsRel为0.28713603，仍参加全部654张平均。平坦结果来自评价器的退化处理，**不表示模型原生预测本身就是常数，也不表示结构恢复成功**。

**3．远端范围裁剪较少，但仍需报告。** 26张图共有249,856个拟合逆深度低于0.1，转换后按最大深度10 m处理，约占全部有效像素的0.160145%。没有拟合逆深度≤0，也没有超过1000的值。这些像素保留在评分中，不能当作缺失或从分母移除；0.160145%不是误差率。没有常数输入触发的退化，唯一退化图是上面的负尺度案例。

| 缓存test位置 / scene | AbsRel | δ1 | 查看与解释 |
|---|---:|---:|---|
| 310 / bathroom_0046 | 0.34122097 | 0.15456070 | 案例图（本地产物：`outputs/part1/dav2_nyuv2/frozen_v1/evaluation/visualization/test_0310.png`）：镜面及其周围区域的GT和预测差异大；全局对齐无法修复局部不一致 |
| 266 / bathroom_0008 | 0.28713603 | 0.53367287 | 案例图（本地产物：`outputs/part1/dav2_nyuv2/frozen_v1/evaluation/visualization/test_0266.png`）：约束后的常数评价结果；不能把它作为正常层次恢复展示 |
| 447 / bedroom_0085 | 0.27964143 | 0.44479795 | 案例图（本地产物：`outputs/part1/dav2_nyuv2/frozen_v1/evaluation/visualization/test_0447.png`）：前景轮廓可辨，但背景/远处层次被压缩，绝对误差仍较大 |

这些观察来自实际查看RGB/GT/预测/误差图，不仅依据分数。没有将所有反光处差异归因于模型，GT在复杂反射场景中的语义也需谨慎解释。完整contact sheet（本地产物：`outputs/part1/dav2_nyuv2/frozen_v1/evaluation/visualization/contact.jpg`）包含固定位置0/326/653和AbsRel最高6图，共9张，不是全部654张可视化。

按场景类型的补充统计如下，完整16类结果见metrics_by_scene_type.csv（本地产物：`outputs/part1/dav2_nyuv2/analysis_v1/metrics_by_scene_type.csv`）。这是同一NYUv2内的描述性分组，不是另一个数据域，也不改主结果的逐图权重。

| 场景类型 | 图像数 | AbsRel | RMSE (m) |
|---|---:|---:|---:|
| bedroom | 191 | 0.04848230 | 0.17945597 |
| living_room | 107 | 0.05599062 | 0.31028422 |
| kitchen | 106 | 0.06265542 | 0.31887104 |
| bathroom | 58 | 0.10348581 | 0.38439111 |
| bookstore | 11 | 0.15352100 | 1.09650118 |

卧室、客厅和厨房占较多样本；浴室和书店在当前结果中更难。书店仅11张，不能据此断言对所有书店场景都失效；每类数量不均也意味着全体均值不能代替逐类表现。

### 运行效率

实际环境为`aiaa3201`、Python 3.10.22、PyTorch 2.1.1+cu121、NVIDIA RTX A6000、FP16；网络实际518×686，参数24,785,089，权重99,218,434 bytes。

| 平均耗时 (ms/图) | 中位数 (ms) | P95 (ms) | 模型FPS | 峰值allocated / reserved (MiB) |
|---:|---:|---:|---:|---:|
| 18.64646556 | 18.59909218 | 19.21165398 | 53.62946649 | 314.9014 / 326.0000 |

耗时取每图三次测量的均值，再跨图统计，包含模型输入处理、设备传输和输出恢复；不含数据读取、GT对齐、保存或绘图。全体三次测量累计36.58436543秒，读取样本至原生保存阶段73.50841505秒，模型加载另为0.66774602秒。73.51秒包含预热、三次测量和I/O，不能当作单遍实时系统的耗时；峰值显存是PyTorch分配/保留值，不是整卡占用。

### 正式产物与核验

```text
outputs/part1/dav2_nyuv2/
  frozen_v1/
    run.json, provenance/              原生身份、资源/源码SHA、命令和真实环境
    predictions/*.npz                  654份原生raw_relative，未GT对齐
    evaluation/                        主评价：654份aligned、CSV/汇总、9张案例与contact
    evaluation_recheck_v1/              用户独立重评的完整产物，结果与主评价一致
    reload_validation.json             用户运行时的654张原始重算记录
  checks_user_v1/                       用户资源检查与重评核验记录
  analysis_v1/                          本次只读复核、完整精度分析、分类统计及清理清单
```

本次对主评价及独立重评均重新执行`verify`，654张的raw、资源/源码、对齐数组/mask、CSV、宏平均和CI通过`rtol=1e-6 / atol=1e-8`核验。另对0/326/653、266/310/447及裁剪较多的45共7图，用独立`numpy.linalg.lstsq`和显式七指标公式复核，通过。

详细来源、SHA和全精度统计见results_analysis.json（本地产物：`outputs/part1/dav2_nyuv2/analysis_v1/results_analysis.json`），实际过程见review.py（本地产物：`outputs/part1/dav2_nyuv2/analysis_v1/review.py`）。本次没有运行模型、训练或下载，也没有改写正式预测和自动运行元数据。

### 项目架构

| 文件 | 职责 |
|---|---|
| `configs/part1/dav2_nyuv2.yaml` | 固定split、模型、crop、对齐与评价设置 |
| `scripts/part1/run_dav2_nyuv2.sh`、`pipelines/part1/dav2_nyuv2.py` | check/predict/evaluate/verify/run环境入口与编排 |
| `src/evaluation/relative_nyuv2.py` | 资源/源码绑定、原生保存、评价、汇总、案例图与重算 |
| `src/metrics/relative_alignment.py` | 逆深度尺度/偏移拟合、约束与退化处理 |
| 既有NYUv2 loader、DA-V2 adapter与depth metrics | 复用数据读取、官方模型推理和七指标，Part 2A协议不变 |

