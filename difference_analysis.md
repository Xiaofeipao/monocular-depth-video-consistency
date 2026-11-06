# 新版课程要求与当前Part 1A、2A、1B、2B的差异

核对日期：**2026-11-02**。要求依据为5页新版[Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf](Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf)，完整译文见[Project3_CN_final.md](Project3_CN_final.md)。旧版6页PDF与[Project3_CN_new.md](Project3_CN_new.md)保留作历史参考。

本文对照课程原文、项目配置、源代码和当前`outputs`记录。**不评价现有Part 3算法、训练、消融或结果，也不调整它们。**下文将课程的经典双目、逐帧视频、静态米制和视频模型分别称为1A、1B、2A、2B；这四个名称是项目的细分方式，PDF使用Part 1/Part 2及其子项。

**1A实施进度补充（2026-11-02）**：四组全有效GT域评价、显式毫米转米和全画面无效/LR诊断已实现，并通过合成流程验证；完整搜索与正式评估仍待用户运行，见[Step 1](docs/step1.md)。默认输出现为`tuning/`与`evaluation/`，辅助指标含Bad-1/2/4、Depth AbsRel/RMSE；原两个1A目录已按用户要求删除，指纹留存于维护记录。下文“当前情况”保留本次最初只读审查的时间范围。

**1B实施进度补充（2026-11-02）**：指定`data/davis_rollercoaster.mp4`的70个真实原始帧已完成DA-V2-S独立逐帧运行、固定色标视频、尺度/边界/时间分析和raw重算，见[Step 2](docs/step2.md)。NYUv2对齐扩展保留；课程100帧素材矛盾不因完成70帧而消失，VDA比较按用户安排留后续2B。未绑定的重复1B入口/配置已清理，现有2B依赖的旧1B文件只读保留；下文最初审查的结果仍按原审计范围理解。

**2B实施进度补充（2026-11-03）**：指定片段的官方VDA-S离线relative/FP16独立流程及另行DA/VDA比较已实现，复用1B指标实现、相同RGB/PTS/尺寸规则、整段固定色标和三个连续局部窗口；资源及合成核验通过。正式VDA预测、同帧比较和观看后的效果分析待用户运行，见分为两部分的[Step 4](docs/step4.md)。streaming与米制视频模式按当前范围暂不开展，旧绑定来源不改写，100帧素材限制仍保留。

**自录2B清理补充（2026-11-03）**：用户明确要求后，旧7段自录2B结果及专用自录工作流/配置已删除，当前课程流程不再保留这些依赖。原始自录视频和旧1B原生结果仅作归档；下文对旧7段的审计是删除前历史，不能据此认为其2B产物仍存在。逐文件SHA和删除范围见`outputs/maintenance/self_recorded_part2b_cleanup_20261103/deletion_manifest.json`。

## 1. 先看结论

现有工作大部分可以保留。**2A基线的科学协议与新版吻合；1A要补全GT区域的评价；1B/2B要补指定官方片段的比较。**自录视频与DA-V2的NYUv2尺度对齐属于额外工作，不再是新版必做项。

| 项目 | 新版要求 | 当前实际情况 | 判断及下一步 |
|---|---|---|---|
| 1A：方法、数据、几何 | 15组Middlebury trainQ；BM或SGBM；正确处理`/16`、`doffs`和毫米转米 | 四组BM/SGBM、15场景、60份预测已存在；几何实现正确 | 可以复用模型输出和几何分析，不必推翻 |
| 1A：评价 | 在有限、正值GT视差上报告MAE与有效预测比例 | 当前主表额外限制为非遮挡GT区域 | **需重评已有预测**；旧分数不能改名当作全GT结果 |
| 2A：NYUv2米制模型 | 选一个模型，RGB-only，654张、课程crop/范围、无测试尺度对齐 | Depth Pro和UniDepthV2-S均已有654张native metric结果，同协议、无额外K | 基线主体满足；选定报告主模型，第二模型作额外对照即可 |
| 1B：逐帧视频 | DA-V2-S独立处理指定片段的同一批帧，输出relative，不做GT对齐 | 有7段自录DA-V2产物；另有NYUv2的GT辅助对齐基准 | 模型与原生relative约定可复用；**仍缺指定片段轨道**；NYUv2对齐表不纳入新版米制主表 |
| 2B：视频模型 | VDA-S、relative、offline、FP16，与DA-V2-S在指定帧上比较 | 有7段自录VDA配对产物，已有同帧报告、浮点预测与固定显示 | 大部分基础设施可复用；**自录结果不能替代指定片段要求** |
| 视频数值诊断 | 只强制定性比较；flow误差、尺度漂移等可选 | 已实现pixel/flow MAE、RMSE、背景std与中位数proxy | 都可保留为补充，不再是课程视频完成的必要前置 |
| 视频素材 | `davis_rollercoaster.mp4`前100帧、原帧率 | 官方文件已在本地，但实测仅70帧；当前GitHub文件与本地为同一Git对象 | **课程文字与素材不一致，需教师确认**，不能靠重复帧补成100帧 |

“当前实际情况”来自本次读到的文件，不沿用README/PLAN里尚未更新的“全部视频等待运行”描述；它也不表示本次重新推理或重算了这些实验。

## 2. 新旧要求最主要的变化

| 主题 | 旧说明与此前实施方向 | 新版要求及影响 |
|---|---|---|
| 视频来源 | 自录固定/运动相机视频；`mandatory clips`来源未列明 | 指定官方`davis_rollercoaster.mp4`，不要求自录或额外数据集；不再等待一份未定义的mandatory列表 |
| 1B尺度对齐 | 允许在benchmark评价中做median或scale-and-shift alignment；本项目据此做了DA-V2-NYUv2 | 1B只做视频relative baseline，不估米、不用测试GT对齐；NYU米制比较由2A独立模型承担 |
| 视频结论 | 要数值时序误差与median-scale drift | 强制定性比较；数值时序误差、漂移、光流与depth-edge P/R均可选；不能用无GT视频宣称米制精度 |
| 视频模型 | Small或Base、relative或metric，可用batch/streaming | 必需轨道明确为**Small、relative、offline、默认FP16**；streaming与metric-video只作可选 |
| 静态模型与分析 | 一个米制模型，并强调四项关键分析 | 仍只需选一个；重点是课程协议下的原始米制精度、绝对尺度/边界误差解释及与改进的公平比较；四项独立分析不再逐项列为必做 |
| NYUv2协议 | 旧PDF指向外部工具/官方协议，本项目自行冻结细节 | 新PDF直接给出795/654、MAT读取方式、crop、范围、等图权重与无对齐，恰与当前2A主协议一致 |
| 双目评价 | 原路线与本项目以非遮挡误差、Bad-2、coverage等为主 | 新版明确要求有限、正值GT上的视差MAE与有效预测比例，需补对应统计 |
| 公平比较 | 容易被理解为各模型内部输入必须完全相同 | 新版要求原方法及其改进同测试/评价/输出设置，同时**保留每个预训练模型自身要求的预处理**；不必强迫Depth Pro与UniDepth内部网络尺寸相同 |
| 困难案例 | 提示镜子、玻璃、天空、模糊、细物体等 | 从提供数据中挑**实际存在的3个代表性困难案例**，不要求凑齐特定场景类型 |
| arXiv | 必须上传并给ID | 改为**可选**；PDF报告与公开代码仍必须 |

新版不是要求把此前额外实验删掉。可以保留它们的真实来源和作用，同时将报告主线收敛到新规定的数据与协议。

## 3. Part 1A：几何正确，主评价区域需要补齐

### 已经吻合的内容

[配置](configs/part1/middlebury_sgbm.yaml)使用Middlebury四分之一分辨率的15个训练场景。磁盘目录名`trainingQ`与PDF使用的`trainQ`指向这套四分之一分辨率训练数据，不是误用了benchmark测试集。

当前有`bm_fixed`、`bm_tuned`、`sgbm_fixed`、`sgbm_tuned`四组，共60份`prediction.npz`。新版只要求BM或SGBM，四组对照与5场景开发/10场景保留划分都属于额外实验，可以继续保留。课程要求的15组整体结果仍需给出，不能只报10个保留场景。

[matching.py](src/stereo/matching.py)先将OpenCV视差除以16；[io.py](src/stereo/io.py)将标定基线由毫米转米；[geometry.py](src/stereo/geometry.py)使用：

\[
z_{\mathrm{m}}=\frac{f_{\mathrm{px}}B_{\mathrm{m}}}{d_{\mathrm{px}}+d_{\mathrm{offs,px}}}.
\]

这是新版要求的等价形式，不需要改成忽略`doffs`的`fB/d`。左右一致性、Bad-1/2/4、深度误差和距离敏感性可继续作为诊断；新版将左右一致性改为可选，不代表必须关掉它。

### 确实存在的评价差异

当前[run_middlebury.py](pipelines/part1/run_middlebury.py)构造的是：

```python
evaluation_mask = scene.non_occluded_mask & target_geometry_valid
```

随后[metrics.py](src/stereo/metrics.py)再筛选有限、正值GT。也就是说，当前只在**非遮挡有效GT子集**上统计，而新版§6.2写的是`trainQ`中**有限、正值GT视差**，没有再要求非遮挡mask。

两个概念必须分开：

- **评价域：**哪些GT像素应纳入统计。新版主域应为`isfinite(GT) & (GT > 0)`，不能额外排除遮挡GT。
- **预测有效性：**某个位置的预测是否可用。现有LR过滤可以继续定义预测有效性，但被过滤的预测要在coverage分母中如实体现。

一种清楚的报告约定是：MAE在主评价域内有有效预测的位置上计算；coverage为该域内有效预测数除以全部有效GT数；同时注明无效预测未计入MAE，不把MAE单独当作整体质量。新版要求说明无效处理方式，没有指定必须给无效预测一个固定MAE惩罚。

下面是**现有非遮挡域、15场景等权平均**，只用于识别当前成果，不是新版全GT分数：

| 方法 | 当前视差MAE（px）↓ | 当前有效预测比例↑ |
|---|---:|---:|
| bm_fixed | 0.97901828 | 54.35600391% |
| bm_tuned | 1.73140054 | 65.94842901% |
| sgbm_fixed | 1.05875355 | 69.66939250% |
| sgbm_tuned | 1.30701556 | 71.68935922% |

最初审查来源：`outputs/part1/middlebury/frozen_v2/metrics_summary.json`（目录后经用户授权删除，文件指纹留存）的各方法`all`字段。旧`reload_validation.json`为60份预测、1,200个指标的既有通过记录；它验证旧协议的一致性，不能证明新版全GT协议已评估。

### 要补什么，是否需要重新运行BM/SGBM

已有NPZ包含`disparity_px`、`raw_disparity_px`、`valid_mask`和`lr_consistent`。**保持预测和有效规则不变时，可以直接用现有预测及GT补算全GT主表，通常无需重新匹配或重新调参。**原非遮挡主表留作额外诊断。

这是后续应新增的独立评价，不应覆盖`frozen_v2`的CSV、JSON或重载记录。本次没有计算全GT新分数，不能预先声称数值会改善或恶化。

## 4. Part 2A：现有两条原生米制基线可直接保留

### 与新版一致的科学协议

核对[Depth Pro配置](configs/part2/depth_pro_nyuv2.yaml)、[UniDepthV2配置](configs/part2/unidepth_v2_nyuv2.yaml)、[NYUv2 loader](src/data/nyuv2.py)及正式summary：

- 标准`testNdxs`，654张；MATLAB索引减1。
- RGB为`images[i].transpose(2,1,0)`，GT为`depths[i].T`，使用填充的米制GT，不用`rawDepths`。
- 输入原始完整RGB，预测输出`480×640`；评价时才取`[45:471,41:601]`。
- `0.001 < GT < 10 m`；仅评价时将预测clip到`[0.001,10] m`，原始浮点预测保留。
- 逐图计算后在654张上等权平均。
- `metric / meter / alignment:none`；主轨道仅RGB，无外部标定K/焦距、无测试GT尺度拟合。

两模型各有654份正式NPZ。新版的三个必报指标如下，**从当前完整精度JSON读取后格式化**：

| 原生米制模型 | AbsRel↓ | RMSE（m）↓ | δ1↑ |
|---|---:|---:|---:|
| Depth Pro | 0.10047157 | 0.39221681 | 0.91517394 |
| UniDepthV2-S | 0.11454373 | 0.39884996 | 0.89217973 |

来源：[Depth Pro summary](outputs/part2/depth_pro_nyuv2/frozen_v1/metrics_summary.json)、[UniDepthV2-S summary](outputs/part2/unidepth_v2_nyuv2/frozen_v1/metrics_summary.json)。现有指标不需为了匹配新版或论文分数重跑。

### 模型选择与尺度/边界分析

新版只需要选一个米制模型，并推荐UniDepthV2 ViT-S，Depth Pro明确允许。现有两个模型都属于允许的方法，不必删掉第二个，也不需要为了推荐顺序放弃已有Depth Pro结果。报告选定一个主模型，另一个可作为额外对照。

当前2A分析聚焦绝对尺度与物体边界：逐图尺度分布、真实距离分桶、边缘P/R/F1、边缘区域误差和具体物体案例。核心定量要求已满足；裁剪前异常值、物体标注支持边缘、区域贡献、局部放大及轮廓深度剖面已在`outputs/part2/scale_boundary_v1/`完成，全量数值重放通过，详见[Step 3](docs/step3.md)。不需要新增目标数据集来满足2A要求。

测试GT驱动的oracle尺度调整只作诊断，不进入RGB-only主表；`DA-V2`在NYUv2对齐后的误差也不能与原生米制结果直接判优。

### 效率与预处理的报告注意事项

正式summary已有推理耗时和GPU峰值；环境JSON分别记录实际PyTorch/CUDA、RTX A6000和精度。Depth Pro与UniDepthV2运行于不同环境，不能把FPS差异描述成所有软件条件完全一致的受控对比。

现有metadata记录原RGB及输出为480×640。最终报告应区分原图/输出尺寸与模型内部预处理尺寸，保留各模型官方流程；未实际记录的内部尺寸不要伪装成本次实测。**视频部分的“Small、默认FP16”规则，不应扩写成所有米制模型也必须使用Small变体；Depth Pro在新版中仍是允许的选项。**

## 5. Part 1B：视频基线可复用，NYUv2对齐变为额外诊断

### 当前自录实验已经有产物

本次读取到了两部分各7段视频的分析summary与verify。每段1B/2B帧数相同，以下状态均来自已有文件：

| 视频ID | 类型 | 各模型帧数 | 相邻帧对 | 1B / 2B分析与既有核验 |
|---|---|---:|---:|---|
| `fixed_scenario1` | fixed_static | 122 | 121 | complete / verified；两部分均有 |
| `fixed_scenario1_people_moving` | fixed_dynamic | 139 | 138 | complete / verified；两部分均有 |
| `fixed_scenario2` | fixed_static | 130 | 129 | complete / verified；两部分均有 |
| `fixed_scenario2_people_moving` | fixed_dynamic | 135 | 134 | complete / verified；两部分均有 |
| `moving_left_to_right` | moving | 141 | 140 | complete / verified；两部分均有 |
| `moving_left_to_right2` | moving | 134 | 133 | complete / verified；两部分均有 |
| `moving_rear_to_front` | moving | 151 | 150 | complete / verified；两部分均有 |

共952帧/945个相邻帧对；DA-V2与VDA各有952份raw NPY。14份summary均为`complete`，14份verify均为`verified`，且verify指向的summary SHA与当前文件一致。**本次只核对记录、文件数量与summary绑定，没有重新计算全部raw指标。**

来源分别为`outputs/part1/video/<ID>/part1/summary.json`、`verify1.json`，以及`outputs/part2/video/<ID>/part2/summary.json`、`verify2.json`。这也说明README、PLAN、Step 2/4中部分“视频待运行”的文字已经落后于实际产物；不能据这些文字判断用户没有运行。

### 新版不再要求DA-V2-NYUv2尺度对齐

新版§5.1要求DA-V2-S用于指定视频的逐帧relative baseline，并明确不估米、不用测试GT对齐。你的NYUv2分支是在654张test上用GT拟合逆深度尺度与偏移，身份为`aligned_relative`。

它是在旧要求下做出的诊断基准，保留真实记录即可，但**不承担新版1B的必做任务，也不能当成新版2A的米制模型结果**。其现有值为：

**AbsRel 0.06171408、RMSE 0.29735612 m、δ1 0.95540346**。这些值使用了测试GT对齐，不是RGB-only原生米制精度。

来源：[DA-V2对齐summary](outputs/part1/dav2_nyuv2/frozen_v1/evaluation/metrics_summary.json)。报告可简述为额外的“相对结构经GT对齐后的诊断”，主表与native metric分开；也可不放入篇幅有限的主报告。无需删除、改数或重新对齐。

### 静止/有人/移动的诊断是否还要做

三类型、自录采集、ROI适用性、pixel/flow MAE/S和RMSE/S、背景std与尺度proxy都可以保留为扩展内容；新版不要求再录制更多自录片段，也不要求这些数值诊断全部通过才处理官方片段。

当前用Small、逐帧独立预测、原生relative、无GT对齐，方法选择正确。仍需将同一个DA-V2-S用于课程指定片段，和VDA在相同帧上比较。

## 6. Part 2B：模型选择正确，正式素材与采样协议需要另建轨道

| 项目 | 当前实现/产物 | 新版要求 | 如何处理 |
|---|---|---|---|
| 模型 | 官方DA-V2-S、VDA-S | 两个Small模型 | 复用已有代码与权重 |
| VDA模式 | relative、offline、FP16 | relative、offline、默认FP16 | 已吻合；不必额外做streaming或metric |
| 输入视频 | 7段self_recorded | 指定`davis_rollercoaster.mp4` | 增加独立课程片段轨道；不把自录改名冒充 |
| 帧序列 | 各片122–151帧，10 FPS选帧 | 指定前100帧、原始帧率 | 不能截取自录100帧替代；需先解决官方文件仅70帧的问题 |
| RGB缓存 | 960×540 | 给出`--max_res 640`起始设置 | 建议课程轨道从640长边开始，保存实际尺寸；不改自录旧配置 |
| 网络输入 | 两模型实际H×W为518×924 | 保留官方预处理；显存不足可减`--input_size`并报告 | 重新准备输入后如实记录实际形状，不把`max_res`与网络输入尺寸混为一谈 |
| 帧对应 | 同一manifest/PTS，2B引用1B | 两模型完全相同的帧 | 现有SHA/manifest设计可以复用 |
| 显示 | TURBO；原生固定[0,50]，每模型首帧固定参考副图 | 同调色板；各模型在整片固定归一化 | 原理吻合；检查饱和情况，不需让两模型颜色对应相同米数 |
| 保存 | float32 NPY、run/summary、固定色标视频 | 浮点预测及彩色可视化；官方起始参数含`--save_npz` | 已具备浮点保存能力；新轨道按课程示例保留NPZ或清楚说明等价浮点格式 |
| 分析 | 定性与pixel/flow/尺度数值 | 定性强制、数值可选 | 主展示先回答闪烁/边缘/运动伪影；无须为课程主视频额外运行RAFT |
| 效率 | DA逐帧计时；VDA离线序列计时和均摊成本 | 每图/每帧时间、峰值显存、硬件/变体/精度/输入 | 新片段另记实际成本；VDA均摊毫秒不称在线首帧延迟 |

`--max_res 640`是新版给出的**起始设置**，不能扩写成任何其他分辨率都绝对违规。但当前960/10 FPS/自录来源这一整套协议，确实不是新版指定片段的运行记录；推荐为课程片段单独固定640起始配置，而不是改写旧结果。

同调色板不等于同米制尺度。当前每模型首帧P5/P95参考在整片固定，不是逐帧归一化；这与新版要求的核心一致。将来展示某方法的改进时，还须沿用该方法原始基线的显示归一化，不能给改进结果单独选一个更好看的范围；此处只解释新规，不审计Part 3实现。

### 为什么不建议直接把现有单视频入口指向官方MP4

原`src/video/clip_workflow.py`（现已清理）在初次审查时的`prepare_clip`写死`source_kind='self_recorded'`，类型也按三类自录诊断组织。若只换`--video`路径，虽可能能运行，却会把官方素材记成自录来源；默认仍是10 FPS、960长边。

项目保留的[pipelines/video/cli.py](pipelines/video/cli.py)有public输入能力，但它的通用默认配置也不是新版指定片段协议。后续应复用底层adapter/保存/绘图，在正确的public来源、原帧率和选帧规则下准备专用入口或配置。可以继续维持1B/2B入口及产物分开，不需要恢复合并入口，更不需要推翻已有7段结果。

## 7. 必须先解决的素材矛盾：要求100帧，提供文件只有70帧

新版§2、§5.2和§6.1都写了`davis_rollercoaster.mp4`前100帧，§6.1进一步要求原始帧率。译文忠实保留这些要求。

### 本地与远端核对证据

| 项目 | 本次读取结果 |
|---|---|
| 本地文件 | `Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4` |
| 分辨率/帧率 | 960×540，24 FPS |
| 时长 | 2.916667秒 |
| 容器帧数`nb_frames` | 70 |
| 实际解码计数`nb_read_frames` | **70**，通过`ffprobe -count_frames`确认 |
| 文件大小 | 1,809,560 bytes |
| 文件SHA256 | `7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8` |
| 本地官方仓库commit | `4f5ae23172ba60fd7bc11ef671cca678842c7072`，跟踪文件无修改 |
| 本地文件Git blob SHA | `f385e1bb30f90b48e0c6a8f485b8ff4862f082a5` |
| 当前上游`main`文件元数据 | 同样为1,809,560 bytes、Git blob SHA `f385e1bb30f90b48e0c6a8f485b8ff4862f082a5` |

本地读文件/解码计数与远端文件元数据核对日期为2026-11-02；远端只读取了[GitHub官方Contents API](https://api.github.com/repos/DepthAnything/Video-Depth-Anything/contents/assets/example_videos/davis_rollercoaster.mp4?ref=main)，没有下载视频。主分支会变化，此处记录的是本次响应。相同Git对象身份表明本地文件并非碰巧截短的另一个版本；当前课程链接对应的上游文件也与本地一致。[上游文件页](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/assets/example_videos/davis_rollercoaster.mp4)

只读复查命令：

```bash
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration \
  -of json Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4
```

### 正确的处理方式

向教师确认：是否接受当前官方文件的完整70帧，还是会提供至少100帧的指定版本。此问题不是靠更换推理参数就能解决。

- `--max_len 100`是最多读取100帧，不会让70帧文件变成100帧；命令正常结束也不等于满足100帧交付。
- 不重复、插帧、循环或拼接来伪造“原始前100帧”，也不自行改用其他示例片段替代指定视频。
- VDA内部为32帧窗口补齐尾部是模型实现细节，最终要裁回真实输入长度，不能把内部padding计作额外交付帧。
- 当前无需重复下载相同Git对象。先确认课程期望的素材/帧数，再由用户启动正式预测。

确认前，可以说“对应模型/流水线已有”，但**不能把新版指定视频任务标为已完成**。

## 8. 报告与提交层面的补充

这部分只核对与1A/2A/1B/2B交付直接相关的要求，不判断现有Part 3是否充分。

1. **报告仍为CVPR模板、6–8页，不计参考文献。**摘要末尾公开GitHub链接、相关工作覆盖第4节阅读列表、消融和失败案例等要求仍在。开始指南建议先读选定模型论文，不等于正式报告可以跳过推荐列表其余论文。
2. **arXiv已改为可选。**`docs/step6.md`仍写必须上传arXiv，应在后续实施整理中同步，不再把取得arXiv ID作为必做门禁。
3. **指定片段RGB/depth并排视频、一个简单应用输出和`depth_results.zip`仍必需。**只完成DA/VDA深度视频不等于完成应用。简单背景虚化或点云视图即可，不需要额外传感器或复杂测距界面。本次检查的非Part 3代码/输出及根目录未发现相应完整应用交付或ZIP；不据此断言所有未检查位置都绝对没有文件。
4. **3个困难案例从提供数据中真实挑选。**不需要为了满足旧版列举的镜子、玻璃、天空等类别再录视频。可以使用NYUv2静态难例，并在指定视频中展示实际出现的伪影；不预先宣称该片段必然含所有类别。
5. **代码公开与视频成果提交是两件事。**继续尊重你“不上传视频和1B/2B运行产物到GitHub”的选择，本次没有改[上传白名单](docs/github_upload_part1b_part2b.md)或上传文件。PDF同时要求公开README含代表性可视化；只发布代码/配置不等于这项展示已经完成。如希望通过外部演示链接展示，应核实课程接受的形式，不能直接宣称无需展示结果。`depth_results.zip`是课程成果交付，不要求把全量预测或视频塞进Git历史。

## 9. 后续工作的顺序

| 顺序 | 要做的事 | 是否需要新模型预测 |
|---|---|---|
| 先确认素材 | 解决“官方70帧/课程100帧”，明确采用的文件和真实帧数 | 不需要；这是课程输入约定问题 |
| 补1A主评价 | 基于现有60份预测，补有限、正值全GT域的MAE/coverage；单独保存新评价，旧表不覆盖 | 通常不需要；不重新调参 |
| 准备课程视频轨道 | public来源、原FPS、确认后的前N帧、建议640长边，1B/2B分别运行与存储 | 准备本身不需要；正式DA/VDA推理由用户启动 |
| 组织2A主表与说明 | 选定主模型，直接复用native结果，保留绝对尺度/边界解释与实际效率 | 不需要重跑现有654张 |
| 完成展示与交付 | 同帧相对深度视频、3个真实困难案例、简单应用、报告及ZIP | 视应用设计决定；不需要训练基础模型 |
| 同步实施说明 | 更新PLAN、plan_1B2B、Step 1/2/4/6和AGENTS第11条的旧门禁/状态 | 不需要；应在确定新轨道后统一整理 |

自录视频和DA-V2-NYUv2对齐记录保留，不必为了课件变化删掉。优先补真正影响合规性的评价域与指定片段，再组织报告，能够复用大部分现有工作。

## 10. 本次核验范围与来源

本次完成了新版5页文字与内嵌链接提取、关键页视觉核对、旧版正文对照、配置/代码阅读、正式文件数量与summary/verify身份检查，以及官方视频实际解码计数与上游文件身份核对。没有启动训练、模型推理、重新下载视频/权重、BM/SGBM搜索或全量指标重算；没有修改任何原始预测、数值配置或自动运行元数据。

译文保留了PDF内全部18个不同链接、12条参考文献和6个备选方向。新增/修改的4份Markdown通过77个本地链接/锚点及10段Bash语法检查；203份科学源码、配置与本次小型来源文件SHA保持不变。检查不涉及重新计算实验指标。

以下SHA用于标识本文引用的具体来源，不是新的实验记录：

| 来源文件 | SHA256 |
|---|---|
| `Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf` | `ea65076917e87febb0c3c67773b9fc6edf304e2821a74266a03f22ef33abc226` |
| `AIAA3201_Project3_Monocular_Metric_Depth_Video_Consistency_Spring2027 (1).pdf` | `8b5f29e7867e7789800497aeec6987b0a338e96b94b874b2b90b723be70d83c7` |
| `outputs/part1/middlebury/frozen_v2/metrics_summary.json` | `0e266b026fb6843486f3be6c3636f7597e47da2d1fabb3f070c19c50249b1199` |
| `outputs/part1/middlebury/frozen_v2/reload_validation.json` | `5e4aa871f12e616f23b7733483e2484c942187da9c7c75922c74c92f41f495ab` |
| `outputs/part2/depth_pro_nyuv2/frozen_v1/metrics_summary.json` | `0755a6501654f10079f4bb90bc251b15bd7c8139974655424145936bb6f705e4` |
| `outputs/part2/unidepth_v2_nyuv2/frozen_v1/metrics_summary.json` | `22a1793a1d9123729830b9bcab699092815551560fe4c39426ccdbc6af6c0f0e` |
| `outputs/part1/dav2_nyuv2/frozen_v1/evaluation/metrics_summary.json` | `60e642a94c291c963ff5a596f2d53be2adf333e88a2b395899f1b1d83bde18c6` |

旧`PLAN.md`、`plan_1B2B.md`、Step文档及AGENTS中的“完成/待运行”仍需要按照新要求和实际产物统一整理。本次只新增译文和差异分析，并在README/implemented增加本次入口和记录，**没有把建议中的新评价或新视频运行写成已经执行**。
