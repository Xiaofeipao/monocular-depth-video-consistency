# Step 3 / Part 2A：原生米制深度、绝对尺度与物体边界

## 任务、状态与评价协议

使用官方预训练的 Depth Pro 与 UniDepthV2-S（ViT-S/14）进行仅 RGB 的单图推理，直接评价输出的米制深度。课程只要求选择一个米制模型；本项目保留两个模型作对照，不训练或微调基础模型。两条654张基线和独立的完整尺度/物体边界分析均已完成，全量数值重放通过。课程依据见[项目说明 §5.2、§6.2](../Project3_CN_final.md)。

| 项目 | 固定设置 |
|---|---|
| 数据与划分 | NYUv2 labeled MAT 的 `images` 和填充的 `depths`；官方 `testNdxs` 654 张，MATLAB 索引减 1；795 张 `trainNdxs` 留作开发 |
| 数组读取 | RGB：`images[i].transpose(2,1,0)`；GT：`depths[i].T`；不使用 `rawDepths` |
| 模型输入 | 完整 480×640 RGB；各模型保留官方预处理 |
| 预测与评价尺寸 | 保存完整 480×640、float32 米制预测；需要恢复尺寸时使用 bilinear、`align_corners=False` |
| 评价 crop | Python 半开区间 `[45:471,41:601]`，426×560 |
| 有效深度 | `0.001 < GT < 10 m`；计算指标时将预测裁剪到 `[0.001,10] m` |
| 正式轨道 | `native metric / meter / alignment:none`，无外部标定信息、无测试 GT 尺度或偏移拟合 |
| 主指标 | AbsRel↓、RMSE↓（m）、δ1↑；654 张逐图计算后等权平均 |
| 补充指标 | SqRel、RMSE-log、δ2/δ3、SI Boundary F1、深度边缘 P/R/F1、边缘区域误差 |
| 区间 | 基线原记录为10,000次图像bootstrap；本页新增误差分析采用10,000次scene成簇bootstrap，均为95% CI |

指标计算的裁剪不改变保存的原生预测。模型内部预测的相机参数随官方输出保存，不作为额外输入。允许跨设备的微小浮点差异，不调整预测或 evaluator 来匹配某个数字，也不宣称精确复现论文分数。

## 运行指令

所有命令从项目根目录执行。Depth Pro 使用 `aiaa3201`，UniDepthV2 与联合分析使用 `aiaa3201-unidepth`。下面是复现命令；已有两条完整基线时，不需要重新预测。

### 1. 小样本检查（首次搭建时）

```bash
./scripts/part2/run_depth_pro_nyuv2.sh --positions 0 326 653 \
  --output-dir outputs/part2/depth_pro_nyuv2/audit3_reproduce_v1
./scripts/part2/run_unidepth_v2_nyuv2.sh --positions 0 326 653 \
  --output-dir outputs/part2/unidepth_v2_nyuv2/audit3_reproduce_v1
```

### 2. 完整原生米制基线

```bash
./scripts/part2/run_depth_pro_nyuv2.sh \
  --output-dir outputs/part2/depth_pro_nyuv2/reproduce_v1
./scripts/part2/run_unidepth_v2_nyuv2.sh \
  --output-dir outputs/part2/unidepth_v2_nyuv2/reproduce_v1
```

正式预测由用户启动。中断后可对相同配置与目录加 `--resume`。`--mode evaluate` 只读取已有预测，但会重写该目录的评估产物和运行记录；如需重评，应先将完整 run 复制到新的目录，保留原记录。

### 3. 完整绝对尺度与物体边界分析

以下两条是本轮实际执行的完整分析与独立重放入口，结果已经存在：

```bash
./scripts/part2/analyze_scale_boundary.sh \
  --config configs/part2/scale_boundary.yaml \
  --output-dir outputs/part2/scale_boundary_v1

./scripts/part2/analyze_scale_boundary.sh \
  --config configs/part2/scale_boundary.yaml \
  --output-dir outputs/part2/scale_boundary_v1 --mode verify
```

`run`只读两条既有`frozen_v1`预测与NYUv2 GT/物体标注，在CPU上完成全部统计、局部剖面、图表和报告，不加载模型。已有目录会拒绝覆盖。`verify`从同一原始数组重放六张逐图表和汇总，并核验输入、配置、源码和产物SHA；浮点值用`rtol=1e-6, atol=1e-8`，身份、整数计数和覆盖必须一致。`--mode verify --hashes-only`只查来源与文件，不能当作全量数值复评。

若需要独立复现整套分析，可使用另一个不存在的新目录：

```bash
./scripts/part2/analyze_scale_boundary.sh \
  --config configs/part2/scale_boundary.yaml \
  --output-dir outputs/part2/scale_boundary_reproduce_v1
./scripts/part2/analyze_scale_boundary.sh \
  --config configs/part2/scale_boundary.yaml \
  --output-dir outputs/part2/scale_boundary_reproduce_v1 --mode verify
```

不需要重新运行前面的模型预测。分析其他复现基线时，复制此分析配置并修改`baselines`，通过`--config`显式选择；输入仍须为完整654张、相同数据SHA与冻结评价协议。失败会留下真实状态，不将局部运行标为完成。

## 原生米制结果

数据来源为两个 `frozen_v1/metrics_summary.json`，以下显示八位小数。AbsRel 是相对误差，例如 0.10047 表示平均相对误差约 10.05%；RMSE 使用米；δ1 是预测与 GT 的比值误差在 1.25 倍以内的像素比例。

| 模型 | AbsRel↓（95% CI） | RMSE↓ m（95% CI） | δ1↑（95% CI） |
|---|---:|---:|---:|
| Depth Pro | 0.10047157 [0.09606078, 0.10510840] | 0.39221681 [0.37050663, 0.41521821] | 0.91517394 [0.90420593, 0.92580502] |
| UniDepthV2-S | 0.11454373 [0.10960142, 0.11970636] | 0.39884996 [0.37819465, 0.42053339] | 0.89217973 [0.87977248, 0.90408387] |

| 补充指标 | Depth Pro | UniDepthV2-S |
|---|---:|---:|
| SqRel↓ | 0.06400258 | 0.06895209 |
| RMSE-log↓ | 0.13533882 | 0.14121805 |
| δ2↑ | 0.97985983 | 0.98073107 |
| δ3↑ | 0.99187774 | 0.99333280 |
| SI Boundary F1↑ | 0.15927408 | 0.11300740 |

七项深度指标已在完整分析中从原始预测重新计算并逐图核对；SI Boundary F1引用冻结baseline CSV并绑定来源SHA，本轮没有重新调用官方SI实现。下文raw/clipped边缘P/R/F1、物体支持边缘和局部剖面均由本轮重新计算。

原生基线仍保留完整654份预测、metadata与五联图，以及失败案例表/总览。展示阈值为任一`AbsRel>0.25`、`δ1<0.60`或`SI<0.01`，分别选出49/77张，可重叠触发；它们不影响654张评分分母，也不是课程及格线。完整尺度/边界分析的12个解释案例另有固定选择记录。

Depth Pro 的 AbsRel、RMSE、δ1 和 SI 平均值更好，UniDepth 的 δ2/δ3 略高。平均值的差异不能直接解释为每张图都更好，也不能只凭两个独立区间的重叠判断模型差异的显著性。SI 与下文固定阈值的边缘 F1 定义不同，数值不直接横比；它们是补充诊断，课程米制主表仍是前三项。

### 效率与输入尺寸

两条记录均使用 NVIDIA RTX A6000。Depth Pro 为 PyTorch `2.1.1+cu121 / float16`；UniDepth 为 `2.7.1+cu126 / official_float16_autocast`。保留各自官方预处理，不强制它们使用相同的网络内部尺寸。

| 模型 | 平均 s/图 | P95 s/图 | 模型 FPS | 峰值 GPU bytes | 含读写保存 FPS | 加载时间 s |
|---|---:|---:|---:|---:|---:|---:|
| Depth Pro | 0.29722507 | 0.30497521 | 3.36445373 | 3973226496 | 3.04906982 | 9.99864572 |
| UniDepthV2-S | 0.01561590 | 0.01582386 | 64.03728774 | 508515328 | 18.00893720 | 0.39910521 |

原 RGB 和恢复后的输出都是 480×640。按固定上游代码，Depth Pro 内部将输入缩放为 1536×1536；UniDepth 默认形状约束对该输入得到 490×644。这两个内部尺寸来自执行版本的代码与 checkpoint 配置推导，并非原运行逐层实测记录。报告中应注明这一来源；如需实测确认，可单独增加不改变预处理的尺寸日志。

模型 FPS 不含完整读写、保存及报告成本。独立环境、内部输入与实现不同，因此这些数值代表各自实际运行成本，不作为统一软件条件下的速度排名。

## 绝对尺度：原始预测、距离误差与整图偏差

完整精度来源：summary.json（本地产物：`outputs/part2/scale_boundary_v1/summary.json`）、逐图表（本地产物：`outputs/part2/scale_boundary_v1/tables/per_image.csv`）。以下raw统计均在固定crop内的有效GT域计算，每模型156,018,240个像素；预测不经评分裁剪。`raw`是这里的模型原生预测，GT仍是课程规定的填充`depths`。

### 原始尺度比与超范围预测

每图`r_i=median(raw_prediction_i/GT_i)`：小于1表示中位数比例偏浅，大于1表示偏深。先逐图取中位数，再在654张上汇总；不要将整图尺度比理解为所有距离、所有物体具有相同比例误差。0.9–1.1是描述性区间。

| 模型 | raw尺度比中位数 | P05–P95 | 0.9–1.1内图像 | 存在>10 m预测的图像 | 超范围像素比例宏平均 | 原始最大预测 m |
|---|---:|---:|---:|---:|---:|---:|
| Depth Pro | 0.95805953 | 0.82894664–1.10450979 | 72.1713% | 32/654 | 0.2517% | 40.96856308 |
| UniDepthV2-S | 1.04779757 | 0.86824602–1.20879561 | 67.4312% | 28/654 | 0.3471% | 27.62650299 |

两个模型在此评价域均没有低于0.001 m的预测，超范围来自大于10 m的长尾。Depth Pro略偏浅、UniDepth略偏深；少数图的局部预测可达到约41/28 m。评分裁剪是固定课程协议，保留raw诊断可以看到这部分长尾。

| 模型 | raw AbsRel | 正式clip后 AbsRel | raw RMSE m | 正式clip后 RMSE m | raw平均有符号误差 m |
|---|---:|---:|---:|---:|---:|
| Depth Pro | 0.10151993 | 0.10047157 | 0.41628871 | 0.39221681 | -0.10719478 |
| UniDepthV2-S | 0.11587880 | 0.11454373 | 0.42148315 | 0.39884996 | 0.07827429 |

这说明裁剪对整图尺度中位数影响较小，但对RMSE长尾有可见影响。raw结果只作诊断，正式成绩仍使用课程clip；不因诊断发现极端值而删除图像或改预测。

尺度分布（本地产物：`outputs/part2/scale_boundary_v1/figures/scale_distribution.png`）

### 不同真实距离的误差

来源：完整三轨道分桶表（本地产物：`outputs/part2/scale_boundary_v1/summary/depth_bins.csv`）。下表使用正式clip轨道；每桶先逐图计算，再对有支持图像等权平均，空桶N/A并保留覆盖。RMSE为逐图RMSE的平均。

| GT距离 m | 图像数 | Depth Pro AbsRel | Depth Pro RMSE m | Depth Pro有符号偏差 | UniDepth AbsRel | UniDepth RMSE m | UniDepth有符号偏差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.001–1 | 254 | 0.17263034 | 0.17519862 | 8.0529% | 0.32732494 | 0.31321858 | 30.8490% |
| 1–2 | 641 | 0.11151014 | 0.21535330 | -0.8247% | 0.15664731 | 0.28023642 | 11.3559% |
| 2–4 | 645 | 0.10467632 | 0.36262549 | -4.7570% | 0.10266586 | 0.35224664 | 2.2124% |
| 4–6 | 415 | 0.13599288 | 0.73962732 | -8.2573% | 0.13110974 | 0.70933059 | -5.6069% |
| 6–10 | 220 | 0.18437289 | 1.41125950 | -12.7110% | 0.18922909 | 1.43332525 | -13.4627% |

UniDepth在1 m以内平均偏深约30.85%，在6–10 m却偏浅约13.46%；Depth Pro相应为+8.05%和-12.71%。两者都有距离向中间压缩的表现；仅将整个深度图乘一个系数，无法同时解释近处和远处误差。不同桶的图像可重叠，图像数不相加作为独立样本数。

距离误差（本地产物：`outputs/part2/scale_boundary_v1/figures/distance_error.png`）

### 整图比例与图内结构分别占什么位置

对每图未裁剪的`e=log(raw/GT)`，令`b=mean(e)`，有精确恒等式：

```text
mean(e²) = b² + mean((e-b)²)
```

`b²`表示整图平均log偏差，余项包含图内结构失真、局部噪声及GT局限。两项先逐图计算再平均。Depth Pro/UniDepth的`mean(b²)/mean(e²)`分别为46.8405% / 47.1039%。其余约53%仍在图内变化中；该份额仅对应logMSE恒等式，不是AbsRel的因果分解。

另计算测试GT辅助的`clip(raw/r_i)`，查看整图比例调整后的残余误差。这条oracle轨道始终只作诊断：

| 模型 | 原生AbsRel | GT辅助后AbsRel | 原生RMSE m | GT辅助后RMSE m |
|---|---:|---:|---:|---:|
| Depth Pro | 0.10047157 | 0.05572485 | 0.39221681 | 0.29245993 |
| UniDepthV2-S | 0.11454373 | 0.06036160 | 0.39884996 | 0.29544132 |

误差降低表明整图比例偏差确实重要；残余误差表明局部问题依然存在。测试GT提供的比例不能用于部署，也不作为模型改进或米制主表成绩。

## 物体边界：轮廓、定位与过渡宽度

### 全部深度边缘与有标注支持的物体边缘

全部深度边缘以四邻接深度比达到1.05/1.10/1.25提取，预测与GT相同。匹配容差1/2/3 px使用Chebyshev距离；原生raw与评分clipped分别计算，以区分结构本身和裁剪影响。下表主诊断取raw、1.05阈值。

NYUv2的`labels`与`instances`提供现成物体标注。组合类别号和实例号得到轮廓，排除label=0；不因实例号为0而直接排除。把距标注轮廓≤2 px的GT深度边缘单独定义为“物体支持边缘”，报告其召回率。这样不会把同一平面上的颜色/语义分界当作必然深度突变。字段依据[NYUv2官方说明](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)。这些标注仅用于评价和选择诊断剖面，不提供给模型。

| 模型 | 容差 px | 全部边缘Precision | 全部边缘Recall | 全部边缘F1 | 物体支持边缘Recall |
|---|---:|---:|---:|---:|---:|
| Depth Pro | 1 | 0.69086452 | 0.47760729 | 0.54976583 | 0.52850031 |
| Depth Pro | 2 | 0.78191925 | 0.55631352 | 0.63423406 | 0.60193238 |
| Depth Pro | 3 | 0.82870670 | 0.60645072 | 0.68447154 | 0.64118925 |
| UniDepthV2-S | 1 | 0.73803256 | 0.39319242 | 0.49899412 | 0.44014298 |
| UniDepthV2-S | 2 | 0.84150526 | 0.46145678 | 0.58015688 | 0.50294686 |
| UniDepthV2-S | 3 | 0.88356587 | 0.50422304 | 0.62521286 | 0.53606579 |

Depth Pro找回的边缘更多，UniDepth预测的边缘Precision更高但漏检更多。物体支持子集有636张图，另18张在该阈值下无支持，保留N/A；不把它们作为0召回混入均值，也不把不完整标注用于惩罚所有预测边缘。

定位诊断另计算GT边缘到最近预测边缘的Euclidean距离，最大记10 px；Depth Pro/UniDepth的逐图均值分别约3.97/4.98 px。它同时受到漏检和错位影响，不能解释为“匹配成功轮廓的纯位置误差”。全部阈值、raw/clipped和覆盖见边缘汇总（本地产物：`outputs/part2/scale_boundary_v1/summary/boundary.csv`）与定位汇总（本地产物：`outputs/part2/scale_boundary_v1/summary/localization.csv`）。1.05处提高容差会提高分数，体现定位容忍度；不同阈值选中不同强度边缘，不据此挑最高分作最终结果。

边界敏感性（本地产物：`outputs/part2/scale_boundary_v1/figures/boundary_sensitivity.png`）

### 边界附近承担多少米制误差

区域带使用Chebyshev距离（方形邻域）。距GT深度边缘≤3 px的区域占有效像素约10.8632%，承担Depth Pro/UniDepth的平方误差份额约24.1910% / 24.6983%。边缘附近的误差密度明显更高；同时，>5 px内部仍承担约69%的平方误差，因此不能只关注轮廓而忽略广域米制偏差。

| 模型 | 边缘≤3 px AbsRel | 内部>5 px AbsRel | 物体支持边缘≤3 px AbsRel | GT辅助后边缘AbsRel | GT辅助后内部AbsRel |
|---|---:|---:|---:|---:|---:|
| Depth Pro | 0.13383808 | 0.09486342 | 0.12565395 | 0.10336904 | 0.04793020 |
| UniDepthV2-S | 0.15495697 | 0.10789522 | 0.15360632 | 0.11138350 | 0.05198069 |

调整整图尺度后，边缘与内部都改善，但边缘残余误差仍明显更高。这把“全图比例不准”和“轮廓局部结构不准”联系起来，支持同时分析两者。误差份额来自每图平方误差和的会计式统计，不表示发现了具体物理原因。

互斥区域[0,1]、(1,3]、(3,5]、>5 px的像素/误差份额可相加；“≤3 px”和“物体支持边缘≤3 px”是重叠区域，不能重复相加。空区域的误差均值为N/A、份额为0；完整覆盖在区域汇总（本地产物：`outputs/part2/scale_boundary_v1/summary/regions.csv`）。

尺度与边界联系（本地产物：`outputs/part2/scale_boundary_v1/figures/scale_boundary_connection.png`）

### 用同一局部剖面区分偏移、模糊与漏检

仅依据GT与物体标注选剖面，两模型共用位置。每图最多3条水平/垂直剖面，半径20 px，两端各5像素的GT log深度标准差≤0.03，GT两侧深度比≥1.10、10–90%过渡宽度≤6 px；剖面间至少60 px，回摆约束固定。共选出513张图上的1,146条剖面，另141张无符合条件的剖面并保留覆盖。

实际曲线始终保留米制深度；另用各条曲线**自身**的左右侧中位数作0–1形状归一化，测50%位置与10–90%宽度。该归一化只用于轮廓形状，不改变任何米制评分。GT和预测两侧顺序、最低预测对比1.05以及交点唯一性均检查；对比太弱、深度次序错误或多次交点均给出状态，几何量为N/A，不写成0。

| 模型 | 可测剖面/全部 | 可测比例 | 具备可测量值的图像 | 位置绝对误差 px | 相对GT宽度增量 px | 预测/GT log对比幅度 |
|---|---:|---:|---:|---:|---:|---:|
| Depth Pro | 1066/1146 | 93.0192% | 503 | 1.81052376 | 2.41359000 | 0.87337304 |
| UniDepthV2-S | 1042/1146 | 90.9250% | 498 | 2.03292803 | 3.19558733 | 0.77354469 |

Depth Pro的不可测状态为弱对比35、交点不唯一35、顺序错误10；UniDepth分别为52、43、9。两模型都存在轮廓变宽与深度台阶幅度减弱，UniDepth更明显。这里的宽度/位置均值是有条件统计：先在一图内平均，再逐图平均；不能忽略不同的有效覆盖来简单判优。

在双方都可测的相同剖面上配对后，位置误差差值（Depth Pro−UniDepth）为-0.17593709 [-0.30439116, -0.05279468] px，宽度增量差值为-0.63567118 [-0.92179151, -0.36834913] px，均有495张图支持。在这一固定GT支持子集上，区间支持Depth Pro更少偏移和更窄的过渡；结论不覆盖所有细杆、曲线或弱边界。

完整剖面逐条表（本地产物：`outputs/part2/scale_boundary_v1/tables/profiles.csv`）保留实际坐标、GT宽度、状态、位置、宽度和对比幅度。水平/垂直剖面是图像坐标方向的局部截面，曲线或斜边的连续法线宽度仍需更专门的几何定义。

### 实际物体与失败案例

产物含12张整图比较、固定坐标局部放大和13组剖面图；没有可选GT剖面的案例另给局部放大。固定示例与按指标选出的困难图见总览（本地产物：`outputs/part2/scale_boundary_v1/figures/case_contact_sheet.jpg`）、选择记录（本地产物：`outputs/part2/scale_boundary_v1/case_selection.json`）。候选用于解释机制，不是随机代表集，也不用于参数选择。

- test_0021，红色椅背边缘，剖面0（本地产物：`outputs/part2/scale_boundary_v1/cases/test_0021_mat_0040/profile_0.png`）：Depth Pro位置误差约0.57 px，UniDepth约2.08 px；UniDepth两侧台阶幅度只有GT的约62.87%，且椅子侧预测偏深。可同时看到位置错位与米制偏差。该图另一剖面上UniDepth位置误差更小，说明同图也有局部优劣。
- test_0545，左侧容器与地面，剖面0（本地产物：`outputs/part2/scale_boundary_v1/cases/test_0545_mat_1234/profile_0.png`）：Depth Pro还能测量，但过渡比GT宽约14.74 px；UniDepth对比太弱，几何量N/A。沙发上沿剖面1（本地产物：`outputs/part2/scale_boundary_v1/cases/test_0545_mat_1234/profile_1.png`）上UniDepth的定位/宽度较好。整图SI较高不能保证每一处物体边界都好。
- test_0302，小圆形物体，剖面0（本地产物：`outputs/part2/scale_boundary_v1/cases/test_0302_mat_0713/profile_0.png`）：GT有明显深度断层，而两模型原生曲线接近平坦；两者log对比幅度只约GT的1.89%/2.75%，都标记弱对比。形状图明确N/A，原生曲线显示漏检，不能赋予“宽度很小”的好成绩。反射表面与填充GT的物理对应仍有限制，解释限定为相对该GT的误差。

## 配对统计与结论范围

新增诊断区间使用215个scene作为重采样簇，重复抽中一个scene时保留其中图像权重；点估计仍为课程要求的逐图宏平均。原baseline文件保留自身的图像级bootstrap。本表是两模型相同图像的直接差值，单看两个独立CI是否重叠不足以判断模型差异。

| 配对量（Depth Pro−UniDepth） | 均值差与95%成簇CI |
|---|---:|
| AbsRel | -0.01407216 [-0.02176904, -0.00628920] |
| RMSE m | -0.00663314 [-0.02506959, 0.01175897] |
| δ1 | 0.02299421 [0.00605070, 0.04002962] |
| 绝对log尺度偏差 | -0.00317835 [-0.01205995, 0.00591018] |
| raw边缘F1，3 px | 0.05925868 [0.05087338, 0.06748808] |
| 物体支持边缘Recall，3 px | 0.10512346 [0.09085584, 0.11933298] |

AbsRel、δ1和这些边界量支持Depth Pro优势；RMSE与绝对log尺度偏差的区间跨零，不能说所有指标都显著更好。GT辅助尺度诊断、物体标注支持范围和剖面可测量覆盖都需要与结果一起报告。

本页已完成两条原生米制基线及尺度/物体边界的全量静态分析。模型对反射、细结构、弱深度突变的困难、填充GT的局限和选择规则的覆盖限制如实保留；不以追加训练或修改预测来抹去失败。应用Demo、Part 3与最终报告整合仍由各自Step负责。

## 架构与产物

| 项目 | 作用 |
|---|---|
| 两条`frozen_v1/`与两份原生模型配置 | 保留核心单图米制工作，完整预测、元数据、效率和原评分不变 |
| `configs/part2/scale_boundary.yaml` | 独立冻结数据/两条基线、尺度、物体边界、剖面与统计协议 |
| `src/analysis/scale_boundary.py` | 无模型的纯分析计算：raw尺度、区域误差、物体支持边缘与剖面 |
| `pipelines/part2/analyze_scale_boundary.py` | 输入/来源校验、654张计算、汇总、来源归档、全量重放与禁止覆盖 |
| `src/visualization/scale_boundary.py` | 七类汇总图、12个案例、局部放大、曲线与中文报告 |
| `scripts/part2/analyze_scale_boundary.sh` | 独立分析/核验入口，环境`aiaa3201-unidepth`，CPU执行 |
| `outputs/part2/scale_boundary_v1/tables/` | 1,308行逐图、19,620行距离分桶、23,544行区域、23,544行边缘、7,848行定位、2,292行剖面记录 |
| `summary.json`与`summary/` | 完整精度点估计、覆盖、配对区间、失败状态和分组汇总 |
| `figures/`与`cases/` | 独立PNG汇总图/总览、案例RGB/GT/预测、误差/匹配图、实际坐标及原生曲线数据 |
| `report.md` | 可单独阅读的中文数值报告 |
| `inputs.json`、`provenance/`、`config.yaml` | 固定输入/权重无关的来源记录、执行时源码与配置 |
| `manifest.json`、`run_metadata.json`、`verification.json` | 产物SHA、真实执行状态/环境/命令，以及独立重放结论 |

### 保留的key_analysis

`configs/part2/key_analysis.yaml`、`pipelines/part2/run_key_analysis.py`与`scripts/part2/run_key_analysis.sh`保持原样，`outputs/part2/key_analysis_v1/`的已有产物也完整保留。它们与上面的完整分析互不覆盖；新分析不依赖其中的外部预测或已算好的诊断分数。

需要单独运行这个入口时，仍可按其原配置选择不存在的输出目录：

```bash
./scripts/part2/run_key_analysis.sh \
  --config configs/part2/key_analysis.yaml \
  --output-dir outputs/part2/key_analysis
```

## 实际验证

完整CPU分析与独立重放已完成，正式米制七指标逐图与两条基线一致；六张数值表和汇总的重放检查通过1,039,777个数值。原始数据/预测只读，完整输入及旧key_analysis/基线记录的保留核对见维护记录（本地产物：`outputs/maintenance/part2_scale_boundary_20261102/README.md`）。没有训练、模型推理或下载。结果来源与实际完成记录见当前结果审计（完整记录保存在源项目，本发布包不包含）、[implemented.md](../implemented.md)。
