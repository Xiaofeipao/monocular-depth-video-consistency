# Step 1 / Part 1A：Middlebury 经典双目

## 基本信息与状态

本步在 Middlebury Stereo v3 的15组 quarter-resolution 训练图像上运行 OpenCV BM 和 SGBM，展示视差与无效区域，使用每个场景自己的标定将视差转换为米制深度，并评价视差精度和有效预测比例。左右一致性作为附加诊断保留。经典双目与 Part 2A 的 NYUv2 米制模型比较分开报告。

配置为 [`configs/part1/middlebury_sgbm.yaml`](../configs/part1/middlebury_sgbm.yaml)，协议标识为 `middlebury_course_v1`。用户已完成两算法完整开发搜索、四组15场景评估与独立核验。另对本次真实输出复核了60份预测、1,200个精度值及全部原尺寸掩码，均通过。误差—有效比例研究也已完成完整搜索、独立场景评估与原始重算，详见后文。

| 实验组 | 方法 | 参数来源 |
|---|---|---|
| `bm_fixed` | StereoBM | 固定经验参数 |
| `bm_tuned` | StereoBM | 仅在5个开发场景上选择 |
| `sgbm_fixed` | StereoSGBM | 固定经验参数 |
| `sgbm_tuned` | StereoSGBM | 仅在同样5个开发场景上选择 |

数据目录为 `data/middlebury/MiddEval3/trainingQ/`。使用该目录图像和 `calib.txt` 的尺寸与内参，不再缩放焦距或视差。

- 开发场景：Adirondack、Jadeplant、Motorcycle、Pipes、PlaytableP。
- 独立评估场景：ArtL、MotorcycleE、Piano、PianoL、Playroom、Playtable、Recycle、Shelves、Teddy、Vintage。

每组分别报告5个开发场景、10个独立评估场景以及全部15场景的逐场景宏平均。全部15场景包含开发数据，因此衡量选参泛化时优先看独立评估组。

## 几何与评价协议

### 视差转换为米制深度

```text
d_px = d_opencv / 16
z_m  = f_px × B_mm / (d_px + doffs_px) / 1000
```

`f_px`、`B_mm`、`doffs_px` 均读取本场景 quarter-resolution `calib.txt`。OpenCV 原始有符号整数仅解码一次；GT PFM 本来就是像素视差，不除以16。PFM 的 scale 符号决定字节序，绝对值不再乘进视差。

`src/stereo/geometry.py` 的 `middlebury_disparity_to_depth` 显式接收毫米基线，使用 float64 计算并保存 float32 米制深度；无效处保存 NaN，并保留布尔掩码。这里直接使用相机标定，不做尺度拟合，也不使用 NYUv2 的 crop 或10米裁剪。

### 哪些像素参与评价

GT评价域为**所有有限且大于0的GT视差像素**，包括其中的遮挡像素。`mask0nocc.png` 作为输入资料保留，但不限制主指标的分母。

一个预测像素必须同时满足：

1. OpenCV 没有返回无效编码，解码后视差有限且大于0。
2. 右图对应坐标在图内、插值所需右视差有效，并满足 `|dL(x) + dR(x-dL)| ≤ 1 px`。
3. `d_px + doffs_px > 0`，可以得到有限正深度。

无效编码为 `(minDisparity - 1) × 16` 及以下。右视差使用负号约定，因此左右检查使用相加。LR阈值固定为1 px，不参与调参。GT只用于评分，不参与匹配或预测有效性判断。

课程必需的两项指标为：

| 指标 | 计算方式 | 如何理解 |
|---|---|---|
| 视差 MAE ↓，单位 px | 在“GT有效且预测有效”的像素上平均 `|d_pred-d_GT|` | 输出了深度的位置有多准确 |
| 有效预测比例 ↑ | GT评价域内有效预测数 ÷ GT评价域像素数 | 本来有GT的位置中，方法能够给出多少有效预测 |

无效预测不虚构为0视差或无限大数值加入MAE，但仍计入比例的分母；两项必须一起看。若某场景没有有效预测，其MAE记为 `null`、比例为0，不能当作完美预测或静默删去；包含它的主MAE宏平均也记为 `null`，同时报告有定义的场景数。

此外，四组正式结果同时报告以下五项辅助指标：

| 指标 | 计算方式 | 无效预测如何处理 |
|---|---|---|
| Bad-1 ↓ | 绝对视差误差 **>1 px** 的像素数，加无效预测数，再除以GT评价域像素数 | 计为失败 |
| Bad-2 ↓ | 同上，阈值为 **>2 px** | 计为失败 |
| Bad-4 ↓ | 同上，阈值为 **>4 px** | 计为失败 |
| Depth AbsRel ↓ | 有效预测处平均 `|z_pred-z_GT|/z_GT` | 不虚构深度误差；结合有效比例解读 |
| Depth RMSE ↓ (m) | 有效预测处 `sqrt(mean((z_pred-z_GT)²))` | 同上 |

Bad-1/2/4使用与MAE相同的全有限正GT视差域，包含遮挡位置；阈值是严格“大于”，误差恰好等于阈值不算bad。保存值为0–1比例，展示时可乘100写为百分比。没有任何有效预测时三项Bad均为1，而MAE/深度误差为 `null`。因此Bad指标同时反映有效预测的错误与输出缺失，不能把它与仅在有效预测上统计的MAE混作同一含义。

先逐场景计算，再对场景等权平均，**不是**把所有场景像素混合平均。CSV同时保存GT、有效预测和无效预测像素数，便于核对分母。

## 参数设置与选择

四组本次实际参数如下：

| 实验组 | `block_size` | `uniqueness_ratio` | `speckle_window_size` | `texture_threshold` |
|---|---:|---:|---:|---:|
| `bm_fixed` | 15 | 10 | 100 | 10 |
| `bm_tuned` | 9 | 0 | 50 | 20 |
| `sgbm_fixed` | 5 | 10 | 100 | 不适用 |
| `sgbm_tuned` | 3 | 10 | 50 | 不适用 |

两者均为 `min_disparity=0`、`speckle_range=2`、`disp12_max_diff=1`。BM 的 `pre_filter_cap=31`；SGBM 为63，使用 `sgbm_3way`，灰度匹配时 `P1=8×block_size²`、`P2=32×block_size²`。每场景依据标定 `ndisp` 向上取16的倍数，四组采用相同规则。

| 算法 | 开发集候选集合 | 候选数 |
|---|---|---:|
| BM | block `{5,9,15,21}` × uniqueness `{0,5,10,15}` × speckle window `{0,50,100}` × texture `{0,10,20}` | 144 |
| SGBM | block `{3,5,7}` × uniqueness `{0,5,10,15}` × speckle window `{0,50,100}` | 36 |

每个候选都在完整5个开发场景上计算同样的主指标。选参规则为：**在开发集平均有效预测比例不低于对应 fixed 组的候选中，选择平均视差MAE最小的一组**。MAE相同则先选有效比例更高者，再按配置中参数顺序的数值升序打破平局。比例比较仅容许 `1e-12` 的数值误差。

这是本项目为兼顾精度与输出数量采用的选参规则，并非课程额外规定的达标门槛。候选集合包含 fixed 参数；tuned 最终与 fixed 相同也是有效结果。只在开发集选择一次，不依据10个独立评估场景或15场景总表再次挑参数。

`tuned` 参数由完整搜索的 `best.json` 加载，不手填进YAML。搜索会保存所有候选、逐场景分数、fixed参考、所选参数、输入SHA256、科学源码和配置；正式评估会校验它们。未完成、参数来源不匹配或证据损坏时停止，不悄悄使用默认参数。

## 用户运行：完整实验

以下三步本次已运行完成，命令保留用于复现；已有目录不覆盖，无需因为图中白边重跑。

从项目根目录依次执行。使用 `aiaa3201`，无需GPU，也不涉及训练。

### 1. 完整开发集搜索

```bash
./scripts/part1/tune_middlebury.sh
```

输出 `outputs/part1/middlebury/tuning/`，包含 BM 的144个候选和 SGBM 的36个候选。两者都成功后，再执行正式四组评估。

### 2. 四组、15场景正式评估

```bash
./scripts/part1/run_middlebury.sh
```

输出 `outputs/part1/middlebury/evaluation/`，共60份预测，自动生成主表、米制深度诊断、全图可视化、距离分析，并从保存数组重算检查。

### 3. 独立核验已保存产物

```bash
./scripts/part1/run_middlebury.sh --mode verify \
  --output-dir outputs/part1/middlebury/evaluation
```

此命令不重新匹配图像。它校验来源/产物SHA256，从原始左右视差重做LR判断、有效掩码、米制转换及全部精度指标，再核对逐场景CSV与宏平均JSON。成功时 `verification.json` 应为 `passed:true`，`prediction_files_checked:60`。浮点比较使用 `rtol=1e-6, atol=1e-8`，不要求跨设备逐位一致。

入口拒绝覆盖已有输出。需要独立复现时，步骤1使用 `--output-dir outputs/part1/middlebury/tuning_repeat/`；步骤2同时指定 `--tuning-dir outputs/part1/middlebury/tuning_repeat/ --output-dir outputs/part1/middlebury/evaluation_repeat/`；步骤3核验对应新目录。失败会保留 `status.json` 和已生成文件，查明原因后换空目录运行，不手工把状态改成完成。

如仅希望单独运行 fixed 对照，可执行以下命令；它不需要搜索，但不代表四组实验已完成：

```bash
./scripts/part1/run_middlebury.sh --experiments bm_fixed sgbm_fixed \
  --output-dir outputs/part1/middlebury/fixed_only
```

## 结果与分析

正式数值以本次运行生成的 `metrics_per_scene.csv`、`metrics_summary.json` 为准。以下四组表从本次`metrics_summary.json`的`all`字段完整精度值格式化，原始完整精度保留在JSON中。

| 实验组 | 视差MAE ↓ (px) | 有效预测比例 ↑ | Bad-1 ↓ | Bad-2 ↓ | Bad-4 ↓ | Depth AbsRel ↓ | Depth RMSE ↓ (m) | 双向时间/场景 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm_fixed` | 1.30384347 | 0.49337005 | 0.55984426 | 0.54537006 | 0.53659726 | 0.01661978 | 0.16746121 | 0.01498008 |
| `bm_tuned` | 1.03000902 | 0.48444870 | 0.55353284 | 0.54475649 | 0.53831721 | 0.01297039 | 0.14687215 | 0.01388909 |
| `sgbm_fixed` | 1.38256272 | 0.63368556 | 0.44526010 | 0.42056597 | 0.40452344 | 0.01565573 | 0.15705546 | 0.02544203 |
| `sgbm_tuned` | 1.30868186 | 0.62352049 | 0.44910917 | 0.42567105 | 0.41083965 | 0.01498266 | 0.15076708 | 0.02424121 |

此表对应15场景宏平均；JSON另有5场景开发组与10场景独立评估组的同样指标。CSV字段分别为 `disparity_mae_px`、`valid_prediction_fraction`、`bad1/bad2/bad4`、`depth_abs_rel`、`depth_rmse_m`；全部参与保存数组的独立核验。辅助指标用于完整分析，不改变开发集的MAE/有效比例选参规则。

本次BM和SGBM的tuned组都降低了15场景MAE和深度误差，但有效预测比例也分别降低约0.89与1.02个百分点。SGBM fixed的有效比例为63.37%，高于BM fixed的49.34%，Bad-1/2/4也更低；BM tuned在所保留的有效像素上MAE最低，不能因此认定它整体最优。

开发集选参约束本次均满足，独立评估组却没有保持同样的有效比例：BM从49.0844%降到47.5516%，SGBM从62.7134%降到61.1808%，对应Bad-2均升高。说明“开发集MAE改善且有效比例不降”没有保证泛化到独立场景，也说明只看有效处误差会漏掉输出缺失的代价。没有按独立评估结果重选参数。

`runtime_seconds` 测量一次左匹配和一次右匹配的 `matcher.compute` 总耗时，不含读图、LR、转换或绘图；它不是整个流水线的端到端时间。`run_metadata.json` 保存CPU型号、逻辑核数、OpenCV/NumPy/Python版本、线程数、各图输入尺寸与实际精度。由于使用CPU，GPU峰值显存记为 `null`（不适用），不能写为实测0。

### 可视化与无效区域

每个 `scene/experiment/visualization.png` 为紧凑的2×3六联图，按从左到右、从上到下依次为：

| 第一行 | 第二行 |
|---|---|
| 1. 左RGB | 4. 绝对视差误差 |
| 2. GT视差 | 5. Matcher invalid：匹配器原始无效区域 |
| 3. 最终预测视差 | 6. Final invalid：最终无效区域 |

GT和最终视差使用本场景相同色标范围；误差图显示0–4 px（更大误差饱和显示，不改变数值评价）。图像保持完整尺寸比例，六幅图的显示大小一致，行列间距与标题留白收紧；色条单独留位，不挤压图像。左右一致性仍保存在独立掩码和原始数组中。

无效区域和LR诊断覆盖**整幅图像**，不受GT是否存在影响。每组另保存原尺寸、无损掩码：

| 文件 | 编码 |
|---|---|
| `matcher_invalid.png` | 255表示OpenCV原始无效编码 |
| `invalid.png` | 255表示最终无效预测，包括LR与几何检验失败 |
| `lr_inconsistent.png` | 255表示匹配器有输出、但LR未通过或无可用对应 |
| `lr_status.png` | 0=匹配器无效，1=LR未通过/无对应，2=LR通过 |

需要一起观察左侧搜索边界、遮挡、低纹理、细结构和重复纹理。LR失败不等于已由GT证实错误；它表示双向对应没有满足设定条件。`prediction.npz` 同时保留原始左右视差和布尔状态，可以核查彩色图隐藏的无效位置。

### 六联图最左侧为什么是白色

本次逐一核对了60份NPZ：原始视差、最终视差、深度、各个布尔mask的高宽都与本场景左RGB相同。四张单独PNG也与对应布尔状态逐像素一致。左边不是被裁掉或没有保存；只是白色无效像素与白色画布融在一起，视觉上像图变窄。六幅图按相同比例和显示大小排列，色条不改变图像轴的尺寸。

按从左到右、从上到下编号：第2图白色表示GT无效位置，第3图白色表示无效预测视差NaN；第4图白色通常是没有可计算误差的位置，不能读成误差为0；第5/6图白色明确表示该像素无效（true），黑色才是该项有效。独立的`lr_status.png`继续用0/1/2保存匹配器无效、LR未通过/无对应、LR通过三种状态。

左侧整条无效带有两个不同原因：

1. **真实对应可能在右图外。** 本项目使用`x_right=x_left-d_left`。若左图x=20、视差d=40，对应x_right=-20，不可能从这对图中匹配。
2. **OpenCV完整搜索范围的边界限制。** 即使某个像素的真实对应在右图内，OpenCV仍可能因完整候选搜索范围无法容纳而不计算。当前OpenCV 4.11.0的SGBM使用`minX=max(minDisparity+numDisparities,0)`作为左计算边界；BM还受匹配窗口边界限制。参见[官方SGBM源码](https://github.com/opencv/opencv/blob/4.11.0/modules/calib3d/src/stereosgbm.cpp#L1519)、[官方BM源码](https://github.com/opencv/opencv/blob/4.11.0/modules/calib3d/src/stereobm.cpp#L1181)。

| 场景 | 图像宽度 | 搜索数量 | SGBM最左完整无效列数 | BM fixed最左完整无效列数 |
|---|---:|---:|---:|---:|
| Adirondack | 718 | 80 | 80 | 86 |
| Jadeplant | 659 | 160 | 160 | 166 |
| PianoL | 707 | 80 | 80 | 86 |

例如Jadeplant的SGBM前160列中有77,333个有效GT像素，其中66,926个按GT计算的右坐标在图内；进一步限定为官方非遮挡mask时仍有47,039个。一个实际例子是`x_left=32, y=228, d_GT=17.40144`，右坐标约14.59856，仍在画面内，却位于OpenCV不计算的带中。因此不能把整条白带解释成“右相机没有拍到”。这里用GT及非遮挡mask只是事后解释边界，主评价和预测有效域没有因此更改。

原始匹配器的无效像素之外，固定1 px的双向检验还会使最终预测进一步稀疏。例如Jadeplant SGBM fixed在完整GT域的原始输出比例约55.74%，最终有效比例约36.12%。这体现OpenCV返回结果与额外可靠性筛选的不同；第6图比第5图的白色无效区域更多是预期行为，不能把所有LR失败都当作真实遮挡。

**实现判断：课程要求得到满足，算法并不因此具有完整覆盖。** 当前运行使用官方OpenCV BM/SGBM、完整15组quarter图、正确单位转换、全有限正GT域的MAE/有效比例与明确无效规则；白带计入有效比例分母和Bad失败数，没有从评价中裁掉。左右一致性是课程可选项，本项目采用它进行显式筛选，须披露对覆盖的代价。没有为了填白边复制图像边缘、插值填洞、缩小视差搜索范围或按GT挑选ROI；本次无需重跑正式搜索/匹配。

本次只读审查的逐场景边界、尺寸、掩码和指标证据位于 `outputs/maintenance/part1a_boundary_review_20261102/review.json`，原始预测与运行元数据保持真实；当前六联图的展示来源另行记录。

### 米制与距离分析

辅助报告深度AbsRel、RMSE及 `[0,2)`、`[2,4)`、`[4,+∞)` 米分层结果。无样本的距离层记为 `null`，宏平均明确列出有定义场景数；这些米制诊断不与NYUv2模型排名混为一表。

`sensitivity.csv/png` 对GT视差注入 ±0.25/0.5/1/2 px，比较精确深度误差和一阶近似 `|Δz| ≈ z²|Δd|/(f×B_m)`，四组共享这份几何分析。`empirical_depth_error_by_distance.csv/png` 则按0.25米分桶，分别统计四组真实预测的视差MAE、深度MAE、有效比例及像素支持量。分桶是域内像素汇总，不等同于主表逐场景宏平均；距离越远越敏感不保证某次实际匹配误差单调上升。

## 误差—有效预测比例与泛化

### 开发集上的两种比较规则

MAE只评价有输出的位置，因此误差下降可能伴随更多空白。除四组主表外，在相同5个开发场景与同一参数网格上，分别冻结以下两种选择，再检查10个独立评估场景：

| 选择 | 开发集约束与目标 |
|---|---|
| `fixed` | 对应算法的固定参数参考 |
| `matched_coverage` | 有效比例与fixed相差不超过**1个百分点**，在其中选MAE最低者；平局时优先有效比例高、候选序号小 |
| `matched_error` | MAE不高于fixed，在其中选有效比例最高者；平局时优先MAE低、候选序号小 |

这两个英文名称指**开发集选参条件**，不表示独立评估场景已经具有相同覆盖率或相同误差。例如开发集fixed有效比例为50%时，matched_coverage允许49%–51%，不是只能等于50%。主实验`tuned`的“有效比例不低于fixed”规则继续单独保留。

完整记录有BM 145行、SGBM 37行：144/36套网格分别另加一行fixed参考，重复参数不删除。逐场景开发分数共910行。参数冻结文件写出后，才加载10个独立评估场景的图像和GT数值；已有独立结果不参与选择。每算法三组，共60份独立评估预测。

本次冻结参数如下；其余参数、quarter标定与固定LR=1 px均沿用前述主协议。候选0为额外fixed，1起按YAML网格顺序枚举。

| 算法 | 选择 | 候选序号 | block | uniqueness | speckle window | texture |
|---|---|---:|---:|---:|---:|---:|
| BM | `fixed` | 0 | 15 | 10 | 100 | 10 |
| BM | `matched_coverage` | 45 | 9 | 0 | 100 | 20 |
| BM | `matched_error` | 40 | 9 | 0 | 50 | 0 |
| SGBM | `fixed` | 0 | 5 | 10 | 100 | — |
| SGBM | `matched_coverage` | 9 | 3 | 10 | 100 | — |
| SGBM | `matched_error` | 2 | 3 | 0 | 50 | — |

### 运行与核验

配置为[configs/part1/middlebury_coverage.yaml](../configs/part1/middlebury_coverage.yaml)，入口为[scripts/part1/study_middlebury_coverage.sh](../scripts/part1/study_middlebury_coverage.sh)。本地完整搜索、60份原生预测、区域统计、图和独立重算已完成，默认目录已有结果；以下命令用于复现，运行入口拒绝覆盖已有目录。

```bash
# 首次完整运行：CPU，aiaa3201，无需GPU或训练。
./scripts/part1/study_middlebury_coverage.sh

# 已有结果可直接只读核验，默认打印报告，不覆盖原verification.json。
./scripts/part1/study_middlebury_coverage.sh --mode verify
```

需要从头再运行时，为run和verify指定同一个**新空目录**：

```bash
./scripts/part1/study_middlebury_coverage.sh   --output-dir outputs/part1/middlebury/coverage_reproduce_v1

./scripts/part1/study_middlebury_coverage.sh --mode verify   --output-dir outputs/part1/middlebury/coverage_reproduce_v1
```

输入、源代码与配置分别保存指纹和快照。verify会重放开发表及选参，从保存的原始左右视差重算LR、最终有效域、米制深度与精度，再重算五类区域表和汇总；不重新调用匹配器。失败保留状态与中间文件，不手改状态或覆盖重跑。

### 独立评估场景的结果

完整精度来源为`coverage_study_v1/held_out.csv`与`summary.json`，以下均为**10个独立评估场景**的等权平均，不与前面的15场景主表混排。Bad-2把无效预测计为失败。

| 算法 | 开发集选择 | 视差MAE↓（px） | 有效预测比例↑ | Bad-2↓ |
|---|---|---:|---:|---:|
| BM | `fixed` | 1.04526212 | 49.084447% | 54.840839% |
| BM | `matched_coverage` | 0.76862694 | 46.188786% | 56.451527% |
| BM | `matched_error` | 0.85488592 | 47.556232% | 55.425266% |
| SGBM | `fixed` | 1.23006771 | 62.713380% | 43.008139% |
| SGBM | `matched_coverage` | 1.16035400 | 60.948217% | 44.158210% |
| SGBM | `matched_error` | 1.43719311 | 63.832454% | 42.693327% |

**BM：更小的条件误差没有消除输出缺失的代价。** matched_coverage的MAE从1.0453降至0.7686，但有效比例从49.08%降至46.19%，下降约2.90个百分点，已超出开发集的±1个百分点条件；Bad-2由54.84%升至56.45%。matched_error同样降低MAE、减少覆盖。因此不能把较低MAE写成“同样覆盖率下全面更好”。

**SGBM：开发集约束不保证评估集仍成立。** matched_coverage略降MAE，也降低覆盖；matched_error将有效比例从62.71%提高到63.83%，但MAE从1.2301升到1.4372，开发集的“误差不增加”未推广到独立场景。其Bad-2却略降，说明输出数量与有效处误差确有取舍，不能用任何一项指标代替整体判断。

这与主实验tuned的泛化观察一致：需要同时报告误差、覆盖率、无效预测代价和独立场景表现。这里的两个规则不是新的匹配算法，也不依据评估表反过来重选参数。

### 分区域表现

五类区域共300行：10场景×2算法×3种选择×5区域。每行额外保留GT、有效与无效像素数，以及Bad-1/2/4。

- `all_GT`：全有限正GT域，与课程主指标一致。
- `non_occluded` / `occluded`：分别使用官方mask的255 / 128标记，再与有效GT相交；0不视为已标注遮挡。
- `low_texture`：左图灰度7×7局部方差<25。
- `depth_boundary`：GT视差的3×3 Sobel x/y梯度绝对和>4。无效GT先置0再计算梯度，因此孔洞附近可能触发；这是深度突变proxy，不能当作纯物体语义边界。

这些区域可以重叠，不能把区域误差或覆盖率相加。空GT区域不参与该区域均值，但保留支持场景数；有GT而没有有效预测时，覆盖率为0、MAE为null，不静默删去该场景的失败。

下面每格为“**MAE（px） / 有效比例**”；各区域本次都有10个支持场景，完整非遮挡结果及计数见CSV。

| 算法 / 选择 | 遮挡区域 | 低纹理区域 | 深度边缘区域 |
|---|---:|---:|---:|
| BM / `fixed` | 7.0822 / 14.79% | 0.8429 / 41.57% | 2.8425 / 43.66% |
| BM / `matched_coverage` | 6.7926 / 12.00% | 0.5205 / 37.72% | 2.6951 / 37.42% |
| BM / `matched_error` | 6.9492 / 12.82% | 0.6124 / 38.90% | 2.7807 / 40.17% |
| SGBM / `fixed` | 6.6603 / 18.98% | 1.6008 / 56.68% | 2.7511 / 63.92% |
| SGBM / `matched_coverage` | 6.1174 / 17.13% | 1.5838 / 55.40% | 2.5183 / 62.02% |
| SGBM / `matched_error` | 6.5199 / 18.32% | 2.0651 / 59.56% | 2.6831 / 64.63% |

遮挡区域只有约12%–19%的有效预测，留下的预测MAE仍约6–7 px；全图均值不能代表这类困难位置已解决。BM在低纹理区域降低条件误差的同时，覆盖也从41.57%降到37.72%；边界区域的matched_coverage覆盖从43.66%降到37.42%，同样存在筛掉困难位置的代价。SGBM matched_error在低纹理处多给出预测，但MAE由1.6008升到2.0651，反映参数取舍对不同区域影响不同。GT仅用于评分与事后分组，不用于预测有效性筛选。

### 开发集误差—覆盖率图

[BM图](../outputs/part1/middlebury/coverage_study_v1/bm_pareto.png)与[SGBM图](../outputs/part1/middlebury/coverage_study_v1/sgbm_pareto.png)均只使用5个开发场景。横轴越右表示有效比例越高，纵轴越低表示有效预测的MAE越小；蓝线连接没有被其他候选同时超过的选择，即Pareto前沿。点落在前沿不代表它能在独立评估场景保持优势，重叠点也不表示少搜索了候选。

### 实际环境与验证范围

本轮使用`aiaa3201`、Python 3.10.22、NumPy 1.24.0、OpenCV 4.11.0，Intel Core Ultra 9 285K、OpenCV单线程，GPU不适用。主表时间仍使用原四组运行记录，不将本轮搜索/绘图总耗时混成受控速度比较。

独立核验通过60份预测、1,200个精度值、300行区域统计与910行开发分数，选参重放通过。原始左右视差、LR/几何有效域、米制转换和区域分母均被检查。同协议证据的6套参数选择相同，182行开发表和300行评估表的964个数值在`rtol=1e-6, atol=1e-8`内一致；本实现沿用float64评分，不为追求逐位一致修改预测。参考副本只在本地结果生成后用于核对，保存于provenance，不进入选参。

完整精度、环境与实际命令见`run_metadata.json`；`verification.json`记录首次核验，独立复核另存在`outputs/maintenance/part1a_coverage_20261104/recheck.json`。原四组运行和搜索不变，本轮501份既有科学文件/产物指纹核对一致。该次1A研究新增10项合成检查通过，当次全项目为1,515项通过，14个既有Matplotlib/PyParsing弃用告警；后续测试集合调整的当前数量见implemented.md。

## 目录与完成门禁

```text
outputs/part1/middlebury/
├── tuning/
│   ├── status.json、summary.json
│   ├── bm/    # best、候选CSV、逐场景分数、fixed参考、provenance、status
│   └── sgbm/
├── coverage_study_v1/
│   ├── development.csv、development_per_scene.csv、selection.json
│   ├── held_out.csv、summary.json、bm_pareto.png、sgbm_pareto.png
│   ├── evaluation/<scene>/<selection>/  # 60份原始NPZ、标定/参数、六联图与掩码
│   ├── provenance/                    # 配置、源码与可选同协议参考副本
│   └── run_metadata.json、status.json、verification.json
└── evaluation/
    ├── <scene>/<experiment>/   # prediction.npz、metadata、六联图、四张掩码
    ├── metrics_per_scene.csv、metrics_summary.json
    ├── sensitivity.csv/png、empirical_depth_error_by_distance.csv/png
    ├── visualization_contact_sheet_<experiment>.jpg  # 每组一张，共四张
    ├── failure_zoom_Jadeplant_sgbm_tuned.png
    ├── provenance/             # 实际配置/科学源码与两份选参记录
    └── run_metadata.json、status.json、reload_validation.json、verification.json
```

完成需要：两次开发集搜索证据完整；四组各15场景；课程主指标与米制转换齐全；60组视差/无效/LR展示齐全；保存数组独立核验通过；主表、独立评估表现、效率与真实困难案例分析完成。本次完整开发搜索、60组正式预测/图与独立重算已完成，上表及边界/精度—有效比例分析已补齐。误差—有效比例研究的开发冻结、60份预测、300行区域结果和重算也已完成。最终课程报告与其他Part分别整合；不把经典双目的覆盖局限宣称为已解决。
