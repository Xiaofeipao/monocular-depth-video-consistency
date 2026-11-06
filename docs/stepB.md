# Part 3 / Direction B：全局米制尺度校准

## 基本信息与状态

研究问题是：在不训练基础模型、不改变局部结构的条件下，用开发集确定一把统一的“尺子”，能改善多少米制误差？对Depth Pro和UniDepthV2-S分别比较RAW与三种全局尺度估计规则，所有规则都使用官方795张开发图拟合一个常数，再固定用于654张测试图。

**两模型开发预测、三规则尺度冻结、完整测试评价、四份全量核验报告与结果分析均已完成。** 三种规则全部报告，不按测试分数选赢家。与[Direction C](stepC.md)独立评价，不执行B+C，也不相加两者收益。

## 方法与评价协议

### 三种尺度估计规则

| 方法 | 开发集估计规则 | 拟合目标 |
|---|---|---|
| RAW | Part 2原生米制预测，不改变尺度 | 未校准参考 |
| B_median_log | 每图取median(log(GT/pred))，795图再取median后exp | 对极值较稳健的典型比例 |
| B_mean_log | 每图取mean(log(GT/pred))，795图再取mean后exp | 图像等权平均平方log误差最小 |
| B_absrel | 对GT/pred取加权中位数，权重为(pred/GT)/本图有效像素数 | 未裁剪预测的图像宏平均AbsRel最小 |

设第i张开发图有n_i个有效像素，p为预测、g为GT：

```text
s_median = exp(median_i(median_j(log(g_ij / p_ij))))
s_mean   = exp(mean_i(mean_j(log(g_ij / p_ij))))
s_absrel = weighted_median(g_ij / p_ij, weights=(p_ij / g_ij) / n_i)
calibrated_test = RAW_test × s
```

第三种利用`abs(s*p-g)/g = (p/g)*abs(s-g/p)`求解，不需要搜索测试尺度。每图先平均，避免有效像素更多的图主导目标；加权中位区间平局取较小端点。每模型、每规则均只有一个尺度，全部测试图共用；没有逐测试图尺度、scale+shift、另一模型预测或测试GT拟合。

**拟合与评分分开：**三种规则都不裁剪开发预测再拟合。B_absrel的最优性只针对未裁剪开发AbsRel，不代表课程裁剪后的指标或测试指标最优；B_mean_log也不保证其他米制指标最佳。

### 固定科学协议

- 数据为`data/nyuv2/nyu_depth_v2_labeled.mat`与`splits.mat`；官方trainNdxs为795张/249场景，testNdxs为654张/215场景，图像与场景均无交集，MATLAB索引减1。
- 模型输入为完整480×640 RGB，只使用RGB，不提供外部焦距或内参；Depth Pro开发推理在`aiaa3201`，UniDepth在`aiaa3201-unidepth`。
- 拟合与评分crop为`[45:471,41:601]`，GT满足`0.001 < GT < 10 m`；预测须正且有限，拟合不裁剪。
- 测试评分时裁剪预测到`[0.001,10] m`，七指标逐图等权平均；保存裁剪前float32米制数组，RAW与Part 2冻结逐图表核对。
- B属于**开发GT校准后的米制结果**，记录`native_metric:false`；`alignment:none`表示没有测试GT对齐，不表示未使用开发标注。Part 2原生基线只读保留。

统计采用按scene成簇的配对bootstrap，10,000次、seed=42，抽整场景并保留图像平均权重。区间是本划分上的探索性描述，未做多重比较校正，不保证跨域效果。三规则的比较受已查看的测试结果启发，报告不将同一654图宣称为完全未受测试信息影响的首次确认；拟合仍只用开发GT，规则不按测试成绩修改。

## 架构与产物

| 文件/目录 | 职责 |
|---|---|
| `configs/part3/direction_b.yaml` | 模型资源、原生基线、开发推理环境与输出根目录 |
| `configs/part3_studies/direction_b.yaml` | RAW/三尺度规则、阶段名称和配对统计设置 |
| `src/refinement/global_scale.py` | 单图中位log比例与尺度应用 |
| `src/part3_studies/estimators.py` | 图像等权的三规则和加权中位数求解 |
| `pipelines/part3/direction_b.py`、`scripts/part3/direction_b.sh` | 795图开发预测、单尺度参考冻结/评价与核验 |
| `pipelines/part3_studies/direction_b.py`、`scripts/part3_studies/direction_b.sh` | 三规则拟合/冻结、四方法评价与全量重算 |
| 两个流水线目录中的`common.py` | 固定协议、身份、配置/源码/输入/父运行SHA、状态与禁止覆盖 |
| `src/refinement/ORIGIN.json`、`MAIN_LICENSE.txt` | 实现来源及MIT许可 |

流水线按职责分阶段：开发预测提供共同输入，单尺度冻结提供B_median_log的独立参考，三规则冻结统一保存参数，测试阶段只应用已冻结常数。代码和目录名与来源记录保持一致。

```text
outputs/part3/direction_b/<model>/
├── development_v1/
│   ├── predictions/train_*.npz       # 795份原生depth_m
│   └── predictions.csv              # 身份、RGB/预测SHA、推理计时与显存
├── frozen_scale_v1/
│   ├── calibration_development.csv  # 单尺度参考的795行中位log比例
│   └── frozen_scale.json            # B_median_log参考尺度
├── evaluation_v1/                   # RAW/B_global单尺度参考结果
├── estimators_fit_v1/
│   ├── development_statistics.csv   # 795行：身份、有效像素、median/mean log比例
│   └── frozen_scales.json           # 三尺度、目标、未用测试GT及冻结时间
└── estimators_evaluation_v1/
    ├── predictions/<method>/       # 三种规则各654份depth_m；RAW只读引用Part 2
    ├── per_image.csv               # 四方法共2616行七指标
    └── summary.json                # 宏平均与相对RAW的scene配对区间
```

各运行都有`status.json`、`run_metadata.json`和`provenance/`，如实保存命令、时间、环境、科学源码及输入/产物指纹。`complete`、`smoke_complete`、`failed`不能互相冒充。`evaluation_v1`中的B_global与B_median_log是同一规则的参考结果；三规则主表统一使用`estimators_evaluation_v1`。

## 完整运行流程

以下按依赖顺序给出完整复现命令，**当前默认目录及四份三规则核验报告均已存在，无需重跑**。命令从项目根目录执行，输出及报告拒绝覆盖。先具备Part 2两模型的654张冻结原生基线；正式训练和长时间运行由用户启动。

### 1. 检查数据、权重、划分和原生基线

```bash
./scripts/part3/direction_b.sh check --model depth_pro
./scripts/part3/direction_b.sh check --model unidepth_v2
```

### 2. 生成共同的795张开发预测

```bash
./scripts/part3/direction_b.sh predict-development --model depth_pro
./scripts/part3/direction_b.sh predict-development --model unidepth_v2
```

这是GPU推理，不是训练。脚本自动为UniDepth选择独立环境，检查实际CUDA并记录真实精度/设备。已有完整开发预测时直接复用，不重复下载或推理。

### 3. 冻结中位log参考和三种尺度

```bash
for model in depth_pro unidepth_v2; do
  ./scripts/part3/direction_b.sh fit --model "$model"
  ./scripts/part3_studies/direction_b.sh check --model "$model"
  ./scripts/part3_studies/direction_b.sh fit --model "$model"
  ./scripts/part3_studies/direction_b.sh verify --model "$model"     --run "outputs/part3/direction_b/$model/estimators_fit_v1"     --report "outputs/maintenance/part3_studies_verify/${model}_b_fit.json"
done
```

单尺度参考是三规则拟合的父运行依赖，不是额外选参轮次；三规则中的B_median_log必须与它一致。fit和fit核验都读取全部795图开发GT，不接触测试GT。缺图、身份不符、非法预测或来源变化时停止，小子集不能冻结。

B_absrel需要对约1.9亿个开发像素比例排序，核验还会重新排序；预计峰值内存为10 GiB量级，建议预留约16 GiB以上可用内存。这是资源估计，不是正式峰值实测。两个模型都冻结并核验后，再开始测试评价。

### 4. 评价并核验全部规则

```bash
for model in depth_pro unidepth_v2; do
  ./scripts/part3_studies/direction_b.sh evaluate --model "$model"
  ./scripts/part3_studies/direction_b.sh verify --model "$model"     --run "outputs/part3/direction_b/$model/estimators_evaluation_v1"     --report "outputs/maintenance/part3_studies_verify/${model}_b_evaluation.json"
done
```

每模型得到RAW及三规则的654图分数和三组校准数组。测试核验重做尺度乘法、全部GT评分、逐图表、宏平均和scene统计；开发尺度求解是否正确由步骤3的fit核验负责，两类报告都保留。步骤3/4不加载基础模型，统一在`aiaa3201`运行。

### 5. 独立重跑与参考轨道

保留共同开发预测及参考尺度，只重算规则/测试时使用新的空目录；两阶段的核验也指向对应目录，报告文件名同样另取：

```bash
./scripts/part3_studies/direction_b.sh fit --model depth_pro   --output-dir outputs/part3/direction_b/depth_pro/estimators_fit_repeat
./scripts/part3_studies/direction_b.sh verify --model depth_pro   --run outputs/part3/direction_b/depth_pro/estimators_fit_repeat   --report outputs/maintenance/part3_studies_verify/depth_pro_b_fit_repeat.json
./scripts/part3_studies/direction_b.sh evaluate --model depth_pro   --frozen-dir outputs/part3/direction_b/depth_pro/estimators_fit_repeat   --output-dir outputs/part3/direction_b/depth_pro/estimators_evaluation_repeat
./scripts/part3_studies/direction_b.sh verify --model depth_pro   --run outputs/part3/direction_b/depth_pro/estimators_evaluation_repeat   --report outputs/maintenance/part3_studies_verify/depth_pro_b_evaluation_repeat.json
```

单尺度评价器也可独立生成/核验`evaluation_v1`参考结果，不是三规则测试的前置依赖；当前该目录已有结果，以下仅供完整复现：

```bash
for model in depth_pro unidepth_v2; do
  ./scripts/part3/direction_b.sh evaluate --model "$model"
  ./scripts/part3/direction_b.sh verify --model "$model"     --run "outputs/part3/direction_b/$model/evaluation_v1"
done
```

`evaluate --limit N`仅用于单列目录的接口检查，不是正式测试。失败保留真实中间产物和状态，不覆盖后改写为首次成功。

## 结果与分析

结果来自上述两模型的冻结尺度、795行开发统计、四方法测试CSV/summary，均为完整精度读取后显示；RAW与Part 2逐图一致，B_median_log与单尺度参考一致。

### 冻结尺度与七项指标

| 模型 | B_median_log | B_mean_log | B_absrel |
|---|---:|---:|---:|
| Depth Pro | `1.0413853532199127` | `1.0397601971598807` | `1.0277159306952304` |
| UniDepthV2-S | `0.9562419672512901` | `0.9552798937095953` | `0.9428846931615024` |

| 模型 | 方法 | AbsRel↓ | SqRel↓ | RMSE↓ (m) | RMSE-log↓ | δ1↑ | δ2↑ | δ3↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Depth Pro | RAW | 0.10047157 | 0.06400258 | 0.39221681 | 0.13533882 | 0.91517394 | 0.97985983 | 0.99187774 |
| Depth Pro | B_median_log | 0.09660522 | 0.06328021 | 0.38033847 | 0.12889871 | 0.92204947 | 0.98070064 | 0.99242283 |
| Depth Pro | B_mean_log | 0.09650993 | 0.06314725 | 0.38022687 | 0.12894608 | 0.92214636 | 0.98068666 | 0.99240837 |
| Depth Pro | B_absrel | 0.09643386 | 0.06257300 | 0.38086288 | 0.12981378 | 0.92219128 | 0.98050557 | 0.99227783 |
| UniDepthV2-S | RAW | 0.11454373 | 0.06895209 | 0.39884996 | 0.14121805 | 0.89217973 | 0.98073107 | 0.99333280 |
| UniDepthV2-S | B_median_log | 0.10203673 | 0.06328816 | 0.38928372 | 0.13495305 | 0.89942320 | 0.98051379 | 0.99270148 |
| UniDepthV2-S | B_mean_log | 0.10194122 | 0.06328762 | 0.38951204 | 0.13496080 | 0.89934747 | 0.98049208 | 0.99268329 |
| UniDepthV2-S | B_absrel | 0.10147545 | 0.06375829 | 0.39417307 | 0.13567229 | 0.89748076 | 0.98016608 | 0.99242588 |

三种规则对Depth Pro都放大距离，对UniDepth都缩小距离。AbsRel目标的尺度分别更接近1（Depth Pro）或缩得更多（UniDepth），说明不同拟合目标确实给出了不同校准，而不是同一公式换名。它们仍然各自只有一个全局常数。

### 相对RAW的收益与失败

| 模型 | 方法 | AbsRel相对降低 | RMSE相对降低 | δ1变化（百分点） | AbsRel改善 / 恶化图数 |
|---|---|---:|---:|---:|---:|
| Depth Pro | B_median_log | 3.8482% | 3.0285% | +0.6876 | 395 / 259 |
| Depth Pro | B_mean_log | 3.9431% | 3.0570% | +0.6972 | 398 / 256 |
| Depth Pro | B_absrel | 4.0188% | 2.8948% | +0.7017 | 418 / 236 |
| UniDepthV2-S | B_median_log | 10.9190% | 2.3985% | +0.7243 | 411 / 243 |
| UniDepthV2-S | B_mean_log | 11.0024% | 2.3412% | +0.7168 | 411 / 243 |
| UniDepthV2-S | B_absrel | 11.4090% | 1.1726% | +0.5301 | 395 / 259 |

三种规则的平均AbsRel都改善，但仍有236–259张图恶化。UniDepth的B_absrel虽有最低平均AbsRel，改善图数却少于另外两规则；“平均误差更低”与“更多图受益”是不同结论。所有UniDepth校准规则的δ2/δ3均略低于RAW，也不能写成七项全部改善。

按215个scene成簇、10,000次、seed42的配对区间，六组AbsRel差值（校准−RAW）均完全小于0。RMSE方面，五组区间完全小于0，UniDepth B_absrel为[-0.01574595, +0.00662512] m，跨零；δ1仅Depth Pro B_absrel的区间完全大于0，其余均跨零。平均值的方向和区间证据应分开报告。这些是本划分上的探索性区间，未做多重比较校正，不是跨域保证。

### 不同估计目标之间的取舍

`B_absrel`在两模型上得到本次最低平均AbsRel，但相对B_median_log只额外降低约0.1774%/0.5501%。它的“开发未裁剪AbsRel最优”不等于“测试全部指标最优”：Depth Pro最低RMSE来自B_mean_log，UniDepth最低RMSE和最高δ1仍来自B_median_log。

下面以B_median_log为参照直接配对，避免只比较各自与RAW的区间。差值为**比较规则−B_median_log**，误差负值较好；两者使用同一654图/215场景。

| 模型 | 比较规则 | AbsRel平均差 | AbsRel 95%区间 | RMSE平均差 (m) | RMSE 95%区间 (m) |
|---|---|---:|---|---:|---|
| Depth Pro | B_mean_log | -0.00009530 | [-0.00021826, 0.00002517] | -0.00011161 | [-0.00043032, 0.00020086] |
| Depth Pro | B_absrel | -0.00017137 | [-0.00119806, 0.00083905] | +0.00052440 | [-0.00214469, 0.00313727] |
| UniDepthV2-S | B_mean_log | -0.00009552 | [-0.00017044, -0.00001935] | +0.00022831 | [0.00004251, 0.00041836] |
| UniDepthV2-S | B_absrel | -0.00056128 | [-0.00158591, 0.00048701] | +0.00488935 | [0.00235653, 0.00749609] |

Depth Pro另外两种规则与中位log规则的这些差值区间都跨零，不能把小幅排名写成稳健胜出。UniDepth平均log规则的AbsRel虽略低，RMSE却略高；AbsRel规则的RMSE比中位log规则增加约0.004889 m，区间完全大于0，而额外AbsRel收益的区间跨零。**三种估计目标具有不同取舍，没有一种规则在全部指标上占优。** 保留三种结果并列报告，不根据这次测试榜单另选尺度。

## 核验与完成范围

每模型795份开发预测、795行开发统计、三个冻结尺度、2616行测试分数及三组各654份校准数组齐全。四份完整核验报告位于`outputs/maintenance/part3_studies_verify/`：

| 报告 | 核验内容 | 当前记录 |
|---|---|---|
| `<model>_b_fit.json` | 从795张开发GT/数组重做三种尺度并核对统计表 | 两模型均passed、formal_complete:true |
| `<model>_b_evaluation.json` | 654图/1962数组、RAW/校准GT指标、汇总及scene统计 | 两模型均passed、formal_complete:true |

元数据确认尺度冻结先于相应测试评价，`test_images_used_for_fitting=0`。[结果复核记录](../outputs/maintenance/part3_studies_results_20261106/review.json)核对完整来源/产物SHA、表格宏平均和配对统计，并抽查两模型位置21的尺度应用；其中开发像素级加权中位数的全量重算由上述fit报告记录，不将表级复核冒充完整排序。

B的静态实验材料与核验已具备。结论限于NYUv2开发到测试划分，不保证所有图改善，也不恢复漏掉的结构；未评价B的边界收益或B+C效果。应用、报告和实际提交见[Step 6](step6.md)。执行时间、版本和命令以各目录真实元数据为准，历史记录见[implemented.md](../implemented.md)。
