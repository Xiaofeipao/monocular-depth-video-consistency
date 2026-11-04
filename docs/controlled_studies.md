# Part 1/2：误差—覆盖率与独立复现核对

## Part 1：选参规则与保留场景结果

`scripts/stereo_coverage_study.py` 复用五个开发场景、十个保留场景与 BM/SGBM 参数网格。
已核对本地与同伴分支的 split、固定参数、搜索网格和固定搜索参数一致，匹配器实现相同。
该补充不改变同伴原有 tuned 选择；另列两种开发集规则：

- matched_coverage：在固定参数覆盖率 ±1 个百分点内，选择开发集 MAE 最低者。
- matched_error：在开发集 MAE 不高于固定参数的候选中，选择覆盖率最高者。

全部规则与参数先写入 selection.json，再读取十个保留场景。开发曲线不使用测试结果。
MAE 只在有效预测处计算，覆盖率分母是全部有限正值 GT，逐场景等权。
重复候选保留，图中重叠不代表缺少候选；五场景网格总计 BM 145、SGBM 37（包含额外固定参考）。

保留场景宏平均：

- BM fixed：MAE 1.045262 px，coverage 49.0844%。
- BM matched_coverage：MAE 0.768627 px，coverage 46.1888%。
- BM matched_error：MAE 0.854886 px，coverage 47.5562%。
- SGBM fixed：MAE 1.230068 px，coverage 62.7134%。
- SGBM matched_coverage：MAE 1.160354 px，coverage 60.9482%。
- SGBM matched_error：MAE 1.437193 px，coverage 63.8325%。

开发集上的匹配条件并未在保留场景上自动成立，不能把较低 MAE 宣称为同覆盖率优势。
SGBM matched_error 的保留场景 MAE 反而上升，属于真实负结果。可借此讨论选参的泛化。
各区域结果见 held_out.csv：non_occluded/occluded 使用数据集 mask；low_texture 使用
7×7 灰度局部方差 <25；depth_boundary 使用 GT Sobel 梯度绝对和 >4。
后两者是启发式诊断，区域可重叠，边界阈值不等于独立人工语义边界。

复现（在仓库根目录、已准备 Middlebury 数据与核心依赖）：

```bash
python scripts/stereo_coverage_study.py --output outputs/stereo_coverage_reproduce_v1
```

现有文件见 [证据](evidence/controlled_studies/stereo/summary.json)，两幅 Pareto 图和
development.csv、held_out.csv、冻结 selection.json 同目录。属于实验补充，不是新双目算法。

## Part 2：逐图核对

比较本地保存的历史同伴 frozen_v1 逐图表与独立运行的全量基线，双方各 654 张。
参考逐图表随证据保留；它不是本次从同伴机器取得的原始预测，不能把它冒称为最新运行的数组。

Depth Pro 的 AbsRel 平均差约 +1.69e-6；UniDepthV2 约 -2.36e-6。
但 UniDepth 的最大单图 δ1 差约 0.09781，不能只凭很小的平均差认定全部差异是浮点噪声。
逐图 CSV、四种指标的差值和来源文件 SHA 均保留，图像身份严格匹配。
我们两条基线的 1308 份预测 SHA 已与本地原始 NPZ 逐份核对。

```bash
python scripts/audit_metric_reproduction.py --reference path/to/reference.csv --candidate path/to/candidate.csv --output outputs/metric_audit_new
```

两条输入表必须为同一 654 张 test 图，sample_id 唯一。核对不执行模型，不能定位
尚未取得的原始预测差异，也不能据此断言 CUDA、插值或依赖版本是差异原因。
附带的 baseline_* 目录是我们的独立运行证据，不替换同伴既有汇总。

## 资源实验与 Part 3 的边界

分辨率/计时/显存实验单独归档，开发集全局尺度校准属于 Part 3 B。
在同伴原始预测上重新验证 C，仍须先取得其 NPY/NPZ；已有指标表不能充当预测输入。
固定测试集不得用于选择尺度或分辨率“最优”参数。
