# AIAA 3201 Project 3

项目研究经典双目、单目米制深度和视频时序一致性。课程要求以[课程PDF](Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf)为准，中文说明见[Project3_CN_final.md](Project3_CN_final.md)。Part 3采用 **B：开发集全局尺度校准**与 **C：固定边缘感知细化**，两项独立评价。

## 当前状态与操作入口

| 内容 | 当前状态 | 文档 |
|---|---|---|
| 公共环境与资源 | 两个推理环境、数据/权重、协议与测试已具备 | [Step 0](docs/step0.md) |
| Part 1A：Middlebury BM/SGBM | 两组fixed各15场景及误差—覆盖率研究完成；30/60份预测重算通过 | [Step 1](docs/step1.md) |
| Part 1B：DA-V2-S逐帧relative | 指定片段70帧运行/分析/重算完成；保留NYUv2对齐扩展 | [Step 2](docs/step2.md) |
| Part 2A：Depth Pro / UniDepthV2-S | 两条654图原生米制基线及尺度/物体边界分析完成 | [Step 3](docs/step3.md) |
| Part 2B：VDA-S offline relative | 当前70帧独立及DA/VDA同帧比较/双方raw重算完成 | [Step 4](docs/step4.md) |
| Part 3 B：开发集全局尺度 | 两模型三规则完整拟合/评价/核验完成，收益与目标取舍已分析 | [Step B](docs/stepB.md) |
| Part 3 C：固定联合滤波 | 两模型六方法完整诊断/核验完成，官方指标与定位/宽度/纹理取舍已分析 | [Step C](docs/stepC.md) |
| 应用、报告与交付 | 待完成，不能以方法实现替代最终交付 | [Step 6](docs/step6.md) |

指定`data/davis_rollercoaster.mp4`实际为70帧、24 FPS，未重复/插帧补足100。需课程确认接受现有完整素材或提供更长版本；已完成70帧实验不等于该门禁通过。DA/VDA颜色均是relative，不代表共享米制尺度。

## 环境与基本用法

命令从项目根目录执行。先按[Step 0](docs/step0.md)准备固定版本上游，再创建Conda环境，不安装到base：

```bash
conda env create -f environment.yml
conda env create -f environment-unidepth.yml
```

主流水线和B/C评价使用`aiaa3201`，UniDepth原生推理使用`aiaa3201-unidepth`。B shell会自动为UniDepth开发推理选择独立环境；C直接读取现有预测，不再推理基础模型。

```bash
./scripts/part3/direction_b.sh check --model depth_pro
./scripts/part3/direction_b.sh check --model unidepth_v2
./scripts/part3/direction_c.sh check --model depth_pro
./scripts/part3/direction_c.sh check --model unidepth_v2
```

完整运行顺序、默认目录与独立重跑命令分别见Step B/C。正式训练和长时间运行由用户启动。输出不覆盖；小子集检查明确为`smoke_complete`，不能用于冻结正式尺度或标记全量完成。

## 固定协议与结果来源

NYUv2使用官方654张test、480×640、crop `[45:471,41:601]`、`0.001 < GT < 10 m`，仅评分时裁剪预测，逐图等权平均。Part 2原生米制轨道保持`alignment:none`、RGB-only，不接受测试GT拟合尺度或额外相机输入。

| Part 2模型 | AbsRel↓ | RMSE↓（m） | δ1↑ |
|---|---:|---:|---:|
| Depth Pro | 0.10047157 | 0.39221681 | 0.91517394 |
| UniDepthV2-S | 0.11454373 | 0.39884996 | 0.89217973 |

完整精度、来源和现有结果见[当前结果](docs/current_results.md)。Middlebury按全有限正GT域联合报告视差MAE与有效预测比例；视频proxy不当作深度准确率或真实尺度漂移真值。

Part 3以两个独立问题组织：B比较三种开发集全局尺度估计，C比较完整法、消融和经典对照，并用定位、宽度与纹理诊断解释边界效果。B每模型每规则只有一个尺度，使用795张开发GT；C六方法固定参数，不训练、不按test调参，也不组合B+C。

| 模型 | B_median_log AbsRel↓ | B_mean_log AbsRel↓ | B_absrel AbsRel↓ | 官方SI F1：RAW → C↑ |
|---|---:|---:|---:|---:|
| Depth Pro | 0.09660522 | 0.09650993 | 0.09643386 | 0.15927408 → 0.17111979 |
| UniDepthV2-S | 0.10203673 | 0.10194122 | 0.10147545 | 0.11300740 → 0.12257544 |

三种尺度规则均降低平均AbsRel，但B_absrel相对中位log规则的额外收益小，UniDepth RMSE还更差，不选为全面赢家。完整C提高官方SI，却降低低阈值召回，并在六法共同可测剖面上加宽约0.55996/0.58970 px；Depth Pro纹理伪边代理有下降证据，UniDepth完整C差值区间跨零，不能称全面锐化。

B流程为“开发预测→参考尺度与三规则冻结→四方法评价/核验”；C流程为“联合滤波/消融预测→经典对照与六法统一评价→定位/宽度/纹理及核验”。完整命令从头到尾分别见Step B/C。当前结果已存在，无需重跑，检查入口包括：

```bash
./scripts/part3_studies/direction_b.sh check --model depth_pro
./scripts/part3_studies/direction_c.sh check --model depth_pro
```

两模型六份完整尺度/诊断核验报告均passed、formal_complete:true；[结果复核](outputs/maintenance/part3_studies_results_20261106/review.json)记录15,135份SHA、全表/scene统计和实际图抽查。联合滤波CPU同公式全量重放的报告留存范围单列在Step C。测试阶段349项通过记录、小型检查和实际实验互相区分，历史见implemented。

官方开发/测试各249/215个scene，无场景交集；尺度仍仅用开发GT拟合。方法比较属于已查看测试结果后的探索性结论，不把该654图称为独立首次确认，也不按分数再选参数。B校准结果与C处理结果都不改写Part 2原生基线。

## Direction C v2：受控边界收窄

独立改进版见[Step C v2](docs/stepC_v2.md)，现有B/C的代码、配置和正式产物保持原位。v2以原C去噪为初值，通过双侧深度结构与RGB支持，受控收窄边界；复用已有795张开发预测，按scene划为636张tune与159张validation，两模型同六候选、同选参规则、各自冻结参数。正式比较RAW、原C、两个经典对照、完整v2及两个消融，联合报告SI、定位、宽度、纹理和米制误差。

两模型完整开发、冻结、654图七方法评价及四份重放报告均已完成并通过。Depth Pro的SI为RAW/C/v2：0.15927408/0.17111979/0.17388612，UniDepth为0.11300740/0.12257544/0.12035916。v2相对原C宽度减少0.42494/0.46976像素，但相对RAW仍宽0.13382/0.12143像素，未实现比RAW更锐的双目标；正式分析、消融、纹理/定位代价及完整命令见Step C v2。实现阶段359项测试通过，本轮不重跑长实验，不组合B+C或覆盖原有结果。新入口`./scripts/part3_c_v2/direction_c.sh`，新产物独立放在`outputs/part3/direction_c_v2/`。

## 项目架构

```text
configs/part1/、part2/、part3/   # 科学协议与任务设置
scripts/part1/、part2/、part3/   # Conda命令入口
pipelines/                      # 推理、评价、来源与核验编排
src/                            # 公共数据/模型/指标/几何、B/C算子
src/refinement/                 # 全局尺度、固定C、消融与main来源/许可
src/part3_studies/              # 尺度估计、经典对照、共同GT诊断
pipelines/part3_studies/        # 三规则/六方法统一评价与来源核验
configs/part3_studies/          # 三尺度规则与六方法诊断的预声明设置
scripts/part3_studies/          # 统一比较Conda入口（aiaa3201）
src/part3_c_v2/                 # 受控边界收窄，独立于现有C
pipelines/part3_c_v2/           # 开发选参/验证、冻结、七方法评价与重放
configs/part3_c_v2/             # v2固定设置、六候选与场景划分
scripts/part3_c_v2/             # v2 CPU后处理入口（aiaa3201）
data/                          # 数据与指定视频（不上传）
checkpoints/                    # 基础模型权重（不上传）
outputs/part1/、part2/           # 已有只读科学产物
outputs/part3/                  # B/C完整正式结果、尺度、消融与案例
outputs/maintenance/            # 小型核验、保护与删除审计
```

详细门禁见[PLAN.md](PLAN.md)，实际实施与验证见[implemented.md](implemented.md)。测试使用`conda run -n aiaa3201 python -m pytest -q`；合成测试不替代真实实验。

完整项目（Part 1/2、B、C v1/v2）通过工作站7897代理更新GitHub分支，见[上传指南](docs/github_upload_project.md)。用`scripts/publishing/export_project.py`生成项目外白名单包，再在独立克隆中完整替换目标分支；视频、data、checkpoints、outputs与第三方仓库均不上传，无需转移到本地电脑。队友分支审查在[analysis_branch.md](github_repository/analysis_branch.md)，main Part 3审查在[analysis.md](github_repository/analysis.md)；借用实现的来源与MIT许可保存在`src/refinement/`，第三方仓库保持独立。

Part 1A入口为`./scripts/part1/run_middlebury.sh`（两组fixed）和`./scripts/part1/study_middlebury_coverage.sh`（matched_coverage/matched_error开发选择）；输出分别为`fixed_v1/`和`coverage_study_v1/`。fixed从已有数组原样整理并重算，研究原始记录与源码快照保留；当前无需重新匹配。完整协议与指令见[Step 1](docs/step1.md)。

[2026-11-14项目清理](docs/project_cleanup.md)已删除旧自录视频、抽帧缓存、旧`outputs/part1/video`、对应预检、过期上传包及项目自有Python缓存；指定DAVIS片段、NYUv2/Middlebury和B/C v1/v2正式结果完整保留。删除前逐文件SHA、来源登记快照与保护核对见`outputs/maintenance/project_cleanup_20261114/`。旧上传说明归档到同一维护目录，当前使用上述完整指南。
