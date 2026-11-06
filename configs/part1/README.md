# Part 1 配置

[`middlebury_sgbm.yaml`](middlebury_sgbm.yaml) 定义Part 1A四组实验、5/10场景划分、quarter标定、毫米转米、固定LR=1 px、全有效GT域的视差MAE与有效预测比例，Bad-1/2/4、Depth AbsRel/RMSE以及距离分析。fixed参数明确给出；tuned参数从完整开发搜索证据加载，不手填占位值。

```bash
./scripts/part1/tune_middlebury.sh
./scripts/part1/run_middlebury.sh
./scripts/part1/run_middlebury.sh --mode verify \
  --output-dir outputs/part1/middlebury/evaluation
```

调参写入 `outputs/part1/middlebury/tuning/`，正式评估写入 `outputs/part1/middlebury/evaluation/`，拒绝覆盖已有目录。BM与SGBM在共同5场景独立选择有效比例不低于各自fixed参考时MAE最小的参数；其余10场景不参与选择。本次完整正式运行与核验已完成，独立复现请使用新空目录，协议、候选、掩码和产物说明见[Step 1](../../docs/step1.md)。

Part 1B当前主线为[dav2_video.yaml](dav2_video.yaml)，入口`scripts/part1/run_dav2_video.sh`，指定`data/davis_rollercoaster.mp4`的连续原始帧、原FPS、640长边、DA-V2-S/FP16。固定整段色标、保存raw及可选时序/边缘诊断，不依赖VDA或RAFT；当前70帧运行/重算已完成，100帧素材问题保留。详见[Step 2](../../docs/step2.md)。

[dav2_nyuv2.yaml](dav2_nyuv2.yaml)为保留的GT辅助aligned_relative扩展，标准654张完整结果位于`outputs/part1/dav2_nyuv2/frozen_v1/`，不修改既有预测或评价。它不作为新版视频主线或Part 2A原生米制结果。旧自录2B需要的共同配置/源码仍只读保留，独立1B重复入口和未绑定配置已清理。

[`middlebury_coverage.yaml`](middlebury_coverage.yaml)引用同一四组基础协议，定义开发集误差—有效比例取舍和区域诊断；入口`./scripts/part1/study_middlebury_coverage.sh`提供完整run与只读verify。独立输出`outputs/part1/middlebury/coverage_study_v1/`，不改变原tuned选择。完整本地搜索、60份原始预测与300行区域表已运行并重算，规则、命令和结果见[Step 1](../../docs/step1.md)。
