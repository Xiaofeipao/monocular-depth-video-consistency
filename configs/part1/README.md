# Part 1 配置

[`middlebury_sgbm.yaml`](middlebury_sgbm.yaml)定义两组fixed基线、quarter标定、LR=1 px、全GT域的MAE/覆盖、Bad与米制/距离诊断。`development_search`保存研究共用网格，不存在独立的第三条选参流程。

```bash
./scripts/part1/run_middlebury.sh
./scripts/part1/run_middlebury.sh --mode verify
./scripts/part1/study_middlebury_coverage.sh
./scripts/part1/study_middlebury_coverage.sh --mode verify
```

前两条写入/核验`outputs/part1/middlebury/fixed_v1/`；后两条使用[`middlebury_coverage.yaml`](middlebury_coverage.yaml)，开发集冻结matched_coverage/matched_error后评价10个独立场景，输出`coverage_study_v1/`。默认目录已完成，run拒绝覆盖；重新运行请指定新空目录。当前30份fixed与60份研究预测均通过原始重算，完整协议/指令/结果见[Step 1](../../docs/step1.md)。

Part 1B当前主线为[dav2_video.yaml](dav2_video.yaml)，入口`scripts/part1/run_dav2_video.sh`，指定`data/davis_rollercoaster.mp4`的连续原始帧、原FPS、640长边、DA-V2-S/FP16。固定整段色标、保存raw及可选时序/边缘诊断，不依赖VDA或RAFT；当前70帧运行/重算已完成，100帧素材问题保留。详见[Step 2](../../docs/step2.md)。

[dav2_nyuv2.yaml](dav2_nyuv2.yaml)为保留的GT辅助aligned_relative扩展，标准654张完整结果位于`outputs/part1/dav2_nyuv2/frozen_v1/`，不修改既有预测或评价。它不作为新版视频主线或Part 2A原生米制结果。旧自录2B需要的共同配置/源码仍只读保留，独立1B重复入口和未绑定配置已清理。

