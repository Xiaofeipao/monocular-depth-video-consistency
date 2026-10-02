# Part 1 配置

`middlebury_sgbm.yaml` 冻结了经典双目 Part 1A 的数据划分、四组实验参数、左右一致性阈值、评估指标、深度分桶和敏感性实验。

一次性复现 BM/SGBM 的单轮 development-set 调参：

```bash
./scripts/part1/tune_middlebury.sh
```

复现四组 15 场景正式结果：

```bash
./scripts/part1/run_middlebury.sh
```

正式输出位于 `outputs/part1/middlebury/frozen_v2/`，调参证据位于 `outputs/part1/middlebury/tuning_v2/`。BM 与 SGBM 都只使用配置声明的 5 个 development scenes 选参；其余 10 个 held-out scenes 不参与参数选择。

逐帧 Depth Anything V2 的 Part 1B 配置将在自录视频恢复后补充。
