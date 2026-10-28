# Step 1 / Part 1A：Middlebury 经典双目

## 状态与协议

Step 1 / Part 1A 已有完成门禁记录；本次只读核对当前产物。冻结配置为 `configs/part1/middlebury_sgbm.yaml`，正式输出为 `outputs/part1/middlebury/frozen_v2/`。

实验分为四组：

- `bm_fixed`：固定经验参数 BM；
- `bm_tuned`：调参后 BM；
- `sgbm_fixed`：固定经验参数 SGBM；
- `sgbm_tuned`：调参后 SGBM。

BM 和 SGBM 使用相同的 5 个 development scenes 选择参数，其余 10 个 held-out scenes 不参与调参。左右一致性阈值固定为 1 px，不参与搜索。

## 运行指令

一次完成 BM 的 144 个候选和 SGBM 的 36 个候选：

```bash
./scripts/part1/tune_middlebury.sh
```

只调一个算法：

```bash
./scripts/part1/tune_middlebury.sh --methods bm
./scripts/part1/tune_middlebury.sh --methods sgbm
```

一次评估四组实验并生成原始预测、指标、可视化、理论敏感性、实际距离分桶和重载验证：

```bash
./scripts/part1/run_middlebury.sh
```

分别运行四组时必须使用不同输出目录：

```bash
./scripts/part1/run_middlebury.sh --experiments bm_fixed \
  --output-dir outputs/part1/middlebury/single_bm_fixed
./scripts/part1/run_middlebury.sh --experiments bm_tuned \
  --output-dir outputs/part1/middlebury/single_bm_tuned
./scripts/part1/run_middlebury.sh --experiments sgbm_fixed \
  --output-dir outputs/part1/middlebury/single_sgbm_fixed
./scripts/part1/run_middlebury.sh --experiments sgbm_tuned \
  --output-dir outputs/part1/middlebury/single_sgbm_tuned
```

这些操作是 OpenCV 经典算法的非学习式参数搜索与评估，不需要 GPU，也不会启动正式训练。

所有复现命令是使用说明，本次整理没有执行。若已有正式目录，运行时用 `--output-dir` 指向新的目录，保存当前冻结产物。

## 参数选择

选择目标为 development scenes 上最低宏平均 Bad-2，依次使用 disparity MAE 和 coverage 打破平局。

| 实验组 | `block_size` | `uniqueness_ratio` | `speckle_window_size` | `texture_threshold` |
|---|---:|---:|---:|---:|
| `bm_fixed` | 15 | 10 | 100 | 10 |
| `bm_tuned` | 15 | 0 | 0 | 10 |
| `sgbm_fixed` | 5 | 10 | 100 | 不适用 |
| `sgbm_tuned` | 3 | 0 | 0 | 不适用 |

## 15 场景结果

Bad-1/2/4 的分母包含 non-occluded GT 内的无效预测，因此不能通过降低 coverage 获得虚假的低 bad-pixel rate；MAE/RMSE 只在有效预测处计算，必须与 coverage 联合解读。

| 实验组 | Disp. MAE (px) | Disp. RMSE (px) | Bad-1 | Bad-2 | Bad-4 | Coverage | Depth AbsRel | Depth RMSE (m) | 双向时间/场景 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm_fixed` | 0.97901828 | 3.80651218 | 0.50281555 | 0.48746252 | 0.47931511 | 0.54356004 | 0.01333963 | 0.14203889 | 0.01555024 |
| `bm_tuned` | 1.73140054 | 5.75321666 | 0.43401738 | 0.40691315 | 0.38988034 | 0.65948429 | 0.02384765 | 0.21063200 | 0.00947368 |
| `sgbm_fixed` | 1.05875355 | 3.97086376 | 0.37351529 | 0.34741863 | 0.33234387 | 0.69669393 | 0.01225990 | 0.12824877 | 0.02495358 |
| `sgbm_tuned` | 1.30701556 | 4.51746390 | 0.36514327 | 0.33732003 | 0.32033001 | 0.71689359 | 0.01536107 | 0.14668840 | 0.02039792 |

BM 调参后 Bad-2 与 coverage 改善，但有效预测处 MAE 和深度误差变差；SGBM 也呈现相同权衡。因此报告必须联合展示 Bad-2、coverage 与有效像素误差。

数值来源为正式目录的 `metrics_summary.json`（`all`：15 场景宏平均）与 `metrics_per_scene.csv`，两者已核对。完整精度与来源 checksum 见 [当前结果审计](current_results.md)。时间列取当前产物，不能沿用早期记录。

## 深度敏感性

`sensitivity.csv/png` 对 GT disparity 注入 ±0.25/0.5/1/2 px。该理论/数值分析只由相机标定和

```text
z = fB / (d + doffs)
```

决定，与 BM/SGBM 参数无关，因此四组共享一份。`empirical_depth_error_by_distance.csv/png` 才分别统计四组实际预测的 depth MAE、disparity MAE、Bad-2 和 coverage 随 GT 距离的变化。

## 正式产物

`outputs/part1/middlebury/frozen_v2/` 包含：

- `metrics_per_scene.csv` 与 `metrics_summary.json`；
- 每个 `scene/experiment` 的 `prediction.npz`、`metadata.json` 和 `visualization.png`；
- 四张 `visualization_contact_sheet_<experiment>.jpg`；
- `failure_zoom_Jadeplant_sgbm_tuned.png`；
- `sensitivity.csv/png`；
- `empirical_depth_error_by_distance.csv/png`；
- `reload_validation.json`：60 个预测文件、1,200 个指标值的重载复算门禁。

调参证据位于 `outputs/part1/middlebury/tuning_v2/`。若有早期 `frozen_v1/`、`tuning_stage1/` 和 `tuning_stage2/`，它们只作为开发历史；本次没有要求重建这些目录。
