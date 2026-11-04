# Step 5 / Part 3：边缘保留细化、消融与应用

## 任务、状态与协议

正式主线为 C：在 Part 2 原生米制预测上做固定 5×5 RGB/log-depth 联合引导的逆深度滤波。
不训练基础模型，不用测试 GT 拟合尺度，不在 654 张测试图上选参数。
C_cuda 为同一公式的 FP32 并行实现；CPU C 保留为参考实现。
既有结果已经完成；下文是新环境的复现流程，不表示本次整理重新执行过模型。

NYUv2 使用填充 `depths`、官方 654 张 test split、480×640、crop
`[45:471,41:601]`、`0.001 < GT < 10`，预测裁剪后逐图等权平均。
输入为未做 GT 对齐的米制预测，不接受 DAV2/VDA relative 输出直接充当米制深度。
本节承接同伴 Step 3 的任务定义；两个分支的文件接口尚未整体合并。

## 输入准备与代码归属

从仓库根目录执行。准备 NumPy/OpenCV/h5py/SciPy/PyTorch、NYU MAT/splits，
以及官方 Depth Pro 的边界评价依赖。CUDA 是 C_cuda 与组件消融的要求，CPU C 可单独运行。
依赖见根目录 requirements 文件和模型说明，不依赖某个私人环境名。

每个基线目录必须包含 `predictions/test_0000_mat_XXXX.npz` 等全部 654 个文件，
其中 `XXXX` 为官方 split 对应 MATLAB one-based 索引，字段为米制 `depth_m`。
先按 [基准复现](benchmark_results.md) 生成基线。接入同伴分支时应检查文件名、字段、
形状与官方索引；不要通过改名把不同预测伪装成同一运行。

- `pipelines/part3/refine_nyuv2.py`：主线入口。
- `scripts/evaluate_nyuv2_refinement.py`：读取基线、细化、课程评分与逐图来源 SHA。
- `src/exploration/edge_aware.py`：CPU/CUDA 的共同算法定义。
- `scripts/ablate_course_refinement.py`：去 RGB、去深度门控两组固定消融。
- `src/metrics/depth.py`、`src/metrics/boundary.py`：米制指标和官方 SI Boundary F1。
- `scripts/build_course_media.py`：已有交付布局上的应用与困难案例生成。

## 用户运行：检查、细化与消融

先检查输入路径和 split；该操作不执行模型、不创建输出。它仅检查输入存在性和 split 数量，
不是对所有 NPZ 内容、模型来源或既有数值的完整独立核验。

```bash
python -m pipelines.part3.refine_nyuv2 --run outputs/full_benchmark_20261029/depth_pro_nyuv2 --output outputs/part3/depth_pro_C_reproduce_v1 --check-only
```

GPU 上生成同公式 C_cuda、逐图评分和可复用浮点预测：

```bash
python -m pipelines.part3.refine_nyuv2 --run outputs/full_benchmark_20261029/depth_pro_nyuv2 --output outputs/part3/depth_pro_C_reproduce_v1 --backend cuda --save-predictions
```

Linux/Bash 也可用 `bash scripts/part3/run_refinement.sh` 传入相同参数。
只有 CPU 时改成 `--backend cpu`；不应将 CPU 耗时引用为 CUDA 实验效率。
UniDepthV2 使用对应基线目录、独立输出目录，参数和评价协议相同。
每次实际运行须使用新输出目录，已有正式输出不覆盖。

组件消融仅在 GPU 环境运行，使用同一批 Depth Pro 原始预测：

```bash
python scripts/ablate_course_refinement.py --run outputs/full_benchmark_20261029/depth_pro_nyuv2 --output outputs/part3/C_ablation_reproduce_v1
```

两条命令都支持 `--mat` 和 `--splits` 指向实际资源。没有重新运行模型的必要：细化读取
已保存的基线。消融不包含 RAW/完整 C，需与同一输入运行的 RAW/C 配对，不能拼接别的运行。

## 产物与核对

细化目录包含 `per_image.csv`、`summary.json`，加 `--save-predictions` 才包含 `predictions/*.npz`。
每张图记录原预测 SHA256；完整运行应有 1308 行 RAW/C 分数和 654 份细化预测（若启用保存）。
消融应有 1308 行两种组件移除结果。汇总按同一方法的 654 行等权计算。
该入口目前没有同伴式 `verify` 子命令；重算应从相同原始预测写入新目录，比较逐图数值和来源 SHA，
不能只比较汇总或把 CSV 一致视作重新计算了 GT 指标。

现有公开证据在 `docs/evidence/full_benchmark_20261029/` 和
`docs/evidence/course_submission/`；这些附件不是运行代码，也不替代下载数据与准备基线。
后处理效率与模型推理效率分开报告，既有 C_cuda 计时见 `docs/evidence/cuda_20261029/`。

## 现有结果、贡献与限制

Depth Pro：AbsRel 0.100473264 → 0.100366985，RMSE 0.392219569 → 0.391542824 m，
δ1 0.915185154 → 0.915454328；SI Boundary F1 0.159268547 → 0.171106615。
UniDepthV2-S：SI Boundary F1 0.113009609 → 0.122560834。
收益主要是边界，米制精度变化很小；不是超过基础模型论文的证据。

Depth Pro 完整 C 的 SI F1 为 0.171106615，去 RGB 为 0.137025218，去深度门控为
0.169701383；完整 C 并非每个精度指标最优。三个困难案例包括 C 实际退化。
iBims/Depth Pro 也出现边界退化，不能把本次 NYU 收益推广为跨域保证。

应用为提供的 NYU 图像 RAW/C 同设置背景虚化及预测相机点云。
三个真实困难案例、局部图、GT/预测与解释见 [交付说明](course_delivery.md)；
媒体生成仍使用该文档的固定已完成交付布局，不随上面的自定义输出自动切换。
自录视频是可选扩展，公开复现不要求获取私人素材。

## 其他方向保留与结论范围

A 光流融合、B 外部测点标定、D 启发式不确定性融合、E 光流后处理加速和 F 应用逻辑
继续由 [Part 3 入口说明](../pipelines/part3/README.md) 进入，不与 C 的图像实验混为一组。
B 缺独立实测标定点；D 的局部残差方差不是已校准的模型置信度；E 不等于主模型 ONNX/量化。
另外已完成使用 NYU 795 张开发图的 [B 全局尺度实验](global_scale_study.md)，
这是另一条有开发标注的校准路径，与外部测点实现分别保留。
最新 D/E 多序列研究见 [D 研究](d_research.md)，不能只引用早期短片结论。

完整结果见 [Part 3 结果](part3_results.md)，论文协议限制见 [论文比较](paper_comparison.md)。
70/100 帧素材例外、报告暂停及本地 ZIP 的状态仍以 [交付说明](course_delivery.md) 为准。
