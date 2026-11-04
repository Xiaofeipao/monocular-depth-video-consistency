# Part 3 B：开发集全局尺度校准

这是已完成的改进实验，独立于 Part 2 原生米制基线和 C 边缘细化。
两条模型分别对官方 795 张 trainNdxs 做 RGB-only GPU 推理；对每张图，在课程 crop
和有效 GT 范围内计算 median(log(GT/pred))，再取 795 张中位数并指数化，得到一个全局尺度。
只保留正且有限预测用于拟合；固定尺度写入 frozen_scale.json 后才评价 testNdxs。
已独立核对两集合无交集。没有逐测试图拟合，也没有测试集参数搜索。

测试使用原先保存的各 654 张预测，RAW 与 B 使用相同的预测文件 SHA、crop、clip 与宏平均。
训练图中间预测没有公开上传；开发逐图比例与预测数组 SHA、冻结尺度、测试逐图分数均保留。

## 完整测试结果

- Depth Pro 全局尺度 1.0413147804；AbsRel 0.100473264 → 0.096599796，
  RMSE 0.392219569 → 0.380340021 m，δ1 0.915185154 → 0.922041795。
- UniDepthV2 全局尺度 0.9561571460；AbsRel 0.114541365 → 0.102022244，
  RMSE 0.398820607 → 0.389266687 m，δ1 0.892364681 → 0.899440251。

AbsRel 相对降低约 3.86% 和 10.93%。这是使用 NYU 开发标注后的域内校准，
不是零样本原生输出，也不是基础模型训练。只改变一个全局尺度，不能修复局部边界、
遮挡或结构错误；尚未评价 B 的跨域泛化和 SI Boundary F1，不能宣称这些方面改善。
C 的边界收益与 B 的米制收益是不同目标；尚未跑 B+C 组合，不把两者收益相加。

## 复现与证据

先准备 `outputs/full_benchmark_20261029/{depth_pro,unidepth_v2}_nyuv2/predictions/` 的
全部基线 NPZ（字段 depth_m），以及 data/nyuv2 中的 MAT/splits、checkpoints 下的权重和
third_party 下的官方代码。数据与权重 SHA 随 protocol.json 记录。

```powershell
. ./scripts/project_cache.ps1
python scripts/metric_resource_calibration_study.py --model depth_pro --output outputs/scale_depth_pro_new
python scripts/metric_resource_calibration_study.py --model unidepth_v2 --output outputs/scale_unidepth_new
```

脚本同时执行 16 图资源检查，该部分属于 Part 2 补充，不混入尺度收益表。
输出目录必须是新的。GPU 原始运行已经完成，不需要为了读取证据重新推理。
开发中间预测的 byte SHA 可能跨设备不同，复现不要求浮点逐位相同。
`docs/evidence/global_scale/` 保存两条完整开发／测试表和冻结尺度；独立复核通过：
795 行开发记录、1308 行 RAW/B 测试分数、尺度公式和三项宏平均均一致。

## 尚未完成的独立来源验证

同伴 frozen_v1 的原始预测尚未取得。在那批预测上运行 C、以及逐图定位复现差异仍待原始数组。
本次使用我们的既有基线进行 B 实验，不能描述成“同伴基线已验证”。
