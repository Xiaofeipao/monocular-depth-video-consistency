# Part 2：米制深度与视频深度模型复现

本部分回答：原生米制模型的尺度是否准确？视频模型是否比 Part 1 逐帧模型更稳定？

## 运行入口

- `run_metric_depth.py`：Depth Pro 单图/逐帧米制预测，记录预测焦距。
- `run_video_depth.py`：Video Depth Anything 离线视频预测；相对和米制权重分开运行。

从仓库根目录运行：

```powershell
python -m pipelines.part2.run_metric_depth --input data/nyuv2/manifest.json --repo third_party/ml-depth-pro --checkpoint checkpoints/depth_pro.pt --output outputs/part2/depthpro_nyu --vis-min 0.1 --vis-max 10
python -m src.evaluate --run outputs/part2/depthpro_nyu --manifest data/nyuv2/manifest.json
python -m pipelines.part2.run_video_depth --input data/videos/static.mp4 --repo third_party/Video-Depth-Anything --checkpoint checkpoints/video_depth_anything_vits.pth --output outputs/part2/vda_static --fps 10 --max-side 640 --vis-min 0 --vis-max 50
python -m pipelines.part2.run_video_depth --input data/videos/static.mp4 --metric --repo third_party/Video-Depth-Anything --checkpoint checkpoints/metric_video_depth_anything_vits.pth --output outputs/part2/metric_static --fps 10 --max-side 640 --vis-min 0.1 --vis-max 10
python -m src.evaluate_video --run outputs/part2/metric_static
```

## 算法实现在哪

- [src/models/foundation.py](../../src/models/foundation.py)：`depth-pro` 和 `vda` 官方实现适配。
- [src/infer.py](../../src/infer.py)：公共图像/视频推理、计时和保存。
- [src/evaluate.py](../../src/evaluate.py)：统一深度与边界评估。
- [src/evaluate_video.py](../../src/evaluate_video.py)：统一时序评估。

预训练网络本体由 `third_party/` 中的官方仓库提供，本项目不复制网络源码。
权重和依赖准备见 [checkpoints/README.md](../../checkpoints/README.md)。
原生米制结果不允许 GT 尺度对齐；比较 DAV2/VDA 时使用一致的输入帧、分辨率与色标。
本部分结果放 `outputs/part2/`。冻结后的米制视频原始预测是 Part 3 的输入，Part 3 不重新推理或修改这些文件。
