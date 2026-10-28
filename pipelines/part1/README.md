# Part 1：几何与逐帧基线

本部分回答：经典双目如何得到米制深度？逐帧单目模型有哪些尺度和闪烁问题？

## 运行入口

- `run_stereo.py`：BM/SGBM、左右一致性、视差转米制深度、视差误差敏感性。
- `run_framewise_depth.py`：Depth Anything V2，每一帧独立预测相对深度。

从仓库根目录运行：

```powershell
python -m pipelines.part1.run_stereo --input data/MiddEval3/trainingQ --output outputs/part1/sgbm --method sgbm
python -m pipelines.part1.run_framewise_depth --input data/videos/static.mp4 --repo third_party/Depth-Anything-V2 --checkpoint checkpoints/depth_anything_v2_vits.pth --output outputs/part1/dav2_static --fps 10 --max-side 640 --vis-min 0 --vis-max 50
python -m src.evaluate_video --run outputs/part1/dav2_static --static-roi 20 80 20 120
```

ROI 按实际静态场景设置；其他参数含义见顶层 README。

## 算法实现在哪

- [src/stereo/run_baseline.py](../../src/stereo/run_baseline.py)：双目实验编排、逐场景指标和敏感性 CSV。
- [src/geometry.py](../../src/geometry.py)：BM/SGBM、左右一致性、`fB/(d+doffs)`、PFM/标定读取。
- [src/models/foundation.py](../../src/models/foundation.py)：`DepthModel` 的 `dav2` 分支。
- [src/infer.py](../../src/infer.py)：公共推理、原始预测保存和固定色标输出。

双目输出是米制深度；DAV2 输出是相对逆深度性质的值，不能称为原生米制深度。
图像/时序评分使用公共 `src.evaluate`、`src.evaluate_video`。
本部分结果放 `outputs/part1/`，作为 Part 2 的比较基线。
