# 2026-10-28：组员成果与本地工作互补

## 来源与合并范围

远端 `main` 原先停留在 `de2ad9a`，`git pull --ff-only origin main` 显示已同步。
组员新成果实际位于 `origin/JiongxuChen`，本次获取的提交为
`1462736a28281ce2df48641c8d6c611b12053529`。本地代码先保存为 `8f5d4c8`，再合并该分支，
保留双方 Git 历史。原始 LICENSE、PLAN 和 `docs/papers/` 全部保留；不沿用组员分支的资料删除。

组员贡献：Step 0 校验、Middlebury 四组/调参、NYUv2 654 张 Depth Pro 和 UniDepthV2-S、
SI boundary、相机内参/尺度/跨类别分析、已生成 JSON/CSV 和来源审计。
本地贡献：逐帧 DAV2/离线 VDA、视频读写、Part 3 A–F、消融、测距、公开数据准备和 TUM 位姿时序评价。
组员原有结果与元数据按原样保留，不伪装为此次在本机重跑的成绩。
根目录 `EXPORT_MANIFEST.json` 是组员上传时的历史导出审计，不是整合后 HEAD 的全仓 checksum；
兼容接口和本轮数据接入的修改以本次 Git diff/提交和各数据 manifest 为准。

## 包结构兼容

双方原先同时存在 `src/metrics.py` 与 `src/metrics/` 等同名结构，会导致 Python 优先导入 package
并丢失另一套 API。本次迁移本地实现到以下文件，package 入口保留兼容导出：

- `src/metrics/video_project.py`：视频/Part 3 原有指标 API；组员 `depth/boundary/temporal.py` 不覆盖。
- `src/models/foundation.py`：原有 `DepthModel`；组员 `DepthProAdapter`/`UniDepthV2Adapter` 保留。
- `src/visualization/video.py`：视频固定色标；组员正式图表保留。
- `src/stereo/run_baseline.py`：本地简洁双目入口；组员正式 Middlebury/tuning 入口保留。

`python -m src.stereo`、`pipelines.part1.run_stereo`、`from src.models import DepthModel` 继续可用。
模型 adapter 延迟到实例化时才加载 PyTorch，导入数据/评分/失败样例选择工具时不初始化 CUDA DLL；
实际模型推理的数值处理不变。
两个已冻结图像评估轨道没有强行合并数值定义：组员官方 SI F1 与本地 log-depth edge F1
名称和结果仍明确分开。新的通用 manifest 支持 exclusive 深度范围和 score-time clip，
避免将 NYUv2 的 `.001 < GT < 10` 改成默认 `.1–10`。

## 数据接入与缺项

实际读取的本地上游提交：DAV2 `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`、
VDA `4f5ae23172ba60fd7bc11ef671cca678842c7072`，与组员 `configs/versions.yaml` 一致。
检查了两仓 README、metric dataset loader、视频 benchmark 提取脚本和公开 assets。
上游的 NYUv2 extractor 主入口存在名称错误且使用占位路径；本次没有直接照搬为可运行入口。

公共样例：从官方 NYUv2 MAT/split 分段提取四对，从 KITTI 官方 selection ZIP 分段提取四对；
TUM fr1/xyz 准备十六组 RGB/深度及插值位姿。所有样例保留来源、采样、哈希和协议。
NYUv2 原始数组只存本地，Git 保留可复现下载脚本、manifest 和校验值；
KITTI/TUM 小型样例带各自许可/署名，不纳入代码 MIT 许可证。

已核对的上游四个示例视频只登记在 `data/manifests/upstream_video_inventory.json`，
不重新标为指定静态/运动片段，也不是学生拍摄视频。用户已同意找不到指定视频时先不使用。
教师 NYUv2/KITTI 特定子集未提供，不能把公开 smoke subset 冒充它们。

TUM 导入：RGB/深度最近时间戳一对一匹配（20ms），位姿按 RGB 时刻平移线性插值/四元数 SLERP，
只在最大 50ms 相邻 GT 位姿间插值；16-bit 深度除以 5000，不重复应用已修正的 ds 比例。
导出 `T_world_camera` 方向、实际时间戳及 ROS 默认 525px 注册图像内参假设。
`src.evaluate_rgbd` 使用 GT 定义的重投影、z-buffer 和可见区作位姿补偿评分；不把二维光流指标称为真几何。
正式 NYUv2/KITTI 评估仍需分别遵循对应的真实 split/crop/mask/max-depth。

## 复现和验证

组员冻结环境与当前 Windows Conda 环境分开记录；未用一个依赖文件覆盖对方环境。
本次整合时 CPU 测试使用带 pytest 的 Python 环境，下载/数据整理使用具备相应依赖的环境；
整合阶段没有运行训练/GPU 推理，后续 GPU 实验另见对应记录。

```powershell
python -m pytest -q
python -m src.prepare_data validate --manifest data/public_samples/kitti/manifest.json
python -m src.prepare_data validate --manifest data/public_samples/nyuv2/manifest.json
python -m src.prepare_data validate --manifest data/public_samples/tum_freiburg1_xyz/manifest.json
python -m src.evaluate_rgbd --run outputs/part2/tum_metric --manifest data/public_samples/tum_freiburg1_xyz/manifest.json
```

最后一条命令需要先生成对应模型预测；下载和格式校验不代表已取得模型精度结果。
上面的 NYUv2 路径是在本机下载后的验证；新 clone 先按 public_samples/README 的命令
生成 `nyuv2_local`，再验证该新目录的 manifest。Git 中 `nyuv2/` 只有 metadata/哈希。
只有依赖官方 `depth_pro.eval.boundary_metrics` 的单项测试在缺少该外部源码时跳过。
