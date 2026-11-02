# 同帧视频输入分辨率检查

比较 CPU FP32 / 252 与 GPU FP32 / 518；分辨率与设备同时变化，不能单独归因于其中一个因素。
四段 self-recorded 与两个 TUM sample 均使用同帧 RGB、同权重、同后处理参数。
后处理仍使用 CPU 实现，GPU 用于 VDA 与独立 RAFT 评分。速度不作跨设备方法优势比较。

## landscape_static，64 帧

- A 的 RAFT warp proxy 降幅：12.37% → 10.66%。
- C 的 RAFT warp proxy 降幅：3.67% → 4.38%。
- D 的 RAFT warp proxy 降幅：17.32% → 16.91%。
- E_half 的 RAFT warp proxy 降幅：12.34% → 10.65%。

运动相机和人物的 warp proxy 包含真实 z 变化；有 GT 时另看 pose-log-error。

## moving，64 帧

- A 的 RAFT warp proxy 降幅：9.91% → 7.75%。
- C 的 RAFT warp proxy 降幅：4.42% → 3.49%。
- D 的 RAFT warp proxy 降幅：15.73% → 12.44%。
- E_half 的 RAFT warp proxy 降幅：10.27% → 8.06%。

运动相机和人物的 warp proxy 包含真实 z 变化；有 GT 时另看 pose-log-error。

## portrait_static，64 帧

- A 的 RAFT warp proxy 降幅：14.42% → 14.22%。
- C 的 RAFT warp proxy 降幅：4.33% → 4.87%。
- D 的 RAFT warp proxy 降幅：16.87% → 16.39%。
- E_half 的 RAFT warp proxy 降幅：14.40% → 14.21%。

运动相机和人物的 warp proxy 包含真实 z 变化；有 GT 时另看 pose-log-error。

## static_with_people_moving，64 帧

- A 的 RAFT warp proxy 降幅：9.43% → 8.33%。
- C 的 RAFT warp proxy 降幅：2.11% → 2.18%。
- D 的 RAFT warp proxy 降幅：12.89% → 10.98%。
- E_half 的 RAFT warp proxy 降幅：9.23% → 8.13%。

运动相机和人物的 warp proxy 包含真实 z 变化；有 GT 时另看 pose-log-error。

## tum_xyz_early，64 帧

原始 VDA AbsRel：0.433432 → 0.129460；δ1：5.567% → 86.589%。
518 下通过精度/边界约束的 pose 指标最优：D。
- A 的 RAFT warp proxy 降幅：3.56% → 3.25%；518 pose-log-error 降幅 1.83%。
- C 的 RAFT warp proxy 降幅：0.58% → 1.01%；518 pose-log-error 降幅 0.86%。
- D 的 RAFT warp proxy 降幅：5.20% → 4.57%；518 pose-log-error 降幅 2.50%。
- E_half 的 RAFT warp proxy 降幅：4.39% → 4.19%；518 pose-log-error 降幅 2.45%。

运动相机和人物的 warp proxy 包含真实 z 变化；有 GT 时另看 pose-log-error。

## tum_xyz_late，54 帧

原始 VDA AbsRel：0.453946 → 0.214268；δ1：7.142% → 69.226%。
518 下通过精度/边界约束的 pose 指标最优：D。
- A 的 RAFT warp proxy 降幅：5.31% → 3.57%；518 pose-log-error 降幅 4.03%。
- C 的 RAFT warp proxy 降幅：0.75% → 1.02%；518 pose-log-error 降幅 1.24%。
- D 的 RAFT warp proxy 降幅：8.32% → 5.86%；518 pose-log-error 降幅 6.30%。
- E_half 的 RAFT warp proxy 降幅：6.92% → 4.81%；518 pose-log-error 降幅 5.76%。

运动相机和人物的 warp proxy 包含真实 z 变化；有 GT 时另看 pose-log-error。

## 解释范围

这是已复现 Part 2 与固定 Part 3 的同帧消融；没有训练、GT 尺度拟合或根据测试集重选参数。
四段手机来自同一场景，两个 sample 来自同一 TUM 序列；不能据此宣布跨场景最优或超过 12 篇论文。
完整指标、逐对表、预测 SHA256、视频及限制分别见 [252 验证](extended_validation.md) 与 [518 验证](extended_validation_gpu518.md)。
