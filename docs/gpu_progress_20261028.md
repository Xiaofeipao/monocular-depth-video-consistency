# GPU progress: 2026-10-28

RTX 5060 Ti 16GB, torch_py311 Python environment, PyTorch 2.12.1+cu132.
CUDA was actually used for DAV2, relative VDA, metric VDA and RAFT-Small inference.
Additional einops/easydict/tqdm dependencies are project-local `.runtime`; no
PyTorch downgrade or environment replacement was performed.

Four Small state dicts match the teammate's existing checksums.sha256 exactly.
Hugging Face's primary endpoint reset the connection; matching artifacts were
retrieved from hf-mirror.com and verified before results were reported.

Real input: first 32 selected frames of the upstream Tokyo-Walk RGB example,
10 FPS, saved image long side <=640, requested network input size 518, FP16.
This is an upstream example, not a mandatory clip or student phone recording.

Saved runs under `outputs/gpu_20261028/`:

- `dav2_tokyo`: 32 raw relative frame predictions; model-only 8.87 FPS.
- `vda_tokyo`: 32 relative video predictions; offline amortized 5.32 FPS.
- `metric_vda_tokyo`: 32 metric video predictions; offline amortized 4.13 FPS.
- `raft_A6_tokyo`: actual GPU bidirectional RAFT + original A6 fusion.
- `directions_tokyo`: all available directions on the same real metric predictions.
- `weight_verification.json`, `relative_comparison.json`: input/weight identity and numerical observations.

These first runs used zero warmup; FPS includes cold inference and is not a
formal efficiency benchmark. VDA used approximately 8.39 GB peak allocated GPU
memory; model latency is offline throughput cost, not causal per-frame latency.

Relative comparison uses the same RGB/FB-valid pixels, original raw values,
and each model's fixed first-frame P95-P5 span. Mean normalized warp error:
DAV2 0.006370; VDA 0.004228 (about 33.6% lower on this short segment).
No per-frame alignment, no test depth, no arbitrary inversion of relative values.

On metric VDA predictions, full-resolution Farneback scoring gives temporal
log-error: RAW 0.012867, A 0.012285, C 0.012705, D 0.011550,
E-half 0.012333, E-adaptive 0.012293. F does not change the depth prediction.
D has the lowest proxy error here (about 10.2% below RAW). B and B+C+D are
explicitly skipped without independent measured landmarks.

There is no GT depth for this clip: AbsRel, boundary accuracy and warning F1
remain unavailable. Smoother predictions do not establish improved accuracy.
Independent Farneback scoring of GPU-RAFT fusion is saved separately. All
methods and negative cases remain in the repository.

GPU-RAFT A6 independently scored with Farneback: temporal log-error 0.012395,
compared with RAW 0.012867 (about 3.7% lower). It did not beat D on this metric.
The four public NYUv2 and four public KITTI samples also completed real GPU DAV2
inference and explicitly labelled inverse-domain scale/shift evaluation. These
are aligned-relative smoke results, not native metric or full benchmark results.

Reproduction (after preparing the four checkpoints):

```powershell
./scripts/run_gpu_video.ps1 -Python D:/Anaconda/envs/torch_py311/python.exe -DAV2Repo D:/Depth-Anything-V2 -VDARepo D:/Video-Depth-Anything -Video D:/Video-Depth-Anything/assets/example_videos/Tokyo-Walk_rgb.mp4 -Output outputs/gpu_repeat
```

TUM acquisition and the earlier Git integration/publishing task are tracked
separately; completing these GPU runs does not imply those steps are finished.

## TUM real RGB-D check (completed subsequently)

Sixteen fr1/xyz RGB/depth/pose samples, spanning 1.70 seconds, are now downloaded
and validated. Metric VDA ran on GPU at the original 640x480 saved resolution.
Native, no-GT-alignment, unclipped results: AbsRel 0.285482, RMSE 0.305843 m,
delta1 0.417334. Pose-compensated temporal MAE is 0.025066 m with 63.4% GT-defined
visible coverage. This short sequence has substantial metric-scale error and
does not constitute a full TUM benchmark.

Real GT Part 3 comparison: A AbsRel 0.285967 / temporal log-error 0.021797;
C 0.285366 / 0.023987; D 0.286335 / 0.021343. RAW temporal log-error 0.024261.
D lowers flow temporal error most (about 12.0%) but delta1 falls from 0.417334
to 0.409591, a 0.774 percentage-point loss, exceeding PLAN's 0.5-point bound.
Therefore D is not selected under the preregistered accuracy guard. A lowers
temporal error about 10.2% and delta1 drops only 0.373 points; A is the current
constrained choice on this tiny GT clip. All alternatives are retained.
The comparison selector now includes the previously omitted PLAN delta1 guard;
this updates selection labels only, never the observed metric values.
