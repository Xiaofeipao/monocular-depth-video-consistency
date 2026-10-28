# Implementation and reproducibility guide

## Scope and validation status

The code covers all three Parts. The integrated teammate branch adds frozen
Middlebury/NYUv2 image experiments and their historical result evidence, without
their original raw predictions or large data/weights. See `docs/current_results.md`
for those runs; they were not re-executed during integration. Locally inspected
DAV2/VDA source checkouts live outside this repository. Public small-data
preparation is described in `data/public_samples/README.md`. Part 3 results
include synthetic diagnostics; initial real GPU video runs are now recorded in
`docs/gpu_progress_20261028.md`. The user has now
authorized GitHub publication of the integrated code and prepared data materials.

Assignment-facing entry points live in `pipelines/part1`, `pipelines/part2` and
`pipelines/part3`, each with its own README mapping entry points to algorithm
implementations. They delegate to the shared `src` package. All commands run
from the repository root using `python -m`; the top-level README groups commands
by Part. Existing `python -m src.*` commands below remain supported.

## Environments

Existing environments inspected on this machine:

- `pytorch_env`: Python 3.13.5, PyTorch 2.12.1 CPU, NumPy, OpenCV, Pillow and
  SciPy present. Core CPU testing uses this environment; no pytest required.
- `torch_py311`: Python 3.11.15, PyTorch 2.12.1+cu132, torchvision, OpenCV,
  NumPy, h5py and SciPy present. A better candidate for official models after
  satisfying each upstream dependency set. CUDA was not available at inspection.
- Both lack timm/einops; full official-model compatibility is not verified.

`python scripts/check_environment.py` reports the environment you actually use.
No dependencies were installed into these existing environments.

Core checks:

```powershell
conda activate env_isaaclab
python -m pytest -q
```

The end-to-end test in `tests/test_pipeline.py` generates synthetic depth with
known GT, runs all seven fusion variants, and checks video, point cloud and
measurement exports in temporary directories. This verifies the software and
cannot substitute for assignment experiments.

## Data and protocol

Use `configs/*.example.json` as schemas. An image manifest contains `samples`
with unique `id`, `rgb`, `depth`, optional `mask`, and `depth_scale`, together
with a fixed `protocol`. Paths are relative to the manifest. The protocol
defines `min_depth`, `max_depth`, optional pixel crop `[top,bottom,left,right]`,
and boundary thresholds. Crop bounds are half-open. RGB is never cropped
before inference: focal/FOV prediction sees the full image. Predictions are
bilinearly resized to GT dimensions at evaluation, then cropped/masked.

Use the instructor's NYUv2/KITTI subset, crop and valid-mask instructions.
The NYUv2 example crop applies to standard 480x640 labeled images; it must be
replaced if the course sample uses another layout. The KITTI example uses
the official 16-bit PNG depth encoding of 1/256 m per unit. RGB/depth pairs
must be registered; the tools do not infer calibration or fix registration.

The optional NYUv2 MATLAB exporter reads the supplied official `testNdxs`
split and requires exactly 654 unique indices. It records source and split
hashes, and uses h5py/SciPy. Do not silently substitute a random split.

Video decoding and FPS sampling are shared between all adapters. For a
30 FPS input sampled at 12 FPS, indices start 0,3,5,8,... rather than using
a rounded stride and reporting the wrong FPS. Video frame directories require
`--fps` and lexically ordered, zero-padded filenames. Videos are decoded into
memory for offline VDA; use `--max-frames` and `--max-side` on constrained
machines. Every compared method must use the same options and input video.
No VDA streaming reproduction is claimed.

## Raw output contract

Each new output directory contains:

```text
metadata.json           scale type, unit, representation, model, checkpoint SHA
config_resolved.json    CLI/configuration
environment.json        Python, platform, dependencies, Git SHA
manifest.json           ordered frame/sample IDs, saved RGB and depth SHA256
depth/000000.npy        float32, full saved-frame resolution
valid/000000.png        finite positive prediction mask
rgb/000000.png          lossless saved BGR/RGB reference image
timing.json             model/postprocess/end-to-end timing as applicable
visualization/          fixed-range outputs
evaluation/<protocol>/  CSV observations and JSON macro means/bootstrap CIs
diagnostics/            fusion weights, error maps, reliable masks (Part 3)
```

Depth Pro and metric VDA: `depth_type=metric`, `unit=meter`,
`representation=depth`, `alignment=none`.
DAV2 and relative VDA: `depth_type=relative`, `unit=arbitrary`,
`representation=inverse_depth`. Their raw values are kept unchanged; higher
values mean nearer structure. No per-frame min/max normalization is applied
to stored arrays. Relative temporal scoring explicitly converts positive
values to inverse values to form a depth-like proxy, without GT alignment.
Zero relative predictions are invalid and their coverage is reported.

Output directories cannot be reused. Refinement writes a new run and records
the parent manifest hash. It never changes baseline predictions. Ground-truth
alignment is stored in an evaluation subdirectory, not in baseline metadata.

## Part 1: geometry

`src.stereo` processes a Middlebury root or one scene, with BM or SGBM. Inputs
must already be rectified. Disparity fixed-point output is divided by 16. A
separate right-to-left matcher uses negative disparities. Left consistency is
`abs(dL(x)+dR(x-dL(x))) <= 1 px`; out-of-bounds and invalid matches are rejected.
Metric depth is `f_px * B_m / (d + doffs)`. Calibration baseline is read in mm
and converted to m. Raw disparity and explicit invalid masks are preserved.

Depth/disparity accuracy on accepted matches is reported alongside coverage;
full-mask bad-1/2/4 rates count rejected predictions as failures. This avoids
presenting a low error on a tiny accepted region without exposing coverage.
Sensitivity CSVs perturb disparity by +/-0.25,0.5,1,2 px, compare exact depth
error with `z^2/(fB) * abs(delta_d)`, and retain invalid perturbations as empty
exact-error entries. The scripts do not rectify uncalibrated image pairs.

## Part 2: official model adapters

See `checkpoints/README.md` for trusted sources and local checkpoint names.
Adapters import official implementations rather than reimplementing model
architectures. Small/vits and Base/vitb are exposed. Checkpoint SHA256, byte
size, parameter count and upstream Git commit are recorded for each run.

Depth Pro uses the official transform, inferred focal length, full input
image and native metric output. No test GT intrinsics or alignment are used.
DAV2 passes OpenCV BGR to `infer_image`. VDA receives an RGB frame stack and
uses official offline inference, including long-video keyframe stitching.
The DAV2 adapter mirrors VDA's input-size adjustment on panoramic clips with
aspect ratio above 1.78 so their framewise/video network grids remain comparable.
VDA's metric flag must match the chosen checkpoint. Do not load an indoor
metric DAV2 checkpoint through the relative DAV2 adapter.

Pure model timing synchronizes CUDA where applicable. Optional `--warmup N`
performs untimed predictions. The default is zero for CPU smoke runs; formal
efficiency experiments should use the planned warmup/sample count, repeat
runs and report hardware. VDA latency is amortized offline clip latency,
not interactive per-frame latency; P50/P95 there are the single amortized
observation. End-to-end time also includes decoding, model loading and output
writing. Peak GPU memory is null on CPU, never reported as zero GPU memory.

## Part 3: controlled exploration

`src.refine` consumes metric video predictions. Relative inverse-depth is
rejected until a separately documented, frozen calibration is supplied.
`TemporalFusion` implements the planned A0-A6 matrix:

- A0: unchanged raw prediction.
- A1: unwarped temporal EMA (inverse-depth domain by default).
- A2: flow-warped history with constant weight, valid bounds only.
- A3: A2 plus forward/backward consistency rejection.
- A4: A3 plus photometric/log-depth disagreement thresholds and soft gates.
- A5: A4 plus dilated RGB/depth boundary protection.
- A6: A5 plus bounded robust historical-scale adjustment toward the current
  metric prediction, shrinkage, scene-cut and low-reliability resets.

Backward sampling uses **current-to-previous** flow in pixel units. Inverse
depth is warped before blending, rather than warping depth then inverting
after bilinear interpolation. Previous validity is also warped; all sampled
neighbours must be valid. Invalid current pixels remain invalid, rather than
being replaced with potentially wrong old geometry. The alpha bound and each
gate are explicit in `FusionConfig`. `--domain depth` provides a domain
ablation. Detailed threshold/sensitivity sweeps can instantiate that dataclass;
select parameters only on development clips, before final testing.

Farneback is available immediately. RAFT-Small accepts a local torchvision
checkpoint. `src.ablate` caches identical flow per saved RGB pair so every
variant uses the same correspondences. Cache-reusing timings are not standalone
runtime measurements; use `src.refine` without a shared cache for runtime.

## Evaluation and avoiding misleading comparisons

Depth evaluator computes AbsRel, SqRel, RMSE, natural-log RMSE, delta1/2/3,
with image macro means and bootstrap 95% intervals. It requires an exact
sample-ID match to the frozen split. GT masks are common across methods;
saved prediction hashes and exact input RGB content are checked before scoring.
nonfinite/nonpositive predictions on valid GT cause a clear error instead
of disappearing from the score. Explicit `--clip-predictions` is available
only when the selected benchmark protocol requires it.

Metric GT alignment is forbidden. Relative evaluation can request
`--alignment median` or `--alignment scale_shift`; alignment is in the
stored representation (inverse-depth for DAV2/VDA), followed by conversion
to depth. Every such result is labelled aligned relative. This is an
explicit evaluation setting, not an official protocol claim for every dataset.

Boundary precision/recall/F1 detects log-depth jumps and permits a fixed
pixel tolerance. Invalid GT neighbours are excluded from edge construction.
This is the assignment-allowed depth-edge metric, **not** Depth Pro's official
SI_boundary_F1. Threshold/tolerance are stored in the protocol. Do not put
these numbers under an official SI_boundary_F1 heading in the report.

Temporal scoring uses non-occluded forward/backward-consistent regions:
mean absolute log-depth difference, local flicker after subtracting median
log difference, median log-scale ratio, cumulative signed scale proxy and
absolute scale variation. Coverage accompanies the error. Empty pairs have
null error, not a misleading zero. These are optical-flow proxies; camera
translation can produce real depth changes, so the cumulative scale curve
is not ground-truth drift. Static clips additionally accept a fixed ROI and
report per-pixel temporal std and median-depth coefficient of variation.
The ROI must actually be static; never use a moving foreground ROI.
Per-transition averages within one clip are descriptive; no bootstrap confidence
intervals are assigned to correlated adjacent transitions.

Independent evaluation flow avoids relying only on the same flow used to
construct the improvement. With a local RAFT checkpoint, for example:

```powershell
python -m src.ablate --run outputs/part2/metric_static --output outputs/part3/crosscheck --fusion-flow raft --flow-checkpoint checkpoints/raft_small.pth --evaluation-flow farneback --static-roi 20 80 20 120
```

If using only Farneback for now, use and document the stable ROI cross-check.
`src.metrics.paired_bootstrap` also supports paired per-image/per-clip
differences; do not resample individual video frames as independent clips.
No significance or real-world improvement is claimed by synthetic tests.

## Application and submission packaging

`src.demo` exports RGB/depth/measurement PNG, colored PLY point cloud and
measurement JSON. It accepts explicit intrinsics at the **saved frame
resolution**, otherwise Depth Pro's focal prediction with an assumed centered
principal point and square pixels. Cropping/resizing calibration is the
caller's responsibility. Measurements use a 5x5 valid-depth median, distinguish
axial depth from ray distance, and support two-point Euclidean distance.

```powershell
python -m src.demo --run outputs/part3/static_A6 --output outputs/demo --intrinsics configs/intrinsics.example.json --clicks configs/measurements.example.json
python -m src.package_submission --manifest configs/submission.example.json --output depth_results.zip
```

Replace example calibration and clicked ground truth with actual measurements.
The GUI is opt-in (`--interactive`); batch export works without a desktop.
Application measurements with GT report MAE/median AE/relative error. Collect
the planned real measurements rather than interpreting an example as data.
ZIP packaging requires static/moving/phone roles, run manifests, videos and
application outputs. It preserves units/configs and includes SHA256 checksums.
It does not automatically verify the instructor's final clip list: list every
mandatory clip explicitly in the submission manifest. The original data and
weights are not included. No upload, git commit or git push is performed.

## Remaining real-data deliverables

Prepare instructor samples, official sources and checkpoints; smoke-test
each real adapter; freeze splits and parameters; run quantitative/qualitative
comparisons and independent-flow/ROI cross-checks; capture difficult scenes
and measurement ground truth; record formal runtime and GPU results when
available; then write the CVPR report based on actual results. Negative
Part 3 outcomes must be retained and explained with the same ablations.
