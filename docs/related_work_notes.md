# Recommended Reading — Notes for Project 3 (Monocular Metric Depth & Video Consistency)

Purpose: working notes for the Related Work section of the CVPR-style report, and a
quick reference while implementing Part 1-3. Each entry states the problem the paper
solves, the technical idea, the numbers worth quoting, and how it is used in this project.

Reading order that matches the report narrative:
classical geometry -> monocular foundations -> metric depth -> video depth & motion -> datasets.

---

## 0. One-line map of the 12 papers

| # | Paper | Venue | One-line takeaway | Used in this project as |
|---|---|---|---|---|
| [1] | Semi-Global Matching | TPAMI 2008 | Fast approximation of a global stereo energy by 1-D path-wise optimization | OpenCV `StereoSGBM`, L-R check, occlusion handling (Part 1) |
| [2] | KITTI benchmark | CVPR 2012 | Real-world driving benchmark; Middlebury-top methods collapse outdoors | Outdoor evaluation set, error definitions (Part 1-2) |
| [3] | DPT | ICCV 2021 | ViT backbone + convolutional decoder for dense prediction | Architectural ancestor of the DPT head used by MiDaS / Depth Anything |
| [4] | Depth Anything | CVPR 2024 | 62M unlabeled images + hard optimization target + DINOv2 alignment | Origin of the "relative, affine-invariant" track |
| [5] | Depth Anything V2 | NeurIPS 2024 | Synthetic labels for precision, pseudo-labeled real data for coverage | Frame-wise relative baseline (Part 1B) |
| [6] | UniDepth | CVPR 2024 | Universal metric depth; camera self-prompting + pseudo-spherical output | Conceptual baseline for "metric without intrinsics" |
| [7] | Depth Pro | ICLR 2025 | Sharp metric depth, absolute scale, 2.25 MP in 0.3 s, focal-length prediction | Main single-image metric model (Part 2A) + boundary metric |
| [8] | UniDepthV2 | arXiv 2025 | UniDepth + edge-guided loss + simpler architecture + uncertainty output | Optional second metric model; uncertainty source (Part 3, Dir. D) |
| [9] | RAFT | ECCV 2020 | Recurrent all-pairs correlation + GRU refinement at full resolution | Optical flow for warping / occlusion / temporal metrics (Part 3) |
| [10] | Video Depth Anything | CVPR 2025 | Spatio-temporal head + temporal gradient loss + keyframe stitching | Main video model (Part 2B) |
| [11] | TUM RGB-D benchmark | IROS 2012 | RGB-D sequences with motion-capture ground-truth poses | Optional set with real poses for drift evaluation |
| [12] | NYU Depth V2 | ECCV 2012 | 1449 labeled indoor RGB-D images with dense depth | Mandatory indoor benchmark (Part 2A) |

---

## 1. Semi-Global Matching (SGM) — Hirschmuller, TPAMI 2008

**Problem.** Dense stereo is hard at occlusions, object boundaries, fine structures,
low/repetitive texture, and under radiometric differences between the two cameras.

**Method — four stages, matching the classic Scharstein-Szeliski taxonomy.**

1. *Matching cost.* Mutual Information (MI) computed pixelwise:
   `MI(I1, I2) = H(I1) + H(I2) - H(I1, I2)`, computed from the intensity distributions
   after warping one image by the current disparity map. MI is insensitive to
   radiometric/illumination changes. Birchfield-Tomasi sampling-insensitive cost is the
   cheap alternative. Explicitly **no** aggregation window is used at this stage, because
   a window assumes constant disparity and therefore blurs exactly the fine structures
   one wants to keep.
2. *Cost aggregation by path-wise optimization.* Define the global energy

   `E(D) = sum_p C(p, D_p) + sum_{q in N_p} P1 * [|D_p - D_q| = 1] + sum_{q in N_p} P2 * [|D_p - D_q| > 1]`

   Minimizing this 2-D energy is NP-hard, so SGM approximates it by minimizing along many
   1-D paths (typically 8 or 16 directions) and summing the path costs. `P1` penalizes
   1-pixel disparity changes (allows slanted/curved surfaces); `P2 > P1` penalizes larger
   jumps with a *constant* penalty, which is what preserves discontinuities. `P2` is
   adapted to the intensity gradient, `P2 = P2' / |I_b(p) - I_b(q)|`, so that real image
   edges may carry a depth jump more cheaply.
3. *Disparity computation.* Winner-takes-all over the aggregated cost, then sub-pixel
   refinement by a quadratic fit through the winning cost and its two neighbours.
4. *Refinement.* Left-right consistency check (`|d_L(x) - d_R(x - d_L)| <= 1`) to detect
   occlusions and mismatches; speckle/peak removal; intensity-consistent disparity
   selection; gap interpolation.

**Reported results.** Top-ranked on Middlebury at the time and best when sub-pixel accuracy
is considered. Complexity is linear in (pixels x disparity range): 1-2 s on typical test
images, versus 20 s to over a minute for graph cuts / belief propagation / layered methods.

**Why it matters here.**
- OpenCV `StereoSGBM` is a direct implementation: `P1`, `P2`, `blockSize`, `numDisparities`,
  `uniquenessRatio`, `speckleWindowSize`, `speckleRange`, `disp12MaxDiff`. Report these
  explicitly in the reproducibility table.
- The sub-pixel quadratic fit explains the 1/16 fixed-point scaling of OpenCV disparity
  output (`disp / 16.0`).
- The L-R consistency check is exactly the Part 1 requirement; the `T[...]` penalty
  structure is the reason SGBM keeps thin structures better than `StereoBM`.
- Failure modes to show: low/repetitive texture and thin structures are still blurred
  because smoothness wins where the data term is uninformative; occlusions are not
  inferred, only detected and rejected.

---

## 2. KITTI Vision Benchmark Suite — Geiger, Lenz, Urtasun, CVPR 2012

**Problem.** Benchmarks were small and laboratory-controlled (Middlebury); methods tuned
on them do not transfer to realistic autonomous-driving conditions.

**Setup.** A station wagon with two stereo camera pairs (grayscale and colour,
1392x512, 10 Hz), a Velodyne HDL-64E lidar (>1M points/s, 100 m range), and an
OXTS RT3003 GPS/GLONASS/IMU localization unit with RTK correction (<5 cm open-sky error).
All sensors calibrated and time-synchronized.

**Released benchmarks.**
- Stereo and optical flow: 389 image pairs (194 train / 195 test), 1240x376 after
  rectification, semi-dense (~50 %) ground truth.
- Visual odometry / SLAM: 22 stereo sequences, 39.2 km total.
- 3D object detection: >200k 3D bounding-box annotations (cars, vans, trucks,
  pedestrians, cyclists, trams) obtained by labelling the lidar point cloud.

**Metrics introduced.**
- Stereo: percentage of outlier pixels, where a pixel is wrong if the disparity error
  exceeds **3 px AND 5 %** of the ground-truth disparity, reported separately for
  background (`D1-bg`), foreground (`D1-fg`) and all pixels (`D1-all`).
- Flow: same 3 px / 5 % criterion (`Fl-all`, later `F1-all`).
- Odometry: average translation (%) and rotation (deg/m) errors, averaged over all
  sub-sequences of a fixed length (100-800 m) and speed, so that drift is measured
  fairly rather than on a single trajectory.

**Key empirical finding.** Methods that rank highest on Middlebury perform below average
on KITTI, because their assumptions (controlled lighting, static scenes, narrow baseline
scales, rich texture) are violated outdoors; non-Lambertian surfaces, large displacements,
and mixed lighting/weather dominate.

**Why it matters here.**
- KITTI is the mandatory outdoor subset in Section 6.1 of the project instruction.
- The "3 px or 5 %" outlier rule is the ancestor of the `Bad-1/Bad-2/Bad-4` disparity
  thresholds in Part 1 — cite KITTI when defining them.
- The sub-sequence averaging idea is a good justification for how to report drift on
  long self-recorded clips: average over windows, not over a single number.
- The "lab benchmark vs. real world" gap is the same argument used to motivate evaluating
  on self-recorded phone video in addition to NYUv2/KITTI.

---

## 3. DPT: Vision Transformers for Dense Prediction — Ranftl, Bochkovskiy, Koltun, ICCV 2021

**Problem.** Convolutional encoders progressively downsample to grow the receptive field.
Downsampling destroys feature resolution and granularity, which are exactly what dense
prediction needs; information lost in the encoder cannot be recovered by the decoder.

**Method.**
- Backbone: a Vision Transformer. After the initial patch embedding the token grid keeps a
  **constant resolution and constant dimensionality across all stages**, and the self-
  attention gives a **global receptive field at every stage**. There is no pyramid.
- *Reassemble*: tokens from several transformer stages (e.g. layers 3, 6, 9, 12) are
  reassembled into image-like feature maps at 1/4 resolution scales.
- *Fusion*: a convolutional decoder progressively fuses these multi-resolution maps and
  upsamples them to full input resolution (Reassemble + Fusion blocks, residual
  convolution units).
- Consequence: fine-grained **and** globally coherent predictions, rather than the usual
  trade-off between the two.

**Results.** Up to 28 % relative improvement in monocular depth over the state-of-the-art
fully-convolutional network; 49.02 % mIoU on ADE20K semantic segmentation (new SOTA);
fine-tuning on NYUv2, KITTI and Pascal Context also sets new SOTA.

**Why it matters here.**
- This is the MiDaS v3.1 architecture and the "DPT head" that Depth Anything inherits;
  Video Depth Anything explicitly replaces the DPT head with its own spatio-temporal head,
  so the report must describe DPT to make that sentence meaningful.
- It is the clean answer to "why does the modern branch of Part 1 look so different from
  StereoSGBM": constant-resolution global attention instead of local windows + smoothing.

---

## 4. Depth Anything — Yang et al., CVPR 2024

**Problem.** Monocular depth needs a foundation model, but labelled depth data cannot be
scaled to tens of millions of images; MiDaS-scale mixes have limited coverage and fail
catastrophically in some scenes.

**Method — three parts.**
1. *Data engine.* 1.5M labelled images from 6 public datasets train an initial teacher;
   the teacher then annotates **~62M unlabeled monocular images** collected from 8 large
   public datasets (SA-1B, Open Images, BDD100K, ...). Unlabeled monocular images are
   cheap, diverse, and produce *denser* labels than lidar (denser than sparse LiDAR returns)
   without running stereo matching.
2. *Hard optimization target.* Naive self-training on pseudo labels gave no gain. The fix:
   perturb the unlabeled views strongly so that reproducing the teacher's prediction
   requires genuinely new visual knowledge, rather than memorizing an easy target.
3. *Semantic auxiliary supervision.* Instead of an auxiliary segmentation head (which loses
   information by decoding into discrete classes), align features with a frozen **DINOv2**
   encoder. This transfers high-level semantic priors and yields a multi-task encoder.

**Results.** Strong zero-shot performance on six public datasets and on in-the-wild photos
(low light, fog, complex scenes, ultra-remote distance). Fine-tuned with metric depth from
NYUv2 and KITTI it sets new SOTA. The better depth model also improves depth-conditioned
ControlNet.

**Why it matters here.**
- It is the paper that makes the **relative vs. metric** distinction concrete: the released
  model is affine-invariant (relative). The project instruction says explicitly:
  "Do not call aligned relative depth metric depth."
- The data-engine idea explains why modern models generalize to self-recorded phone video —
  a useful point when arguing that failure cases are intrinsic (mirrors, glass, sky) rather
  than a domain-shift artifact.

---

## 5. Depth Anything V2 — Yang et al., NeurIPS 2024

**Motivation.** V1 is robust and efficient but coarse and weak on transparent/reflective
objects; Stable-Diffusion-based models (Marigold) are detailed but slow, heavy and
unstable in complex scenes. V2 wants all properties at once.

**Three findings (the paper is organized as Q&A).**

- **Q1/A1.** Coarse depth is *not* intrinsic to discriminative models. The decisive change
  is to **replace all labelled real images with synthetic images**. Real labels have two
  defects: (i) label noise — depth sensors fail on transparent objects, stereo fails on
  repetitive/textureless patterns, SfM fails on dynamic objects; (ii) ignored details —
  real GT depth is coarse and misses thin structures. Synthetic data is pixel-exact and
  complete, and a discriminative model trained on it produces extremely fine detail.
- **Q2/A2.** Synthetic-only training has its own problems: limited scene diversity and a
  domain gap to real images, which is why prior work kept real data.
- **Q3/A3.** Bridge the two: **scale up a teacher trained only on synthetic data**, then
  distill into student models using **large-scale pseudo-labeled real images**. The
  students get synthetic precision + real-world coverage. V1's heavy augmentation trick
  and the DINOv2 alignment loss are no longer needed — the recipe is simpler.

**Also contributed.** A new evaluation benchmark with precise annotations and diverse
scenes, because existing test sets (notably NYUv2) are noisy and too narrow to reveal the
true quality of modern models.

**Scale and speed.** Models from 24.8M to 1.3B parameters. On a V100 the Small model is
about 25M params / 60 ms and the Large model about 335M params / 213 ms, versus 891M-948M
params and 2.1-5.2 s for Marigold / DepthFM: more than 10x faster and more accurate.
Fine-tuning with metric labels gives the metric variants.

**Why it matters here.**
- **Part 1B baseline**: Depth Anything V2-Small, run frame by frame. Save the raw
  float32 relative depth, not the colourized PNG.
- The Figure 3 argument (real GT noise at transparent objects, repetitive patterns,
  dynamic objects) is a direct, citable justification for the project's categorized
  failure-case section.
- Their critique of noisy test sets is a legitimate reason to report both NYUv2 numbers
  and self-recorded qualitative/clip-level evidence.

---

## 6. UniDepth — Piccinelli et al., CVPR 2024

**Problem.** Monocular metric depth estimation (MMDE) is accurate only inside the training
domain. Existing methods need camera intrinsics at test time, and injecting noisy camera
parameters directly injects error.

**Task definition.** *Universal* MMDE: predict metric 3D for each pixel from a single image
with **no** external information (intrinsics optional, can be added if available).

**Method.**
- **Self-promptable camera module**: predicts a dense, non-parametric camera representation
  (a per-pixel camera embedding) that *conditions* the depth features. This replaces the
  usual "concatenate the intrinsics" design.
- **Pseudo-spherical output representation** `(theta, phi, z)`: azimuth and elevation angles
  encode the camera ray and `z` the depth. In this space the camera plane and the depth axis
  are orthogonal, so the depth gradient `eps_z` does **not** flow back into the camera
  module. Camera components are embedded with Laplace spherical harmonics. Without this
  disentanglement, joint training is unstable and scale-ambiguous.
- **Geometric invariance loss**: sample two geometric augmentations of the same image
  (simulating two different apparent cameras); the camera-conditioned depth features must be
  mutually consistent.

**Results.** Zero-shot evaluation on ten datasets; superior to prior work, including
methods trained directly on the testing domains.

**Why it matters here.**
- It defines the problem that Depth Pro then solves differently — a clean contrast for
  Related Work: UniDepth predicts a full camera representation; Depth Pro predicts a single
  focal length / field of view from intermediate features.
- Relevant to Part 3 Direction B (metric scale calibration): UniDepth shows that camera
  and depth can be estimated jointly, and that decoupling them matters.

---

## 7. Depth Pro — Bochkovskii et al., ICLR 2025 (Apple)

**Goal.** Zero-shot metric monocular depth with (a) absolute scale, (b) no camera
intrinsics required, (c) sharp high-frequency boundaries, (d) low latency: a 2.25-megapixel
depth map in 0.3 s on a V100.

**Four contributions.**
1. **Boundary evaluation metrics from matting data.** Pixel-accurate depth ground truth
   near boundaries is impossible to obtain in the wild, so instead they use existing
   high-quality **matting / saliency / segmentation** annotations as binary foreground-
   background maps and measure how well the predicted depth boundary traces them
   (boundary recall / F1). This is the *scale-invariant boundary F1* the project uses as
   its boundary metric — cite this when defining it.
2. **Efficient multi-scale ViT.** The whole network runs at a **fixed 1536x1536**
   resolution (2.25 MP). The image is downsampled once to 1536x1536, then split into
   384x384 patches at several scales; the two finest scales use overlapping patches
   (25 and 9 patches) to avoid seams. A global pass on the 384x384 downsample gives global
   context; the patch passes give high-frequency detail. Fixed resolution guarantees
   constant runtime and avoids OOM.
3. **Training protocol.** A curriculum plus losses that produce sharp depth while training
   jointly on real datasets (realistic but coarse and wrong near boundaries) and synthetic
   datasets (pixel-exact but limited realism).
4. **Focal length estimation.** A field-of-view / focal-length head on intermediate
   features of the depth network. This is what makes metric scale possible without
   intrinsics, and it is state of the art in cross-domain single-image focal length
   estimation.

**Results (as claimed).** Multiplicative improvement in boundary recall over prior work
while being one to two orders of magnitude faster than the detail-focused methods
(Marigold, PatchFusion); boundary recall vs. runtime figure is the headline plot.

**Why it matters here.**
- **Primary Part 2A model.** Predict metric depth without test-set scale alignment; report
  AbsRel, SqRel, RMSE, RMSE-log, delta1/2/3 and boundary F1.
- The released code returns `focallength_px`; this is the intrinsics source for the demo's
  back-projection `X = (u - c_x) Z / f_x`, `Y = (v - c_y) Z / f_y`, with the fallback
  priority "real calibration > reliable metadata focal length > Depth Pro prediction".
- Caveat to write down for the report: feeding a *cropped* image changes the field of view,
  so predict on the uncropped image and apply the NYUv2 crop at evaluation time only.

---

## 8. UniDepthV2 — Piccinelli et al., arXiv:2502.20110, 2025

**Relation to UniDepth.** Same task (universal MMDE, single image, no intrinsics) and the
same three core mechanisms: self-promptable camera module, pseudo-spherical `(theta, phi, z)`
output space, geometric invariance loss.

**What is new.**
1. **Edge-guided loss** — improves localization and sharpness of depth discontinuities.
   Ground-truth depth is sparse and incomplete near edges, which is precisely why MMDE
   predictions are blurry there; the new loss supervises local structure around edges.
2. **Revisited, simplified, more efficient architecture.**
3. **Additional uncertainty output** — a per-pixel confidence that downstream tasks can use.

**Results.** Zero-shot evaluation on ten depth datasets; claims superior performance and
generalization over UniDepth.

**Why it matters here.**
- Optional second native-metric method (the instruction allows "UniDepthV2 **or** Depth Pro";
  Depth Pro is the main choice, UniDepthV2 is the extension).
- The uncertainty output is directly useful for Part 3 Direction D (uncertainty-aware
  smoothing) and for the reliability gate in the planned temporal fusion. It also gives an
  alternative to deriving confidence from test-time augmentation or cross-model disagreement.

---

## 9. RAFT — Teed and Deng, ECCV 2020

**Problem.** Optical flow is still limited by fast-moving small objects, occlusions,
motion blur and textureless regions.

**Architecture — three components.**
1. **Feature encoder** extracts per-pixel features from both images (at 1/8 resolution);
   a separate **context encoder** extracts features from the first image only.
2. **All-pairs correlation**: a 4-D `H x W x H x W` correlation volume of inner products
   between all pairs of feature vectors, then pooled to build a multi-scale pyramid of
   volumes.
3. **Recurrent update operator**: a convolutional GRU (2.7M parameters) that looks up
   correlation values around the current flow estimate and iteratively refines a
   **single, fixed, high-resolution flow field** initialized to zero. Weights are tied
   across iterations, so it can be applied 100+ times at inference without diverging.

**Design choices that matter.**
- No coarse-to-fine cascade: maintaining one high-resolution flow field avoids unrecoverable
  coarse errors and does not miss small fast-moving objects; it also trains with ~10x fewer
  iterations.
- The architecture mirrors classical optimization: features replace the hand-crafted data
  term, the GRU replaces the iterative optimizer, but both are learned.

**Results.** KITTI `F1-all` 5.10 % (16 % relative reduction over the previous 6.10 %);
Sintel final-pass EPE 2.855 px (30 % reduction over 4.098 px). Trained only on synthetic
data it still reaches 5.04 px EPE on KITTI (40 % better than the best prior network trained
the same way), showing strong cross-dataset generalization. 1088x436 at 10 FPS on a 1080Ti;
a version with 1/5 of the parameters runs at 20 FPS and still beats all prior methods on Sintel.

**Why it matters here.**
- The flow estimator for Part 3 Direction A: warp the previous depth with
  `F_{t-1->t}`, reject occlusions with forward-backward consistency
  (`|F_fwd(x) + F_bwd(x + F_fwd(x))|` small), and fuse with the current prediction.
- Also the basis of the mandatory temporal-consistency metric: flow-aligned temporal error
  on valid non-occluded pixels.
- **Evaluation-hygiene note (must be stated in the report):** if the same RAFT flow is used
  both to build the fusion and to compute the metric, the metric can be "gamed". Add a
  cross-check with a second flow method (Farneback) or a manually selected stable ROI.

---

## 10. Video Depth Anything — Chen et al., CVPR 2025 (ByteDance)

**Problem.** Depth Anything V2 is trained on static images and flickers in video. Prior
fixes either rely on test-time optimization, or on optical flow / camera poses whose errors
propagate, or repurpose video diffusion models (fine detail, but slow and limited to short
clips, typically < 10 s).

**Method.**
- Base: Depth Anything V2, backbone frozen in spirit; replace the **DPT head** with a
  lightweight **spatio-temporal head (STH)** containing four temporal attention layers
  applied along the temporal dimension at each spatial position. Temporal attention is put
  **only in the head**, so the limited video data cannot corrupt the image representation.
- **Temporal gradient matching loss**: constrain the temporal gradient of the predicted
  depth to match the temporal gradient of the ground truth. Combined with the usual
  scale-shift-invariant loss and spatial gradient matching loss. No geometric prior
  (no flow, no pose) and no generative prior.
- Training: 730K video frames with depth annotations plus 0.62M unlabeled images with
  self-training, as in Depth Anything V2.
- **Super-long video inference**: segment-wise. Each new segment is concatenated with
  **8 overlapping frames and 2 key frames** from the previous segment, giving 32 frames;
  overlapping frames are progressively interpolated between consecutive windows
  (depth stitching) so quality, consistency and generalization hold for arbitrarily long
  videos (demonstrated on a 196 s / 4690-frame clip).

**Results.** SOTA zero-shot video depth on five datasets; best temporal consistency
(temporal alignment error, TAE) on all of them; smallest model runs at 30 FPS; only a
marginal drop on image benchmarks versus Depth Anything V2. The paper's Figure 1 plots
accuracy (delta1), consistency and latency (A100) simultaneously.

**Why it matters here.**
- **Primary Part 2B model**: Small checkpoint, FP16 preferred, no Large model.
- The mandatory comparison "frame-wise Depth Anything V2 vs. Video Depth Anything at
  identical resolution, visualization range and evaluation frames" comes directly from here;
  report both the relative and the metric checkpoint separately.
- Offline vs. streaming inference is an official distinction — report them separately
  (streaming trades accuracy for latency), which also feeds the efficiency table.
- Since the model's own loss is a *temporal gradient* loss, the project's own temporal
  metric should be reported honestly as a flow-aligned proxy, not as ground-truth geometry.

---

## 11. TUM RGB-D Benchmark — Sturm et al., IROS 2012

**Contribution.** A benchmark for RGB-D SLAM / visual odometry: 39 sequences recorded with
a Microsoft Kinect at full sensor resolution **640x480 at 30 Hz**, with **ground-truth
camera poses** from a motion-capture system of 8 high-speed tracking cameras at 100 Hz,
carefully calibrated and time-synchronized with the Kinect.

**Content.** Two environments: an office (`fr1`) and an industrial hall (`fr2`). Motions
range from slow "debugging" movements to long trajectories with and without loop closures.
Most sequences are handheld with unconstrained 6-DOF motion; some are from a Kinect mounted
on a Pioneer robot navigating a cluttered indoor space.

**Metrics and tooling.** Two automatic metrics with published evaluation tools:
- **RPE (relative pose error)** — drift over a fixed time/distance interval, for odometry.
- **ATE (absolute trajectory error)** — global pose error after alignment, for full SLAM.

Data is released under CC-BY 3.0.

**Why it matters here.** This is the "optional but recommended" set in Section 6.1, and it
is the only item on the reading list that supplies **synchronized depth and true camera
poses**. That makes it the one place where the project can evaluate:
- metric depth accuracy against real sensor depth, and
- **temporal consistency with pose-warped ground truth** rather than an optical-flow proxy.
It is also the natural place to measure scale drift honestly, which is otherwise impossible
on self-recorded phone video without ground truth.

---

## 12. NYU Depth V2 — Silberman et al., ECCV 2012

**Dataset.** 1449 densely labelled RGB-D images from 464 diverse indoor scenes (recorded in
three different cities) with a Kinect. Each image has a dense pixelwise depth label; a
subset has detailed segmentation and support-relation annotations.

**The paper's own method** (useful context, not needed for the project): a physical scene
parse. Inpaint the raw Kinect depth to fill sensor holes, compute surface normals, align the
point cloud to the room by finding three dominant orthogonal directions, fit planes with
RANSAC, build a hierarchical segmentation, classify regions into structural classes
(ground / permanent structures / furniture / props), and infer physical support relations
with an integer-programming formulation that combines geometric constraints and statistical
priors. Depth cues handle the geometry; appearance cues handle object identity.

**Protocol note (important).** The *dataset* is this paper; the standard **654-image test
split, the Eigen crop, and the depth metrics AbsRel / SqRel / RMSE / RMSE-log / delta1-3**
come from the follow-up Eigen et al. line of work. Cite both. The commonly used protocol is:
evaluate on the 654-image test split with the Eigen crop, valid depth range about 0.1-10 m,
and report the mean over images of per-image metric means.

**Why it matters here.**
- Mandatory indoor benchmark (Section 6.1): "follow the supplied crop, valid-depth mask and
  maximum-depth settings".
- Two consequences to record: (i) Kinect depth has holes and is unreliable at edges, which
  is why a valid mask is mandatory and why boundary metrics need care; (ii) NYUv2 labels are
  themselves noisy, which is the Depth Anything V2 argument for a cleaner benchmark — worth
  acknowledging in the limitations section.

---

## 13. Suggested Related Work narrative (4 paragraphs)

1. **Classical stereo geometry.** Start from the Scharstein-Szeliski four-stage taxonomy and
   SGM [1] as the practical approximation to a global stereo energy (MI cost, path-wise
   optimization, P1/P2, L-R consistency). Then note that progress measured on laboratory
   benchmarks does not transfer: KITTI [2] showed Middlebury-top methods drop to below
   average outdoors, and introduced the 3 px / 5 % outlier criterion and sub-sequence
   averaging for drift. This motivates both the Part 1 geometry study and the use of
   in-the-wild footage.
2. **Monocular foundation models.** DPT [3] replaced the downsampling convolutional encoder
   with a constant-resolution ViT plus a convolutional decoder, giving fine-grained and
   globally coherent dense prediction. Depth Anything [4] scaled data to ~62M unlabeled
   images via a data engine with a hard optimization target and DINOv2 feature alignment.
   Depth Anything V2 [5] showed that the key to fine detail is *precise synthetic* labels,
   with large-scale pseudo-labeled real images as the bridge, and criticized the noise in
   existing test sets. These models are affine-invariant: their output is **relative**.
3. **Metric depth.** UniDepth [6] introduced universal MMDE — metric 3D from one image with
   no intrinsics — using a self-promptable camera module, a pseudo-spherical output space and
   a geometric invariance loss. UniDepthV2 [8] adds an edge-guided loss, a simpler
   architecture and an uncertainty output. Depth Pro [7] instead predicts the focal length
   from intermediate features, runs a fixed 1536x1536 multi-scale ViT for sharp boundaries at
   0.3 s, and contributed boundary metrics derived from matting annotations. The contrast
   (camera representation vs. focal length; boundary sharpness vs. latency) is the axis along
   which the project compares metric models.
4. **Video depth and motion.** RAFT [9] provides the flow machinery — all-pairs 4-D
   correlation plus a recurrent GRU refinement of a single high-resolution flow field — used
   both for forward-backward occlusion detection and as an evaluation tool. Video Depth
   Anything [10] adds a spatio-temporal head and a temporal gradient matching loss to Depth
   Anything V2, with keyframe-based segment stitching for arbitrarily long videos, avoiding
   flow and pose priors entirely. Our Part 3 sits exactly in the gap: it *does* use explicit
   flow-based reliability gating, and asks whether that buys anything over the model's
   implicit temporal consistency.

## 14. Cross-cutting facts to keep straight while writing

| Fact | Consequence for the report |
|---|---|
| Depth Anything / V2 base models are affine-invariant | Never put them in the same column as native metric models without an `aligned` marker |
| Depth Pro predicts focal length | Metric scale without intrinsics; also the demo's intrinsics fallback |
| UniDepthV2 outputs uncertainty | A principled confidence source for Direction D |
| Video Depth Anything has relative and metric checkpoints | Evaluate and report the two tracks separately |
| RAFT is used for both fusion and evaluation | Add an independent-flow or fixed-ROI cross-check |
| NYUv2 and KITTI GT depth are themselves noisy | Failure cases are partly label artifacts; say so in limitations |
| TUM RGB-D is the only source with true camera poses | The only place drift can be measured without a proxy |
