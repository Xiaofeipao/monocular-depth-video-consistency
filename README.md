# Project 3: Monocular Metric Depth Estimation & Video Consistency

AIAA 3201 — Introduction to Computer Vision  
Term Project Instruction  
Fall 2026

## Overview

This project focuses on estimating dense depth from a single RGB image or an ordinary monocular video, recovering a meaningful metric scale, and maintaining stable predictions over time. The goal is to connect classical geometry with modern depth foundation models and video temporal modeling.

The final system should support an application such as point-cloud visualization, distance measurement, depth-aware effects, or obstacle warning.

> Experimental verification required: evaluate whether the final system actually improves depth quality and temporal consistency on real image/video samples.

### Core Task

Given a single RGB image or monocular video, the system must:

- estimate dense depth,
- recover an approximate metric scale,
- maintain temporal consistency across frames,
- produce a meaningful downstream application output.

### Key Concepts

- Projective geometry: disparity, focal length, camera intrinsics, metric scale
- Monocular depth: relative depth vs. zero-shot metric depth
- Foundation models: large-scale pretraining and cross-dataset generalization
- Video consistency: temporal propagation, optical flow, scale drift, and occlusion handling

> Experimental verification required: confirm the practical effect of these concepts on real benchmarks and videos.

### Group Size

- 2 students per group

---

## Project Goal

We aim to build a monocular depth estimation pipeline that combines:

- frame-wise depth estimation from a strong backbone,
- temporal consistency modeling for video sequences,
- optional calibration or flow-based refinement,
- evaluation on both image and video benchmarks.

The final project should demonstrate both quantitative depth quality and visually stable video results.
> Experimental verification required: determine which combination of backbone + temporal refinement is actually better on the provided data.
---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── checkpoints/
│   └── README.md
├── data/
│   ├── nyuv2_sample/
│   ├── kitti_sample/
│   ├── videos/
│   └── depth_results/
├── src/
│   ├── models/
│   ├── datasets/
│   ├── utils/
│   ├── infer_image.py
│   ├── infer_video.py
│   ├── evaluate.py
│   └── train.py
├── notebooks/
│   └── demo.ipynb
├── results/
│   ├── sample_depths/
│   └── demo_videos/
└── report/
    └── paper.pdf
```

> TODO: Update this structure if the project layout differs during implementation.

---

## Environment Setup

```bash
conda create -n depth python=3.10 -y
conda activate depth
pip install -r requirements.txt
```

### Tested Environment

- Python: 3.10
- CUDA: TODO
- GPU: TODO
- PyTorch: TODO
- OS: TODO

> Experimental verification required: record the actual runtime environment, GPU memory usage, and compatibility of the chosen setup.

---

## Data Preparation

Create a local dataset folder similar to the following:

```text
data/
├── nyuv2_sample/         # instructor-provided NYUv2 subset
├── kitti_sample/         # instructor-provided KITTI subset
├── videos/
│   ├── static_camera/    # static-camera clip
│   ├── moving_camera/    # moving-camera clip
│   └── phone_clip.mp4    # custom 10–30s video
├── depth_results/        # output directory (git-ignored)
└── annotations/          # optional metadata or masks
```

### Recommended Practice

- Use short clips with stable lighting and enough texture.
- Keep a fixed evaluation split for fair comparisons.
- Save raw floating-point depth maps in addition to colored visualizations.

> Experimental verification required: check how different video conditions (static vs moving camera, lighting, motion blur) affect prediction quality.

---

## Checkpoints

Download the pretrained models and place them under `checkpoints/`.

| Model | Purpose | Download Link |
| --- | --- | --- |
| Depth Anything V2 | Frame-wise depth backbone | TODO |
| UniDepthV2 | Metric depth estimation | TODO |
| Video Depth Anything | Temporal consistency refinement | TODO |
| RAFT | Optional optical flow guidance | TODO |

---

## Quick Start

### Single-image depth estimation

```bash
python src/infer_image.py --input path/to/image.jpg --output results/depth.png
```

### Video depth estimation

```bash
python src/infer_video.py --input path/to/video.mp4 --output results/video_depth.mp4
```

### Evaluation

```bash
python src/evaluate.py --dataset nyuv2 --method uni_depth_v2
```

### Typical metrics reported

- AbsRel
- SqRel
- RMSE
- RMSE-log
- δ1 / δ2 / δ3
- Temporal consistency error
- FPS / runtime

> Experimental verification required: these metrics must be measured and compared across all baseline and improved methods.

---

## Submission Requirements

### 1) PDF Report (Mandatory)

- Format: CVPR LaTeX template
- Length: 6–8 pages excluding references
- You must upload the final report to arXiv
- Change the document status from “REVIEW version” to “CAMERA-READY version”
- Include your arXiv ID on the first page of the Canvas submission

> Experimental verification required: quantitative experiments and ablations must be included in the report with evidence, not just qualitative claims.

#### Report content requirements

1. Abstract & Introduction  
   Background, motivation, problem definition, and solution overview. Include the public GitHub repository link at the end of the abstract.
2. Related Work  
   Review and cite at least all papers in the Recommended Reading List.
3. Method  
   Explain the technical roadmap clearly with diagrams and examples.
4. Experiments  
   Present quantitative tables, qualitative comparisons, ablation studies, and failure cases.
5. Conclusion  
   Summarize findings, limitations, and future work.

### 2) Code (Mandatory)

- Upload the code to a public GitHub repository with a clear `README.md`
- Include dependencies, data preparation, checkpoint instructions, training/evaluation commands, and representative visual results
- Do not submit raw code files to Canvas

### 3) Depth Demo & Video (Mandatory)

- Submit processed videos for all mandatory clips
- Provide a runnable or recorded application demo
- The demo must show the RGB input, predicted depth, and one application output such as:
  - point cloud,
  - click-to-measure interface,
  - 2.5D parallax,
  - depth-aware rendering,
  - or obstacle warning
- Pack outputs into `depth_results.zip`

> Experimental verification required: confirm that the downstream application actually works on real input and that the depth output supports the target use case.

---

## Important Tips for Success

- Method flexibility is encouraged: the suggested roadmap is not mandatory if the final task is well addressed.
- All three project parts are mandatory: if Part 3 does not improve the main metric, explain why with ablations and failure analysis.
- Visual quality matters. Clean diagrams, overlays, zoomed comparisons, and failure visualizations are important.
- The provided NYUv2/KITTI sample and mandatory video clips should be completed for a passing grade.
- Use identical data splits, preprocessing, and evaluation settings across methods for fair comparisons.
- Include runtime, GPU memory, and test-time refinement details in the final report.
- Correctly cite all datasets, pretrained models, and external implementations used.

---

## Guide to Start

1. Literature review  
   Read the recommended papers to understand pinhole geometry, monocular scale ambiguity, optical flow, and depth evaluation.
2. Code familiarization  
   Run the official demos for Depth Anything V2 and Video Depth Anything.
3. Experimentation  
   Run the methods on provided samples, save raw depth maps, and diagnose flicker or scale drift.
4. Writing & submission  
   Organize results, write the report, upload to arXiv, and submit required project files.

---

## Recommended Reading List

- Classical geometry: Semi-Global Matching [1], KITTI benchmark [2]
- Monocular foundations: DPT [3], Depth Anything [4], Depth Anything V2 [5]
- Metric depth: UniDepth [6], Depth Pro [7], UniDepthV2 [8]
- Video depth and motion: RAFT [9], Video Depth Anything [10]

---

## Implementation Roadmap

### Part 1: Baseline — Geometry & Frame-Wise Depth

#### 1. Classical Stereo: Disparity to Depth

- Rectify stereo images.
- Use OpenCV `StereoBM` or `StereoSGBM`.
- Visualize disparity and invalid regions.
- Convert disparity to depth with $z = fB/d$.
- Study sensitivity to disparity errors.

> Experimental verification required: quantitatively verify how disparity error propagates into depth error at different distances.

#### 2. Monocular Frame Baseline

- Run a relative depth model such as MiDaS, DPT, or Depth Anything V2.
- Apply median or scale-and-shift alignment only when required by benchmark evaluation.
- Measure temporal flicker on static-camera clips and scale drift on moving-camera clips.

Expected result: frame-wise models can recover scene structure but may suffer from incorrect metric scale, unstable boundaries, and temporal flicker.

> Experimental verification required: compare frame-wise depth quality and temporal stability on the provided clips.

### Part 2: SOTA Reproduction — Metric & Temporally Consistent Depth

#### 1. Single-Image Metric Depth

- Use UniDepthV2 or Depth Pro.
- Predict metric depth without test-set alignment when reporting official metric-depth results.
- Compare boundary quality, scale accuracy, and domain generalization.

> Experimental verification required: confirm whether metric-depth prediction provides more useful absolute scale than relative depth models.

#### 2. Video Depth

- Use Video Depth Anything for temporally coherent depth prediction on short clips.
- Compare against frame-wise Depth Anything V2 under identical resolutions and evaluation settings.
- Prefer the Small model and FP16 if GPU memory is limited.

Expected result: video models reduce flicker and scale drift, although thin structures, reflective surfaces, and rapid motion remain hard cases.

> Experimental verification required: measure whether the temporal model reduces flicker and scale drift in practice.

### Part 3: Exploration — Calibration, Consistency & Application

Possible directions:

- Direction A: Flow-guided fusion with RAFT or Farneback
- Direction B: Metric scale calibration via known-size object, camera height, or ground plane
- Direction C: Edge-aware refinement using RGB gradients or semantic boundaries
- Direction D: Uncertainty-aware smoothing
- Direction E: Efficient deployment with ONNX/quantization
- Direction F: Depth-aware downstream application

> Experimental verification required: choose one direction, run controlled ablations, and verify whether the proposed improvement is genuinely beneficial.

> TODO: Choose and implement one direction and report ablation results.

---

## Dataset & Evaluation

### Datasets

1. Image benchmark sample (mandatory): NYUv2 indoor subset and KITTI outdoor subset  
2. Video clips (mandatory): static-camera and moving-camera clips, plus one 10–30 second phone video  
3. Optional but recommended: TUM RGB-D or ScanNet sequences

### Metrics (Mandatory)

- Metric depth accuracy: AbsRel, SqRel, RMSE, RMSE-log, δ1/δ2/δ3
- Boundary quality: depth-edge precision/recall or similar boundary metric
- Temporal consistency: flow-aligned temporal error and per-frame median scale drift
- Efficiency: FPS, latency, GPU memory, model size, input resolution
- Qualitative evaluation: RGB, depth map, point cloud or app output, and failure cases

> Experimental verification required: all listed metrics must be measured and compared on the same data, not just visually inspected.

> Use a fixed depth visualization range for every method on the same video clip.

---

## Example Result Layout

| RGB Input | Predicted Depth | Application Output |
| --- | --- | --- |
| TODO | TODO | TODO |
| TODO | TODO | TODO |

### Common Failure Cases

- mirrors and reflective surfaces
- glass and transparent materials
- thin structures and distant small objects
- motion blur and severe occlusions
- sky regions and textureless areas

> Experimental verification required: explicitly test and document which failure modes appear in the actual predictions and how severe they are.

---

## References

[1] Hirschmüller, H. “Stereo Processing by Semiglobal Matching and Mutual Information.” TPAMI, 2008.

[2] Geiger, A., et al. “Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite.” CVPR, 2012.

[3] Ranftl, R., et al. “Vision Transformers for Dense Prediction.” ICCV, 2021.

[4] Yang, L., et al. “Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data.” CVPR, 2024.

[5] Yang, L., et al. “Depth Anything V2.” NeurIPS, 2024.

[6] Piccinelli, L., et al. “UniDepth: Universal Monocular Metric Depth Estimation.” CVPR, 2024.

[7] Bochkovskii, A., et al. “Depth Pro: Sharp Monocular Metric Depth in Less Than a Second.” ICLR, 2025.

[8] Piccinelli, L., et al. “UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler.” arXiv:2502.20110, 2025.

[9] Teed, Z., and Deng, J. “RAFT: Recurrent All-Pairs Field Transforms for Optical Flow.” ECCV, 2020.

[10] Chen, S., et al. “Video Depth Anything: Consistent Depth Estimation for Super-Long Videos.” CVPR, 2025.

[11] Sturm, J., et al. “A Benchmark for the Evaluation of RGB-D SLAM Systems.” IROS, 2012.

[12] Silberman, N., et al. “Indoor Segmentation and Support Inference from RGBD Images.” ECCV, 2012.

---

## Project Status

This repository is intended to serve as the project implementation and documentation for the course assignment.

### TODO Checklist

- [ ] Fill in team member names
- [ ] Add public GitHub link in final report
- [ ] Complete environment and dependency versions
- [ ] Download required checkpoints
- [ ] Prepare dataset folders
- [ ] Implement inference scripts
- [ ] Run evaluations and record metrics
- [ ] Add result figures and failure-case analysis
- [ ] Upload final arXiv paper and submit project files

> Experimental verification required: the checklist items related to metrics, temporal consistency, and failure analysis must be completed with measurements, not assumptions.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Notes

This README follows the official project instruction and is kept intentionally structured so that missing project-specific details can be filled in as TODOs while the implementation progresses.
