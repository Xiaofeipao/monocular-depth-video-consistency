# Project 3：单目度量深度估计与视频一致性

**AIAA 3201——计算机视觉导论**  
**课程项目说明**  
**2027年春季学期**

> 译文来源：[新版课程PDF](Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf)，共5页。核对日期：2026-11-02。PDF SHA256：`ea65076917e87febb0c3c67773b9fc6edf304e2821a74266a03f22ef33abc226`。
>
> 本文按原文保留要求、可选项、编号及下载链接。与当前项目的差异、素材问题及待办见源项目的 difference_analysis.md（未收入本发布包）；本包当前状态见README.md，不将项目自己的实施选择加入课程要求。

## 1. 项目概述

**核心任务：**在Middlebury上学习双目几何，在NYU Depth V2上评估**一个预训练度量深度模型**，并在提供的示例片段上比较逐图像深度与视频深度。之后实现**一项改进**，并展示一个简单应用。静态图像用于定量评价，视频用于定性比较。

**核心概念：**

- **投影几何：**视差、焦距、相机内参与度量尺度。
- **单目深度：**相对深度与零样本度量深度。
- **基础模型：**大规模预训练与跨数据集泛化。
- **视频一致性：**时序传播、光流、尺度漂移与遮挡。

**小组人数：**每组2人。

## 2. 提交要求

### PDF报告（必须）

- **格式：**使用[CVPR LaTeX模板](https://github.com/cvpr-org/author-kit)。正文必须为**6–8页，不包括参考文献**。
- **上传arXiv（可选）：**可以将最终报告上传到arXiv。请记得将模板中的“REVIEW version”改为“CAMERA-READY version”。如果上传，请在Canvas提交材料的第一页清楚注明arXiv ID。
- **内容要求：**
  1. **摘要与引言：**介绍背景、动机、问题定义与解决方案概览。**在摘要末尾明确写出公开GitHub仓库链接。**
  2. **相关工作：**至少回顾并引用第4节“推荐阅读列表”中的全部论文，鼓励扩展到列表之外。
  3. **方法：**用自己的语言解释技术路线，使用高质量的流程图和可视化示例辅助说明。
  4. **实验：**用表格展示定量结果，用图像展示定性比较，并包含消融实验和按类别整理的失败案例。
  5. **结论：**总结发现、局限性和具体的未来研究方向。

### 代码（必须）

将代码上传到**公开GitHub仓库**，提供清晰的`README.md`，包含依赖、数据准备、模型权重、训练与评估命令，以及有代表性的可视化结果。**不要向Canvas提交原始代码文件。**

### 深度Demo与视频（必须）

对第6.1节链接仓库中`davis_rollercoaster.mp4`的**前100帧**，提交RGB与深度的并排比较。

还需从提供的图像或片段生成**一种简单应用输出**，例如深度感知背景虚化或点云视图。录制好的Demo即可满足这一展示形式。

将输出打包为`depth_results.zip`。

### 成功完成项目的重要提示

- **方法灵活性：**可以遵循建议路线，也可以设计其他技术上合理的方法。只要完成核心任务并遵守评价协议，不必严格照搬“实现路线”。
- **完整性：**完成Part 1和Part 2，然后在Part 3选择**一项改进**，与未经改动的方法比较。如果精度没有改善，应通过受控实验和失败案例解释原因。完全忽略Part 3将被评为“Incomplete”。
- **可视化：**整洁的流程图、预测叠加图、局部放大比较和失败可视化是计算机视觉课程的重要内容，会显著影响展示成绩。
- **数据范围：**双目练习使用Middlebury；单目定量评价使用NYU Depth V2；演示和定性比较只需指定的仓库视频。**不要求自录视频或额外数据集。**评分重点是想法、测得的效果及分析。
- **公平比较：**原方法与改进方法使用相同测试图像、评价mask、输出尺寸和指标。保留各预训练模型要求的预处理，报告输入尺寸、运行时间与GPU峰值显存。
- **引用：**正确引用所有数据集、预训练模型、外部仓库和借用的实现组件。

## 3. 开始指南

为帮助高效开展项目，建议使用以下流程：

1. **Step 1：文献阅读。**从双目几何、Depth Anything V2 [5]、选定的度量深度模型[7或8]和Video Depth Anything [10]开始。只有选择时序融合改进时，才需要进一步关注光流方法。
2. **Step 2：熟悉代码。**克隆并运行[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)和[Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)的官方仓库。先运行提供的Demo，再用一个样本检查输入/输出格式和预训练checkpoint。
3. **Step 3：实验。**按照第6.1节的链接下载资源。先运行单张图像，再评估静态测试集，最后处理指定的100帧片段。除彩色可视化外，还要保存**浮点深度预测**。
4. **Step 4：写作与提交。**将定量结果、消融和失败案例整理成表格与图像，使用CVPR模板撰写报告，引用全部参考文献，并将规定文件提交到Canvas。

## 4. 推荐阅读列表

- **经典几何：**Semi-Global Matching [1]、KITTI benchmark [2]。
- **单目基础模型：**DPT [3]、Depth Anything [4]、Depth Anything V2 [5]。
- **度量深度：**UniDepth [6]、Depth Pro [7]、UniDepthV2 [8]。
- **视频深度与运动：**RAFT [9]、Video Depth Anything [10]。

## 5. 实现路线

### 5.1 Part 1：基线——“几何与逐帧深度”

**目标：**理解深度的几何含义，并在使用时序模型之前建立逐帧基线。

#### 1. 经典双目：从视差到深度

- **Step 1：校正后的双目图像。**在Middlebury的15组`trainQ`图像对上运行OpenCV StereoBM或StereoSGBM。展示视差和无效区域；左右一致性属于**可选诊断**。
- **Step 2：转换为度量深度。**使用各组`trainQ/calib.txt`：深度以米计，等于焦距乘以基线、除以`disparity + doffs`，然后再除以1000，因为标定中的基线单位为毫米。OpenCV的定点视差必须先除以16。使用四分之一分辨率的标定参数，并排除无效视差。

按原文文字写为：

\[
d_{\mathrm{px}}=\frac{d_{\mathrm{OpenCV}}}{16},\qquad
z_{\mathrm{m}}=\frac{f_{\mathrm{px}} B_{\mathrm{mm}}}{d_{\mathrm{px}}+d_{\mathrm{offs,px}}}\times\frac{1}{1000}.
\]

[Middlebury标定定义](https://vision.middlebury.edu/stereo/data/scenes2014/)

#### 2. 单目逐帧基线

- **相对深度：**在相同的100个视频帧上，逐帧独立运行**Depth Anything V2-Small**。这是视频的逐帧基线，**不用于在NYU上作度量深度预测**。
- **尺度约定：**将输出标注为“relative depth”（相对深度）。不要估算米制深度，也不要使用测试GT对它做对齐。Part 2的度量深度评价使用独立的度量模型。
- **时序诊断：**在提供的片段上，将逐帧结果与Video Depth Anything比较，描述可见的闪烁、边缘不稳定和运动伪影。**不需要相机位姿标注或时序GT。**

**预期结果：**逐帧模型应恢复较可信的场景结构，但可能存在度量尺度不正确、边界不稳定与明显时序闪烁。

### 5.2 Part 2：SOTA复现——“度量深度与时间一致深度”

**目标：**无需从头训练基础模型，复现近期公开模型，实现零样本度量深度与长视频一致深度。

#### 1. 单图度量深度：UniDepthV2 [8]或Depth Pro [7]

- **要求：**选择**一个**度量模型，建议从**UniDepthV2 ViT-S**开始。加载：

  ```python
  UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")
  ```

  然后遵循仓库的`infer(rgb)`示例。Depth Pro也可作为替代。采用**仅RGB输入的推理**，直接在NYU上评价原生米制输出，**不使用测试集尺度拟合**。
- **关键学习点：**使用相同的NYU协议比较所选度量模型与自己的改进，解释绝对尺度和物体边界上的误差。**不要求训练基础模型。**

[UniDepthV2代码](https://github.com/lpiccinelli-eth/UniDepth) · [ViT-S checkpoint](https://huggingface.co/lpiccinelli/unidepth-v2-vits14) · [替代方案Depth Pro](https://github.com/apple/ml-depth-pro)

#### 2. 视频深度：Video Depth Anything [10]

- **系统：**使用**Video Depth Anything Small相对深度checkpoint**，处理`davis_rollercoaster.mp4`的**前100帧**。采用**离线推理**；streaming和米制视频模式为可选项。
- **必须比较：**在**完全相同的帧**上展示逐帧Depth Anything V2-Small与Video Depth Anything-Small。使用相同调色板，并且每种方法的归一化在整段视频上保持固定。输出标注为相对深度；**颜色不代表两方法共享的米制尺度**。
- **资源规则：**使用Small模型和视频仓库默认的FP16。起始参数采用：

  ```text
  --encoder vits --max_len 100 --max_res 640 --save_npz
  ```

  如果显存不足，可减小`--input_size`，并报告具体设置。一个方法与它自己的改进之间应保持分辨率固定。

**预期结果：**观察视频建模是否减少闪烁，并识别仍存在的伪影。片段不提供深度GT，因此**视觉观察不能证明米制精度**。

### 5.3 Part 3：探索——“标定、一致性与应用”

**目标：**选择**一项轻量改进**。以下方向是不同备选路线。最简单的路线是在静态图像上做细化，并在NYU上评价。视频改进也可以使用受控的定性比较和运行时间测量；**不需要带标注的视频集合**。

- **Direction A：光流引导融合。**使用RAFT或Farnebäck光流，将上一帧深度warp到当前帧；通过前后向一致性排除遮挡，再按置信度权重与当前预测融合。
- **Direction B：度量尺度标定。**仅用NYU训练/开发图像拟合**一个全局尺度**，然后固定该尺度，在测试划分上评价。解释改进能否泛化；不得针对测试GT逐张拟合独立尺度。
- **Direction C：边缘感知细化。**利用RGB梯度、语义边界或分割mask锐化深度不连续边界，避免将图像纹理复制进深度。
- **Direction D：不确定性感知平滑。**通过模型置信度、测试时增强或跨模型分歧估计不确定性，只平滑不确定且时间上不一致的区域。
- **Direction E：高效部署。**将Small模型导出为ONNX/Core ML，进行量化或使用动态分辨率，报告速度、内存与精度的权衡。
- **Direction F：深度感知应用。**改进背景虚化或基于深度的物体排序等简单效果，与使用未经改动深度模型的同一应用比较。复杂界面、额外传感器和新增标注数据均为可选项。

## 6. 数据集与评估

### 6.1 数据集与所需下载内容

#### 1. Middlebury Stereo v3：双目练习

下载四分之一分辨率输入ZIP（约30 MB）和左视图GT ZIP（约13 MB），无需填写申请表。

只使用15组`trainQ`图像对、对应的`calib.txt`以及`disp0GT.pfm`真值，**不要求benchmark测试集**。

[输入ZIP](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip) · [GT ZIP](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip) · [benchmark说明](https://vision.middlebury.edu/stereo/submit3/)

#### 2. NYU Depth V2：度量评价

下载labeled `.mat`文件（约2.8 GB）及下方的`splits.mat`，无需填写申请表。

- 使用`trainNdxs`的**795张图像**进行开发，使用`testNdxs`的**654张图像**进行评价。
- Python中使用MATLAB索引时，先减1。
- 使用h5py，RGB读取方式为`f["images"][i].transpose(2,1,0)`。
- 米制填充深度的读取方式为`f["depths"][i].T`。保持浮点类型。
- 链接中的loader只用于示范`.mat`读取；评价应使用`depths`和下方课程协议，**不要采用该loader的`rawDepths`或对齐评价方案**。
- **无需下载raw视频集合。**

[标注数据](https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) · [官方划分文件](https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat) · [数据字段说明](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) · [Python loader示例](https://gist.github.com/ranftlr/a1c7a24ebb24ce0e2f2ace5bce917022)

#### 3. 视频与Small权重

必须使用的片段为Video Depth Anything仓库克隆后包含的：

```text
assets/example_videos/davis_rollercoaster.mp4
```

按**原始帧率**使用其**前100帧**。

将下面两个相对深度Small模型的权重分别下载到对应仓库的`checkpoints/`目录。依照链接README中的安装与推理命令操作；视频部分使用Part 2给出的参数。其他示例片段属于可选内容。

- [示例视频目录](https://github.com/DepthAnything/Video-Depth-Anything/tree/main/assets/example_videos)
- [指定片段下载](https://github.com/DepthAnything/Video-Depth-Anything/raw/refs/heads/main/assets/example_videos/davis_rollercoaster.mp4)
- [逐帧模型代码](https://github.com/DepthAnything/Depth-Anything-V2)
- [逐帧Small权重](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true)
- [视频模型代码](https://github.com/DepthAnything/Video-Depth-Anything)
- [视频Small权重](https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true)

### 6.2 评价指标（必须）

- **度量深度准确性：**在NYU上报告**AbsRel（主要指标）、RMSE和δ1**。将预测恢复到`480 × 640`，在Python crop `[45:471, 41:601]`中、GT深度处于`0.001–10 m`范围的像素上评价；将预测裁剪到该范围。对全部654张测试图像逐图计算指标后等权平均，**不做尺度对齐**。应比较本组在课程协议下得到的结果，不宣称精确复现论文中的分数。
- **双目练习：**在`trainQ`中**有限、正值GT视差**上，报告以像素计的**平均绝对视差误差**和**有效预测比例**。说明如何处理无效预测。这项练习与NYU度量模型比较分开。
- **视频比较：仅作定性比较。**展示相同的帧，讨论时序平滑性、边缘稳定性与运动伪影。时序误差、尺度漂移、光流评价和深度边缘precision/recall等数值指标均为**可选项**。
- **效率：**报告每图/每帧推理时间和GPU峰值显存，同时说明硬件、模型变体、数值精度和输入尺寸。
- **定性评价：**展示RGB、预测深度、选定的应用输出，以及提供数据中**实际出现的3个代表性困难案例**。解释局限性，不要求必须出现某几类场景。

**注意：**禁止对视频每一帧分别归一化，因为这可能掩盖闪烁。归一化应在整段视频上保持固定，并且改进方法使用未经改动方法的归一化设置。

## 参考文献

[1] Hirschmüller, H. “Stereo Processing by Semiglobal Matching and Mutual Information.” TPAMI 2008.

[2] Geiger, A., et al. “Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite.” CVPR 2012.

[3] Ranftl, R., et al. “Vision Transformers for Dense Prediction.” ICCV 2021.

[4] Yang, L., et al. “Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data.” CVPR 2024.

[5] Yang, L., et al. “Depth Anything V2.” NeurIPS 2024.

[6] Piccinelli, L., et al. “UniDepth: Universal Monocular Metric Depth Estimation.” CVPR 2024.

[7] Bochkovskii, A., et al. “Depth Pro: Sharp Monocular Metric Depth in Less Than a Second.” ICLR 2025.

[8] Piccinelli, L., et al. “UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler.” arXiv:2502.20110, 2025.

[9] Teed, Z., and Deng, J. “RAFT: Recurrent All-Pairs Field Transforms for Optical Flow.” ECCV 2020.

[10] Chen, S., et al. “Video Depth Anything: Consistent Depth Estimation for Super-Long Videos.” CVPR 2025.

[11] Sturm, J., et al. “A Benchmark for the Evaluation of RGB-D SLAM Systems.” IROS 2012.

[12] Silberman, N., et al. “Indoor Segmentation and Support Inference from RGBD Images.” ECCV 2012.
