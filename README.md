# AIAA 3201：Part 1 与 Part 2

本分支为经典双目、单目深度与视频时序比较的代码和说明快照。
仅包含 Part 1/2 及相关公共依赖，不包含数据、视频、权重和运行产物。

## 复现前需要准备什么

本分支只提供项目自有代码、配置、测试和说明。克隆后没有 data、checkpoints、outputs，
也没有官方模型仓库；请按下面顺序准备。所有路径均相对于本分支的项目根目录，
也就是包含 configs、scripts 和 src 的目录，后续命令均从这里执行。

### 1. 运行工具与环境

完整复现按 Linux / Bash、Conda、Python 3.10 和 NVIDIA CUDA GPU 准备。
数据下载/解压工具包括 git、wget、unzip；视频读写依赖随项目环境安装。
Windows Git Bash 可用于上传代码，实际实验建议使用 Linux 或具备GPU支持的WSL2；
后者未在本项目单独验证。Middlebury经典匹配和部分统计使用CPU，
但默认单目/视频模型配置使用CUDA和FP16，不能把它们直接当作CPU运行配置。
原实验显存和耗时只是参考，不能当作所有设备的最低资源要求。

先取得下一节的固定版本官方代码，再按 [Step 0：Conda环境](docs/step0.md#conda-环境)
创建 environment.yml 对应的 aiaa3201，以及 environment-unidepth.yml 对应的
 aiaa3201-unidepth。Depth Pro、DA-V2和VDA使用前者；UniDepth与联合尺度/边界分析
使用后者。两个环境分别安装各自依赖，不合并安装到base。
运行模型前在对应环境确认PyTorch能识别CUDA；具体检查见Step 0。

### 2. 官方模型代码

把四个官方仓库单独放到以下位置；它们不包含在本分支里。
固定commit和上游地址见 [configs/versions.yaml](configs/versions.yaml)，
克隆与checkout命令见 [Step 0](docs/step0.md#通用复现顺序)。

| 项目根目录下的位置 | 对应官方仓库 | 使用任务 |
|---|---|---|
| ml-depth-pro/ | apple/ml-depth-pro | 2A Depth Pro及官方边界评价依赖 |
| Depth-Anything-V2/ | DepthAnything/Depth-Anything-V2 | 1B逐帧与NYUv2扩展 |
| Video-Depth-Anything/ | DepthAnything/Video-Depth-Anything | 2B视频深度及指定素材来源 |
| UniDepth/ | lpiccinelli-eth/UniDepth | 2A UniDepth |

先放好 ml-depth-pro 再创建主环境，因为 environment.yml 会安装本地的
 -e ./ml-depth-pro。已有官方仓库先核对版本与自己的修改，不要直接覆盖或reset。

### 3. 数据放在哪里

| 数据 | 放置位置 | 应包含什么、用于什么 |
|---|---|---|
| Middlebury Stereo v3 Quarter | data/middlebury/MiddEval3/trainingQ/ | 15个场景子目录；每个场景含im0.png、im1.png、disp0GT.pfm、calib.txt、mask0nocc.png，用于1A |
| NYU Depth V2 labeled | data/nyuv2/nyu_depth_v2_labeled.mat | 完整1449张labeled MAT；程序读取RGB、填充深度及分析所需标注，不是只下载RGB图片 |
| NYUv2官方split | data/nyuv2/splits.mat | 官方795 train / 654 test划分；用于2A及1B的NYUv2对齐扩展 |
| 指定视频 | data/davis_rollercoaster.mp4 | 1B与2B使用同一文件；不需要另采自录视频 |

Middlebury和NYUv2的下载、解压及检查方法见 [Step 0：数据与权重](docs/step0.md#数据与权重)。
已有这些资源时可按表放好后校验，不必重新下载。不要多套一层目录，也不要换用Middlebury
其他分辨率或NYUv2其他split来匹配本项目的结果。

指定视频来自固定版本 Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4，
将该文件原样复制到 data/davis_rollercoaster.mp4；不要转码、裁剪、重采样或补帧。
预期SHA256为7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8，
对应配置为 [dav2_video.yaml](configs/part1/dav2_video.yaml) 和
[vda_video.yaml](configs/part2/vda_video.yaml)。此版本只有70个真实帧、24 FPS，
课程100帧要求待确认。不能以复制帧凑数或修改预期SHA绕过检查。

### 4. 预训练权重放在哪里

| 模型 | 项目根目录下的权重路径 | 使用任务 |
|---|---|---|
| DA-V2-Small relative | checkpoints/depth_anything_v2/depth_anything_v2_vits.pth | 1B视频与NYUv2扩展 |
| Depth Pro | checkpoints/depth_pro/depth_pro.pt | 2A |
| UniDepthV2-S | checkpoints/unidepth_v2_vits14/model.safetensors 和同目录config.json | 2A；两份文件均需准备 |
| VDA-Small relative | checkpoints/video_depth_anything/video_depth_anything_vits.pth | 2B；不要换成metric视频权重 |

来源、下载脚本与校验步骤见Step 0；固定SHA256和模型版本以 configs/part1、configs/part2
的对应YAML为准。准备脚本可能下载额外兼容资源；当前2B不需要RAFT、metric视频权重、
相机位姿或视频深度GT。1A的BM/SGBM不需要预训练权重。
手工放入已有资源时也需核对固定SHA，而不是仅用文件名或自己新生成的校验清单证明来源。

### 5. 哪些目录由程序生成

- data/video_frames/part1b_davis_v1/：按规定帧号和原PTS生成的RGB缓存，不必自己抽帧；2B会核对并复用或创建兼容缓存。
- data/manifests/：数据准备/验证产生的来源和检查记录，不是另一个需要手工寻找的数据集。
- outputs/part1/：本地生成的Middlebury、DA-V2视频和NYUv2扩展结果。
- outputs/part2/：本地生成的两条米制基线、尺度/边界分析、VDA独立结果与配对比较。

这些目录均不随本分支上传。程序生成后应保留原始预测和真实运行记录，
已有正式目录不要覆盖；复现实验使用步骤文档指定的新目录。
配对比较必须先具备1B和2B两条预测；完整2A尺度/边界分析必须先具备两条654张基线。
如果使用Step 3的 reproduce_v1 等新目录，分析配置中的 baselines 也必须指向新结果，
不能直接使用默认绑定 frozen_v1 的配置。Step 3已说明如何复制配置并指定 --config。

### 6. 开始实验前的检查

按Step 0完成环境、资源版本/SHA、数据结构与相关测试检查；
再按Step 2和Step 4的 check 入口核对视频、权重、官方代码及CUDA。
新克隆仓库没有任何历史预测，不能从 verify 或结果比较命令直接开始。
各任务首次运行、分析与核验的完整指令见下方Step文档，不在README重复维护另一套实验命令。

## 实验顺序与完整指令

1. 阅读 [Step 0](docs/step0.md)：先克隆固定版本的四个官方仓库，再创建环境。
   主环境为 aiaa3201，UniDepth 为 aiaa3201-unidepth，不安装到 base。
2. [Part 1A](docs/step1.md)：Middlebury BM/SGBM、开发集选参和独立评估。
3. [Part 1B](docs/step2.md)：指定片段 DA-V2-S 逐帧 relative 与 NYUv2 对齐扩展。
4. [Part 2A](docs/step3.md)：Depth Pro/UniDepth 原生米制基线与尺度/边界分析。
5. [Part 2B](docs/step4.md)：VDA-S 离线 relative 及 DA/VDA 同帧比较。

## 当前状态与协议

1A/2A 已完成；1B/2B 已完成现有官方素材的70帧实验与原始重算。
课程100帧素材要求仍待确认，不重复或插帧补足。
NYUv2 保持标准654张、冻结crop、有效范围与 alignment:none 的原生米制主轨道；
GT辅助对齐仅作诊断。视频没有深度GT，变化proxy不是深度准确率或真实尺度漂移真值。

Step文档保留已有指标与分析；其中“已在本地”“已经核验”和运行效率描述指源实验项目，
不表示刚克隆本分支就具备数据或通过核验。文档提到的 outputs 路径需在本地运行生成，
历史维护脚本和审计附件没有随本包上传。新复现应自行记录实际环境和核验结果。
第三方仓库按 configs/versions.yaml 固定版本单独获取，第三方许可证随各上游仓库取得。
下载脚本可能准备额外兼容资源，不代表当前2B需要米制视频模式或RAFT。
正式训练与长时预测由用户启动，不要覆盖已有正式输出目录。

## 目录

configs 为配置与协议，scripts 为命令入口，pipelines 为任务编排；
src 为公共实现，tests 为相关回归测试，docs 为完整步骤说明。
课程要求见 Project3_CN_final.md 与根目录课程PDF。
安装好官方依赖和相应环境后，可运行 `conda run -n aiaa3201 python -m pytest`。
本包测试范围小于源项目全部测试，不能沿用全项目测试数量。

发布文件清单与SHA256见 PART1_PART2_EXPORT.json；完成范围见 implemented.md。

## 补充实验与公开证据

[误差与覆盖率、独立基线逐图核对](docs/controlled_studies.md)：开发集冻结选参、十场景验证、Pareto 图，以及独立 NYUv2 运行证据；不替换上述原有基线。
