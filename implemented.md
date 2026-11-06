# 实际完成与验证

当前方法说明见README与Step B/C；历史执行记录未改写，调整前的完整说明归档在`outputs/maintenance/part3_bc_rebuild_20261104/implemented_before.md`。此处保留Part 1/2的实际记录，新B/C仅按本轮真实执行补记。

## 2026-10-01 — Step 0 自动化基础（历史）

- 建立 `AGENTS.md`、主环境 `aiaa3201`（当时 Python 3.10.21）、环境 YAML、直接依赖和早期完整包清单，记录四个上游 commit；UniDepth 的依赖冲突被识别，未加入主环境。
- 建立按 Part 编排、`src/` 共享实现的结构，冻结数据与输出契约。metric/relative/aligned-relative 区分明确；schema 后续升级为 v2，允许几何预测的 mask 外 NaN。
- 实现七项深度指标、`doffs` 几何、边界 P/R/F1、target-to-source backward warp、FB consistency 与时序/尺度 proxy；pytest 限定项目测试。当时结果为 **26 passed**，下载脚本语法和资源 checksum 检查通过。
- 下载 Middlebury 输入/GT ZIP、NYUv2 split 和权重，验证 15 场景与 795/654 划分。NYUv2 官方下载速度缓慢，完整 labeled MAT 改从公开镜像获取；ZIP 检查通过，MAT 为有效 HDF5、长度 2,972,037,809 bytes，前 5,343,837 bytes 与官方部分下载一致。只删除了当时冗余的不完整 `.part` 文件。
- 来源与清单记入 `data/manifests/`、`checkpoints/checksums.sha256`；RAFT 从项目内已验证 checkpoint 读取，不隐式下载。
- 当时完成 DA-V2-S、Depth Pro 的单图 smoke、VDA relative/metric 的八帧 smoke 与 RAFT 图像对 smoke，没有训练或更新权重。当前 `outputs/step0/` 有后续环境和 smoke 产物，不能仅根据路径名认为它们仍是当时的文件。
- 核验参考 evaluator 的有效范围与 crop；标准 654 张 crop 内最小正 GT 为 0.71330047 m，所有有限正深度共 156,018,240 像素，没有 GT≥10 m。采用 `0.001–10 m`；视频协议因输入暂缓仍为 `candidate_v1`。
- 公开数据入口门禁当时通过。自录视频、mandatory clips 与 dev/test 帧 manifest 没有完成，不把 Step 0 整体标记完成。

## 2026-10-01 — Part 1A 初版与几何修正（历史）

- 实现 PFM/标定读取、MATLAB/图像方向检查、baseline 毫米转米、OpenCV 1/16 视差解码、左右双向匹配与 1 px 亚像素一致性检查，使用 `z=fB/(d+doffs)`。
- 初次调参 Bad-2 接近 100% 后停止冻结，定位到 PFM scale 被错误重复应用：payload 已是像素视差，负 scale 符号表示小端序。修正解析后重做检查，未修改 GT。
- 固定 development scenes：Adirondack、Jadeplant、Motorcycle、Pipes、PlaytableP；其余 10 场景不参与调参。早期两阶段 SGBM 搜索、30 份预测及 600 个重载指标门禁，后来由四组 `frozen_v2` 取代。
- 未放宽 LR 阈值追求分数，因为 invalid-as-bad 的 Bad-2 会随放宽阈值单调不增。人工查看 contact sheet、Jadeplant 局部和 PianoL 等失败样本，记录方向、尺度与遮挡/薄结构问题。
- 完成 ±0.25/0.5/1/2 px 扰动、精确深度误差与一阶理论比较；当时测试 **35 passed**。没有正式训练。

## 2026-10-02 — Part 1A 四组 frozen_v2（历史开发）

- 明确拆分 `bm_fixed/bm_tuned/sgbm_fixed/sgbm_tuned`，分别使用实验 ID、目录与 metadata。
- BM 一轮搜索 144 个候选，SGBM 一轮 36 个；同一 5 场景、固定 `LR=1 px`，按最低 Bad-2、MAE、coverage 排序。BM 选择 `block_size=15,uniqueness_ratio=0,speckle_window_size=0,texture_threshold=10`；SGBM 选择 `block_size=3,uniqueness_ratio=0,speckle_window_size=0`。
- 在 CPU 完成 15 场景×4 组，保存 60 份预测、metadata、六联图、四张 contact sheet、逐场景与宏平均指标、局部失败图和元数据。
- 几何扰动理论由标定决定，四组共享；另增四组实际预测的距离分桶误差、Bad-2 与 coverage。调参降低 Bad-2 并提高 coverage，但 valid-only MAE/RMSE 可能上升，需联合解释。
- 当时人工检查 BM tuned contact sheet 与经验曲线，重载 60 份预测、1,200 个指标，`rtol=1e-6,atol=1e-8` 通过；测试 **36 passed**。
- 本节记录方法与当时门禁；当前数值和时间取下方当前产物审计，不将后续重跑记作本日执行。

## 2026-10-02 — Depth Pro × NYUv2（历史开发）

- 按用户要求先进行无需视频的 Step 3，Step 2 仍暂缓。复用已验证数据/split/checkpoint，无新增数据下载，没有训练、微调或更新权重。
- 新增流式 `src/data/nyuv2.py`，恢复 MATLAB/HDF5 维度顺序；新增 Depth Pro adapter、pipeline、shell 入口、五联图与官方 `SI_boundary_F1` 包装。
- 使用完整 RGB、官方 transform、FP16、预测焦距，保存 crop/指标裁剪前 `float32` 米制深度；统一 evaluator 不做 GT 对齐。入口支持 `infer/infer-evaluate/evaluate`、固定位置审计、小子集和 `--resume`。
- 首次三图 smoke 因执行器继承多个任务而并发重复运行；当时不把该 smoke 当作正式证据，保留失败记录，后续改用单进程执行。未把这次失败描述为成功门禁；早期开发目录无需重建。
- positions 0、326、653 当时完成独立 NumPy 七指标审计和人工方向/crop/色标检查。Boundary F1 合成测试发现官方分别平均四个方向，单方向完美边界为 0.25；保留上游定义，测试改用四方向中心跳变。
- 完成 654 张冻结 baseline、10,000 次图像级 bootstrap、分类失败分析和 654 张五联图。失败阈值是 `AbsRel>0.25` 或 `δ1<0.60` 或 `SI Boundary F1<0.01`；仅阈值失败进入独立 contact sheet。
- 后续 CPU evaluate 重建评估图表，没有改写 prediction。Part 2A 不再生成额外 `reload_validation.json`，保留三样本公式审计。新增阈值筛选测试，当时结果 **41 passed**。
- 当时人工观察强亮窗、反光、密集结构与较远室内区域的失败。该人工检查属于历史，当前重新生成的图像未在本次全量复查。

## 2026-10-03 — UniDepthV2-S baseline（历史开发）

- 先建立同协议第二条 baseline，再做相机/边界/尺度/目标域四项分析；没有把普通指标或场景分类当作完整因果与跨域分析。
- 建立 `aiaa3201-unidepth`，新增其环境/依赖文件。使用 Python 3.10.21、PyTorch 2.7.1+cu126、torchvision 0.22.1+cu126、NumPy 2.2.6、xformers 0.0.31.post1；未改主环境。
- 使用 `lpiccinelli/unidepth-v2-vits14`，revision `038c238f06c87b6c2f5b3749fd51fbf442b1f218`。首次官方下载遇 TLS 失败，随后使用 `hf-mirror.com`；配置和权重 SHA256 分别为 `ecc1f898690debe387d10329cca5d9d66a0a447ce83c3a2a7ae3673c7ce43cfe`、`93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648`。未编译只用于 3D 评估的 KNN 扩展。
- 新增 adapter/pipeline/script/config，共享 NYUv2 split/crop/range/metrics/bootstrap/失败阈值；正式轨道不输入 K，保存模型预测 3×3 内参。支持同样的模式、三样本审计与续跑。
- 公共 evaluator 同时核对预测焦距或内参，绘图标题改为配置驱动；增加协议一致性测试。
- 当时先通过三样本门禁，再完成 654 张推理/评估；全部 raw depth/K、metadata、CSV 与 visualization stem 经核对，人工检查正常 kitchen、镜面 bedroom 与失败总览。没有训练或微调，当时测试 **42 passed**。

## 2026-10-03 — 四项联合关键分析（历史开发）

- 新增 `key_analysis.yaml`、分析入口与脚本，支持 `all/infer-external/analyze` 和 `--resume`，显式区分 native、calibrated、oracle 诊断。
- 扩展 adapter 的外部相机接口。发现 UniDepth 在传入 K 后仍返回未使用的辅助相机头预测，因此保存本次真正构造 rays 的输入 K。
- Depth Pro calibrated f 用官方精确关系 `depth_native*f_calibrated/f_predicted`；检查 raw depth 未触及内部 clamp 后使用闭式重缩放。UniDepth 外部 K 当时进行独立 654 张诊断推理。焦距误差、主点误差与尺度误差相关性仅作归因线索。
- 实现固定比值阈值 1.05、1/2/3 px Chebyshev edge P/R/F1、边缘区域误差与 20 个确定性选择案例；当时人工查看定量图和八联图。
- 实现每图尺度比、距离分桶、oracle 单乘法尺度诊断；GT 驱动的结果禁止进入正式主表。
- 按用户要求未增加第二目标域：只有 NYUv2 迁移与域内稳健性，类别排名要求 `n>=5`；Depth Pro 完整训练组成未逐图核实，未宣称严格排除了 NYUv2 训练重叠。
- 保存联合逐图、边界、深度分桶、场景类型与配对统计；调用 metadata 区分本次执行和累计完整性。没有训练，当时测试 **43 passed**。

## 2026-10-04 — 分析说明澄清（历史）

- 说明 `all` 依赖完整 baseline，运行外部 K 诊断和分析而不重生成 baseline；当时实际由 `infer-external` 与 `analyze` 分阶段生成。
- 解释每图 oracle 公式与 GT 泄漏限制，逐图胜负计数不表示微调后表现，也不反映差距大小。
- 说明各 JSON/CSV/PNG/NPZ 用途；仅修改文档，没有执行实验。上述分阶段历史与当前后续 `all` 调用记录分别保留。

## 2026-10-12 — Step 文档拆分（历史）

- 新建 `docs/README.md` 与 Step 0–6 文档，根 README 精简为状态、导航、快速开始、核心结果、协议和目录。
- 未完成步骤只记录依赖、计划与门禁。该次任务仅重组中文文档，没有运行实验或修改数据、预测、指标。

## 2026-10-13 — 当前产物审计与项目说明整理（本次）

### 实际操作与范围

- 阅读课程 PDF、中文译文、`AGENTS.md`、完整计划、现有说明、配置、运行入口、adapter 与评估/分析实现；确认经典几何、原生米制、relative 视频和 Part 3 的依赖关系。
- 整理工作规则、相对路径示例和代码版本说明；版本文件保留上游 commit，移除失效资源快照。代码检索确认 `runtime_snapshot` 与路径模板没有项目调用方，没有机械改动冻结科学配置。
- 提取通用上游版本、环境检查、独立 SHA256、复现顺序与输出检查说明到 `docs/step0.md`，移除失效的跨设备说明。
- 中文 README、PLAN、Step 1/3 和辅助说明从当前完整精度文件生成表格，新增 `docs/current_results.md`，保留具体来源 SHA256。
- 项目代码仅改 `scripts/smoke_models.py` 的过时报错文字与 `scripts/prepare_data.sh` 的来源注释；未更改模型、预测或评估实现。
- 没有运行推理、训练、微调、重新下载、参数搜索或全量原始预测复评；读取现有运行记录不代表本次执行它们。

### 当前 baseline 指标

来源为以下正式目录的 `metrics_summary.json`，本文直接保留完整精度：

```text
outputs/part1/middlebury/frozen_v2/
outputs/part2/depth_pro_nyuv2/frozen_v1/
outputs/part2/unidepth_v2_nyuv2/frozen_v1/
```

| 指标 | Depth Pro | UniDepthV2-S |
|---|---:|---:|
| abs_rel | 0.10047157333839377 | 0.11454372957333221 |
| sq_rel | 0.06400258360254008 | 0.06895209489957993 |
| rmse | 0.3922168142032252 | 0.39884995782725563 |
| rmse_log | 0.13533881856042604 | 0.14121805082915398 |
| delta1 | 0.915173937355017 | 0.8921797284727733 |
| delta2 | 0.9798598292097128 | 0.9807310734949964 |
| delta3 | 0.9918777381413865 | 0.9933327987804502 |
| si_boundary_f1 | 0.15927408060093043 | 0.11300739651910406 |

| Middlebury 实验 | Disp. MAE (px) | Bad-2 | Coverage | Depth AbsRel | Depth RMSE (m) |
|---|---:|---:|---:|---:|---:|
| bm_fixed | 0.9790182810919639 | 0.48746251528646606 | 0.543560039101519 | 0.013339634814117755 | 0.1420388864510264 |
| bm_tuned | 1.7314005389377825 | 0.40691314701337505 | 0.6594842901060117 | 0.0238476467201213 | 0.21063200280676336 |
| sgbm_fixed | 1.0587535544461757 | 0.347418625607543 | 0.6966939250439883 | 0.012259898983278608 | 0.12824877441986693 |
| sgbm_tuned | 1.307015561428136 | 0.3373200307999628 | 0.7168935922221646 | 0.015361069375918145 | 0.14668839580267548 |

当前运行记录的 UTC 时间分别为：Middlebury `2026-10-12T14:00:49.480361+00:00`；Depth Pro `2026-10-12T14:11:06.991986+00:00`；UniDepthV2-S `2026-10-13T04:47:20.413561+00:00`。这是已有输出的记录时间，不是本次启动命令的时间。

### 来源一致性与外部 K

- 只读审计使用 `aiaa3201` 的 Python/NumPy/CSV/JSON/YAML 工具，临时脚本不调用模型或 evaluator。当前 split 654 张身份与逐图 CSV 对应。
- 两条 baseline 和分析总表逐图比较七项深度指标及 SI Boundary F1，每模型 5,232 值，共 10,464 值，最大绝对差为 0；baseline 宏平均与逐图 CSV 均值一致。相机四轨道、尺度、oracle、边缘区域、P/R/F1 与逐图胜负也核对通过。
- 60 份 Middlebury 预测、60 行逐场景表与各组宏平均一致；原有 `reload_validation.json` 记载 60 文件/1,200 值、`rtol=1e-6,atol=1e-8` 通过。本次只读取该 gate。
- 两条 baseline 的配置 SHA256 与各 `run_metadata.json` 一致；分析配置 SHA256 为 `64a7f57dc9816db6a45981755b303c61bbbb00711eb6bcc3b1f336b78266f94b`，与调用记录一致，resolved YAML 语义相同。完整哈希见 [当前结果审计](docs/current_results.md)。
- 当前分析 `invocation_metadata.json` 记载 UTC `2026-10-13T04:50:08.613642+00:00`、`mode:all`，本次推理/分析均执行且输出完整。外部 K 摘要记载新推理 654、`resumed:0`，PyTorch/CUDA/GPU/精度环境与当前 UniDepth baseline 相同。
- 所有 654 份外部 K metadata 标记 `external_intrinsics_provided:true`、`metric/meter/alignment:none` 与固定标定 K。每轨道抽查 positions 0/326/653（共 9 份 NPZ），验证 depth 的尺寸/dtype/有限正值与外部 K 一致性。
- 因此当前四项分析确实引用当前 UniDepth baseline，外部 K 有当前环境重新生成的证据；无需为确认这两点启动重算。未对全部原始数组再做逐像素评估，不把文件关联审计写成一次新全量评估。
- 当前深度 Pro/UniDepth 失败数 49/77、分别触发计数 17/30/18 与 26/36/39，已从逐图 CSV 重新统计；边界 3,924 行、距离桶 4,350 行、场景类型 32 行符合数量约束。

### 环境、检查与失败记录

- 用 `conda run -n aiaa3201` 与 `conda run -n aiaa3201-unidepth` 读取包版本：主环境 Python 3.10.22、torch 2.1.1+cu121、torchvision 0.16.1+cu121、NumPy 1.24.0、OpenCV 4.11.0.86、SciPy 1.15.3、h5py 3.14.0；隔离环境 Python 3.10.21、torch 2.7.1+cu126、torchvision 0.22.1+cu126、NumPy 2.2.6、xformers 0.0.31.post1。没有安装或导出新环境。
- 根目录 `git status --short` 返回“不是 git 仓库”；本次用修改前副本与文件清单比较，不报告虚假的 Git clean。
- 首轮临时只读审计把 Middlebury 空距离桶 CSV 空字符串当作 float，报 `ValueError`；修正临时脚本跳过空桶后全部审计通过，没有改动 CSV。
- 检查命令 `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/aiaa3201-mpl-audit conda run -n aiaa3201 python -m pytest -q -p no:cacheprovider`：**43 passed，14 warnings**；warnings 均为现有 Matplotlib/PyParsing 弃用提示。缓存放在临时目录，禁用第三方字节码写入。
- 检查 81 份项目自有文本：21 份 Markdown 的本地链接与代码围栏检查通过，37 份 Python 的 AST 解析通过，11 份 YAML 可解析，7 份 shell 入口及当前操作文档中 26 个 Bash 代码块的 `bash -n` 通过。项目自有说明与入口的失效平台/路径引用为 0，已移除说明的引用为 0。
- 对照任务开始时的清单，第三方仓库、`data/`、`checkpoints/` 与所有自动输出文件的尺寸/mtime 未变，小型文本的 SHA256 未变；`outputs/README.md` 是本次明确修改的手写目录说明，不属于自动运行元数据。
- 受保护的 `data/manifests/sources.yaml` 保留当时下载来源的原文；`data/`、`checkpoints/` 中既有说明也未改写，中文当前使用说明集中在根 README 与 `docs/`。这些历史材料不构成当前执行要求。

### 本次文件变更

- 根目录：`AGENTS.md`、`PLAN.md`、`README.md`、`implemented.md`。
- 说明：`docs/README.md`、`docs/step0.md`、`docs/step1.md`、`docs/step3.md`；新增 `docs/current_results.md`；移除过期跨设备说明。
- 配置说明与模板：`configs/versions.yaml`、`configs/paths.example.yaml`、`configs/part2/README.md`。Middlebury、NYUv2 与关键分析的冻结配置未改动。
- 辅助入口与说明：`scripts/prepare_data.sh`（注释）、`scripts/smoke_models.py`（报错文字）、`src/README.md`、`pipelines/README.md`、`tests/README.md`、`outputs/README.md`。

以上变更包括 17 份修改、1 份新增和 1 份移除；没有把旧运行改记为本次实验。计划中的视差敏感性公式补全了 `doffs`，与既有实现一致，没有修改几何代码或重算结果。

### 剩余事项

自录视频与 mandatory clips 按用户要求继续暂缓，Step 0 视频门禁、Step 2/4/5/6 未完成。后续需由用户恢复输入并冻结 dev/test manifest，随后进行逐帧/视频比较、Part 3 消融、Demo 实测、报告和最终提交。无第二目标域结论仍限定 NYUv2；本次不新增目标数据。正式训练与微调继续只能由用户启动。

## 2026-10-28 — Step 0、Part 1A/2A 的 GitHub 分支上传说明

- 阅读工作规则、计划及 Step 0/1/3，核对公共包初始化、流水线导入、配置测试和实际小型结果清单。项目根目录 `git status` 返回“不是 git 仓库”；没有在完整项目初始化 Git 或执行删除操作。
- 新增中文 `docs/github_upload_part1a_part2a.md`，根 README 增加入口。说明包含上传/排除范围、两环境、第三方固定版本、资源 checksum、科学协议、部分结果证据的复现限制，以及先清空目标 branch 再上传的完整命令。
- 采用独立导出和独立克隆：先检查导出内容，再用删除提交清空目标 branch 的当前文件树并推送，随后提交和推送范围内文件。历史提交保留，push 显式指定唯一目标 ref；说明使用仓库/branch 占位值，由用户填写并手动执行。
- 文档内导出逻辑已在 `/tmp` 实际检查：默认 118 个文件、2,674,630 字节；附带少量图时 131 个文件、约 9.97 MiB。文件限于 Step 0 与 Part 1A/2A；没有数据、权重、第三方源码、原始预测或其他研究分支的实现/产物。
- 导出副本生成单独 README、`.gitignore` 和 `EXPORT_MANIFEST.json`。三份说明副本整理范围与链接；其他复制文件的源/目标 SHA256 一致，包括数值配置、完整精度 JSON/CSV 和自动运行元数据。没有改写项目原 Step 0/1/3、来源清单或结果文件。
- 在临时本地 bare 仓库和克隆中演练清空/上传：已跟踪隐藏文件删除、空文件树提交、118 文件完整暂存、显式 ref 推送与另一分支 SHA 保持不变均通过；没有 `160000` 嵌套仓库条目。没有连接或修改实际 GitHub 仓库。
- 检查 9 份原/导出 Markdown 的 52 个实际本地链接、38 个 Bash 代码块、40 份导出 Python AST、13 份 YAML 和 7 份 shell 入口，均通过。首轮链接检查将代码示例中生成 README 的相对链接当成正文，产生四项误报；排除围栏代码后检查通过，未为通过检查修改这些正确的链接。
- 本次仅编写说明与进行短时文件/Git 流程检查，未运行 pytest、模型 smoke、推理、训练、下载、参数搜索或全量评价。实际仓库地址、branch 名称、认证和远端推送由用户按指南处理，当前没有宣称已经清空或上传远端。

## 2026-10-28 — GitHub 认证排查与本地电脑传输包

- 用户反馈第 6 步 `git ls-remote` 返回 `Permission denied (publickey)`。按其日志和 GitHub 官方文档判断，已连接 SSH 服务，失败发生在账户公钥认证；其提供的 ED25519 主机指纹与官方公布值一致。没有将其记为工作站无法联网，也没有将只读命令失败记为已清空分支。
- 扩充上传指南：推荐下载导出包后在本地 Git Bash/macOS/Linux 终端认证并继续第 6–8 步；补充 VS Code 下载、SCP、校验/解压、本地变量设置与 HTTPS 登录。保留用户已填写的仓库与 branch，不修改其选择。
- 补充 7900 反向隧道的可选说明：本地代理监听端口与工作站入口区分、在本地启动 `ssh -R`、工作站设置 HTTP(S) 代理并改用 HTTPS URL；网络代理和仓库认证分别处理。本次没有启动隧道、登录 GitHub、实测 7900/VPN 或对实际远端执行 Git 命令。
- 按已检查的范围清单重新导出文件，准备 `outputs/github_upload/step0_part1a_part2a_20261028.tar.gz` 和 `.sha256`。包内 118 个文件，压缩包 907,496 字节，SHA256 为 `f6cae64ce27db8a2aab5724cabbaee218e0ea319ff306b8cdad8bbdf15ef065c`；115 个复制来源 SHA256 核对通过，差异仅限三份导出说明副本。原始数据、权重、科学源码、数值配置、预测及自动运行元数据未改写。
- 实际检查压缩包 SHA、解压路径和包内文件；临时 Git 索引完整暂存 118 个文件、没有 `160000` 条目，七个 shell 入口均为 `100755`。指南在上传阶段增加显式可执行位设置，便于从 Windows 上传后在其他环境运行。
- 指南 13 个 Bash 代码块、导出脚本 AST、指南/README 42 个本地链接检查通过。同步根 README 的说明入口；没有安装依赖、运行 pytest、模型、搜索、训练或评价。

## 2026-10-28 — Part 1B/2B 中文独立计划

- 新增根目录 `plan_1B2B.md`，在 README 和总 PLAN 添加入口。重新读取最新版课程 PDF 的六页内容、中文译文相关章节、Step 2/4、公共候选配置、时序函数及测试，并查看固定上游的视频推理实现；没有改动课程文件或第三方源码。
- 计划主体为 DA-V2-S 逐帧 relative 与离线 VDA-S relative 的同输入比较，米制视频、其他模型和 streaming 按需另列。视频来自真实自录 RGB，mandatory clips 来源需另行确认；四段独立 dev/test、时长、FPS 和分辨率是实施建议，不写成课程硬性要求。
- 明确共享解码/精确帧身份、实际输入张量、backward flow、双向一致性及共同有效域；补充 relative 的尺度/偏移、零值、中位数 ratio 与真实运动的限制。计划保留原生输出，以固定参考跨度比较相对时序误差，不以逐帧 alignment 掩盖闪烁，也不把无 GT 视频 proxy 称为米制准确率。
- 记录官方离线窗口/重叠/关键帧复用/补帧及预测间 scale/shift 拼接需披露，区分 GT alignment；规划固定显示、逐片汇总、效率、结构保真与失败案例。实施顺序和完成门禁区分工程检查、冻结、正式运行、重载核验及最终指定片段交付。
- 目录、入口和后续工作均明确为待实现方案，没有提供虚构的可执行脚本、填入不存在的视频或宣称视频阶段完成。完整视频推理及长时运行仍由用户启动，现有视频暂缓状态和静态结果保持不变。
- 检查新计划、总 PLAN 和 README 的 57 个本地链接、8 个既有 Bash 代码块及 Markdown 围栏，均通过；未启动 pytest、视频采集/解码、光流、模型推理、训练、下载或其他实验，也未改配置、数据、权重、预测或自动运行元数据。

## 2026-10-29 — GitHub main 克隆的贡献、差异与融合审查

- 只读审查 `github_repository/monocular-depth-video-consistency/`，固定快照为 main `2c3e26abbfe9194c139c07a916e4f90818c12ceb`。新增外置中文分析 `github_repository/Analysis.md`，根 README 添加入口；没有修改克隆、合并代码或 fetch/push。
- 按 Git 历史区分 JiongxuChen 的静态上传/视频计划，以及 Xiaofeipao 的视频工程、整合、A–F/变体、应用与 CUDA 工作。确认 `bc39453` 实际合并双方父提交，原有包冲突通过兼容导出处理；不将静态历史分数记为队友本次重跑。
- 历史导出清单的 115 份源文件 SHA 仍与当前项目匹配；克隆 44 份静态输出规范化换行后与当前结果相同。308 个 tracked 文件中，68 个与 HEAD 字节一致，240 个仅 CRLF/LF 不同，无实质未提交内容或模式变化。分析区分原字节哈希与换行规范化，不改写元数据迁就 SHA。
- 读取视频/光流/时序/位姿/应用源码及队友提交的 JSON/CSV，核算六组证据对应关系、CUDA 五轮均值/倍率和 TUM16/后续32帧的时间/ID 不重叠。说明 Tokyo32、TUM 短片及 C_cuda 的收益和限制：相对稳定性、后处理加速、同场景复核均不等于正式自录/全基准完成；双方 C 不是同一算子，也不能混用官方 SI 与自定义 edge F1。
- 识别融合前需处理的事项：课程说明版本落后、简洁双目 PFM 幅值处理、relative CLI 取倒数与固定跨度摘要不同、一次解码/VFR/共同 mask/原生中位数曲线未齐、实际输入与效率元数据、条件性 CPU 设备问题，以及外置 raw/分析脚本缺失。没有直接修补队友源码。
- 对一个 Adirondack GT PFM 做短时只读比较：scale `0.003922`，正式读取器的有效正视差中位数 `29.596343994140625 px`，队友简洁读取器 `0.11607686430215836 px`，证实后者在当前 Middlebury 数据上乘了不应再乘的幅值。未运行匹配、重新评价冻结结果或改动输入文件。
- 在克隆 cwd 使用主环境执行 `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/aiaa3201-team-analysis-mpl conda run --no-capture-output -n aiaa3201 python -m pytest -q -p no:cacheprovider`：**101 passed，14 个现有 Matplotlib/PyParsing 弃用提示，测试耗时 2.35 s**。测试覆盖代码/合成联调，未运行真实深度模型或 GPU 复验；本环境可取得官方 Depth Pro 指标源码，不重写队友历史环境中 100 passed/1 skipped 的记录。
- 收尾核对全部 308 份克隆 tracked 文件的 SHA256、size、mtime 均未改变；分析/README 的 79 个本地链接、9 个 Bash 代码块和五行 TUM 主表与完整精度源格式化结果一致，检查通过。没有安装包、下载、训练、真实视频推理、正式数组复评或改动已有数据/权重/原始预测。
- 建议按“统一当前要求/证据 → 复用视频工程并补共同协议 → 接入冻结静态 C → 按需 C＋A 受控比较 → 应用实测”推进；融合和新增实验尚未执行，完整运行继续交用户启动。

## 2026-10-29 — Part 1B/2B 专项完成度与复现审查

- 新增中文 `github_repository/Analysis_1B2B.md`，README 增加入口。只讨论逐帧 DA-V2-S 与 relative VDA-S，区分课程必做项、高质量复现建议和可选扩展；结论是视频工程基础可复用，但正式自录比较与主要结果复现链尚未完成。
- 重新对照最新版课程 PDF 的视频要求、独立视频计划、队友模型/读写/评价实现及已提交证据。说明固定相机与运动相机、自录正式输入、完整片段比较、稳定性/漂移/效率/失败分析和指定片段确认的缺项，不将建议段数、时长或额外 RGB-D 数据写成课程硬性要求。
- 核实 Tokyo32 的固定跨度摘要与现有评分 CLI 使用不同计算方法；完整精度摘要对应 VDA 误差降低 `33.62775724618706%`，只代表该短片上的稳定性诊断。原评分生成脚本、完整旧 run 和逐对记录未随克隆交付，不能据此宣称已独立复评历史分数；新文档列出具体材料请求和拿不到时的独立新实验路径。
- 确认根项目已有两种官方源码、relative 权重、RAFT 权重和 Tokyo 示例；源码版本与既有权重 SHA 记录可对应队友记录。本次仅重新哈希 2,251,450 字节的 Tokyo 视频，SHA256 为 `097f16c33dd8c8d1d2a24d9ea31a90b76bd0ee324b958a47385183e3547a63a8`，没有重算大权重哈希或重新测试 CUDA。
- 给出根项目分阶段接入方案：独立 adapter、共享帧身份、原生 relative 保存、共同 flow/mask、固定跨度配对评价、Tokyo 工程闭环、正式自录实验与重载核验。待开发路径明确标注，现有只读预检命令与尚不存在的视频入口分开，不提供虚构的完整运行命令，也不整段执行队友混合任务脚本。
- 检查三份 Markdown 的 55 个本地链接、围栏、新文档两个 Bash 代码块及内嵌 Python 语法；主表四个数值和误差降幅与源 JSON 一致，两条权重 SHA 与现有清单一致。克隆相对 HEAD 的忽略行尾差异检查为空；未修改克隆、移植代码、安装依赖、下载、运行推理/光流/训练或重新评价数组，没有重复执行上一轮测试。

本次完成专项审查和复现路线，根项目视频流水线仍待实现；正式视频长运行继续由用户启动，不将说明文档的完成记为 Part 1B/2B 实验完成。

## 2026-10-29 — 根项目 Part 1B/2B 视频流水线实现

本次按用户授权独立实现视频主线，先使用已有公开示例，后续接入自录素材；不开展其他研究任务，不依赖队友尚缺的原 run/评分脚本，也不将本项目新分数记为队友旧结果复评。

- 新增 `src/video/` 的帧准备、公共读写/来源、冻结与运行、双向光流、relative配对评价、报告和分组汇总。ffprobe扫描真实PTS后建立共同RGB缓存，保留VFR、旋转、源帧号、时间戳、SHA、来源/session/ROI；两模型只读相同缓存。原视频不改写，缓存/协议/run/评价拒绝覆盖，失败保存实际错误。
- 新增独立DA-V2-S和离线VDA-S adapter，使用本地固定版本/权重、相同官方预处理与实际网络尺寸。DA张量显式放到请求设备，VDA保留官方32/10/22窗口、关键帧、补帧、8帧插值和预测间scale/shift，最终只保存T个有效输出。原生float32为relative/inverse_depth_like/arbitrary，零值保留，不取倒数、不逐帧对齐或伪称米制。
- 主环境实际为Python3.10.22、torch2.1.1+cu121、torchvision0.16.1+cu121、NumPy1.24.0、OpenCV4.11.0。CUDA可用性为false，`nvidia-smi`驱动通信失败；默认GPU预检按设计返回2，真实记录见 `outputs/video_engineering/preflight_gpu_v1.json`。没有将历史CUDA环境记为本次环境或静默回退。xFormers小CPU张量探针实际失败后，项目adapter为显式CPU请求绑定官方PyTorch参考attention并关闭官方motion对象的xFormers开关，记录实例数量；第三方文件、模型参数和GPU路径不改。
- 主协议为 `configs/video/relative_video.yaml`，公开输入登记在 `configs/video/public_clips.yaml`；两个Part配置指向同一协议。确认旧全局视频段仅由配置测试引用后，更新其relative主指标索引，正深度log/AbsRel仅列辅助诊断；NYUv2与Middlebury的数值协议不改。冻结JSON保存配置、资源/输入/源码SHA、原字节源码base64快照和内容校验，科学内容改变需新版本。
- 新增 `pipelines/video/cli.py` 和 `scripts/video/run.sh`，支持check/prepare/freeze/infer/flow/evaluate/verify/summarize；两个Part另有薄Python/shell入口。共同backward warp、FB一致和双方有限插值支持形成实际保存的mask；固定第0帧P5/P95跨度归一化MAE/RMSE，不改变raw。主平均全部非空pair，另列像素加权；低覆盖仅质量提示，空mask/零跨度/不兼容median ratio记null及原因。固定背景诊断要求fixed及预声明ROI，来源/split/相机分组等权汇总并拒绝重复源及session dev/test泄漏。
- 报告保存七联曲线、原生固定0–50和参考帧固定0–1的两种同步视频、flow/mask差异视频、真实最差帧对、饱和比例、窗口实际混合区间与中文分析。原生中位数、固定跨度变化和ratio分别保存；无GT时不声称准确率或真实尺度漂移。效率保存预热、整片重复计时、首个测量pass预测、参数/权重大小、实际尺寸、GPU峰值及cached-RGB流程总耗时；离线摊销不称在线延迟。

实际短运行与验证：

1. 独立重新计算DA-V2/VDA/RAFT三份权重SHA，固定commit和原视频SHA检查通过；现有资源直接复用，没有下载、安装依赖或网络代理操作。公开Tokyo原片保持 `097f16c33dd8c8d1d2a24d9ea31a90b76bd0ee324b958a47385183e3547a63a8`；另核对rollercoaster来源/SHA并查看真实帧，登记为moving/public/development。
2. `prepare --public-clip tokyo_walk --max-frames 33 --max-long-edge 160` 创建 `data/video_frames/engineering_tokyo33_v1/`；随后冻结CPU/FP32、两线程、input_size98、零预热/一次计时、显式Farneback。两模型真实本地权重推理分别约0.9653/1.6902秒，实际网络均98×168，评价画布90×160，VDA两个窗口/21个尾补帧，最终各33份raw。工程输出在 `outputs/video_engineering/cpu_tokyo33_v1/`，不代表完整公开片或默认GPU性能。
3. 共同flow32对、评分32对/逐帧33行、mask、三种MP4、PNG和中文报告全部生成。主MAE为DA-V2 `0.03035770617969271`、VDA `0.009776797626062192`；平均覆盖率 `0.9732421874999999`，无空/低覆盖。背景项因moving/无ROI正确不适用。这些只作为CPU小尺寸工程结果，不与队友Tokyo32或正式GPU协议混合排名。
4. 协议、两模型run、评价及单片汇总均通过。`verify --evaluation`核对216个关联文件，从保存raw/flow重新warp并重算逐对/逐帧、mask、参考、背景与汇总；实际执行模型为false，数值容差rtol1e-6/atol1e-8，原文件SHA不变。视频导出后实际重解码检查帧数/尺寸，人工查看生成曲线。
5. 另用真实RAFT-Small本地权重对两个不同源帧做CPU、两次更新的双向接口检查，128×160补边网格恢复90×160，有限输出通过；真实脚本、两个flow数组、源码/输入/权重校验及环境记录在 `outputs/video_engineering/raft_cpu_pair_v1/`。不是默认12次更新全片或GPU验证。
6. 新增152项视频测试，覆盖真实CFR/VFR/旋转编码、共同官方变换、CPU参考attention、官方拼接加合成forward的1/22/31/32/33/55帧、位移/padding、首pass副本保护、协议/权重/源码/raw篡改、掩码/空值/固定跨度性质、真实闪烁不被抹除、ROI、中位数、重载计算、MP4与clip分组。集成测试发现首pass数组复用问题后改为独立副本；非法CPU FP16改为冻结前拒绝，对应测试同步，未改变科学评分以追求分数。
7. `MPLCONFIGDIR=/tmp/aiaa3201-video-tests conda run --no-capture-output -n aiaa3201 python -m pytest -q`：**1,383 passed，15 warnings，31.68秒**；14项为既有Matplotlib/PyParsing弃用，1项为NVML不可用提示。随后仅更新全局视频配置索引，相关7项配置测试通过，不重写为第二次完整测试。未运行训练、完整公开片预测或全量视频实验。

中文README、PLAN、独立视频计划、Step0/2/4、文档/源码/流水线/配置/测试/输出索引已同步；专项材料审查保留历史判断并增加当前实现入口。Step4提供从预检到双公开视频汇总的完整命令，Step2说明自录session/ROI接入。默认518/FP16/RAFT全片、真实fixed/moving自录、人工失败分析、mandatory clips及最终课程交付仍未完成，由用户后续启动/补齐。现有Part1A/2A、其他研究源码与结果、第三方仓库及GitHub克隆没有修改。

## 2026-10-29 — Part 1B独立说明与完成状态核对

- 对照中文版§5.1，并从当前PDF重新提取原文，核对逐帧相对预测、benchmark限定对齐、固定相机flicker、运动相机scale drift和逐帧变化展示要求。
- 重写 `docs/step2.md`，仅说明DA-V2-S的输入、独立预测、原生保存、文件核验、实际33帧记录、自录接入与验收；移除其他视频模型、比较流程和Step4链接。提供现有真实入口能执行的两个公开片段准备/冻结/DA-V2/文件核验命令，不虚构独立分析命令。
- 源码核对发现现有时序评价/绘图入口为成对评价，尚不能只读取DA-V2运行结果形成完整分析。因此明确列出单模型分析/导出尚缺，及固定相机素材、完整运动片段曲线、效率和失败分析待完成；不将预测文件核验等同科学分析完成。README与PLAN相应状态同步。
- 本次仅阅读已有33帧run及核验记录，并修改说明；未重新验证原始数组、执行推理/光流/评价/训练、下载或运行pytest。历史测试与工程运行记录保留，不改写已有预测、协议或自动元数据。
- 收尾检查四份Markdown的70个本地链接、Step2两个Bash代码块及围栏通过；确认Step2不含其他视频模型或Step4引用，33帧与98×168等字段对应已有DA-V2运行记录。

## 2026-10-29 — Part 1B独立分析、导出与自录操作补齐

- 新增 `src/video/framewise.py`、`framewise_report.py`、`framewise_roi.py`，及 `pipelines/part1/framewise_video.py`、`scripts/part1/video.sh`。只读取一份DA-V2原生预测和RGB flow，提供独立分析/图表视频/原始重算、背景ROI派生及按片段汇总；没有复制预测来伪造第二个模型，也不要求其他深度模型的运行产物。
- 新增 `configs/part1/video_framewise_standalone.yaml`，仅登记DA-V2-S与评价用RAFT。默认10 FPS、RGB长边960、input_size518/FP16、一次预热/三次整片计时、RAFT12次更新；原通用配置/代码与已保存协议保持不变。两个预检实际核对DA权重，GPU条件另核对RAFT权重SHA；实际CUDA仍为false，GPU预检返回2，不启动推理或静默回退。
- 主误差保持全图backward warp、FB一致与有限插值支持，参考帧全图P5/P95跨度整段固定；全部非空pair主均值、像素加权另列，低覆盖仅标记。背景ROI只用于fixed背景诊断，不缩小全图主误差或固定空间中位数的取样域；记录raw中位数、固定跨度变化、可用时ratio/log proxy、背景相邻变化和总体标准差ddof=0。无GT不声称准确率或真实米制漂移，零值/零跨度/空mask等按实际记录。
- ROI入口支持多个包含矩形、排除矩形及PNG mask，从已有缓存复制原RGB字节并派生新manifest/二值mask/预览，原缓存不改、不重新解码；预览同时画出排除孔洞边界。单模型汇总按来源/split/相机分组等权，不把N/A填零，拒绝重复原片、session泄漏及同组混用科学协议。
- 独立报告保存六联曲线PNG/PDF、两种固定色标RGB/深度视频、时序差异/mask视频、背景标准差图/数组、较高误差帧对、饱和比例及中文说明。verify核验来源/源码原字节快照、文件SHA，再从raw/flow重算CSV、mask、背景和汇总，亦核对header、模型/效率元数据与显示来源绑定，不执行模型。
- 新增 `tests/test_framewise_video.py`与`tests/test_framewise_video_cli.py` 共42项，使用合成小数据验证已知平移方向、固定正尺度/偏移不变性、闪烁不被对齐抹除、合法零值/有限插值支持、背景std真值、ROI与主指标分离、空值、固定空间中位数、低覆盖不剔除、clip等权/N/A、ROI原件保护/孔洞预览、MP4重解码、来源/重封装CSV/header篡改与单模型CLI。没有加载真实深度或光流模型。
- 最终 `MPLCONFIGDIR=/tmp/aiaa3201-part1b-tests conda run --no-capture-output -n aiaa3201 python -m pytest -q`：**1,425 passed，15 warnings，32.37秒**。14项为已有Matplotlib/PyParsing弃用，1项为NVML不可用提示；没有启动正式训练、完整公开片预测、分割或其他长实验。

实际派生验证：

- 复用 `outputs/video_engineering/cpu_tokyo33_v1/dav2/` 与已有Farneback flow，仅重新分析33个真实原生预测，不重新预测或估计flow。集成阶段的首次派生 `outputs/part1/video/standalone_engineering_v1/tokyo33/analysis/` 保留当时源码快照；最终代码的活跃工程证据为同级 `analysis_final_v1/`，没有覆盖原记录。
- 最终单模型MAE `0.03035770617969271`、RMSE `0.04044592770550371`，全精度来自该次summary；平均coverage `0.9732421874999999`，32对均有效，无低覆盖。134份原始来源文件与45项分析/报告文件关联，三种MP4均实际重解码；人工查看曲线。moving/无ROI的背景项正确不适用，不伪造fixed实测。
- `scripts/part1/video.sh verify --analysis .../analysis_final_v1 --output .../verify_analysis_final_v1.json` 实际通过，33帧/32对从raw重算、mask/背景重算、`models_executed:false`；再生成 `standalone_engineering_v1/summary_final_v1/`。这是160×90输入的工程核验，不是960画布、GPU或自录正式结果。

重写中文Step2，保持只讲Part1B：给出完整公开与自录运行链、先看RGB再选ROI的操作、输出解释与未完成门禁；解释 `data/video_frames/engineering_tokyo33_v1` 是160×90的33帧CPU缓存，原片1280×720未改，正式输入应从原片重提，不将低清图放大冒充新输入。自录建议每段10–15秒、1080p/30 FPS/横屏、普通1×镜头，近中远层次/室内光照、fixed与moving拍法和独立session划分；距离/段数只是建议，不是课程规定、GT或模型有效深度范围。

README、PLAN、独立视频计划、文档/源码/流水线/配置/测试/输出索引同步。没有修改第三方、原始数据/权重/预测/自动运行元数据或其他研究实现。完整公开实验与真实fixed/moving自录、人工失败分析仍由用户后续运行/补充；实现门禁通过不等于课程正式实验全部完成。

收尾核对12份说明的140个本地链接、17个Bash代码块（Step2四块）、六份新增Python语法和新shell入口，通过；Step2不含其他视频模型/Step4引用，Step4仅修复指向新自录章节的链接。两个上游仓库tracked状态均为空。本次未修改旧科学源码、原片或已有预测；默认960输入的全片结果尚未生成。

## 2026-10-29 — Part 1B完整公开结果核验、分析与说明整理

用户已执行Step 2公开实验的全部指令。本次读取`outputs/part1/video/public_v1/`及`data/video_frames/part1b_public_v1/`，没有启动新的深度预测、光流估计、训练或下载。

- Tokyo Walk：原片1280×720/15 FPS/195帧，13秒；按目标10 FPS选130帧，缓存960×540，129个相邻对。DAVIS rollercoaster：原片960×540/24 FPS/70帧，2.916667秒；选29帧，28对。均为`public/development/moving`，不是自录或未观察的test。
- 两片段实际记录均为`aiaa3201`、NVIDIA RTX A6000、DA-V2-S/FP16、网络输入518×924、RAFT-Small/FP32/12次更新；一次预热、三次完整序列计时。资源commit/SHA、实际形状、原生relative表示、alignment:none、参考帧P5/P95、全图主误差与背景ROI分离均保持原协议。
- 从各片段`analysis/summary.json`读取完整精度：Tokyo逐对平均MAE/S=`0.0064841222907683115`、RMSE/S=`0.012907316466668475`、平均coverage=`0.9556336431715954`；rollercoaster为`0.026655437701475177`、`0.0748300369612884`、`0.5794155781525572`。片段等权均值为`0.016569779996121744`、`0.043868676713978434`，并非157对混合平均。
- 模型FPS来自原运行记录，分别为`42.3985651259517`、`42.024389619791435`；均摊延迟`23.585703832885393`、`23.795705518803032` ms/帧；不含flow/评分，不称完整系统FPS。完整计时范围与峰值显存说明写入Step 2。
- 独立只读复核保存为`outputs/part1/video/public_v1/review_v1/review.py`与`outputs/part1/video/public_v1/review_v1/review.json`。实际命令为`env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/aiaa3201-part1b-review conda run --no-capture-output -n aiaa3201 python outputs/part1/video/public_v1/review_v1/review.py`：协议/资源、两片段DA-V2原生输出与运行记录、两份单模型分析均通过；159帧/157对的CSV、mask及汇总从raw/flow重算，容差`rtol=1e-6, atol=1e-8`，`models_executed:false`。
- 六个MP4逐帧解码、帧数/尺寸检查通过：两种深度视频各130/29帧，差异视频129/28帧；现有分组汇总和CSV SHA核对通过。复核前后675份既有公开运行文件SHA一致，源码/协议/原生预测/自动运行元数据未覆盖。
- 扫描原生数组确认rollercoaster每帧零值占68.0284%–76.0066%，29帧全图中位数均为0；ratio及log变化不可评价（0/29），并非零漂移。曲线前两栏为零，ratio栏留空；CSV/JSON的N/A保持真实。分组log指标仅有Tokyo一个可评价片段。moving且无ROI，两个背景闪烁诊断均不适用。
- 实际查看两段起中末对照、曲线及高变化案例图。Step 2记录Tokyo人物/建筑轮廓与反光地面变化、rollercoaster交叉细杆与遮挡困难；最低coverage分别90.2153%、37.0774%。区分预测变化、真实运动/构图与flow误差，不将无GT稳定性诊断写成准确率或真实米制漂移。

`docs/step2.md`按状态与范围、公开片段协议/完整命令/结果/分析/产物核验、自录待完成部分重新组织。保留自录10–15秒/1080p建议、近中远层次/光照与固定/移动拍法，以及准备、预先选ROI、冻结、预测、flow、分析、verify、summarize的完整命令；原有自录接入锚点继续可用。

同步README、PLAN、plan_1B2B、docs索引、configs/part1索引、pipelines索引、outputs索引和data说明。当前说明移除Tokyo33低清工程结果与“公开全片未运行”的过期状态，统一指向当前公开产物。检查后，旧`engineering_tokyo33_v1`缓存仍被9份历史JSON/探针脚本直接引用，因此保留其输入、原生预测及自动元数据；旧执行历史与Part 2B独立工程证据未改写。共同双模型640画布的配置/入口仍有用途，保留并明确它不是当前单模型960结果的运行协议，不直接混算。

本次未修改科学实现、第三方仓库、权重或Part 3。九份相关说明的116个本地链接（含锚点）、14个Bash代码块（Step 2四块）与Step 2主表数字一致性检查通过；加入本记录后最终复核10份说明、120个本地链接、15个Bash代码块全部通过，复核脚本及结果来源SHA仍一致。未为说明文字修改重复运行全项目测试。公开实验门禁通过，自录fixed/moving及后续课程交付仍未完成。

## 2026-10-29 — Step 2精简与课程适用范围说明

重新阅读课程PDF原文：§5.1要求逐帧relative预测、固定/运动镜头时序诊断，未限定视频来源；尺度对齐仅在benchmark评估按规定需要时使用。结合后续自录短视频与相同帧比较要求，正式素材采用自录固定/运动视频；Tokyo Walk和rollercoaster是额外公开开发验证，不是指定必做的两段素材。普通自录RGB无深度GT，保持alignment:none，不逐帧调尺度。

按用户要求将Step 2由328行精简到255行：删除公开结果的逐对案例表、冗长曲线/效率/显示分析及重复产物表，保留主要结果、适用性限制、原始来源链接；完整公开命令折叠保留。自录拍摄、背景ROI、接入运行及完成门禁正文与命令原文保留，补充自动与人工工作的界限：程序生成预测、指标、图视频和核验，用户仍需真实拍摄/登记、按RGB选ROI，并查看和解释结果。

README同步上述范围；本次仅修改三份Markdown，没有修改代码、协议、数据、预测或自动运行元数据，没有运行模型、光流、训练、下载或重新评估。

核对三份说明的56个本地链接及13个Bash代码块通过；Step 2四个命令块与精简前逐字一致，自录拍摄至完成门禁正文保持一致，主要结果仍与原summary全精度数值格式化后一致。

## 2026-10-29 — Part 1B：NYUv2相对深度尺度对齐与自录诊断

按用户确定的范围，Step 2只组织两个分支：DA-V2-S在NYUv2上进行GT辅助的相对深度对齐基准；自录固定/运动视频进行原生relative预测与时序诊断。删除当前说明中的两段公开示例命令、指标表与分析，已有原始视频/RGB缓存、预测与自动元数据及旧执行历史保留真实，不为整理说明而删除。通用配对入口和public_clips登记仍被Part 2B使用，保留，不修改第三方仓库。

新增实现：

- `configs/part1/dav2_nyuv2.yaml`、`scripts/part1/run_dav2_nyuv2.sh`、`pipelines/part1/dav2_nyuv2.py`提供check/predict/evaluate/verify/run、显式设备/精度、固定样本选择及预测恢复。主环境aiaa3201，默认CUDA/FP16、input_size518、一次预热、每图三次计时，保留第一次原生预测。标准654张test和部分smoke身份分别记录，禁止覆盖既有run/evaluation。
- `src/metrics/relative_alignment.py`实现逆深度域每图`a*r+b≈1/GT`的非负尺度最小二乘。拟合与评分统一使用`[45:471,41:601]`内有限且0.001<GT<10的像素。原生零值合法；退化/非正斜率采用约束下常数解并记录，继续保留该图；拟合逆深度裁剪到[0.1,1000]后取倒数，不因非正拟合值删除有效GT像素。raw不改写，结果明确为aligned_relative、GT used，不宣称native metric或论文官方表格复现。
- `src/evaluation/relative_nyuv2.py`复用NYUv2 loader、官方DA-V2 relative adapter和七项depth metrics，独立保存raw、执行时源码/输入/权重SHA、命令/环境与效率、GT辅助对齐数组/mask、CSV、宏平均及10,000次图像bootstrap CI、固定样本与AbsRel最高案例。原生推理只将RGB交模型；GT仅用于独立评价，不训练模型。verify从raw/GT重新拟合并核对对齐数组、mask、CSV、宏平均与CI，绑定源文件/科学源码，记录models_executed:false。
- 新增两份测试共24项，覆盖已知仿射真值与独立lstsq、固定正仿射变化、crop/有效GT、零值、常数/负斜率、非正逆深度、有效像素完整保留、完整合成生命周期、真实/对齐身份、损坏与禁止覆盖、恢复条件及CLI。开发检查发现CSV中的布尔值不能按浮点读取，已修正并通过最终检查。合成测试不加载真实模型。

实际执行与验证：

1. `./scripts/part1/run_dav2_nyuv2.sh check --output outputs/part1/dav2_nyuv2/checks_v1/resources_gpu.json`：数据/权重SHA、仓库commit与654张split检查后，实际设备检查返回not_ready/退出2，原因为当前工具进程CUDA不可用；没有静默改CPU或启动GPU运行。
2. 显式执行`env OPENBLAS_NUM_THREADS=1 ./scripts/part1/run_dav2_nyuv2.sh run --device cpu --precision float32 --positions 0 326 653 --output-dir outputs/part1/dav2_nyuv2/smoke_cpu_v1`。真实NYUv2三图与已有relative Small权重完成预测、对齐评价、案例图、原生重算；网络实际518×686，输出480×640，run/status complete、reload_validation verified、subset_smoke_only。检查实际contact sheet，GT辅助与非native身份显示正确。
3. 三图仅用于工程验证：无退化拟合、无逆深度裁剪；完整精度小样本宏平均AbsRel=`0.04054090078097255`、RMSE=`0.20255161078571812`、δ1=`0.9763288061703554`。这些不是654张或GPU结果，不进入正式主表、不代表泛化结论。来源为`smoke_cpu_v1/evaluation/metrics_summary.json`，原始重算容差rtol=1e-6/atol=1e-8。
4. 相关92项测试通过后，最终`env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/aiaa3201-relative-tests OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n aiaa3201 python -m pytest -q`为 **1,449 passed、15 warnings、34.18秒**；14项已有Matplotlib/PyParsing弃用、1项NVML不可用提示。真实权重短测另有已有TypedStorage弃用提示。没有训练、微调、下载、完整654张推理或自录实验。

Step 2给出两分支的协议、架构、用户完整运行/恢复/重算命令及完成门禁，保留自录拍摄、固定ROI和全流程。README、PLAN、plan_1B2B、docs/配置/源码/流水线/测试/数据/输出索引同步；共享candidate配置的relative分支改为指向实际Part 1B配置，Part 2A的native metric协议和既有科学源码不变。654张正式结果及真实fixed/moving仍待用户启动，不预先标记Part 1B完成。

收尾另执行独立`checks_v1/independent_audit.py`，用通用`numpy.linalg.lstsq`和显式七指标公式重算真实三图，系数、有效像素与CSV/宏平均通过rtol=1e-6/atol=1e-8，models_executed:false，结果与脚本SHA保存在`independent_audit.json`。共享candidate配置调整后，7项配置测试通过。13份说明的134个本地链接与28个Bash代码块（Step 2五块）检查通过；对照修改前文档确认自录拍摄、ROI、命令与分组说明原文保留，Step 2无公开示例章节/路径。没有删除已有原始数据、预测或自动元数据。

## 2026-10-29 — Part 1B完整NYUv2结果分析与授权清理

用户已运行Step 2的完整654张预测、评价、重算和独立重评。本次只读检查正式`outputs/part1/dav2_nyuv2/frozen_v1/`，未执行新模型预测、训练、下载或全套测试，没有改动科学实现、配置、第三方源码或Part 2A/3结果。

- 正式run为complete/full_654，positions为0–653，654份未对齐raw_relative、主评价654份aligned/mask、独立重评654份aligned/mask均完整。标准split、crop、有效GT范围、relative权重及SHA绑定通过；每图238,560个有效像素，共156,018,240个，215个scene。
- 实际用户运行记录为aiaa3201、NVIDIA RTX A6000、FP16、input_size518/网络518×686，输出480×640；一次预热、每图三次计时，保留第一次预测。模型均摊18.64646556 ms/图，FPS=53.62946649，PyTorch峰值allocated/reserved为330,198,016/341,835,776 bytes；模型阶段范围与全部I/O/对齐/绘图区分，未用旧CPU检查耗时替代。
- 从正式summary的完整精度读取：AbsRel=`0.061714078193980906`、SqRel=`0.03791698783859191`、RMSE=`0.2973561231848489`、RMSE-log=`0.09352265717944204`、δ1=`0.9554034579546596`、δ2=`0.9878371464772324`、δ3=`0.9957887616217179`。Step 2表按8位小数显示并列图像级bootstrap CI。两份summary的七指标/CI/协议/效率一致，逐图CSV逐字一致。
- 1张退化：test位置266/bathroom_0008/MAT650的无约束尺度−0.04683307366095048，非负约束解a=0、b=0.5356401276710163，对齐后常数深度约1.86692510 m；不是模型raw本身常数，亦非正常结构恢复。AbsRel=0.287136031044823，未丢弃。26张图合计249,856个拟合逆深度低于0.1的像素按10 m处理，占0.160145%；无非正逆深度/超过1000的值，像素均保留评分。
- 真实查看test 266、310、447的五联图：记录负尺度约束退化、浴室镜面区域局部差异、卧室背景深度压缩。主contact保留固定0/326/653及最高AbsRel六图共9图。补充16类场景统计与误差分布，注明类数量不均、图像级CI和GT辅助对齐条件，不与native metric做无GT公平排名。

实际新核验与派生产物：

1. 两次执行`run_dav2_nyuv2.sh verify --evaluation-dir ...`，分别指向正式`evaluation`和`evaluation_recheck_v1`，记录为`analysis_v1/verify_original.json`与`verify_recheck.json`。两套654张从raw/GT重算对齐、数组、mask、CSV、宏平均与CI通过，rtol=1e-6/atol=1e-8，models_executed:false；未覆盖用户原核验记录。
2. 新增`analysis_v1/review.py`只读分析脚本，生成`results_analysis.json`、`metrics_by_scene_type.csv`与`formal_file_hashes.json`。对0/326/653、266/310/447和裁剪较多的45共7图，用独立lstsq与显式七指标公式复核通过；全654张主表重新聚合通过。分类/分布只是既有结果分析，不是新预测或选参。
3. 当前正式输出只保留`frozen_v1/`、`checks_user_v1/`、`analysis_v1/`；完整重评保留在正式run内，未删除其绑定数组或改写自动元数据。

按本次用户明确要求清理，删除以下8个废弃目录：

- `outputs/part1/dav2_nyuv2/smoke_cpu_v1/`
- `outputs/part1/dav2_nyuv2/smoke_gpu_v1/`
- `outputs/part1/dav2_nyuv2/checks_v1/`
- `outputs/part1/video/public_v1/`
- `outputs/part1/video/standalone_engineering_v1/`
- `outputs/video_engineering/`
- `data/video_frames/engineering_tokyo33_v1/`
- `data/video_frames/part1b_public_v1/`

共1,216个文件、1,960,217,802 bytes，删除前逐文件路径/大小/SHA记录在[cleanup_manifest.json](outputs/part1/dav2_nyuv2/analysis_v1/cleanup_manifest.json)，实际操作脚本为`analysis_v1/cleanup.py`。删除前后2,002份正式run及用户检查文件SHA完全相同。此次授权仅用于这些废弃烟测、视频缓存/派生及工程检查；NYUv2原数据、split、权重、正式预测、来源快照、独立重评和其他Part结果未改动。第三方仓库自带视频与供Part 2B使用的public_clips登记保留。自录所需实现和维护用回归测试保留。

Step 2去掉烟测命令/状态，保留新目录完整复现、恢复、重评与当前只读核验指令；补充正式结果、CI、行为/案例/场景分析、效率和完成门禁。自录拍摄、ROI与运行段落原文保留，Part 1B整体仍待自录。README、PLAN、plan_1B2B、current_results及相关索引同步；Step 4仅清理指向已删除工程文件的说明和过期复用指引，不改Part 2B科学实现。历史执行条目的事实保留，失效本地链接改为路径文字，不将旧运行改写成未发生的新记录。

收尾核对15份相关说明的154个本地链接（含锚点）、30个Bash代码块，全部通过；确认Step 2无烟测或公开示例流程，自录完整段落逐字保留，主表数值与全精度summary格式化一致。再次核对2,002份正式文件SHA未变，清理清单中的8个目录均不存在；没有为文档整理重复跑模型或全套测试。

## 2026-10-30 — 六段自录视频的Part 1B适用性检查

重新核对课程PDF §5.1/§5.2及Step 2协议，检查`data/self_recorded/`六段实际输入。执行`conda run --no-capture-output -n aiaa3201 python outputs/part1/video/self_recorded_review_v1/review.py`，使用ffprobe读取流/容器信息与逐帧PTS，以OpenCV完整解码，并派生每片9帧缩略图和首/中/尾全尺寸JPEG。实际查看六张缩略图及人物活动、横向运动、前进片段的中间全尺寸图；没有将抽样查看写成完整逐帧人工观看。

- 四段固定视频分别366/415/388/405帧，两段运动分别410/439帧，共2,423帧；视频流时长依次12.20/13.83/12.93/13.50/14.14/15.14秒。解码帧数与逐帧PTS条数相同，全部时间戳严格递增；源文件检查前后SHA256相同。
- 用户报告手机设置1080p/30 FPS；实际六文件均1280×720/HEVC，固定片段30 FPS，两段运动约29 FPS。视频流码率约0.15–0.51 Mbps，容器带`wxMeta`标记，部分细节发软，提示可能经历传输/导出处理；不能确认具体压缩链路，不将手机设置写成实际文件属性。720p和约29 FPS仍可接入既定10 FPS/长边960协议，正式实验优先保留并使用手机原片，通常无需为场景内容重拍。
- 四段固定片段约2 FPS抽样，以第一帧为参考进行前后向LK特征检查及RANSAC部分仿射拟合，640×360检查尺寸下估计的图像中心最大位移依次为0.3512/0.1931/0.2367/0.1012 px，支持相机稳定。这里只是RGB背景主导的近似检查，不是相机位姿GT，不用于重映射正式视频，也不是深度闪烁结果。
- 场景具有桌椅、沙发、货架、物体遮挡与近远层次，覆盖固定静态、固定有人活动、横向扫视、向前移动，可用于Part 1B。第二固定场景有人取走桌上物品，该处不作为稳定背景；固定ROI须避开整段人物活动路径。横向视野更换和前进导致真实距离变化，后续中位数变化只能作尺度proxy，不能全部叫预测漂移。没有从画面估计实测距离或假设纯平移/精确相机轨迹。

本次新增仅为`outputs/part1/video/self_recorded_review_v1/`的素材检查脚本、结构化记录及派生JPEG；没有运行深度模型、RAFT、训练、下载或生成正式实验成绩，没有修改源视频、既有预测、自动运行元数据或科学实现。真实session、输入版本、划分和背景ROI尚未冻结，Part 1B视频门禁保持未完成。

更新中文`data/self_recorded/README.md`列出六文件及适用性，README、PLAN、Step 2和本文件同步素材到位状态；Step 2接入命令仍明确是四个示例文件名，不能直接当作当前六文件的实际运行记录。目录说明删除过时的六类/15–30秒/20个测距点要求，未改变课程PDF、既定评价协议或历史记录。

收尾重新读取六视频校验SHA、解码数/PTS条数与递增性，全部通过；五份更新说明的82个本地文件链接和15个Bash代码块语法检查通过。未重复运行全项目测试。

## 2026-10-30 — 1080p自录输入核验与六段正式运行说明

用户将六段输入更新为1920×1080。本次未改动源视频，重新完整解码并检查SHA、PTS和实际流属性，结果独立保存到`outputs/part1/video/self_recorded_review_v2/`；旧`review_v1/`元数据和历史条目保持原执行含义，不用于当前输入核验。

- 执行`conda run --no-capture-output -n aiaa3201 python outputs/part1/video/self_recorded_review_v2/review.py`，六段均为H.264/1920×1080/名义30 FPS，固定片段实际平均30 FPS，两段运动约29.94 FPS。源帧数依次366/415/388/405/423/453，共2,450帧，解码数与PTS条数相同，时间戳均严格递增，源SHA在检查前后及文档收尾校验时一致。
- 按现有`_probe_video/_selection_indices`规则只计算选帧计划，没有导出正式RGB缓存。目标10 FPS时依次122/139/130/135/141/151帧，共818帧/812帧对；计划保存为`selection_plan.json`。缓存尺寸仍为960×540，不将1080p源尺寸误写成模型实际输入；既定518×924网络尺寸、米制/relative区分和alignment:none不变。
- 执行`./scripts/part1/video.sh check --output outputs/part1/video/self_recorded_review_v2/resource_check.json`，DA-V2仓库commit/未修改状态、DA-V2与RAFT权重SHA通过，实际CUDA可用、设备RTX A6000，ready_for_requested_device=true。只检查资源，没有加载预测模型或计算RAFT。
- 为固定片段检查全帧RGB和约2 FPS绿色ROI预览。人物活动片段在约4秒/8秒会遮挡立柱候选区域，因此没有将该候选冻结为稳定背景。`roi_rgb_review.json`及候选图保留实际检查结果；`roi_rgb_map.py`只查看输入变化，不做深度评价。最终背景主诊断使用两段无人fixed：场景1矩形`160 170 800 480`、场景2`100 180 780 500`，均在960×540坐标内；全帧RGB检查及预览保存在`declared_roi_rgb_review.json`与对应contact图，没有按深度分数挑ROI。两段人物片保留全图时序/遮挡补充，整段固定背景指标N/A；两段moving无固定ROI，不将N/A补零。

本次采用固定算法和协议，六段统一test用于评价，不人为构造开发集或宣称盲测。真实session未提供，`intake_provenance.json`将`session_unconfirmed_batch1`明确解释为保守合并的待确认批次，不冒充真实独立拍摄session；手机型号/镜头/防抖/曝光等未知字段保留null。后续来源信息在prepare前补充；不以不同文件名制造独立性，也不将同一休息区的视角差异作为多场景泛化证据。

Step 2的接入段已改为六段实际文件的完整第1–8步：变量和资源检查、全片prepare、两段roi、逐片freeze、DA-V2逐帧预测、RAFT双向光流、分析与三类核验、六段汇总。明确主要耗时步骤、输出结构、N/A适用性、结束后的查看重点和失败/中断时保留记录、另建阶段目录的做法；NYUv2既有正式结果不重跑。正式缓存位置`data/video_frames/part1b_self_v1/`，正式run位置`outputs/part1/video/self_recorded_v1/`，本次两处均未创建。README、PLAN和中文自录目录说明同步当前1080p来源、ROI与运行状态；既有科学源码、配置、权重和第三方仓库没有修改。

实际验证：五份说明的本地链接与Bash语法检查通过；将8个正式运行代码块置于临时目录，仅替换入口为参数记录器，完整展开52次CLI调用（check 1、prepare 6、roi 2、freeze 6、predict 6、flow 6、analyze 6、verify 18、summarize 1），逐一通过真实argparse解析，并核对每片输入、相机类别、session、来源说明、冻结所选source/with_roi路径及ROI坐标。此项是命令构造检查，不冒称真实执行了52次实验；结果见`documentation_validation.json`。未运行正式预测、光流、训练、下载或全项目测试，Part 1B自录结果和案例解释仍待用户运行后完成。

## 2026-10-30 — Part 1B六段自录完整结果分析

用户已完成Step 2第1–8步，正式输出为`outputs/part1/video/self_recorded_v1/`。本次只读分析已有预测、光流、mask、JSON/CSV与图视频，没有运行新深度模型、RAFT、训练、下载或修改科学源码、配置、原数据和自动元数据；没有将本次分析写成新的用户推理记录。

执行`conda run --no-capture-output -n aiaa3201 python outputs/part1/video/self_recorded_analysis_v1/review.py`。六套run/flow/analysis均complete、输入与冻结源码/资源绑定通过；818份原生relative、812帧对、18个视频、18份用户核验及六段汇总齐全。所有帧对非空，无低于5%coverage的帧对。检查4,211份不同绑定文件、9,527,876,284 bytes的SHA全部通过；完整frame/pair CSV重新聚合一致，实际源视频SHA与冻结输入一致。每段首/最高MAE/末尾三对共18对，用独立OpenCV backward remap、FB阈值和显式MAE/RMSE公式复核，mask逐像素相等、数值通过rtol=1e-6/atol=1e-8；两段无人背景252帧/198,400和217,600像素的标准差独立重算一致。用户原有verify及summarize已完整从raw重算，本次不冒称另做了六段全量数值重建。

完整精度来源为各片`analysis/summary.json`，派生JSON、文件校验清单和只读脚本独立保存于`outputs/part1/video/self_recorded_analysis_v1/`。关键结果：

- 无人fixed两段时序MAE/S为`0.005094131026783188 / 0.0015279530098708733`，背景std/S为`0.004539373842827352 / 0.002361263657646324`。整体层次较稳定，存在小幅可测变化，场景1高于场景2，不夸大为严重闪烁，也不将全部成因归为模型随机性。
- 人物fixed两段时序MAE/S为`0.009432451002721073 / 0.006179972758101934`，背景项按预声明保持N/A。只用既有CSV按当前帧时间补充[0,3.5)、[3.5,9)、[9,结束)描述性分段，没有丢帧或改变全片分数；活动期变化与对应失效都增加，离开后相邻变化下降，不代表恢复到相同深度值或真实场景复原。
- 移动两段时序MAE/S为`0.02148980265387661 / 0.01029940463559312`；全图中位数比值范围分别`0.7777149321266968–2.168552036199095 / 0.81100611450806–1.0594774874930517`。横向显示器退出和桌椅进入、前进真实距离变化均影响统计，不能解读为米制漂移真值或百分比准确率。
- 实际查看六张首/中/尾图、五张高变化帧对图、两张背景标准差图及横向移动曲线。记录场景1静态8.0秒小幅广域变化、人物场景1的4.4秒和场景2的3.9秒背景/遮挡变化、横向3.83秒高变化以及前进13.63秒货架/细杆变化。没有声称完整人工逐帧观看全部视频。场景2人物刚进入时变化扩展到可见桌面、地面和立柱，支持上下文敏感的解释，但不排除成像、真实遮挡或光流误差。
- 实际A6000/FP16、网络518×924/输出540×960，DA-V2约23.02–24.29 ms/帧、41.17–43.45模型FPS，allocated约328.57–329.40 MiB、reserved404 MiB；RAFT约76.44–81.41 ms/双向帧对、allocated877.19 MiB/reserved1,094 MiB。分别报告计时边界，不把模型FPS当整套系统实时能力。raw显示无越界；固定参考显示约9.50%–19.71%像素饱和，只影响显示，不修改评分数组。

Step 2按基本状态/协议、真实六段安排、完整复现命令、产物索引、主表、固定/人物/移动解释、显示效率和核验范围组织，保留全部八步；拍摄建议折叠为补充参考。当前正式实验已完成，复现的两个根变量改为新`part1b_self_reproduce_v1`/`self_recorded_reproduce_v1`，避免照抄覆盖当前正式产物。README、PLAN、docs索引、current_results完整精度表、plan_1B2B及数据说明同步完成状态；失效的“自录待采集”链接随新标题更新。历史记录不改写，原自动逐段analysis.md不覆盖。

Part 1B两分支实验、诊断、效率及案例分析门禁已满足，无需补跑；真实session与手机设置未知信息仍保留待确认，mandatory clips来源、最终Demo/报告交付仍未关闭。本次未推进Part 2B或Part 3实验。

收尾验证：9份更新说明的145个本地链接（含Markdown锚点）、20个Bash代码块语法通过；8个复现代码块在临时目录展开52次调用并通过真实CLI参数解析，确认使用新的reproduce目录及正确的source/with_roi选择。Step 2六行主表、中位数范围和效率值均与全精度JSON格式化一致；30份正式JSON元数据再次检查SHA不变。记录保存为`outputs/part1/video/self_recorded_analysis_v1/documentation_validation.json`；没有重复运行全项目测试或模型。

## 2026-10-30 — Part 2B六段自录的官方VDA同条件比较

重新阅读课程PDF §5.2、§6.2，按用户确定的联系组织：复用Part 1B发现的背景闪烁、人物活动变化和运动相机尺度proxy，检查视频模型是否缓解以及是否牺牲结构。主体固定官方VDA-Small、relative、离线batch、FP16，不训练/微调，不增加metric/Base/Large/streaming或额外公开视频任务，不预设一定改善。

新增独立实现：

- `scripts/part2/video.sh`与`pipelines/part2/video.py`提供check/prepare/predict/evaluate/verify/summarize，支持逐片执行、已完整run核验后跳过及禁止覆盖。主环境aiaa3201，正式六段长时工作由用户启动。
- `pipelines/part2/video_common.py`检查父协议/源SHA/manifest/DA-V2/flow/原分析/实际尺寸精度及用户核验。prepare只读复用六段818帧、812对RAFT和原ROI，在独立比较清单中绑定原DA/flow协议与新VDA协议。原元数据、协议SHA和执行身份不改写，不复制旧run伪装为新推理；新增源码/配置/输入一并封存。
- `pipelines/part2/video_evaluation.py`复用原measure_single数学定义，并在完整有限float32前提下逐像素核对相同全图mask，核验DA侧与原Part 1B指标/中位数一致。背景ROI仅用于两段无人fixed，不限制主误差或中位数。保存双方原生ratio、固定跨度中位数变化、时序MAE/RMSE、背景std、coverage、窗口附近观察及正/负降低率；零基线/N/A不虚构百分比，不删难帧。verify从双方原始数组与共同flow重算，summary按无人fixed/人物fixed/moving分组等权保留负结果。
- `pipelines/part2/video_report.py`生成共同原生[0,50]、固定参考[0,1]、差分[0,0.1]的同步视频、共同mask、曲线、首/中/尾与改善最大/恶化最大/高变化案例、背景标准差对照、显示饱和比例和自动中文说明。效率保存实际参数/尺寸/精度/显存与三次整段计时，DA复用原计时；记录硬件/库版本是否匹配和分开调用限制，不把离线均摊成本当在线延迟。
- `configs/part2/video_depth_anything.yaml`改为六段自录专用配置，网络518×924、缓存540×960、FP16，继承父推理/评价/显示设置，仅增加VDA-S资源。被Part 1B冻结绑定的共享src/video、模型adapter、pipelines/video原文件保持不变，不修改第三方仓库。

实际检查：

1. `./scripts/part2/video.sh check --output outputs/part2/video_depth_anything/checks_v1/resources.json`返回ready；六段帧数/来源/相机/ROI、DA/VDA固定commit及干净工作树、三个权重SHA通过。实际CUDA可用、RTX A6000、torch2.1.1+cu121。所需权重均已有，没有下载或安装包，不涉及代理网络。
2. 执行独立`/tmp/aiaa3201_vda_short_check.py`（原文副本保存在checks_v1/short_vda_check.py）：只对场景1原缓存首33帧运行一次未预热官方VDA接口，实际两个32帧窗口，网络518×924/FP16，返回33×540×960有限float32，耗时1.9192109536379576秒、峰值allocated2,857,305,088 bytes。原生数组与源帧身份/SHA/model/window/环境记录保存为checks_v1/vda_33frames.npy和vda_33frames.json。它是双窗口工程检查，不作为六段正式时序分数或效率主表，不启动完整818帧预测。
3. 新增`tests/test_part2b_self_recorded.py`的19项合成真实文件检查：父产物原字节不变、不同真实协议的兼容绑定、全图/ROI职责、已知改善与恶化、正仿射单位、固定偏移与ratio限制、零值/零跨度/N/A、不同输入/尺寸/精度/设备/官方窗口/非有限输出拒绝、seal/父文件/mask/CSV篡改、覆盖拒绝、仅路由VDA、同步视频帧数与显示界限、双方原始重算及汇总。合成生命周期不加载真实模型。
4. 相关回归222项先通过；最终源码完善后执行`PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/aiaa3201-part2b-tests OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n aiaa3201 python -m pytest -q`，**1,467 passed、14 warnings、35.20秒**。告警为现有Matplotlib/PyParsing弃用；真实VDA短检查另有已有TypedStorage提示。资源检查和GPU短测与合成测试分开记录。
5. 最终核对13份相关说明的130个本地链接（含锚点）、21个Bash代码块，六步命令在临时目录展开并通过真实argparse解析；从/tmp运行新shell入口help成功。再次校验Part 1B绑定的4,211份文件、9,527,876,284 bytes，SHA全部不变；正式Part 2B的self_recorded_v1目录尚未创建。核验记录为checks_v1/final_validation.json。

清理与文档：删除未被任何六段Part 1B协议绑定的两个冗余转发文件`pipelines/part2/run_video_depth.py`和`scripts/part2/run_video_depth.sh`，删除前原文/字节数/SHA保存在checks_v1/cleanup_manifest.json。当前Part 2B配置去掉未使用的metric索引分支；同步两个配置测试的职责。Step 4重写为当前六段的基本状态、科学协议、完整六步运行/恢复、结果等待与分析规则、架构/产物和门禁，移除旧公开片段/640画布流程。plan_1B2B重写为当前单图基准、六段已完成1B和新2B的衔接；README、PLAN、文档/配置/流水线/输出/源码/测试索引同步。AGENTS第11条根据用户当前授权更新自录到位和1B完成状态，保留用户启动正式训练/长时预测及mandatory待核实规则。旧共享代码及其兼容配置被保留用于父结果核验，不再作为当前2B主入口；data、权重、用户修改、第三方仓库、原始预测与自动运行元数据均未删除或改写。

下一步由用户运行Step 4六步，正式输出`outputs/part2/video_depth_anything/self_recorded_v1/`；只新增官方VDA，六段预测/比较、完整原始重算、人工失败分析尚未完成，当前不声称VDA已改善flicker或scale drift。Part 1B已完成实验及分析保持有效，Part 2A/3未运行新实验，最终Demo/报告与mandatory clips继续单列门禁。

收尾另用真实六段执行仅元数据阶段`./scripts/part2/video.sh prepare --output-dir outputs/part2/video_depth_anything/checks_v1/binding_check`：父产物完整SHA检查、六套独立VDA协议和比较清单生成成功；随后逐片load_comparison重新验证818帧、父/新协议、配置和源码快照，全部通过，记录为checks_v1/binding_check_result.json。它是实际父产物衔接检查，没有VDA预测，不占用正式self_recorded_v1目录，不将其当作六段实验完成。该真实衔接检查结束后没有再修改科学源码。


## 2026-10-30 — Part 2B六段正式结果核查与分析

用户已完成Step 4六步，正式产物为`outputs/part2/video_depth_anything/self_recorded_v1/`。六段VDA-Small原生relative/FP16共818帧，812对同光流评价、18个同步/差分视频、六份verify及三种条件汇总齐全；不重新生成DA-V2、光流或任何模型预测，原Part 1B协议/元数据保持真实。

本次执行`conda run --no-capture-output -n aiaa3201 python outputs/part2/video_depth_anything/analysis_v1/review.py`，通过6,796份来源/产物SHA（12,278,531,365 bytes）、全部逐帧/逐对CSV聚合、812个双方mask逐像素比较、输入/配置/源码/窗口/FP16及DA基线一致性；原Part 1B的4,211份保护文件完全不变。独立以OpenCV backward remap和显式公式抽查27个不同帧对，双模型共54项MAE/RMSE及mask；两模型两段无人ROI的完整背景std和相邻变化共4项复核通过，容差rtol=1e-6/atol=1e-8。18个MP4帧数与首尾解码通过，核查结果和逐文件SHA保存在该analysis_v1目录。六份用户verify及summarize已经完整重算；本次抽查没有冒称再次全量重算六片。

正式结论：六段MAE/S降低54.06%–80.84%，RMSE/S也均下降；三种条件的MAE组均值降低72.83%/57.85%/73.81%。24/812个帧对的VDA MAE仍更大；人物固定1的4.6 s局部恶化、5.8 s RMSE尖峰均保留。无人固定1背景std降低25.83%，无人固定2增加31.92%，两段均值仅降低6.07%。固定跨度中位数绝对偏离只在横向片减少，另五片增加；原生ratio也同方向，不据此宣称米制尺度漂移已解决。离线使用未来帧、人物/视野/真实运动与无GT限制明确披露。

实际效率：A6000、相同PyTorch/CUDA、518×924/FP16；DA约41.17–43.45 FPS，VDA约33.68–38.54 FPS，VDA峰值allocated约2.66–2.77 GiB。DA是已有计时、VDA为新调用，不声称严格同时受控；全片均摊ms不是在线延迟。原生[0,50]显示无越界，固定参考副图的逐片饱和比例保留；显示截断不用于评分。

人工查看六段首中尾、六份曲线和代表性差分；从已有两个人物同步视频抽出指定帧供读者对照，生成逻辑保存在analysis_v1/extract_review_frames.py，未修改原MP4。证据支持背景大范围跳动减少，人物和细边缘仍困难；没有把颜色差异等同几何准确率、把全部残差叫作拖影，或声称已覆盖镜面/玻璃/天空/强模糊等全部场景。

Step 4按状态/固定协议、完整六步复现与恢复、正式表格、背景和尺度proxy、实际案例、成本/显示、架构与完成门禁整理。复现默认使用新self_recorded_reproduce_v1及checks_reproduce_v1，不覆盖正式结果。README、PLAN、plan_1B2B、docs/outputs索引、current_results的全精度来源同步；本条之前的执行历史保留，没有把待运行的历史条目改写成新运行。科学源码、冻结配置、第三方仓库、原始数组与自动元数据均未改动，没有训练、推理、RAFT或下载任务。实现阶段最近的1,467项测试不在本轮重复运行。

本批Part 2B实验、核验和代表性分析门禁通过，无需补跑；改善目标部分达成。mandatory clips来源核实、素材覆盖限制、最终报告/应用Demo与视频打包仍需整合，本次不推进Part 3。

收尾验证：8份相关Markdown的136个本地链接（含锚点）、16个Bash代码块语法通过；Step 4六步参数经真实argparse解析，未执行命令对应的实验。主表8位显示、完整精度记录与正式JSON一致，新分析脚本AST通过；文档修改后再次核对6,796份原来源/产物SHA均不变。记录为`outputs/part2/video_depth_anything/analysis_v1/documentation_validation.json`。


## 2026-10-30 — 三类型单视频流程与全图逐像素诊断

按用户要求，Part 1B/2B以单段视频为单位组织，不要求先收齐一批后再按功能分批运行。新增`pipelines/video/clip.py`、`src/video/clip_workflow.py`、`clip_analysis.py`和`clip_report.py`；Part 1/2各自的video_clip薄入口与原shell分别默认完成逐帧或配对分支，共同`scripts/video.sh`默认完成同视频两步。原来的四份固定六段Part 2B编排源码和对应旧专项测试退役，配置别名改指共同科学协议；底层官方adapter、frames/runner/flow、既有数学算子和第三方仓库未修改。

新增共同`configs/video/self_recorded.yaml`和独立`clips.yaml`：登记三类型、源路径、session与描述，不把六段/帧数写死在方法或调度器里。当前六段预计818帧/812对；这不是新流程已有结果。新视频可直接--video/--type，首次准备后记住登记，后续只需ID。--task选择单项，默认补齐依赖，--no-deps严格只执行所选任务。完整产物统一`outputs/video/<ID>/`，data缓存每片一次；已完成阶段验证后复用，失败/不完整记录不覆盖。commands按真实调用区分执行、复用、尝试/完成模型阶段及失败，不把失败推理记成从未尝试。

无人固定且场景静止的背景统计使用全画面，不要求矩形ROI；固定有人和移动片的全图背景std记N/A。所有类型增加无光流、无FB mask、同坐标全图pixel MAE/S及RMSE/S，同时保留同光流对应的两项指标；各自逐帧对等权。固定首帧P5/P95跨度、原生relative/arbitrary/alignment:none、合法零值、非有限失败、零跨度N/A、coverage仅标记保持明确。保留原生median ratio与固定跨度delta、实际效率和官方离线窗口；逐像素变化混入真实运动、尺度proxy受真实距离/视野影响，不当作准确率或米制漂移真值。

报告共用单/双模型布局，保存固定原生/首帧参考深度、pixel/flow差分两种视频、两类案例、曲线、适用全图背景std、饱和计数、中文类型说明与成本；两模型输入/尺寸/精度/官方窗口一致，mask逐像素一致，DA在两步桥接一致。verify重算双方raw/flow、pixel/flow/背景/median、CSV与数组，不运行模型。汇总按三类型等视频、任意片数、N/A不补零，拒绝重复源或混合科学协议。

实际验证：新增`test_video_clip_workflow.py`的28项合成测试通过（真实临时MP4/PNG/NPY，未加载模型）；平移对照确认光流误差为零时全图pixel变化仍非零，说明两者任务不同。初次该对照中“滚动平移仍有残差”的测试假设不成立，已按有效对应的解析关系更正断言，没有改预测或评价去制造残差。完整`conda run --no-capture-output -n aiaa3201 python -m pytest -q`为1,476 passed、14个既有Matplotlib/PyParsing弃用告警，39.91秒。随后仅补强失败尝试日志与报告色标校验，相关专项再次检查，结果另见维护验证记录。

实际运行`./scripts/part1/video.sh check --output outputs/maintenance/video_reorganization_20261030/resources.json`：CUDA可用，RTX A6000、torch2.1.1+cu121；DA-V2/VDA官方commit与干净工作树、三个已有权重SHA通过，没有下载、安装或执行模型。对六段源视频仅重读SHA及ffprobe流信息，保存`data/self_recorded/source_inventory.json`，与原登记SHA一致；未声称本轮重新全片解码。未知手机/session等信息为null。

按本轮“不要保留原版运行结果和产物”的授权执行`outputs/maintenance/video_reorganization_20261030/cleanup.py`，清理`outputs/part1/video/`、`outputs/part2/video_depth_anything/`和`data/video_frames/part1b_self_v1/`，共7,158份文件、12,430,970,008 bytes。删除前逐文件路径/字节数/SHA记录于cleanup_manifest.json，未把原自动元数据改写成新执行；仅保留清理指纹与原有implemented执行历史。删除前后7,493份保护文件SHA一致，覆盖Part 1A、DA-V2-NYUv2、两条Part 2A、四项关键分析、六段原视频和相关权重；Part 3没有改实现、数据或实验产物。退役源码为pipelines/part2/video.py、video_common.py、video_evaluation.py、video_report.py与旧专项测试，均为项目自有文件。

Step 2保留已完成NYUv2协议/命令/结果，重写视频段为类型分工、全图指标、每片完整命令、单项/续跑、新视频、架构与待运行门禁，拍摄指导继续保留。Step 4改为同视频配对完整命令、共用依赖、三类解释、报告与待运行门禁；不填入已清理协议的背景分数。README、PLAN、plan_1B2B、AGENTS当前状态、current_results及数据/配置/源码/流水线/测试/输出索引同步；历史条目的完成含义不改写，已清理链接改为文字路径。没有启动正式DA/VDA/RAFT、训练、下载或Part 3实验；新视频的真实效果仍待用户运行。

收尾实际验证：最终单视频专项28 passed、14 warnings、5.52秒；包括最后补强的失败尝试日志、类型与manifest/session一致性及参考域标签。19份相关说明的176个本地链接/锚点、39段Bash语法通过；Step 2/4全部当前视频和可选任务命令通过真实argparse与依赖计划检查，未执行正式阶段。三个shell从项目目录外、带空格参数的环境/路径路由检查通过。再次核对7,493份保护SHA不变，三个退役目录均不存在，`outputs/video/`尚未创建；记录在维护目录的`validation.json`，最后文件指纹另存`final_state.json`。


## 2026-10-30 — Part 1B/2B独立入口与产物

按用户要求，1B与2B分别运行和存储。`scripts/part1/video.sh`调用`pipelines/part1/video_clip.py`，只调度prepare、dav2、flow、analyze1、verify1，默认写入`outputs/part1/video/<ID>/`。`scripts/part2/video.sh`调用`pipelines/part2/video_clip.py`，只调度绑定输入/冻结自身协议、vda、analyze2、verify2，默认写入`outputs/part2/video/<ID>/`。两个入口各自提供check/plan/run/summarize和本部分任务选项；移除合并入口`scripts/video.sh`和--part/both模式。`pipelines/video/clip.py`仅保留参数定义及预检/汇总辅助，模型adapter和数学实现继续共用。

2B必须引用已完整完成且通过raw重算的1B；缺少输入直接报错，不运行DA/RAFT，不创建伪完成输出。--part1-root明确1B输入根，--output-root只控制2B结果，两根不能相同或互相包含。2B自己的clip.json.baseline绑定1B目录及clip/protocol、DA run、flow清单、分析summary和verify1的SHA，VDA使用自己的protocol与同一RGB manifest。两份协议分别真实记录调用，文件SHA无需相同；科学配置、RGB/PTS、实际模型尺寸/精度、固定显示、flow mask及DA分析桥接须一致。2B只读1B，不向其commands、报告、预测、flow或verify写入；2B比较所需mask/背景等派生物保存在自身part2目录，不复制DA原始预测。

fixed_dynamic和moving的补充项明确写为不经光流、无FB mask、全图同坐标的逐像素MAE/S、RMSE/S。三类型任务、无人固定全图背景、真实运动与尺度proxy解释保持一致；两类运动视频不把逐像素变化单独解释为纯flicker。Step 2/4同步六段运行、各自汇总路径、单项任务、新视频先1B后2B及自定义输入根说明，README/PLAN/plan_1B2B与相关架构/配置/数据/测试/输出索引同步。NYUv2和其他静态科学配置/结果未修改，Part 3未改动。

实际验证：首次专项为39 passed、3 failed，涉及两处测试仍按共目录定位，以及argparse将--part缩写为--part1-root；已修正测试定位并对两入口/子命令禁用长参数缩写。随后专项42 passed、14 warnings；补充CLI真实调度与输出根嵌套拒绝后，全项目`conda run --no-capture-output -n aiaa3201 python -m pytest -q`为**1,484 passed、14 warnings、41.70秒**。警告为既有Matplotlib/PyParsing弃用提示。测试使用临时MP4/PNG/NPY和明确合成预测，验证2B缺失1B门禁、独立任务选择、完整1B所有文件SHA在2B运行后不变（包括commands）、来源/核验篡改检测、同DA桥接、raw重算和独立CLI调度，没有加载正式DA/VDA/RAFT权重。

本次未创建正式1B/2B视频产物，未启动真实视频预测、训练、下载或安装；实际效果待用户运行。独立维护记录位于`outputs/maintenance/video_parts_separation_20261030/`：全项目pytest原始日志、文档命令/链接与shell路由、保护文件核验、当前文件SHA另存其中。上轮清理/资源/执行记录与本文件历史段落保留当时含义，没有改写成此次执行。


## 2026-11-02 — 新版课程说明翻译与Part 1/2差异核对

完整读取根目录`Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf`（5页，SHA256 `ea65076917e87febb0c3c67773b9fc6edf304e2821a74266a03f22ef33abc226`），提取正文与内嵌链接，核对第3/4/5页关键条目，并与旧版正文比较。新增中文`Project3_CN_final.md`，保留六个备选方向和12条参考文献，明确arXiv可选、指定片段、原FPS、relative/无GT对齐及课程NYUv2协议；没有将本地实现选择扩写成课程要求。

新增`difference_analysis.md`，分别核对1A、2A、1B、2B，未评价现有Part 3实现。1A几何与15场景/60份预测可复用，但当前评价代码使用non_occluded mask，需另补有限正值全GT域的MAE/coverage。2A两模型各654份native输出、RGB-only、crop/range/等图权重和alignment:none与新版一致，主表数值从当前完整精度JSON格式化；外部K/oracle仍仅诊断。旧DA-V2-NYUv2对齐基准和四项分析列为扩展，不改写或删除。

实际读取7段自录两部分的14份summary与14份verify：均有complete/verified，verify指向的summary SHA匹配；各模型有952份raw NPY、945相邻帧对。它们使用10 FPS、960长边的旧self_recorded轨道，不是新版原FPS/指定片段轨道。这里是本次读到的既有产物状态，不将它们写成2026-10-30开发阶段已经发生的新执行，也没有重新执行原raw重算门禁。

对官方`davis_rollercoaster.mp4`执行只读ffprobe实际解码计数：960×540、24 FPS、70帧、2.916667秒，SHA256 `7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8`。本地冻结commit `4f5ae23172ba60fd7bc11ef671cca678842c7072`跟踪文件无修改；视频Git blob为`f385e1bb30f90b48e0c6a8f485b8ff4862f082a5`。通过工作站直连GitHub官方Contents API只读main文件元数据，返回同一Git blob与1,809,560字节，未下载视频。因此“前100帧”与当前提供文件冲突，差异分析明确保留待教师确认，不插帧、循环或将VDA内部padding当作新输入帧。

README更新新版权威入口及1A/1B/2B状态区别；旧PDF/译文、PLAN、plan_1B2B、Step文档、科学配置及自动运行元数据保留原样，待确认新轨道后再统一改造。继续尊重GitHub不上传视频和1B/2B产物的偏好；本次没有更改上传白名单或推送。只做文档/来源核验，没有训练、模型推理、BM/SGBM搜索、全量指标复评或资源下载；本轮没有重新执行pytest。

收尾文档核验通过：译文保留PDF内全部18个不同URL、12条参考文献和6个方向；4份新增/修改Markdown的77个本地链接/锚点、10段Bash语法通过。203份科学源码、配置与本次小型来源文件SHA保持不变。该检查不包含重新计算1A/NYUv2/视频指标，也不把旧verify记录算作本次raw复评。


## 2026-11-02 — Part 1A课程评价与四组复现流程

Part 1A保持`bm_fixed/bm_tuned/sgbm_fixed/sgbm_tuned`，fixed经验参数和共同5/10场景划分保留。`middlebury_disparity_to_depth`显式接收quarter标定的毫米基线，按`f_px×B_mm/(d_px+doffs_px)/1000`计算米制深度；OpenCV定点仅除以16一次，GT PFM不重复换算。对非正/非有限视差、非正分母和无效标定作检查，保存float32米制结果与布尔mask，不做尺度拟合。

主指标统一为所有有限正GT视差域的视差MAE与有效预测比例，分母不受non_occluded mask限制。无效预测不进入MAE、保留在比例分母；零有效预测MAE为null，宏平均不静默跳过该场景。米制/距离分析单独作为辅助诊断，主表和搜索不再包含Bad-1/2/4或视差RMSE。

BM144候选、SGBM36候选只使用共同5个开发场景，按“有效比例不低于各自fixed参考时MAE最小”、有效比例与确定性参数顺序选参，LR固定1 px。YAML不手填tuned参数；正式运行校验完整候选、参数来源、配置、科学源码、数据和产物SHA。`src/stereo/protocol.py`提供绑定/归档，两个pipeline及shell入口分别执行搜索与四组评估；输出拒绝覆盖，失败如实记录。评估的`--mode verify`不运行匹配器，从保存左右视差重做LR、mask、米制深度和全部精度指标，并核对场景覆盖与宏平均。

八联图包含RGB、LR前视差、GT、最终视差、误差、整图匹配器无效、整图最终无效、整图LR状态；额外保存原尺寸matcher_invalid/invalid/lr_inconsistent/lr_status无损PNG。四组总览、局部困难案例、理论±视差扰动和四组实际距离分层继续保留，使用对应全GT域。

实际验证：专项26项通过；完成最终指标整理后运行`conda run --no-capture-output -n aiaa3201 python -m pytest -q`，为**1,492 passed、14 warnings、42.72秒**，警告为既有Matplotlib/PyParsing弃用提示。新合成生命周期覆盖两次小候选搜索、四组15个合成身份、参数缺失、fixed独立运行、禁止覆盖、失败记录、来源/候选与NPZ篡改检测、原始数组重算、无定义MAE与整图掩码，未把它作为真实算法结果。

实际仅对已有Adirondack/sgbm_fixed缓存做单样本转换与八联图检查：本场景`f=1038.018 px,B=176.252 mm,doffs=53.271 px`，与已有保存米制深度最大绝对差0.0 m，符合两种单位写法的等价关系；此数字仅是一个缓存样本的检查，不要求跨设备逐位相同。图像已查看，视差、整图无效与LR状态显示正常。派生检查位于`outputs/maintenance/part1a_course_20261102/visual_check/`，明确标注缓存来源，没有启动匹配器。

Step 1重写为中文协议、四组设置、完整搜索/运行/独立核验命令、结果入口、全图可视化与门禁；README/PLAN及1A相关配置、源码/流水线/测试/输出索引同步。current_results保留非遮挡域历史表并明确其来源，difference_analysis增加1A当前实施进度；历史执行记录不改写。

完整144/36开发搜索与15×4正式运行交用户启动，预定目录为`outputs/part1/middlebury/tuning_course_v1/`和`course_v1/`，本轮两目录均未创建。没有启动训练、模型推理、全量匹配/参数搜索或下载。原有1A输出198份文件逐一按大小/SHA校验不变（含原始预测与自动元数据），data/checkpoints/第三方仓库和Part 1B/2A/2B/3科学实现未修改。维护核验与保护清单保存在`outputs/maintenance/part1a_course_20261102/`。


本轮收尾文档检查：13份说明中的176个本地文件链接与28段Bash语法通过；两个正式入口的Shell语法与实际`--help`调用通过。原有198份1A文件大小及SHA不变，正式输出/搜索目录均未创建。结果存入上述维护目录`validation.json`，没有将链接检查写成全部锚点或完整正式实验验证。


## 2026-11-02 — Part 1A辅助指标与输出目录

按用户要求，四组正式结果同时保存视差MAE、有效预测比例、Bad-1/2/4、Depth AbsRel和Depth RMSE。Bad阈值严格为绝对误差>1/2/4 px，GT域与主指标相同；无效预测计为失败、保留在完整GT分母。Depth AbsRel/RMSE原已实现，继续在有效预测处计算并明确进入Step 1结果表。新增Bad项参与逐场景CSV、5/10/15场景宏平均与独立重算；无有效预测时Bad均为1，MAE和深度误差为null。主指标和开发集选择规则不变，不用新增辅助指标事后更换选参目标。

配置默认目录改为`outputs/part1/middlebury/tuning/`与`outputs/part1/middlebury/evaluation/`，不带版本后缀。README、PLAN、Step 1及配置/输出/当前结果索引、Step 0、GitHub导出说明同步，独立复现仍要求新空目录。内部协议标识保留作机器可读元数据，不作为输出目录名。

用户明确要求删除`frozen_v2/`与`tuning_v2/`，实际执行维护目录中的`cleanup.py`，先保存逐文件路径/大小/SHA，再删除这两个确切目录，共198文件、126,677,772 bytes。清单和状态见`outputs/maintenance/part1a_metrics_cleanup_20261102/cleanup_manifest.json`，没有复制一份旧预测到其他目录；数据、权重、第三方和其他Part产物未删除。上轮“198份未改动”的记录保持当时含义，本轮新增真实删除记录。

实际运行全项目`conda run --no-capture-output -n aiaa3201 python -m pytest -q`：**1,493 passed、14个既有弃用告警、42.05秒**。新增严格阈值/无效计失败测试，合成完整流程核对七项精度指标均写入汇总并完成raw重算。没有运行正式搜索、匹配、训练、下载；两个新正式目录未创建，用户下一步仍按Step 1的三条指令执行。


## 2026-11-02 — Part 1A完整运行与左侧白边审查

用户已完成三条正式指令。本次读取实际`tuning/`与`evaluation/`，核对当前配置、科学源码、所有输入及产物SHA，重放两算法完整候选选择：BM144、SGBM36候选完整，所选BM为block9/uniqueness0/speckle50/texture20，SGBM为block3/uniqueness10/speckle50。读取与复核是本次工作，正式匹配与搜索由用户执行，不追记为Codex执行。

用`outputs/maintenance/part1a_boundary_review_20261102/review.py`只读重算60份预测的LR、有效掩码、米制转换与1,200个精度值，逐场景/宏平均一致；全部NPZ数组与原始RGB同尺寸，240份独立无效/LR PNG与状态逐像素一致。查看本次Jadeplant SGBM八联图，另对全部15场景四组统计完整无效左边界。没有发现图像裁窄、坐标平移或单位解码错误。

实际OpenCV为4.11.0，核对对应官方SGBM/BM源码：完整搜索范围限制使SGBM的左边界为minDisparity+numDisparities（此处min=0），BM还受窗口限制。Adirondack SGBM左80列无效、BM fixed左86列；Jadeplant分别160和166列。不能仅归因为真实视场不重叠：Jadeplant左160列中47,039个官方GT非遮挡像素的右对应坐标仍在图内，例如(x=32,y=228,d=17.40144)对应右x≈14.59856。该GT诊断未参与预测、选参或主评价掩码。

第2/4图NaN视差、第5图无可评价误差显示为空白；第6/7二值图白色表示无效。画布留白与无效白色混合导致视觉上像窄图，图像数组未丢列。固定1 px双向筛选进一步降低有效比例，是明确的后处理规则，非课程强制项；例如Jadeplant SGBM fixed的全GT域原始输出55.74%、最终有效36.12%。该局限在Step 1披露，不通过填洞、裁GT分母、缩小搜索或修改预测消除白带。

Step 1、README、PLAN、当前结果及配置/输出索引更新为用户真实完成状态，补齐实际七指标表、参数、独立评估覆盖下降和白边成因。15场景MAE/有效比例为BM fixed 1.3038434691676977/0.49337005169720183、BM tuned 1.030009022193071/0.4844487040496428；SGBM fixed 1.3825627218309424/0.6336855599347191、SGBM tuned 1.3086818590909965/0.6235204897544366。两算法tuned在开发集满足选择约束，但独立评估覆盖均降低，不宣称全面改善。

审查脚本初次因独立文件导入路径缺失而失败，补充根目录到sys.path后通过；科学流水线未改。没有运行正式匹配、参数搜索、训练、下载或重跑pytest；上一轮1,493项通过记录保留。本轮实际验证是已有真实产物的SHA、尺寸/PNG和原始数组重算。正式输出的prediction、metadata、图像、CSV、run_metadata、verification等原字节未改；审查脚本和报告独立保存在维护目录。


## 2026-11-02 — Part 1A紧凑六联图直接替换

按用户要求，六图固定为左RGB、GT视差、最终预测视差、绝对视差误差、matcher invalid、final invalid。`src/visualization/stereo.py`改为按输入宽高比布置等尺寸图像轴，缩小标题/行列空白，单独为色条留位；误差色条紧邻误差图，两个无效图不使用连续色条。保留完整图像与NaN/无效带，不拉伸或裁剪内容；独立LR状态和四份mask继续保存。

实际从已有60份NPZ重新绘图，直接替换原`evaluation/<scene>/<experiment>/visualization.png`共60张，以及四张`visualization_contact_sheet_*.jpg`。查看了正式Adirondack BM与Jadeplant SGBM的六图，次序、第五/第六图、色条和布局正常。没有启动匹配、参数搜索、训练或模型推理；没有要求用户另跑绘图步骤。

原`run_metadata.json`、status、原verification、预测、逐场景metadata、指标JSON/CSV、四张独立mask及其他辅助图均保持原SHA。原运行记录仍描述实际初次执行，未伪造为当时生成六图。新增`evaluation/visualization_refresh.json`仅绑定本次展示源码和64张合成图的新旧SHA；核验器支持这种明确的展示更新，只允许合成图/总览变化，并比较科学流水线AST，拒绝借展示更新改变评分或预测。当前tuning证据及其执行时源码快照未修改；未来新实验正常从完整搜索开始绑定当前源码。

本轮维护脚本与执行前文件指纹位于`outputs/maintenance/part1a_six_panel_20261102/`。执行后确认只有64张授权图像变化，其余387份原有文件SHA不变；保存数组再核验60文件、1,200个精度值通过。Step 1更新六图顺序、无效白色解释与目录说明，未加入临时绘图操作；“用户运行：完整实验”整个章节逐字节未变。README及当前展示说明同步为六图。

实际测试：专项6项通过；全项目`conda run --no-capture-output -n aiaa3201 python -m pytest -q`为**1,494 passed、14个既有弃用告警、42.37秒**。新增测试覆盖授权图像替换后原核验流程可用、运行记录SHA不变、指标文件不得通过展示更新白名单。独立重算及测试没有改写正式运行的原始预测或评分。

## 2026-11-02 — Part 2A原生米制、绝对尺度与物体边界

按课程PDF §5.2与§6.2核对：允许的预训练模型、RGB-only、NYUv2标准654张、固定crop/有效范围、逐图宏平均与alignment:none均与两条基线一致。AbsRel/RMSE/δ1及实际效率已有产物，不需要重新推理来满足核心定量要求。Depth Pro为0.10047157333839377 / 0.3922168142032252 m / 0.915173937355017；UniDepthV2-S为0.11454372957333221 / 0.39884995782725563 m / 0.8921797284727733。

`run_key_analysis.py`与`key_analysis.yaml`只保留绝对尺度和边界分析：删除外部相机推理、标定对照/相关性分析与场景类别汇总；基线评估不再生成场景类别表或聚合相机参数分布。两个项目自有模型adapter只暴露RGB输入，保持官方原生推理与模型附加输出；未修改第三方仓库。联合分析在CPU读取完整native基线，逐图核对身份与七项指标，并绑定数据/配置/源码/预测SHA；新目录生成、禁止覆盖、失败状态如实记录。脚本使用aiaa3201-unidepth，不需要额外模型轨道。冻结的两份baseline YAML不改，原始预测、生成记录与执行时配置全部保留真实字节。

Step 3按任务/协议、完整命令、米制结果、效率、绝对尺度、物体边界、具体案例与待补工作组织。读取当前完整精度JSON/CSV补充距离桶RMSE和有符号偏差：UniDepth在0.001–1 m平均偏深30.849032739258014%，6–10 m偏浅13.462662500706674%；Depth Pro相应为+8.052854420029187%与-12.710993516335622%。说明单一整图比例不足以解释不同距离的误差，oracle测试GT调整仅用于诊断。查看了test_0021椅背/扶手、test_0545沙发/灯/细桌腿与test_0302墙面/小物体三个八联图，补充漏检、伪边缘、细结构及GT局限的说明，没有按指标排名推断所有物体的质量。

仍可完善裁剪前尺度/超范围比例、固定坐标的轮廓局部放大和深度剖面、全局尺度与局部边界的联系。内部网络尺寸来自上游代码和checkpoint配置推导，明确区分于原运行实测。未将这些待补诊断写成已有结果，也未开展Part 3修改。

同步README、PLAN、配置/流水线/输出索引、Step 0、当前结果审计与difference_analysis的2A说明，上传指南使用尺度/边界白名单并包含三张小型案例图；没有生成上传包、连接GitHub或更改远端。主文档以当前方法组织，不添加说明更新的叙事；实际执行历史不改写。

实际验证：新增12项合成/回归测试覆盖无相机输入的完整分析、已知尺度偏差与oracle分离、缺失/重复/错位身份、指标不一致、无效预测、禁止覆盖、失败记录、协议保护和官方RGB-only调用。专项20项通过；最终全项目`conda run --no-capture-output -n aiaa3201 python -m pytest -q`为**1,506 passed，14个既有弃用告警，42.47秒**。实际aiaa3201-unidepth分析CLI的`--help`与shell语法通过，没有启动模型。

来源审查保存于`outputs/maintenance/part2a_review_20261102/`：654张联合表与每模型5,232个native/SI值最大差0；两模型各3份原始NPZ的七指标重算最大差0；边缘表3,924行、距离分桶表4,350行；1,993份原有JSON/CSV/YAML SHA未变。没有运行训练、正式预测、全量评分或下载，也没有改写原始自动元数据。新分析入口的完整真实数据重跑仍由用户按Step 3命令启动。

收尾检查：167个当前Markdown本地链接与40段Bash/Python示例语法通过；两个冻结baseline YAML与整理前逐字节一致；当前2A实现/配置/主说明不再引用外部相机推理或场景汇总入口。差异分析对课程要求的历史对照、其他Part自身的统计及原始执行历史不作为待删除实现。

## 2026-11-02 — Part 2A完整绝对尺度与物体边界分析

保留两个单图度量模型、两个原生frozen_v1目录，以及现有key_analysis配置/源码/入口和全部生成记录。新增独立`configs/part2/scale_boundary.yaml`、`src/analysis/scale_boundary.py`、`src/visualization/scale_boundary.py`、`pipelines/part2/analyze_scale_boundary.py`、`scripts/part2/analyze_scale_boundary.sh`；正式输出为`outputs/part2/scale_boundary_v1/`。两个模型共用图像、GT域、规则和诊断条件，不选参、不训练、不重新推理；第三方仓库、数据、权重与原始预测不修改。

实际在aiaa3201-unidepth环境用CPU完成654张全量分析。每模型156,018,240个有效GT像素，七项正式深度指标逐图与冻结CSV一致。SI从冻结CSV引用并绑定输入SHA，不追记为本次重新调用官方SI；其余尺度、三阈值/三容差raw/clipped边缘、物体支持召回、定位、区域和剖面均从原始数组重新计算。原生米制正式轨道始终alignment:none；crop、split、有效范围、裁剪与逐图等权保持不变。

尺度主体包括raw/clipped对照、每图raw中位数比例、超范围比例与极值、五个真实距离桶、米制/相对有符号误差、log误差的整图均值平方与中心化方差恒等式，以及raw中位数GT辅助调整后的残余误差。Depth Pro/UniDepth分别32/28张有>10 m预测，像素比例宏平均0.0025169300717659674 / 0.0034713248912434853，最大40.968563079833984 / 27.626502990722656 m。raw RMSE为0.4162887145650984 / 0.42148314854145946 m，正式clip后仍为0.3922168142032252 / 0.39884995782725563 m。raw尺度比中位数0.9580595290246076 / 1.0477975681064606；整图log偏差平方项占raw logMSE的0.4684048166760037 / 0.471039448637656。oracle只作测试GT辅助诊断，不作为算法改进或因果误差分解。

物体边界使用现有MAT的labels/instances，类别与实例组合识别轮廓，label=0排除；实例0不直接当无标注。物体支持深度边缘是距标注轮廓≤2 px的GT深度边缘，只报该子集召回，不用不完整物体标注惩罚所有预测边缘。主阈值1.05有636图支持、18图N/A；raw物体边缘Recall@3为0.6411892504296968 / 0.5360657922402186。全部深度边缘用1.05/1.10/1.25三阈值和1/2/3 px Chebyshev匹配；Euclidean截断10 px距离单列，不混用距离。边缘≤3 px占0.10863240733903934像素比例，承担0.2419104627603672 / 0.24698292495705676的平方误差；互斥区域贡献可逐图相加。GT尺度调整后边缘AbsRel仍为0.10336904236921644 / 0.11138349710809184，高于内部0.04793019944453446 / 0.05198068906247826。

仅按GT与标注的固定条件选择513张图上的1,146条共同水平/垂直剖面，半径20、稳定端点5、GT对比≥1.10、GT宽度≤6 px、间距60。预测自身两侧归一化仅测形状；弱对比、次序错误或交点不唯一记N/A，保留失败和覆盖。Depth Pro为1,066可测/35弱对比/35不唯一/10错序，UniDepth为1,042/52/43/9；条件均值先按图内平均再图像平均。两模型位置绝对误差为1.8105237616762482 / 2.0329280250007282 px，宽度增量为2.4135899959476474 / 3.1955873256104987 px；双方都可测的共同剖面495张图上配对差值区间支持Depth Pro较少偏移和过渡模糊，不外推到所有曲线/细结构。

正式产物包括1,308行逐图、19,620行分桶、23,544行区域、23,544行边缘、7,848行定位、2,292行剖面；完整精度JSON/CSV、七类汇总图、12个案例、13组局部剖面/米制曲线和实际坐标、中文report、输入/源码/配置归档及产物SHA齐全。查看了test_0021椅背、test_0545容器/沙发、test_0302小圆形物体的实际放大与曲线；弱对比形状明确N/A，不归一化放大噪声后赋予几何成绩。代表图和指标选择有偏、物体标注/填充GT局限、剖面覆盖及轴向限制均在Step 3说明。

区间使用10,000次scene成簇bootstrap并保持图像权重，点估计仍为课程图像宏平均；原baseline图像级bootstrap记录保留。配对AbsRel、δ1与这些边界量支持Depth Pro优势；RMSE和绝对log尺度偏差的区间跨零，未宣称所有指标显著更好。

真实完整命令：

```bash
./scripts/part2/analyze_scale_boundary.sh --config configs/part2/scale_boundary.yaml \
  --output-dir outputs/part2/scale_boundary_v1
./scripts/part2/analyze_scale_boundary.sh --config configs/part2/scale_boundary.yaml \
  --output-dir outputs/part2/scale_boundary_v1 --mode verify
```

正式完整运行耗时166.6226069466211秒（CPU分析/绘图，非模型效率测试），全量数值重放核对1,039,777个值与汇总，71个产物SHA通过。浮点rtol=1e-6、atol=1e-8，身份、整数计数和覆盖严格一致；没有为匹配旧数值改预测或评估。仅SHA模式明确不代表数值重放。16项新增测试覆盖已知尺度/裁剪、log恒等式、标注组合、空域、边缘距离、位置/模糊/次序/对比、GT选择、成簇图像权重、真实HDF5合成生命周期与绘图/全量重放、覆盖/损坏和数值容差。最终主环境全套为**1,522 passed、14个既有Matplotlib/PyParsing弃用告警、46.34秒**。

实现核验归档在`outputs/maintenance/part2_scale_boundary_20261102/first_full_check/`，保存第一次完整运行/重放的原记录；随后完善汇总容差与N/A显示，再生成正式目录。评分公式和完整精度科学汇总一致，不把首次执行的源码/时间改写成最终版本。保留指纹检查2,000个原有记录/源码SHA全部一致，原版key_analysis与两条baseline未覆盖。来源检查曾误用base解释器而缺h5py，改用指定环境后通过；未向base安装包。没有训练、模型推理、下载或视频/Part 3改动。

Step 3已按协议、完整命令、正式米制、尺度、物体边界、具体局部图、配对统计、架构与核验组织。README、PLAN、当前结果、配置/流水线/输出索引与difference_analysis同步真实完成状态；上传指南补足新分析源码、测试和小型证据白名单，没有导出包、推送或修改远端。

收尾文档核验：186个本地链接/图像路径和33段Bash/Python示例语法通过；七份执行时源码均与当前代码及归档SHA一致，71个正式产物与manifest一致。实际12个案例包含13组剖面图，共37份PNG/JPG，文档按实际计数。原有2,000项保留指纹再次通过；GitHub现成包仍按原生成记录保留，没有把它宣称为包含新增分析的现成发布包。


## 2026-11-02 — Part 1B：指定片段的独立逐帧基线

重新阅读新版PDF §5.1、§5.2、§6.1–6.2。1B主线改为官方DA-V2-Small在指定davis_rollercoaster的连续原始帧上独立预测，输出relative/arbitrary、alignment:none，无视频GT、无米制拟合、无时间平滑；DA-V2-NYUv2的654张GT辅助aligned_relative扩展保留，不变更原始预测、评价或科学源码。按用户安排，本轮不运行VDA，也不作DA/VDA比较。

本地`data/davis_rollercoaster.mp4`SHA256为`7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8`，960×540、24 FPS、70个真实帧。当前原始索引0–69全部处理，不重复/插帧/循环凑100。run/protocol/verification均如实保留`requirement_100_frames_satisfied:false`及待课程确认状态；完成当前70帧工作不等于课程100帧要求或VDA比较门禁已完成。

新增配置`configs/part1/dav2_video.yaml`、入口`pipelines/part1/dav2_video.py`/`scripts/part1/run_dav2_video.sh`，独立提供check/run/analyze/verify。新增`src/evaluation/course_video_inputs.py`直接按解码索引保存原帧/PTS，不使用FPS采样网格，避免小数PTS舍入导致原始帧被跳过；`src/evaluation/framewise_clip.py`负责资源/配置/源码绑定、DA独立运行、原生浮点保存、计时、分析与原始重算。`src/metrics/video_diagnostics.py`提供整段固定单位、可选Farneback/FB和边缘变化proxy；`src/visualization/framewise_clip.py`生成固定色标RGB/relative/预测边缘视频、总览、曲线和连续案例。不依赖VDA/RAFT权重，不修改第三方仓库。

实际先执行`./scripts/part1/run_dav2_video.sh check --output outputs/maintenance/part1b_course_20261102/resources.json`，确认CUDA可用、A6000、torch2.1.1+cu121、官方DA代码/权重和源视频身份。随后实际执行`./scripts/part1/run_dav2_video.sh run`，对2.917秒短片完成70帧预测/分析/原始重算；没有启动长时预测、训练、下载或包安装。完整日志保留于本轮维护目录。

正式当前产物为`outputs/part1/dav2_video/rollercoaster_v1/`，缓存`data/video_frames/part1b_davis_v1/`。70份360×640 float32 NPY、69个相邻帧对、完整源索引/PTS、实际518×924网络/FP16、run/protocol及源码归档齐全。平均21.434968071324483 ms/帧，46.652740357369204 FPS；峰值allocated/reserved为345,405,440/423,624,704 bytes，加载0.7489040461368859秒；准备输入到保存原生预测6.599679694976658秒，含预热和3次模型测量，不是单遍实时系统耗时。模型计时与解码/绘图区分。

原始预测不裁剪；整段P1/P99范围固定为[0,2.685546875]，TURBO、无逐帧归一化。16128000个像素中160584个只在显示上界饱和，原始数值未改。正仿射2D+5及相应显示范围的演示最大归一化差5.947459835731195e-08，只用于说明相对数值不能确定米制尺度，没有作为新预测写入。视频无GT，不虚构AbsRel/米制误差。后续2B的同RGB/PTS/画布、DA原始身份与固定显示约定保存在`analysis/handoff_2b.json`，`vda_executed:false`。

辅助帧对等权MAE/S、RMSE/S分别为同坐标0.09631663770266184/0.2370281232179769、Farneback对应0.012942993213330214/0.04071482743622585；光流有效比例0.9111152626811594，边缘变化proxy 0.0033902655239424736。它们定位可视候选，不作为时序GT、边界准确率或米制漂移，真实运动/遮挡/光流近似和天空大区域会影响解释。可选诊断没有修改模型预测。

实际查看全段总览及65–69、52–56、57–61连续帧的RGB/relative与局部放大：主体支架大轮廓总体连贯，运动列车、交叉细杆和显露区域有局部颜色/厚度/轮廓变化；不强行描述成全画面剧烈随机闪烁。三例来源于同一短片，不称独立多场景覆盖，没有视频GT也不把预测梯度叠加当作真实边界。只读放大图、帧号/PTS/框/SHA及实际复绘脚本保存在`review/`，原分析和自动元数据未改写。

依赖检查记录30份原有绑定科学源码/配置SHA；清理了6份未被任何保留run绑定的项目文件：`configs/part1/video_framewise.yaml`、`video_framewise_standalone.yaml`，`pipelines/part1/framewise_video.py`、`run_framewise_depth.py`，`scripts/part1/run_framewise_depth.sh`和`tests/test_framewise_video_cli.py`。删除前指纹与原因在本轮`cleanup.json`。`outputs/part1/video/`等现有2B冻结的绝对路径/源码依赖只读保留，不为清理1B破坏2B复核；原视频、缓存、NYUv2、Part 1A/2A/3预测、权重和第三方文件均未删除。通用数值测试改从共同配置构造显式仅DA夹具，不恢复重复入口或改变评价公式。

新增11项测试使用临时MP4/PNG/NPY和明确合成模型，覆盖24 FPS连续索引、短源不补帧、零跨度N/A/合法零、正仿射约定、已知平移、完整生命周期/原始重算、来源/raw/CSV/交接/帧数/尺寸篡改及不覆盖；未加载正式深度权重。专项11 passed。首次全项目为1 failed、1,525 passed：旧单模型夹具改用共同配置后同时含dav2/vda，已显式保留dav2以恢复原测试意图，没有改推理或评价去匹配数字。最终全项目为**1,526 passed、14个既有Matplotlib/PyParsing弃用告警、47.99秒**；两次真实日志分别保留`pytest_initial.log`和`pytest.log`。

清理后30份保护源码SHA全数不变，NYUv2仍有654份原生NPZ。代表性既有自录2B fixed_scenario1的122帧来源/raw/分析绑定只读校验通过，没有重算旧指标或执行VDA。Step 2重写为指定片段的协议、完整指令、实际结果、三项分析、架构与门禁，原NYUv2扩展审计描述保留其原执行含义；README、PLAN、plan_1B2B、AGENTS、相关数据/配置/源码/测试/输出索引同步，Step 4仅添加兼容范围说明。课程100帧、后续VDA比较、应用Demo/报告/ZIP仍未完成；不重写任何旧自动执行记录。

## 2026-11-03 — Part 1B诊断图与预测边缘说明

对照实际`diagnostics.png`、绘图与指标实现，在`docs/step2.md`补充四张子图的顺序、时间坐标、CSV字段、定义和阅读方式。只读聚合当前70行逐帧、69行逐帧对CSV：70个预测中位数均为0；第66→67帧（PTS 2.791667秒）的边缘变化最高，为0.008278967429034223；光流覆盖范围为0.8736805555555556–0.9548871527777778。说明零中位数不能证明无尺度漂移，图中仅画RMSE/S而非MAE/S，边缘变化与光流覆盖均不是准确率。

补充红色预测边缘由固定跨度归一化、Gaussian平滑及Sobel梯度阈值提取的过程，区分预测深度分界、物体轮廓与GT；说明叠加图没有人为加粗，2像素匹配容差不控制绘制线宽，红带粗细还受跳变幅度、过渡形状和插值等影响。README同步说明入口。本次只修改三份Markdown，不运行模型、重绘图像或改写任何原始预测、分析产物与自动元数据；文档链接及所引用CSV数值检查通过。

## 2026-11-03 — Part 2B：指定片段VDA离线相对深度与单独比较

重新读取新版PDF §5.2与§6.1–6.2，使用官方VDA-Small relative、offline、FP16，前100真实帧上限/原FPS/640画布/518输入；streaming及米制视频模式不进入本次入口。源视频实际70帧、24 FPS，内部窗口补齐不充当课程100帧。当前实施准备和资源/合成验证完成，**未运行指定片段的正式VDA预测或正式DA/VDA比较，不存在新的VDA效果分数**；正式预测命令交由用户启动。

新增`configs/part2/vda_video.yaml`、`scripts/part2/run_vda_video.sh`和`pipelines/part2/vda_video.py`，提供check/predict/run/analyze/verify，以及另行compare/verify-comparison。新增`src/evaluation/temporal_clip.py`独立调用已有官方VDA adapter，不要求DA预测；现有RGB缓存逐张校验后只读复用，缺少缓存时按连续原始帧创建。原生结果另存`outputs/part2/vda_video/rollercoaster_v1/`，整段float32`depths.npz`与逐帧NPY逐值核验；协议、科学源码归档、实际窗口/网格、帧序/PTS、加载和整段均摊计时、实际显存与环境随运行记录。VDA窗口保持官方32/10/22/8和预测间衔接，不修改第三方代码、权重或相对输出。

分析直接复用1B的`framewise_clip.measure`和`video_diagnostics`，保存同坐标/对应后MAE/S、RMSE/S、中位数/范围/边缘比例、边缘变化与光流覆盖、固定全段P1/P99参考及正仿射尺度约定演示。新增`src/visualization/temporal_clip.py`制作VDA独立RGB/relative/预测边缘视频、固定色标/饱和计数、四子图、自动连续候选及65–69、52–56、57–61同坐标局部观察；裁剪只影响展示。生成的分析说明不预写视觉改善，需正式运行后观看真实产物。

另行compare只读DA/VDA完整运行、分析及1B交接，核对源SHA、逐张RGB SHA、真实PTS/帧号、画布、实际网络输入/精度和指标/显示规则，保存于独立`outputs/part2/video_comparison/rollercoaster_v1/`。逐项比较仅使用双方共同可评价帧对，保存DA−VDA、降低率和共同数量，保留负值、零分母与N/A。输出同帧深度/预测边缘视频、四子图、固定三窗口和最大/最小差候选；不把变化较小、原生中位数或颜色解释为真实尺度/边界准确率。verify-comparison从双方raw重算全部数值；不启动1B、不改写两条输入。VDA整段均摊成本与DA单帧时间明确区分，VDA整段分位数不称逐帧延迟分布。

实际执行`./scripts/part2/run_vda_video.sh check --output outputs/maintenance/part2b_course_20261103/resources.json`：源/relative权重SHA、VDA官方commit与干净工作树通过；aiaa3201、torch2.1.1+cu121、CUDA可用、RTX A6000。没有下载、安装包、模型推理或训练。随后只读运行1B的verify，原有70帧/69对完整raw重算通过，记录另存本轮`part1b_recheck.json`与日志，原verification未覆盖。

新增15项`test_course_part2b.py`合成验证，覆盖独立VDA无需DA、缓存和双方输出只读、真实小文件/NPZ、官方窗口记录、固定色标MP4/四子图/局部图、同协议比较、负降低率、零跨度/N/A、模式限制、失败保留、不覆盖，以及raw/NPZ/RGB/配置/窗口/CSV篡改拒绝；重新封装CSV的SHA仍无法通过原生重算。首轮新旧专项24 passed；补充负结果及比较CSV重封装检查后，最终全项目为**1,541 passed、14个既有Matplotlib/PyParsing弃用告警、57.40秒**，真实日志为`pytest_targeted_initial.log`与`pytest_full.log`。测试使用合成模型，不作为VDA实际效果。

Step 4重写为两个主板块：第一部分独立描述VDA任务/协议/完整指令/产物/诊断，第二部分再引入1B作同帧比较。README、PLAN、plan_1B2B及相关配置/流水线/输出/测试/源码/文档索引和当前结果状态同步；差异分析新增本次进度，不改写最初审查与旧执行历史。核验170个本地文档链接、Step 4的6段Bash示例语法和独立CLI帮助通过；另将12条文档VDA命令交给真实CLI解析并替换工作函数，确认参数接受，不运行模型或写入运行产物。

清理审计确认上轮删除的6份无绑定旧1B重复文件仍不存在；剩余旧1B/2B入口、共同源码和配置由旧自录run或公共回归引用，本轮没有再删除必要依赖。68份既有科学源码/配置及当前1B运行/分析记录SHA全部不变，见`preserved_before.json`与`preserved_after.json`。原视频、缓存、NYUv2、静态结果、旧自录原生预测和自动元数据均未修改；两个新正式输出目录尚不存在。100帧素材确认、正式VDA/比较及人工结果分析、应用Demo/报告/ZIP仍需完成。

## 2026-11-03 — 按用户要求清理旧自录2B

用户明确要求删除旧自录2B后，清理`outputs/part2/video/`中7段自录VDA及比较产物。同步删除11份不再使用的专用项目文件：`scripts/part1/video.sh`、`pipelines/part1/video_clip.py`、`scripts/part2/video.sh`、`pipelines/part2/video_clip.py`、`pipelines/video/clip.py`、`src/video/clip_workflow.py`、`clip_analysis.py`、`clip_report.py`、`configs/video/self_recorded.yaml`、`configs/part2/video_depth_anything.yaml`和`tests/test_video_clip_workflow.py`。这些文件构成旧自录工作流，当前指定片段1B/2B不依赖它们；公共视频组件、官方adapter及新入口均保留。

删除前逐文件记录SHA256、字节和源文件快照，清单位于`outputs/maintenance/self_recorded_part2b_cleanup_20261103/deletion_manifest.json`。本次共删除3,034份文件、3,310,712,339字节，约3.08334 GiB；其中3,023份为旧2B产物，11份为上述项目文件。保留原始自录视频、RGB缓存、`clips.yaml`来源登记及旧1B原生结果作归档，不将旧1B视为仍有运行入口的维护轨道。当前指定片段、NYUv2和其他Part原始预测不变；不改写已发生的执行历史或剩余自动元数据。

公共配置测试改为核对当前VDA独立offline relative协议，以及和DA相同的源/原FPS/前100帧上限/640画布/FP16/诊断与显示规则。移除旧工作流对应的36项回归后，全项目实际为**1,505 passed、14个既有Matplotlib/PyParsing弃用告警、48.47秒**，日志保存于本轮`pytest.log`。另只读运行当前1B verify，70帧/69对原始重算通过；181份当前输入、raw、分析及科学源码SHA与删除前一致。没有推理、训练、下载或第三方改动。

AGENTS第11条、README、PLAN、Step 2/4、视频配置/源码/输出/测试索引及相关说明更新为当前状态；旧上传指南标注其自录代码包已过期，避免用旧包覆盖当前实现。差异审计保留原审计历史，另补充删除说明；链接检查发现1A/2A上传指南的4个旧相对链接错误并修正。原始素材说明改为来源归档，不要求继续采集自录视频。课程指定素材仅70帧及正式VDA待运行等门禁不因清理而改变。

## 2026-11-03 — Part 2B第一部分：VDA独立结果分析

用户完成指定片段VDA独立实验后，本轮只读检查`outputs/part2/vda_video/rollercoaster_v1/`，没有重新预测、训练、重绘原分析或运行第二部分配对比较。70份360×640非负有限float32 NPY与整段NPZ逐值一致；run、协议、官方代码/权重、同源RGB/PTS、窗口与图视频/CSV/summary附件完整。实际执行`./scripts/part2/run_vda_video.sh verify --run outputs/part2/vda_video/rollercoaster_v1 --output outputs/maintenance/part2b_standalone_review_20261103/recheck.json`，70帧/69对原始重算通过，rtol=1e-6、atol=1e-8，原verification未覆盖。

完整精度来自原`analysis/summary.json`，SHA为`b55b4af78227b8649adc0e7384f026bc6a5df7154f00f65d6fe8d9872d06334c`；CSV独立等权聚合一致。同坐标MAE/S、RMSE/S为0.09746970822436357、0.24548739393117433；对应后为0.010502717301660868、0.039509406605546166；光流有效覆盖0.9111152626811594，边缘变化proxy 0.0034054039117231223，六项均有69个可评价帧对。固定跨度5.251484932899474，原始范围[0,7.399656295776367]；仅显示上界饱和161,280/16,128,000像素，即1%，不改原生值或评分域。

实际环境aiaa3201、Python3.10.22、torch2.1.1+cu121、RTX A6000/FP16，四个518×924窗口恢复360×640；原70帧、24 FPS，官方内部补齐28帧后裁回真实帧数。原计时平均整段2.052508618837843秒，均摊29.321551697683475 ms/真实帧、34.104607092775524 FPS，allocated/reserved为2689.2197265625/3138 MiB；参数29,035,137、权重116,440,756字节。加载0.71959113702178秒，预测阶段9.908725869841874秒含完整预热、三遍计时与保存，不称单遍或streaming时延。

实际查看全段总览、诊断图、65–69/52–56/57–61连续局部图与26–30/65–69自动候选：主体轨道和支架轮廓连贯，车体/乘客及交叉遮挡处仍有粗色块与细节合并，远处旗杆和薄结构较弱；不强行判为全画面严重闪烁，也不将平滑称为准确。边缘变化最高在27→28帧（PTS1.166667秒），0.007172854349193791；对应后RMSE最高在66→67帧，0.04847949901472186。真实运动/显露、光流近似和2像素匹配容差均单独解释，没有视频GT，不给出绝对深度准确率或拖影/滞后量化结论。

从raw核对，全部70帧中位数等于该帧最小值，每帧67.6315%–76.0516%的像素落在低值平台。中位数由0分段升到0.010361052118241787、0.018883058801293373，最后0.019235486164689064；变化段与官方输出24–31、46–53、68–75混合区间一致，最后只保留68–69。依据大面积背景平台和官方scale/shift/插值解释数值约定，不将首末差/S=0.0036628661055814324写成真实尺度漂移；每窗拟合系数未单独保存，不虚构具体scale/shift。补充描述统计、视频70帧/24 FPS完整解码、来源SHA及实际只读脚本在本轮`results_review.json`、`review_stats.py`和日志中。

Step 4第一部分新增实际效率/六项指标、固定色标/饱和、三项分析及三个连续困难窗口；首次指令保留，明确默认目录已完成、不要再次覆盖运行。第二部分仅标注暂未运行、表格待配对核验，没有填入双方收益或创建比较目录。README、PLAN、plan_1B2B、当前结果与相关状态索引同步独立完成状态，不把整个Part 2B或100帧素材门禁标为完成。109份原运行/分析/预测与科学源码保护SHA保持一致；本轮仅结果复核和说明更新，不重复无变更代码的全项目测试，最近完整测试仍为清理后的1,505 passed。

## 2026-11-03 — Part 2B第二部分：同帧DA/VDA比较分析

用户完成第二部分指令后，本轮只读检查`outputs/part2/video_comparison/rollercoaster_v1/`及两条绑定原生预测。比较summary SHA为`8497e93fc3dfc16d84d7946c80a47f52b54d6975d7b1db4712ad37cbfdc2446c`；双方run和独立analysis身份与当前1B/VDA一致。实际再次执行`./scripts/part2/run_vda_video.sh verify-comparison --comparison outputs/part2/video_comparison/rollercoaster_v1 --output outputs/maintenance/part2b_comparison_review_20261103/recheck.json`，70帧/69对、双方raw、CSV/汇总与附件完整重算通过，rtol=1e-6、atol=1e-8；没有重新预测、训练、下载或覆盖原verification。

本次完整精度的DA−VDA差值/降低率为：同坐标MAE/S −0.0011530705217017295 / −1.1971665012449484%，同坐标RMSE/S −0.008459270713197437 / −3.5688890408241067%；对应后MAE/S +0.0024402759116693463 / +18.854030682454994%，对应后RMSE/S +0.0012054208306796835 / +2.9606433493247852%；边缘变化proxy −0.000015138387780648781 / −0.44652513715340636%。五项均有69个共同可评价帧对，无N/A或按分数排除；双方光流覆盖均为0.9111152626811594。逐帧范围/中位数/边缘比例及各项均值与两条独立CSV/summary逐项一致。

独立聚合比较CSV得到VDA较低/较高/相同帧对数：同坐标MAE为32/37/0、RMSE为19/50/0，对应后MAE为69/0/0、RMSE为49/20/0，边缘变化为37/32/0。因此收益主要是对应后平均变化降低，不称18.85%的纯闪烁已消除；较大残余变化的改善有限，整体边缘近乎持平且略差。边缘均值绝对增加0.0015138387780648781个百分点，不将其写成GT边界准确率或显著退化；帧对相关，不作为独立多场景显著性。

实际查看双方诊断图、52–56/57–61/65–69固定连续局部图及27–31/1–5/35–39自动正负候选；从现有预测边缘视频只读导出2/3、28/29、36/37、66/67帧供观察，维护图不改正式视频。主体结构与多数轮廓接近，遮挡处车体/乘客及杆件仍合并，细杆和旗杆问题没有全面解决。最大对应后RMSE改善28→29为0.04225216136654611→0.03578321320935884；最大退化2→3为0.03766651597529336→0.04179528876610432。边缘最大改善66→67为0.008278967429034223→0.005147789157621041；最大退化36→37为0.00446337707928357→0.006623872544219034。帧对指标是全图有效域，不虚构某个物体的单独误差贡献；固定三窗口与正负候选均保留。

DA原生中位数全0、VDA窗口相关平台/缓升不作为真实尺度漂移排名。两模型各自固定P1/P99跨度2.685546875与5.251484932899474，同TURBO不代表共同米制尺度，无视频深度/时序GT时不声称米制准确率或真实漂移改善。两个正式比较视频均完整解码70帧/24 FPS。

实际A6000/FP16、518×924网络/360×640输出下，DA/VDA均摊21.434968071324483/29.321551697683475 ms/真实帧，增加36.79307382272052%；FPS46.652740357369204/34.104607092775524，降低26.89688358812902%。峰值allocated329.404296875/2689.2197265625 MiB，约8.163887818327355倍；reserved404/3138 MiB。两条记录分别在11月2日和3日执行，不称同一时刻受控负载或相同在线延迟分布。

Step 4第二部分填入实际表格、逐对计数、三项通俗分析、成本、局部正负案例和边界条件，完整运行指令保留并注明默认比较目录已完成，不重复覆盖运行。README、PLAN、AGENTS、plan_1B2B及相关状态索引/当前结果同步为当前70帧独立与配对工作完成；Step 2仅更新跨Part比较门禁。222份原生预测、运行/报告及科学源码保护SHA一致；补充统计/实际脚本和日志保存在`outputs/maintenance/part2b_comparison_review_20261103/`。本轮不改科学实现，不重复全项目测试，最近完整测试仍为1,505 passed。课程100真实帧素材确认、应用Demo、报告与最终打包继续保留独立门禁。

## 2026-11-03 — Part 1/2 GitHub完整分支替换指南

新增中文`docs/github_upload_part1_part2.md`，提供工作站白名单导出、传回本地电脑、HTTPS克隆指定分支、删除旧工作区文件（保留.git）、复制发布快照、暂存文件集合与SHA256逐项核对、一次提交与普通push、远端核验和失败处理。示例沿用此前仓库的JiongxuChen分支，地址与目标分支需用户执行时确认；不是本次实际远端操作。README新增入口。

导出范围为Part 1/2代码、公共依赖、29份相关测试、两套环境与配置、Step 0–4和当前课程说明；不包含data、checkpoints、outputs、官方仓库副本、队友克隆或Part 3实现。发布副本生成专用README/实施范围摘要，未上传outputs链接转成本地产物路径，修正Step 0旧视频状态及未随包提供的审计链接；源文档、原始预测与自动元数据不改写。保留现有70帧实验已完成、课程100帧素材待确认的边界。

在`/tmp/part12-guide-final-d2a0mjd1/`实际验证导出128份文件加1份清单；129份Git暂存内容与SHA256全部匹配，87份Python AST、12份脚本Bash语法、指南Bash示例和发布包21处本地Markdown链接通过。临时Git仓库预置普通旧文件与隐藏旧文件，完整替换后两者均删除，提交并推送到临时本地bare仓库成功；没有操作GitHub。另为Git Bash提供cygpath路径转换以供Windows Python读取环境变量；Windows实机执行未测试。没有训练、预测、下载或修改科学实现，不重复运行无变更的全项目实验测试。

## 2026-11-03 — 补全GitHub发布包README的复现准备说明

按用户补充要求，扩展`docs/github_upload_part1_part2.md`内的README生成模板。发布分支首页现在按准备顺序说明Linux/Bash、Conda、CUDA与两个隔离环境、四个官方仓库及固定版本、Middlebury/NYUv2/split/指定视频的准确放置路径、四模型权重文件（含UniDepth的config.json）、资源SHA核对、自动生成目录，以及两条视频预测先于配对比较、两条完整米制基线先于联合分析的依赖。具体实验命令继续引用Step 1–4；注明新克隆仓库没有历史结果、不能直接从verify开始，以及使用新基线目录时要同步分析配置。不上传视频、数据、权重或outputs的范围不变。根README同步入口说明。

重新执行指南的导出及临时Git目录替换：128份发布文件加1份清单，129份暂存文件SHA全部一致，旧文件删除通过；27处本地Markdown文件链接和全部Bash示例语法检查通过。临时验证包位于`/tmp/part12-readme-check-rnrshjqb/`。仅修改文档，未运行模型、下载或操作GitHub；Windows/WSL2未作实机复现验证，已在发布README中说明。

## 2026-11-04 — 队友对Part 1/2分支补充的审查

新增`github_repository/analysis.md`，只读审查重新克隆的JiongxuChen分支HEAD `548420f8e1d36f7b7663e3552d0ccca6e8fd7ea7`，以用户上传后的`9aaa869`为比较基点。两个队友提交`8b42ab3`/`548420f`共32文件变化、15,014行新增，主要为1A覆盖率取舍及2A独立基线/资源证据，没有改动原1B/2B主线或模型适配器。文档解释新规则、实际数据、正负结果、材料缺漏和融合顺序；README补充入口。

实际核验25份证据文件SHA全部一致，两份历史参考CSV的解析内容与本地当前654张基线完全一致；重算两模型四指标的逐图差异和汇总。使用aiaa3201运行新增逐图审计入口两次，summary与附件完全一致，临时文件保存在`/tmp/teammate-review-20261104-gnxi3iqd/`。另从182行开发表复核6组选择，300行保留场景表复核全部汇总，两模型各96行资源表的数量/均值一致，4份新增Python源码语法解析通过。未进行模型预测、匹配搜索、训练或下载，不将表格核验称为完整推理复现。

发现资源记录指向本克隆不可得的`a717683`和main上的原始执行脚本，环境附件中的两模型适配器SHA与当前文件不符，CRLF/LF统一后仍不符；third_party路径也与本项目根目录上游约定不一致。明确其来源缺口，不认定结果错误，不替换本地主基线或编造新运行历史。UniDepth单图test_0139_mat_0351的δ1差0.09780767940979207需双方数组进一步追查；当前仅指标表不能归因。指出删除中文说明/发布清单属于用户上传后的提交，避免错归队友，并提示docs/evidence包含轻量运行结果，与纯代码上传偏好需要统一。没有修改克隆仓库、第三方实现、数据、权重、原生预测或自动元数据，未执行融合；现有70帧与100帧课程素材门禁不变。

## 2026-11-04 — Part 1A误差—有效比例研究的本地实现与复现

在现有架构增加`src/analysis/stereo_coverage.py`、`pipelines/part1/study_middlebury_coverage.py`、`configs/part1/middlebury_coverage.yaml`和`scripts/part1/study_middlebury_coverage.sh`。分析/选择与任务编排分开，复用原1A的measure、匹配、几何、六联图/掩码保存和原始数组重算；未复制队友的通用src/io或改动主四组科学源码。主协议、tuned选参和旧产物保持独立，不为匹配参考数值修改预测。

实际执行`./scripts/part1/study_middlebury_coverage.sh --reference-dir github_repository/monocular-depth-video-consistency/docs/evidence/controlled_studies/stereo`。使用本地Middlebury的固定5/10场景与144/36网格，各额外计一次fixed，共182行开发表、910行逐场景分数；冻结覆盖率±1个百分点时最小MAE、MAE不高于fixed时最大覆盖两条规则后，再加载10个独立场景。6组实际参数与参考相同，独立输出`outputs/part1/middlebury/coverage_study_v1/`含60份原始左右视差/米制深度/布尔状态NPZ、标定/参数、六联图/原尺寸掩码、300行五区域表、完整像素计数/Bad指标、Pareto图及源码/配置/输入指纹和快照。参考副本在本地结果生成后才复制并核对，不用于选参。

首次原生核验通过，另实际执行`./scripts/part1/study_middlebury_coverage.sh --mode verify --report outputs/maintenance/part1a_coverage_20261104/recheck.json`：60份预测、1,200个精度值、300行区域统计、910行开发分数与选参重放通过；从raw重做LR/有效域/米制转换，空GT与无有效预测区分且不静默丢弃失败。参考182行开发表与300行评估表共964个数值在rtol=1e-6、atol=1e-8内一致，沿用本地float64评分，允许参考float32造成的微小差异。两份正式运行/核验元数据均保持真实，新的verify默认只打印或另存报告，不覆盖首次verification。

本地十场景全GT宏平均MAE/有效比例依次为：BM fixed `1.0452621248185863 / 0.49084447426807964`，matched_coverage `0.768626937376369 / 0.46188785671262966`，matched_error `0.8548859233976142 / 0.4755623182048505`；SGBM fixed `1.2300677081942617 / 0.6271337989379588`，matched_coverage `1.160354001911018 / 0.6094821729799482`，matched_error `1.4371931074173774 / 0.6383245401033675`。BM条件误差降低但覆盖损失、Bad-2上升；SGBM matched_error增加覆盖却提高MAE。遮挡覆盖仅约12%–19%，低纹理与边界区域也有取舍；不宣称开发约束已泛化或匹配困难已经解决。

实际环境aiaa3201、Python3.10.22、NumPy1.24.0、OpenCV4.11.0、Intel Core Ultra9 285K、OpenCV单线程，仅CPU经典非学习式搜索/评估，不涉及训练、模型预测或下载。原主运行/搜索及相关科学文件共501份SHA保留核对一致，克隆分支工作区仍干净。原始新运行元数据SHA `d76457b8b8f0a644699bada6ddcdaeccafafb4d0fb282693b7c699053071e90a`，汇总SHA `06f11c9dd02b51911bff552eb6d14f0e17517cc017ea88356443eaa83221efd2`；真实日志/保护清单/审查统计在`outputs/maintenance/part1a_coverage_20261104/`。

新增10项合成回归通过，覆盖开发/评估顺序、两条规则、稳定平局、非法/无定义候选、空域/缺失、60份预测重算、禁止覆盖、SHA重封装后错误区域身份及原始NPZ篡改拒绝。全项目实际`1,515 passed、14个既有Matplotlib/PyParsing弃用告警、51.82秒`。实际查看SGBM Pareto图，Step 1与指南Bash语法、本地正文链接检查通过。发布指南白名单加入新测试，重新导出确认133份发布文件加1份清单包含全部新实现/配置/入口/测试，临时包在`/tmp/part1a-coverage-export-oze8srlh/`；未操作GitHub。

Step 1按研究问题自然组织规则、完整命令、本地结果、区域分析、图的读法及目录；README、PLAN、当前结果、配置/源码/流水线/测试/文档索引同步。github_repository分析另记1A本地复现进展，上轮2A资源来源缺口仍保留；本次不涉及2A资源或1B/2B更改。没有重写既有执行历史，最终报告与其他Part门禁不变。

## 2026-11-04 — Part 3 B/C清理、接入与小型验证

按用户明确指令清理旧Part 3的项目自有算法、流水线、脚本、配置、测试、图文和运行产物：删除src/refinement旧实现、pipelines/part3、scripts/part3、configs/part3、docs/step5.md、方向专用图、分割环境说明及依赖、outputs/part3、checkpoints/part3和旧分割路线专属checkpoints/segmentation；第三方SAM2仓库和现有Conda环境不做卸载或改写。删除前逐文件记录54,067份文件、45,903,275,493字节（约42.75076 GiB）的SHA/字节，见`outputs/maintenance/part3_bc_rebuild_20261104/deletion_manifest.json`。调整前完整implemented/PLAN仅保留为该目录下的历史审计文本，不重写成未发生的执行，现行说明不再介绍旧研究版本。

从main HEAD `528e33e`接入固定B/C。`src/refinement/edge_aware.py`为main 5×5 RGB/log-depth联合引导逆深度滤波的LF逐字副本，CPU/CUDA保留FP32原公式；`ablation.py`补齐同设置的CPU去RGB/去深度门控与后端选择。`global_scale.py`实现每图有效crop中位log(GT/pred)，再取官方795图中位数并指数化的全局尺度。来源commit/原始与LF SHA在ORIGIN.json，MIT许可及Fei Gao版权保留在MAIN_LICENSE.txt。没有改第三方源码或队友克隆。

新架构为`pipelines/part3/common.py`、`direction_b.py`、`direction_c.py`，对应两份YAML与两份Conda shell入口。读取当前本地654张native baseline及其metadata/逐图表，沿用标准split、crop、有效深度与预测裁剪、宏平均和SI实现；RAW评分与原Part 2冻结表逐图核对。资源路径使用根目录约定，UniDepth的795图开发推理由shell自动选择aiaa3201-unidepth，其他评价使用aiaa3201。B预测、拟合/冻结、评价分阶段不可覆盖，保存795图开发数组与来源；小子集不能冻结，B结果明确native_metric:false及开发GT校准来源，不替换Part 2原生结果。C独立运行RAW/C/两个固定消融，保存三组预测、scene配对统计和事后案例，不选参、迁移或组合B+C。verify默认只打印/显式另存报告，从父运行/原数组/GT重算，不覆盖原运行记录。

实际通过Depth Pro的B资源预检与UniDepth的C资源预检：数据/权重SHA、官方795/654无交集、上游commit和两条654份native baseline身份均通过。实际运行UniDepth开发命令的--help，验证独立环境的CLI导入与路由，不加载模型。B的真实795图推理、尺度冻结和654图评价均未启动；main数值只作参考，不写为本地新结果。

专项`tests/test_part3_bc.py`11项通过：main CPU代码指纹/逐值一致、孔洞保留、完整795图尺度公式/非法输入与小子集拒绝、合成795开发→冻结→小子集评价/原始重算、开发身份篡改和禁止覆盖、C完整/两个消融/scene统计及外层SHA重新封装后的数组篡改拒绝。CUDA小数组测试先检查实际可用性，三个CPU/CUDA方法均通过rtol=1e-6、atol=1e-8对照；没有推理基础模型或训练。

另分别对Depth Pro/UniDepth各2张真实图运行CPU C与两个消融，并保存6份预测及GT分数，各自verify重算通过；目录为本轮维护目录的`c_depth_pro_final/`、`c_unidepth_final/`，独立报告为各`*_recheck.json`。均明确`formal_complete:false`，不把这4张检查写为654图结果。开发过程首次两图检查记录也保留在维护区，原记录不改为最终源码版本。

清理旧方向测试后的当前全项目实际为**338 passed、14个既有Matplotlib/PyParsing弃用告警、25.27秒**，不是原1,515项集合的漏跑；旧方向专用测试已随实现清理。55处现行说明本地链接、B/C shell及文档Bash示例语法检查通过。实际指令/日志、删除清单、输入保护与review.json都保存在本轮维护目录。formal_b_started/formal_c_started均为false。

README、PLAN、AGENTS、文档/源码/配置/测试/输出/流水线索引及current_results整理为B/C；新增中文docs/stepB.md、docs/stepC.md，给出完整首次运行、核验、独立重跑、产物字段、科学区分和待完成状态。Step 6修正为应用/报告/实际提交独立门禁及arXiv可选，不再要求旧方向迁移实验或虚构mandatory素材来源。main审查中的主项目对接说明同步当前状态。Part 1/2、课程原始资源与原始运行元数据不改；正式运行交用户后，才能据本地完整结果填写B/C主表。

## 2026-11-04 — 用户正式运行后：B/C结果分析与只读复核

用户反馈Step B/C全部指令运行结束。本次读取当前正式输出完整精度，不重新运行基础模型、不训练、不下载，不修改科学代码/配置、第三方仓库、原预测或自动元数据。两模型B均有795份开发预测、795行校准表、一个冻结尺度、654份B测试预测和1308行RAW/B分数；C各有2616行四方法分数、三种方法各654份预测、215场景统计和三个案例。八个正式阶段均为complete、formal_complete:true；B冻结时间早于相应测试评价。

只读脚本/完整精度报告/保护指纹在`outputs/maintenance/part3_bc_results_20261104/`。实际执行`conda run --no-capture-output -n aiaa3201 python outputs/maintenance/part3_bc_results_20261104/review.py`：9,673份去重来源/输入/产物SHA与运行记录一致，科学源码匹配；795行开发身份/预测指纹、开发/测试无交集、从保存log比例表复算尺度、全部654图方法身份/预测覆盖、RAW与冻结Part 2及宏平均均通过；重新计算10,000次scene bootstrap和案例选择一致。对每模型三个自动案例共六张实际图，CPU重做C/两消融共18份数组，从保存数组和GT重算八指标，并核对六份B数组为RAW×scale，均在rtol=1e-6、atol=1e-8内通过。另查看两模型各一张最高收益与最大退化四面板，未凭缩略图将指标变化虚构成局部机制证据。

B本地尺度为Depth Pro `1.0413853532199127`、UniDepth `0.9562419672512901`；测试AbsRel为`0.09660522364809511 / 0.10203673364140407`，相对RAW降低`3.8482025928633354% / 10.918970404155575%`；RMSE为`0.38033847184009667 / 0.3892837235816476`。分析保留259/243张AbsRel恶化和UniDepth的δ2/δ3均值略降，说明开发校准、同数据集泛化与原生零样本的区别，不作边界/B+C/跨域保证。

C本地完整法AbsRel为`0.1003652890283648 / 0.11442974504564071`，F1为`0.17111979364956253 / 0.12257543890814086`，相对RAW F1提高`7.437313719808669% / 8.466739951326385%`，米制收益很小。完整C边界476/440张改善、176/210张恶化、2/4张持平；scene区间与退化案例完整报告。去RGB的F1为`0.13702177190084192 / 0.09354581224491552`，均退化。去深度门控为`0.16971214681034838 / 0.1341396336526837`，UniDepth高于完整C且七项米制均值也较好；不声称完整法在所有模型/指标都最好，不据test另选设置。没有单独量化纹理复制或细杆恢复，不将总体F1收益写成这些问题已解决。

C真实运行记录为RTX A6000、PyTorch2.1.1+cu121、FP32；完整法平均后处理`16.681415901094994 / 16.623332631554945 ms/图`，峰值allocated `285732864 bytes`。文档区分传输/同步与模型/IO/评分，不报端到端FPS或系统总显存。

Step B/C补充正式七/八指标表、冻结尺度、直观收益解释、逐图改善/退化、消融、215场景区间、实际案例和成本，保留完整首次运行/核验/独立重跑指令。README、PLAN、docs索引/current_results、Step 6当前结果状态与outputs说明同步。用户已执行的verify默认只打印，目录没有保存全量重放报告；本轮未重复795图GT比例/全部654图C CPU与GT重放，明确将用户反馈、产物完整性和本轮抽查分开，不写成独立全量重放门禁通过。需要可审阅报告时只需verify加--report，正式预测无需重跑。Demo、最终报告/提交及70/100帧素材门禁保持独立。此次仅文档与只读分析，未重复运行全项目测试。

文档检查通过：9份Markdown、68处本地链接、14个Bash代码块语法；改文档后再次读取9,673份保护文件，SHA全部不变，报告为本轮维护目录`documentation_checks.json`。另从真实created_at_utc确认两模型均开发推理→冻结→测试评价有序；未以文档日期替代自动执行时间。

## 2026-11-06 — B/C改进空间审查

按用户请求审查课程PDF原文、AGENTS/PLAN、Step B/C、B拟合/评价实现、C CPU/CUDA权重公式、边界指标和Part 2A可复用分析工具。原文B要求开发集一个全局尺度、C要求改善不连续边界并避免纹理复制，故审查区分总体指标、边界机制证据及实验可靠性。未更改算法、配置、参数或任何正式预测/自动元数据；未启动训练、推理、参数搜索或大规模评价。额外查阅He等Guided Image Filtering作者资料（https://people.csail.mit.edu/kaiming/eccv10/index.html），仅作为可补充的经典滤波对照来源，不宣称其在本数据上优于现方法。

实际执行`conda run --no-capture-output -n aiaa3201 python outputs/maintenance/part3_bc_opportunities_20261106/review.py`，只读数据场景引用、split及B/C逐图CSV。官方开发795图对应249场景、测试654图对应215场景，场景交集为0。开发图逐图GT/预测中位比例的P10/P90分别为Depth Pro `0.9321227310040486 / 1.1744068146297069`、UniDepth `0.8535597604932231 / 1.0761506204215643`，显示同模型图间的尺度差异，一个全局数值无法逐图同时匹配。这是诊断比例，不作为测试逐图校准方法。

保持215场景、10,000次重采样、seed42，补算完整C减去C_no_depth_gate的探索性配对差值：Depth Pro的F1均值差`0.0014076468392141715`，95%区间`[-0.00041709880096512243, 0.003220161491539894]`；UniDepth均值差`-0.01156419474454283`，区间`[-0.013914640431275464, -0.00924786866071834]`。前者不能据此确认门控带来稳定边界收益，后者在此划分上支持完整法边界较差；不作跨域或未校正多重比较的强结论。脚本、完整指标和输入SHA保存在同目录review.json，原CSV不改写。

建议优先补齐C的边缘precision/recall、定位、GT固定剖面宽度与平坦GT/强RGB纹理区域的伪边缘诊断，复用2A工具并保留无定义/失败覆盖；补充RAW/简单平滑/经典RGB引导/现C的受控对照。若开展新方法，只在官方795图内按scene划分开发/验证，两模型同实现、同候选与规则而分别选参，研究门控强弱和处理强度；不将测试上有利的消融直接选作新最终法。B可比较少量仍输出一个全局尺度的开发集估计规则，补尺度稳定性、开发样本量和分场景/深度效果；不改成逐测试图校准。全部为后续建议，未开展或预报收益，也未自动组合B+C。

已查看过654图测试结果，新增验证流程不能抹掉这一事实。后续应将795图中的场景留出用于选择与确认，冻结后在654图重评时记录为后续探索，不宣称测试从未被查看。README同步审查状态，当前PLAN/Step B/C的固定方法与正式结果保持不变；本轮未因只读统计重复运行全项目测试。

## 2026-11-06 — B三尺度规则与C经典对照/诊断实现

按用户新指令保留原B两层中位log规则、C固定法/两消融与全部正式产物，增加两种全局尺度规则和两个经典对照、定位/宽度/纹理诊断。未开展门控参数搜索、B+C、重新推理、下载、模型训练或全量补充评价。此前冻结方法的测试评价本身正常；补充设计受到已有测试分析启发，因此记录为后续探索，不把654图包装成从未查看的新确认集。本轮三规则全部报告，不根据test成绩挑赢家；未来若选择参数仍须在795图内按scene开发/验证。

新增独立`src/part3_studies/`（estimators/classical/diagnostics）、`pipelines/part3_studies/`（common及B/C）、两份`configs/part3_studies/`YAML、两份可执行`scripts/part3_studies/`Conda入口及11项回归。原`src/refinement/`、`pipelines/part3/`、脚本/冻结YAML未改，避免向原动态源码清单添加文件而使旧输出核验失效。新来源记录包含冻结实现和扩展实现，保存父运行、输入/产物SHA、配置/源码快照、真实命令/环境及test_results_previously_inspected:true；新旧输出均拒绝覆盖。

B复用真实795份开发预测：`B_median_log`重现原尺度；`B_mean_log`取每图mean(log(GT/pred))再等权平均指数化；`B_absrel`对GT/pred以(pred/GT)/本图有效像素数加权取中位数，精确优化未裁剪的图像宏平均AbsRel。三个目标都不裁剪开发预测、不用测试GT拟合。正式评分依旧clip，明确不将未裁剪最优性写成课程裁剪指标/测试最优性。全局像素排序的内存需求在Step B中标为估计，完整fit/verify/evaluate交用户。新增输出名estimators_fit_v1与estimators_evaluation_v1，分别795行开发统计/三尺度与654图四方法表/三组数组；scene配对区间相对RAW，全量尺度与测试数组重算分别核验并另存报告。

C复用原RAW/C/两个消融数组，固定增加`Gaussian_log`（5×5，σ1.5）和`RGB_guided_log`（彩色He局部线性模型，r2/ε.01，RGB[0,1]）。二者在log-depth域处理再exp，CPU float64计算/float32输出，不需ximgproc或新包；不读GT作为处理输入。与原C逆深度域及Guided的有效9×9支撑差异明确披露，作为经典方法对照而非组件消融。作者资料https://people.csail.mit.edu/kaiming/eccv10/index.html。

六方法共用2A的三深度比/三容差P/R/F1、物体标注支持、定位/漏边/伪边；共同GT/标注剖面的10–90宽度、位置和对比/顺序状态，N/A与覆盖保留，宽度变化只在双方共同可测剖面比较。纹理区域由GT 7×7局部比≤1.02、有效域/边界排除与RGB固定Sobel阈值定义，低纹理对照同列；统计掩码数量、预测边缘比例/误差和scene配对区间。全量新增diagnostics_v1将有每模型六方法3924行、两组经典预测、五类CSV、固定共同位置图和源码/来源；尚未运行。CPU经典计时与原CUDA C记录保留后端，不能直接推断算法速度优劣或端到端FPS。

实际验证：首次编译发现B写JSON语句多余右括号，已修正；最终专项`tests/test_part3_studies.py tests/test_part3_bc.py`为22 passed、3.35秒。新增11项覆盖不同像素数下图像等权AbsRel目标、log目标差异、完整795份合成开发拟合/重算、Guided的独立逐窗口回归/常量解、纹理空域/N/A及伪边注入、固定GT剖面对模糊响应、共同可测失败计数、来源集合不变、不可覆盖、六方法数组/GT重算及外层SHA重封装后的错误数组拒绝。全项目实际349 passed、14个既有Matplotlib/PyParsing告警、26.41秒；没有用合成795图作为正式B结果。

实际运行Depth Pro新B check通过，核对资源、完整795开发/原尺度与654基线，不加载模型。新C分别对Depth Pro/UniDepth首1张真实图运行并verify通过，每份含六方法分数、两个经典数组、五类诊断表及三种图，均formal_complete:false。实际查看Depth Pro的边缘叠图、共同GT剖面与纹理掩码图；两模型各3条GT剖面一致供六方法使用。小样本和报告在`outputs/maintenance/part3_studies_build_20261106/c_{depth_pro,unidepth}_smoke/`及对应`*_recheck.json`，不报其分数为正式改进。

README、PLAN、AGENTS、Step B/C、当前结果和文档/配置/源码/流水线/测试/输出索引同步新增结构与待用户状态。Step B/C后半部分提供完整check→fit/run→verify→evaluate→verify命令，显式--report留存且拒绝覆盖；补充运行无需重做基础模型预测。原正式结果段与历史执行记录保留。正式长时运行全部交用户，尚不能宣称新增方法改善了指标。

最终只读保护/文档检查通过：14份Markdown、80处本地链接、23个Bash代码块及两个可执行shell语法有效，9,695份去重受保护源码/原产物/输入文件SHA保持一致。报告为维护目录`final_checks.json`；正式B/C补充、训练及新基础模型推理启动标志均为false。原B/C来源核验仍可使用原入口，未改写旧执行记录。

## 2026-11-06 — 用户完整运行后三尺度B与六方法C结果审计

用户反馈B/C补充命令均已完成。读取两模型estimators_fit_v1、estimators_evaluation_v1和diagnostics_v1，以及maintenance/part3_studies_verify下六份完整报告：两份fit均795图/三尺度重新拟合通过，两份B评价均654图/1962数组重算通过，两份C均654图/1308新经典数组CPU重放及六方法全部GT/诊断/统计重算通过，全部passed、formal_complete:true。没有改动算法/配置、原预测、第三方仓库或自动元数据，没有启动新训练、推理、全量排序或滤波。

只读脚本/保护指纹/完整精度复核和六法共同剖面统计在`outputs/maintenance/part3_studies_results_20261106/`。实际执行review.py（aiaa3201），12个正式含父阶段、15,135份去重来源/输入/产物/报告SHA通过；三规则/六方法全部身份/覆盖、RAW与Part 2冻结、B_median_log与原B一致；795行开发身份及两种log尺度公式、全部宏平均、scene区间重算一致，冻结早于测试评价。首次审查脚本误用了target_edge_pixels字段，修正为现有gt_edge_pixels后完整通过；科学产物未受影响。额外对两模型测试位置21（三尺度数组、两个经典CPU数组及GT评分）实际抽查通过；本轮未重新执行795图大内存加权中位数或654图全量滤波，完整重放以用户留存的真实verify报告为证。

B的三个尺度：Depth Pro `1.0413853532199127 / 1.0397601971598807 / 1.0277159306952304`，UniDepth `0.9562419672512901 / 0.9552798937095953 / 0.9428846931615024`。三规则AbsRel为Depth Pro `0.09660522364809511 / 0.0965099253773059 / 0.09643385612402633`，UniDepth `0.10203673364140407 / 0.10194121863893553 / 0.10147545468113502`。最低均值均来自AbsRel规则，但比原规则额外降低仅约0.1774%/0.5501%，两组差值scene区间跨零；UniDepth AbsRel规则的RMSE比原规则高`0.0048893501737579 m`、区间`[0.002356531072936579,0.007496085254227028]`。不以最低AbsRel均值宣布全面赢家或重选规则，开发未裁剪目标与正式clip评分保持区分。

C每模型3924行主表、35316边界行、11772定位行、6876剖面行、7848纹理行、两个经典654数组组及12张共同位置图齐全。原四方法/RAW的主指标不变。Depth Pro Guided的AbsRel/RMSE更低但SI低于RAW；UniDepth Guided的SI增量区间跨零。完整C在1.05/1px诊断中P提高、R下降，补充F1由`0.5497658290519466→0.5228890541109205`和`0.49899411626071244→0.4719192456093704`（RAW完整精度来自当次JSON/CSV）；GT到预测边缘距离分别增加`0.428991350490006 / 0.3155041092507797 px`，区间全正。636图有GT/物体标注支持的边缘召回，18图保留N/A。三阈值三容差中，完整C在1.05/1.10均降低F1，在1.25均提高，解释不同边缘强度与官方方向性10阈值加权SI定义的取舍，不改评价器追分，不将SI提高写成全面锐化。

每模型GT固定剖面1146条/513图，141图无合格GT剖面，六方法坐标与GT属性一致。完整状态计数保留。进一步取六法全部可测的相同支持：Depth Pro1031条/499图/198场景，UniDepth1018条/494图/199场景。图内平均后按图等权，完整C宽度增加`0.5599554313432596 / 0.5897039850226321 px`，95%区间分别`[0.4990036892374084,0.6222593070438754] / [0.5293828369478378,0.6525321257555978]`；中心误差降低`0.03889333337279865 / 0.07486343187279897 px`，区间全负。支持中心略准但过渡更宽；完整C比其他后处理保留较窄形状，却不比RAW更锐。此子集不能外推654图；原summary两两共同可测统计完整保留，不改写成新的全量值。

纹理区7,725,362像素与低纹理对照34,161,616像素六法一致，654图均有可测区域。Depth Pro完整C图像宏平均伪边代理由`0.02032273857248179`降为`0.018687207896156625`，区间有下降证据；UniDepth由`0.006239909452578719`变为`0.0061501385884426465`，区间跨零。UniDepth去门控代理增量`0.00034249294702151645`、区间`[0.00010207181318788184,0.0005870722723703167]`，同时过渡更宽，不能称去门控在所有边界性质上胜出。代理不是纹理复制的完美真值，不用抹边获得较低代理就宣称更好。

新高斯CPU平均约1.68ms/图、Guided约67–68ms/图，原完整C CUDA约16.6–16.7ms；不同后端/变换域/支撑和旧时序记录均标明，不作公平算法速度或端到端FPS结论。查看两模型位置21的剖面图、Depth Pro位置302边缘叠图、UniDepth位置545共同纹理掩码；仅按可见结构与对应分数解释，不从掩码图臆断某方法已产生伪边。

Step B/C后半补充完整正式表、参数、收益/失败、区间、六法共同可测支持与案例解释，保留全部完整复现指令并提醒默认目录已存在。README、PLAN、索引、当前结果和架构状态同步补充门禁已具备；旧C全量CPU同公式重放的报告留存范围单列，不把本次经典对照核验冒充它。原执行历史、构建时349项测试与小型formal_complete:false检查均保留。本轮科学代码不变，不重复跑全项目测试，Demo/报告/提交与70/100帧素材问题保持独立。

最终文档/保护复查通过：11份Markdown、75处本地链接、22个Bash代码块语法、43张表列数和全部B/C正式主表格式化数字匹配；15,135份受保护文件SHA再次一致。脚本与报告为本轮维护目录check_docs.py、documentation_checks.json。没有新增或覆盖原始运行元数据。

## 2026-11-06 — B/C统一文档与项目说明

按用户要求，将Step B和Step C按完整研究任务重组，不再按研究增量分成两套方法/命令/结果。Step B从开头统一介绍RAW/三尺度规则，依次说明协议、源码分工/产物、共同开发预测→参考与三规则冻结→统一测试/核验、完整结果和目标取舍。Step C从开头统一六方法、定位/宽度/纹理协议，按联合滤波/消融预测→经典对照与六法统一评价→核验组织命令，再统一呈现六法八指标、组件/场景统计、定位/阈值、共同剖面与状态、纹理、实际案例和成本。重复主表、重复运行章节和分阶段研究叙事合并；保留所有尺度规则/对照/消融、实际收益与负结果。

现有脚本分工与真实依赖保留：三规则fit仍需要795图开发与中位log参考冻结，六法诊断仍读取联合滤波/消融预测；没有为了文档统一移动或改写已绑定科学源码、脚本、YAML、数组或自动元数据。完整默认命令明确只供复现、目录已有无需重跑，独立重跑使用新目录/报告；B单规则参考评价作为可独立核验的参考轨道解释，不冒充第四种校准方法。C同公式CPU重放与统一诊断核验范围分开，未另存的全量滤波报告没有因整理而改为已保存。

README、PLAN、AGENTS、当前结果页、源码/流水线/配置/输出/测试说明统一为三规则B与六方法C的职责结构，不用“补充实验/后续工作”组织当前研究。implemented已有执行历史和原始审计文件保留，不重写成一次执行或假称测试从未被查看。测试集启发设计的探索性范围仍在协议中如实披露，不按test另选参数或组合B+C。

文档整理前的14份说明快照和SHA在`outputs/maintenance/part3_docs_unified_20261106/before/`及documents_before.json；统一文档脚本unify_docs.py只写读者说明。指标从现有完整精度报告/CSV保留，C统一六法主表含全部八指标，计时从原表汇总均值/中位/P95并标明后端及GPU allocated，未重新测量或运行实验。本轮未启动训练、推理、参数搜索或全量评价，代码未变，不重复运行全项目测试。

整理后检查通过：14份说明、84处本地链接、16个Bash代码块语法、38张表列数及B/C完整主表数字一致；15,135份受保护科学源码/输入/产物/报告SHA保持不变。check.py和checks.json保存在同一维护目录，记录13份实际改动说明；Step B从324行合并为211行，Step C从453行合并为329行。没有删除科学结果或改写运行时间/历史元数据。

## 2026-11-06 — 完整项目GitHub分支更新说明

新增`docs/github_upload_project.md`，覆盖Part 1/2及Direction B/C，沿用仓库与JiongxuChen分支示例，提供白名单导出、SHA256清单、可选传到本地电脑、独立克隆、完整替换工作树、暂存字节核对、普通提交/push与远端确认的完整命令。保留提交历史，不强制推送，不操作其他分支。README与docs索引增加完整项目入口，原Part 1/2专用指南保留其限定范围。

发布包包含项目自有源码、脚本、配置、测试、环境与课程/研究说明，明确同时保留part3和part3_studies两组职责目录，以及refinement来源和MIT许可。data、checkpoints、outputs、第三方仓库和github_repository整个目录不导出；已有指标保留在说明中，冻结尺度等独立结果文件不上传，文档本地资源链接与新目录复现依赖另作说明。包内生成专用.gitignore、保留字节的.gitattributes及PUBLISHING.md，不修改原始源码或运行元数据。

文档生成期间临时多行文本结束标记冲突，Shell误读后续文字，已中止；没有清空、提交或推送远端。随后直接提取最终文档的13段Bash与两段Python进行语法检查，并在临时本地Git仓库验证：186文件导出约0.89 MiB、无资源/预测文件、两组B/C目录齐全；旧普通/隐藏文件删除、暂存内容与清单一致、可执行位设置、历史父提交和另一分支保留；篡改包会在删除前被拒绝。检查记录在`/tmp/aiaa3201-publish-check-g6i2bfd7/checks.json`，属于临时发布验证，不是科学实验。本轮未运行模型、训练或正式评价，尚未替用户发布分支。
