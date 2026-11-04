# Part 1/2 与 Part 3 的本地衔接

比较基准：本地主线 `a717683` 与同伴 `JiongxuChen` 分支 `9aaa869`。
新增 Part 1/2 已追加到同伴分支 `JiongxuChen`（`548420f`），没有整支覆盖；
Part 3 在 main 单独提交。没有创建新分支。
同伴分支的[受控实验与证据](https://github.com/Xiaofeipao/monocular-depth-video-consistency/blob/JiongxuChen/docs/controlled_studies.md)
包含开发集冻结选参、十场景验证、逐图指标核对、原始基线 SHA 和 16 图资源测量。

## 分工与入口

同伴 Step 1–4 对应 1A 双目、1B DAV2、2A 米制、2B VDA；本地主线新增
[Step 5](step5.md) 承接 Part 3。主线仍保留已有 Step 1/3，尚未用同伴新版覆盖。
Part 3 正式主线为 C，其余 A–F 与研究实验保留。

## 已补充的范围与其他可复用材料

- Part 1 全 GT 复评分：`scripts/rescore_course_stereo.py` 与
  `docs/evidence/course_submission/stereo/`。从既有视差重算
  全部有限正值 GT 与 non-occluded 两种口径，适合核对旧结果；同伴新版已有全 GT
  主协议，不应再把旧 tuned 参数结果当作新版 tuned 结果覆盖过去。
- Part 2 公开数值附件：`docs/evidence/full_benchmark_20261029/` 中逐图 CSV、汇总与
  原预测来源信息。它们是我们独立运行的证据，同伴需保留自己的运行身份；不能将我们
  的逐图表与其接近但不完全相同的基线汇总拼接。

上述两类证据及新受控实验已放在同伴分支 `docs/evidence/controlled_studies/`，
两条基线 1308 份原始预测 SHA 逐份核验通过。逐图比较保留 UniDepth 的局部大差异，
不再仅凭很小的平均差推断浮点原因。以下回放/应用工具仍属可复用建议，没有同步到同伴分支。
- 指定视频原始结果打包与 CPU 回放：`scripts/replay_course_video.py`、
  `src/package_submission.py`、ZIP 校验说明，便于没有 GPU 的成员检查已保存预测。
  两套视频输出的目录和字段不同，需要显式适配，不能直接交叉调用打包器。
- 应用与困难案例：`scripts/build_course_media.py`、`docs/figures/course_delivery/`、
  `docs/evidence/course_submission/difficult_cases.json`。可补齐总体交付；这部分同时
  依赖 Part 3 C 的预测，不能声称仅靠同伴 Part 1/2 即能生成全部媒体。
- 统计与跨域限制：`scripts/audit_refinement_statistics.py`、`docs/paper_comparison.md`、
  iBims/自录/TUM 扩展说明。用于支撑报告讨论，属于可选扩展，不增加课程必需数据。

## 优先吸收同伴的实现

1. 双目参数搜索的固定参考覆盖率门槛、源码/数据哈希及独立核验。
2. 2A 的距离分组、物体支持边缘、轮廓位移和过渡宽度分析。
3. 1B/2B 的原始 PTS/RGB 身份检查、连续案例、源码绑定、预热及重复计时与完整重算。

这些实现尚未移植到主线，不能把同伴文档里的已验证状态写成我们的新验证结果。
先对齐输入文件、协议和保存格式，再运行针对性的核验；新实验写入新目录。
DAV2 的 NYU GT 辅助对齐扩展仅作诊断，不能加入原生米制排行榜。

## 合并前仍需处理

同伴分支是 Part 1/2 的范围导出，整支合并会包含大量删除与目录接口变动；应选择性迁移。
其根目录课程 PDF 已被跟踪，本地 ignore 无法屏蔽已跟踪文件，不能将该 PDF 纳入后续提交。
其 README 引用的已删除导出清单及中文要求文件需修正。
保留我们的 Part 3、正式证据、原始预测、本地自录视频及交付 ZIP；私人素材不公开上传。
