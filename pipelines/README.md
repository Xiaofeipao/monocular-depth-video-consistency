# 流水线

- `part1/run_middlebury.py`、`tune_middlebury.py`：四组经典双目和开发集选择。
- `part1/study_middlebury_coverage.py`：误差—覆盖率冻结规则、十场景评估和原始重算。
- `part1/dav2_video.py`：指定片段逐帧DA-V2-S、分析与核验。
- `part1/dav2_nyuv2.py`：保留的NYUv2相对深度对齐扩展。
- `part2/run_depth_pro_nyuv2.py`、`run_unidepth_v2_nyuv2.py`：两模型原生米制基线。
- `part2/run_key_analysis.py`、`analyze_scale_boundary.py`：静态尺度/边界分析与重放。
- `part2/vda_video.py`：指定片段VDA独立实验及DA/VDA配对比较。
- `part3/common.py`：B/C固定协议、身份、源码/输入/父运行来源和禁止覆盖。
- `part3/direction_b.py`：开发预测、拟合全局尺度、冻结、测试及独立核验。
- `part3/direction_c.py`：固定C、去RGB/去深度门控、场景统计、案例及CPU同公式重算。
- `part3_studies/direction_b.py`：复用795张开发预测，三种全局尺度规则全部拟合、冻结、评价与核验。
- `part3_studies/direction_c.py`：组织六方法统一评价，读取联合滤波/消融、生成经典对照并做共同GT定位/宽度/纹理诊断。
- `part3_studies/common.py`：统一比较的源码/父运行来源和scene配对统计，保持各阶段来源可核验。

Part 1/2及B/C完整运行、核验报告与分析已具备；新运行仍由用户启动，不自动调参或组合B+C。入口与完整命令见[文档索引](../docs/README.md)。
