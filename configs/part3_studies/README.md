# Part 3：方法比较与诊断配置

- `direction_b.yaml`：声明RAW/三种全局尺度规则，使用795图共同开发预测及参考校准，三尺度分别冻结后对654图全部报告。
- `direction_c.yaml`：固定六方法集合、两个经典log-depth对照、共同GT诊断、RGB纹理阈值和共同展示位置；读取联合滤波/消融预测，边缘与剖面定义只读引用2A。

本目录与`configs/part3/`的模型/算子配置按数据依赖衔接，组成统一B/C流程。求解/经典算子/诊断、流水线与入口分别为`src/part3_studies/`、`pipelines/part3_studies/`、`scripts/part3_studies/`，与refinement及part3生成阶段共同保存完整来源。方法和参数固定，源码绑定后不移动或修改已生成结果。

完整运行、资源说明、输出字段与门禁见[Step B](../../docs/stepB.md)、[Step C](../../docs/stepC.md)。正式结果与六份完整尺度/诊断报告均具备，不改变配置追逐测试分数；独立运行仍由用户启动。
