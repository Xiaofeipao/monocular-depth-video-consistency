# Part 3：模型与算子配置

- [direction_b.yaml](direction_b.yaml)：模型资源/基线/推理环境，795图共同开发预测和中位log参考校准。
- [direction_c.yaml](direction_c.yaml)：固定5×5联合滤波、去RGB与去深度门控，生成统一评价需要的三组预测。

两配置只读引用Part 2模型协议与frozen_v1原始预测；不更改Part 2 split/crop/范围、预测或运行元数据。B是开发GT校准，不能称untouched native metric；C不读取GT、不使用B预测。固定C设置不在测试集搜索。

模型资源路径以根目录为准，UniDepth开发推理自动使用独立环境；评价统一使用aiaa3201。完成命令、独立重跑和状态见[Step B](../../docs/stepB.md)、[Step C](../../docs/stepC.md)。

三尺度规则、经典对照与六法定位/宽度/纹理评价由[part3_studies](../part3_studies/README.md)配置。两目录按职责组成完整B/C流程，配置与已绑定源码指纹保持一致；不通过修改参数追逐测试结果。

动态片段的目标/背景/边界检查见[动态实验](../../docs/dynamic_clips.md)；与 NYU 主表分别报告。
