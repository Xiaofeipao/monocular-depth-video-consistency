# Part 1B与Part 2B：按视频实施计划

## 目标和状态

课程依据项目PDF §5.1、§5.2、§6.2（本地保留：`AIAA3201_Project3_Monocular_Metric_Depth_Video_Consistency_Spring2027 (1).pdf`）。NYUv2的DA-V2-S尺度对齐基准已完成，aligned_relative身份和既有数据/产物不变。视频使用原生relative、无GT、alignment:none；官方Small与FP16，不训练。

当前六段原片已登记，单视频调度、三类型评价、固定色标报告、续跑与raw重算已实现并通过合成验证；正式预测、评价与真实结果解释由用户运行后完成。完整命令分别见[Step 2](docs/step2.md)、[Step 4](docs/step4.md)。

## 每段视频的任务

| 类型 | 闪烁/一致性主诊断 | 尺度与补充 | 背景时间std |
|---|---|---|---|
| fixed_static：固定无人且静止 | 全图逐像素MAE/S、RMSE/S | 光流指标及中位数曲线 | 全画面，不画矩形ROI |
| fixed_dynamic：固定有人/动态 | 光流对应MAE/S、RMSE/S及人物遮挡案例 | 逐像素变化含真实运动，中位数受人物占比影响 | N/A |
| moving：相机移动 | 光流对应误差、中位数比值与固定跨度变化 | 逐像素项含视差，proxy混合真实距离和视野变化 | N/A |

所有类型保留原生相对预测、两组时序指标、coverage、曲线、显示饱和、实际成本及案例；三类仅在任务适用性与解释重点上不同。S固定为每模型首帧全图P95−P5，整片不变；pixel使用同一屏幕坐标和全图，不采用FB mask。flow使用全图有效非遮挡对应。不能因为有效域不同直接比较pixel与flow大小来判优，也不能把中位数变化全归为真实米制漂移。

## 单视频运行与复用

共同配置`configs/video/self_recorded.yaml`，登记`configs/video/clips.yaml`。逐帧入口`scripts/part1/video.sh`，视频比较入口`scripts/part2/video.sh`；各自的`run --clip ID`只完成本部分的适用任务，各自存储结果。2B要求该视频1B已经完成，只读引用，不启动或改写1B。

1. prepare：只解码一次，保存真实PTS/RGB SHA，冻结源、类型、配置、代码与官方资源。
2. dav2/flow：同一视频生成逐帧相对深度和双向RAFT。
3. analyze1/verify1：全图逐像素、同光流、适用背景、尺度proxy、图视频和原始重算。
4. vda：同帧、同网络/输出尺寸、FP16、官方离线窗口；不重复DA/flow。
5. analyze2/verify2：同条件两模型比较，DA桥接和mask逐像素核验，完整raw重算。
6. summarize：可选；任意已完成视频按三类型等片段汇总，N/A不补零，不运行模型。

两个入口各自补齐本部分的依赖；指定--task只做本部分某类任务，--no-deps严格要求依赖已经存在。2B缺少完整1B输入时直接报错。再次运行同ID时，完整产物通过核验才复用；失败/不完整目录不覆盖，不手改status。源、类型、配置或科学源码改变时另用ID或新的输出/缓存根目录，不混用同一run。

新视频可通过--video/--type直接接入，第一次保存登记，后续只需ID；也可预先在YAML加一项。真实session已知时记录，未知保留null，不从文件名推断独立场景或来源。输入既不写死六段，也不固定818帧；每片以实际manifest为准。

## 共同协议与架构

10 FPS、长边≤960；16:9网络518×924/输出540×960，Small/FP16。RAFT-Small/FP32/12次更新、FB≤1 px；有限零值合法，NaN/Inf失败，零参考跨度N/A。主评价全图，所有非空帧对等权，低coverage仅标记。TURBO原生[0,50]、固定参考副图[0,1]、两种差分[0,0.1]，显示截断不改变指标。

VDA为官方32/10/22窗口及8帧混合，预测间scale/shift无GT。离线使用未来帧，均摊耗时不等于在线首帧延迟，模型FPS不含RAFT/解码/保存/绘图。原生中位数比值受任意零点影响；固定跨度变化补充解释，均非真实米制漂移真值。

```text
data/self_recorded/<ID>.mp4
data/video_frames/self_recorded/<ID>/manifest.json + rgb/
outputs/part1/video/<ID>/
  clip.json + protocol.json + commands/
  dav2/ + flow/ + part1/ + verify1.json
outputs/part2/video/<ID>/
  clip.json（baseline只读引用1B）+ protocol.json + commands/
  vda/ + part2/ + verify2.json
outputs/part1/video_summary/<汇总快照>/
outputs/part2/video_summary/<汇总快照>/
```

`pipelines/part1/video_clip.py`与`pipelines/part2/video_clip.py`提供两个独立CLI，`pipelines/video/clip.py`只复用参数定义与辅助，不提供运行入口；`src/video/clip_workflow.py`的两个独立调度函数分别冻结与记录，`clip_analysis.py`计算/核验，`clip_report.py`生成单/双模型报告；官方adapter、RGB/RAFT和底层数值算子复用。1B保留DA原始预测和光流，2B保留VDA原始预测与比较报告；两部分报告各自保存，相同输入与基线通过SHA引用。

## 完成门禁

- NYUv2完整654张、GT辅助对齐与重算：已完成，不重复运行。
- 视频实现、合成文件生命周期、资源/真实CUDA：已验证，不等同正式视频完成。
- 六段各自的实际预测、两组时序指标、适用背景与中位数、同条件比较及raw重算：待用户正式运行。
- 实际收益/恶化、细结构/遮挡、效率和限制分析：待产物；不预设VDA在所有项取胜。
- mandatory clips来源确认、报告、应用Demo和最终打包：单列。当前素材没有覆盖的镜面/玻璃/天空/强模糊不能称已验证。

正式训练、微调和长时预测继续由用户启动；现有资源校验优先，无需重新下载。
