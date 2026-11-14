# 发布范围与复现

本分支包含Part 1/2、Direction B、C v1与v2的项目自有实现、测试、环境和说明。
不包含data、checkpoints、outputs、第三方仓库、队友仓库副本或缓存。
结果数值保留在文档，预测/视频/图像/CSV/冻结参数JSON和原始运行元数据不上传。
文档指向这些本地资源的链接需在恢复相应资源后打开，GitHub文件树中不存在。
先按configs/versions.yaml与docs/step0.md克隆固定上游，再创建环境、准备数据与权重。
各Step给出完整生成/评价/核验命令；只克隆本分支无法直接verify未上传的正式产物。
B/C的part3、part3_studies和part3_c_v2目录都是必需组成，来源及MIT许可完整保留。
不按测试GT选参，不训练，不因发布而改变70/100帧素材、Demo和报告等门禁。
当前使用方法以README和各Step为准。
