# AllReduce分阶段调用关系与分支

[返回阅读指南](../READING_GUIDE.zh-CN.md)。以下关系由固定快照逐函数审读；路径定位已转换为带逐行注释的固定提交行号。Host→AICPU为runtime发射关系，虚调用按本例注册的executor/template/UB传输类型展开。树内简写行号对应审读快照S行；树后定位表和正文链接对应新增注释后的L行。

## 入口与算法编排

- [第1页](CALL_RELATIONS-1.part-01.zh-CN.md)：AllReduce：算法选择、资源请求和 Mesh OneShot 展开；可复现的讲解条件；主调用链及调用位置；入口每条控制路径；旧选择器完整引擎分流；另有8节



## 公共调度与设备入口

- [第1页](CALL_RELATIONS-2.part-01.zh-CN.md)：AllReduce dispatch / launch / resource 调用关系实证；Host 算法选择、执行及资源树；AICPU资源准备深入到 HCOMM 公共ABI；输入/输出依赖与保序发射；Device入口与执行器编排/缓存树；另有6节



## HCOMM资源与数据任务

- [第1页](CALL_RELATIONS-3.part-01.zh-CN.md)：HCOMM 调用关系与分支导读（固定基线 f8af6a3）；1. 通道协议为何进入 UB Lite；2. 资源准备主干；3. 本地任务、远端数据和通知如何进入 SQ；4. 提交、批模式与完成边界；另有16节

- [第2页](CALL_RELATIONS-3.part-02.zh-CN.md)：MyRank::BatchCreateChannels；MyRank::BatchCreateSockets；MyRank::BatchServerInitForChannels；MyRank::BatchGetSocketsForChannels；MyRank::GetEndpointPairFromChannel；另有24节

- [第3页](CALL_RELATIONS-3.part-03.zh-CN.md)：HcommChannelNotifyWaitOnThread；HcommBatchTransferOnThread；HcommBatchModeStart；HcommBatchModeEnd；HcommSetLaunchMode；另有29节

- [第4页](CALL_RELATIONS-3.part-04.zh-CN.md)：UbConnLite::LaunchOneWqe；StreamLite::GetRtsq；RtsqA5::NotifyWait；RtsqA5::NotifyRecordLoc；RtsqA5::SdmaCopy；另有38节
