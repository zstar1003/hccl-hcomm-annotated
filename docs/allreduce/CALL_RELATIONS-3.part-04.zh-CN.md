# AllReduce分阶段调用关系与分支

[返回阅读指南](../READING_GUIDE.zh-CN.md)。以下关系由固定快照逐函数审读；路径定位已转换为带逐行注释的固定提交行号。Host→AICPU为runtime发射关系，虚调用按本例注册的executor/template/UB传输类型展开。树内简写行号对应审读快照S行；树后定位表和正文链接对应新增注释后的L行。

[返回调用关系导航](CALL_RELATIONS.zh-CN.md)。HCOMM资源与数据任务，第4/4页。

### UbConnLite::LaunchOneWqe

定义：[ub_conn_lite.cc:L353–L395](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L353-L395)。维护16位 UB PI，定位 UB SQ 环槽并在未缓存锁定时复制 WQE。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [ub_conn_lite.cc:L357](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L357)；`HCCL_INFO("[UbConnLite::%s] start, opCode[%s]", __func__, opCode.Describe().c_str());` | Describe（未展开边界）；c_str（未展开边界） | 记录UbConnLite::LaunchOneWqe的状态/性能诊断，字段包含UB读写操作码的Describe字段；日志本身不执行传输。 |
| [ub_conn_lite.cc:L378](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L378)；`auto ret = memcpy_sp(va, SQE_SIZE_64, sqe, SQE_SIZE_64);` | memcpy_sp（未展开边界） | 设置当前调用状态为/按`memcpy_sp(va, SQE_SIZE_64, sqe, SQE_SIZE_64)`（当前WQE写入地址、当前UB WQE结构）；把WQE/SQE数据复制到设备映射队列内存，失败抛异常。 |
| [ub_conn_lite.cc:L382](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L382)；`THROW<InternalException>(StringFormat("[UbConnLite::%s] memcpy_sp failed, ret = %d", __func__, ret));` | StringFormat（未展开边界） | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [ub_conn_lite.cc:L363](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L363) | `if (sqOffset < sqDepth_ && (sqOffset + 1) >= sqDepth_) {` | 仅当`(sqOffset < sqDepth_ && (sqOffset + 1) >= sqDepth_)`（UB SQ环内WQE槽位、RTSQ或UB SQ深度）成立时进入此分支。 |
| [ub_conn_lite.cc:L376](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L376) | `if (!dwqeCacheLocked_) {` | 仅当`(!dwqeCacheLocked_)`（WQE直接写队列的缓存锁定状态）成立时进入此分支。 |
| [ub_conn_lite.cc:L380](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L380) | `if (UNLIKELY(ret != 0)) {` | 仅当`(UNLIKELY(ret != 0))`（当前调用状态）成立时进入此分支。 |
| [ub_conn_lite.cc:L382](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L382) | `THROW<InternalException>(StringFormat("[UbConnLite::%s] memcpy_sp failed, ret = %d", __func__, ret));` | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |


### StreamLite::GetRtsq

定义：[stream_lite.cc:L47](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/stream_lite.cc#L47)。返回 StreamLite 持有的执行队列对象；具体 950 队列为 RtsqA5。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [stream_lite.cc:L47](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/stream_lite.cc#L47)；`RtsqBase* StreamLite::GetRtsq() const { return rtsq.get(); }` | get（未展开边界） | 本行定义并直接执行StreamLite::GetRtsq：返回 StreamLite 持有的执行队列对象；具体 950 队列为 RtsqA5；调用get，使用执行队列对象的get字段。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### RtsqA5::NotifyWait

定义：[rtsq_a5.cc:L500–L514](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L500-L514)。构造有秒级超时的通知等待 SQE，再更新待提交计数。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_a5.cc:L504](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L504)；`BuildA5SqeNotifyWait(streamId_, taskId_, notifyId, timeout, GetCurrSqeBuffer());` | BuildA5SqeNotifyWait → [sqe_build_a5.h:L60–L90](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L60-L90)；GetCurrSqeBuffer → [rtsq_a5.cc:L451–L459](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L451-L459) | 在本地SQE缓存编码通知ID、超时及流/任务ID；取得下一个本地SQE缓存位置并记录对应RTSQ目标地址；传入/处理运行时执行流编号、队列当前任务编号、硬件通知ID、超时秒数。 |
| [rtsq_a5.cc:L512](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L512)；`RefreshInfo();` | RefreshInfo → [rtsq_a5.cc:L464–L495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L464-L495) | 推进任务编号和待提交数，EAGER立即提交或达到内部阈值提交。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### RtsqA5::NotifyRecordLoc

定义：[rtsq_a5.cc:L517–L527](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L517-L527)。构造本地通知记录 SQE并更新提交状态。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_a5.cc:L521](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L521)；`BuildA5SqeNotifyRecord(streamId_, taskId_, notifyId, GetCurrSqeBuffer());` | BuildA5SqeNotifyRecord → [sqe_build_a5.h:L93–L117](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L93-L117)；GetCurrSqeBuffer → [rtsq_a5.cc:L451–L459](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L451-L459) | 在本地SQE缓存编码本地通知记录；取得下一个本地SQE缓存位置并记录对应RTSQ目标地址；传入/处理运行时执行流编号、队列当前任务编号、硬件通知ID。 |
| [rtsq_a5.cc:L525](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L525)；`RefreshInfo();` | RefreshInfo → [rtsq_a5.cc:L464–L495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L464-L495) | 推进任务编号和待提交数，EAGER立即提交或达到内部阈值提交。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### RtsqA5::SdmaCopy

定义：[rtsq_a5.cc:L558–L575](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L558-L575)。构造无归约 SDMA SQE并更新提交状态。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_a5.cc:L565](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L565)；`BuildA5SqeSdmaCopy(streamId_, taskId_, dstAddr, srcAddr, size, RTSQ_A5_PART_ID, 0, GetCurrSqeBuffer());` | BuildA5SqeSdmaCopy → [sqe_build_a5.h:L196–L241](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L196-L241)；GetCurrSqeBuffer → [rtsq_a5.cc:L451–L459](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L451-L459) | 在本地SQE缓存编码SDMA源/目标、字节长度与归约码；取得下一个本地SQE缓存位置并记录对应RTSQ目标地址；传入/处理运行时执行流编号、队列当前任务编号、目标地址整数表示、源地址整数表示、字节容量或单片字节数。 |
| [rtsq_a5.cc:L573](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L573)；`RefreshInfo();` | RefreshInfo → [rtsq_a5.cc:L464–L495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L464-L495) | 推进任务编号和待提交数，EAGER立即提交或达到内部阈值提交。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### RtsqA5::SdmaReduce

定义：[rtsq_a5.cc:L592–L628](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L592-L628)。转换归约/类型编码后构造 SDMA SQE并更新提交状态。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_a5.cc:L600](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L600)；`ReduceOpToStarsOpKindMap.find(reduceIn.reduceOp) == ReduceOpToStarsOpKindMap.end()` | find（未展开边界）；end（未展开边界） | 补全本分支/循环判断的`ReduceOpToStarsOpKindMap.find(reduceIn.reduceOp) == ReduceOpToStarsOpKindMap.end()`（底层归约类型/操作描述的reduceOp字段），和前面条件共同决定是否进入后续路径。 |
| [rtsq_a5.cc:L602](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L602)；`\|\| DataTypeToStarsDataTypeMap.find(reduceIn.dataType) == DataTypeToStarsDataTypeMap.end())) {` | find（未展开边界）；end（未展开边界） | 补全本分支/循环判断的`\|\| DataTypeToStarsDataTypeMap.find(reduceIn.dataType) == DataTypeToStarsDataTypeMap.end()))`（底层归约类型/操作描述的dataType字段），和前面条件共同决定是否进入后续路径。 |
| [rtsq_a5.cc:L604](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L604)；`THROW<InternalException>(StringFormat(` | StringFormat（未展开边界） | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |
| [rtsq_a5.cc:L606](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L606)；`"Sdma does not support reduceOp %s dataType %s", reduceIn.reduceOp.Describe().c_str(),` | Describe（未展开边界）；c_str（未展开边界） | 为当前RtsqA5::SdmaReduce诊断/异常表达式提供格式文本，将报告底层归约类型/操作描述的reduceOp.Describe字段；这一物理行没有数据搬运副作用。 |
| [rtsq_a5.cc:L608](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L608)；`reduceIn.dataType.Describe().c_str()));` | Describe（未展开边界）；c_str（未展开边界） | 为组装带上下文的错误或状态文本；取得对象诊断文本用于日志补入`reduceIn.dataType.Describe().c_str()))`（底层归约类型/操作描述的dataType.Describe字段）；本行是参数/结构化初始化续行。 |
| [rtsq_a5.cc:L613](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L613)；`u8 op = static_cast<u8>(ReduceOpToStarsOpKindMap.at(reduceIn.reduceOp));` | at（未展开边界） | 设置u8 op为/按`static_cast<u8>(ReduceOpToStarsOpKindMap.at(reduceIn.reduceOp))`（底层归约类型/操作描述的reduceOp字段）；调用at，使用底层归约类型/操作描述的reduceOp字段。 |
| [rtsq_a5.cc:L615](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L615)；`u8 type = static_cast<u8>(DataTypeToStarsDataTypeMap.at(reduceIn.dataType));` | at（未展开边界） | 设置Thread类型为/按`static_cast<u8>(DataTypeToStarsDataTypeMap.at(reduceIn.dataType))`（底层归约类型/操作描述的dataType字段）；调用at，使用底层归约类型/操作描述的dataType字段。 |
| [rtsq_a5.cc:L618](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L618)；`BuildA5SqeSdmaCopy(streamId_, taskId_, dstAddr, srcAddr, size, RTSQ_A5_PART_ID, (op \| type), GetCurrSqeBuffer());` | BuildA5SqeSdmaCopy → [sqe_build_a5.h:L196–L241](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L196-L241)；GetCurrSqeBuffer → [rtsq_a5.cc:L451–L459](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L451-L459) | 在本地SQE缓存编码SDMA源/目标、字节长度与归约码；取得下一个本地SQE缓存位置并记录对应RTSQ目标地址；传入/处理运行时执行流编号、队列当前任务编号、目标地址整数表示、源地址整数表示、字节容量或单片字节数、Thread类型。 |
| [rtsq_a5.cc:L626](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L626)；`RefreshInfo();` | RefreshInfo → [rtsq_a5.cc:L464–L495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L464-L495) | 推进任务编号和待提交数，EAGER立即提交或达到内部阈值提交。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [rtsq_a5.cc:L598](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L598) | `if (UNLIKELY(` | 仅当`(UNLIKELY(`成立时进入此分支。 |
| [rtsq_a5.cc:L604](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L604) | `THROW<InternalException>(StringFormat(` | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |


### RtsqA5::UbDbSend

定义：[rtsq_a5.cc:L660–L675](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L660-L675)。把 jetty 与16位 UB PI编码成 RTSQ Doorbell SQE，再更新提交状态。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_a5.cc:L665](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L665)；`BuildA5SqeUbDbSend(streamId_, taskId_, jettyLiteId, piValue, GetCurrSqeBuffer());` | BuildA5SqeUbDbSend → [sqe_build_a5.h:L244–L276](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L244-L276)；GetCurrSqeBuffer → [rtsq_a5.cc:L451–L459](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L451-L459) | 在本地SQE缓存编码UB jetty与生产指针Doorbell；取得下一个本地SQE缓存位置并记录对应RTSQ目标地址；传入/处理运行时执行流编号、队列当前任务编号、UB jetty的die/function/jetty标识、16位UB jetty生产指针。 |
| [rtsq_a5.cc:L673](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L673)；`RefreshInfo();` | RefreshInfo → [rtsq_a5.cc:L464–L495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L464-L495) | 推进任务编号和待提交数，EAGER立即提交或达到内部阈值提交。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### RtsqA5::RefreshInfo

定义：[rtsq_a5.cc:L464–L495](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L464-L495)。推进 SQE/Task计数：EAGER可立即提交，BATCH 达内部阈值仍会提交。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_a5.cc:L468](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L468)；`SetTaskIdBySqeId();` | SetTaskIdBySqeId（未展开边界） | 按当前SQE位置刷新taskId。 |
| [rtsq_a5.cc:L475](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L475)；`if (launchFlag_ && !IsBatchLaunchMode()) {` | IsBatchLaunchMode（未展开边界） | 仅当`(launchFlag_ && !IsBatchLaunchMode())`成立时进入此分支；调用IsBatchLaunchMode。 |
| [rtsq_a5.cc:L477](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L477)；`LaunchTask();` | LaunchTask → [rtsq_a5.cc:L288–L379](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L288-L379) | 将已生成的任务提交到具体RTSQ队列，不等同于全部任务完成。 |
| [rtsq_a5.cc:L493](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L493)；`LaunchTask();` | LaunchTask → [rtsq_a5.cc:L288–L379](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L288-L379) | 将已生成的任务提交到具体RTSQ队列，不等同于全部任务完成。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [rtsq_a5.cc:L473](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L473) | `#ifdef CCL_KERNEL_AICPU` | 编译条件`ifdef CCL_KERNEL_AICPU`限定后续实现，区分Host/设备或构建能力分支。 |
| [rtsq_a5.cc:L475](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L475) | `if (launchFlag_ && !IsBatchLaunchMode()) {` | 仅当`(launchFlag_ && !IsBatchLaunchMode())`成立时进入此分支；调用IsBatchLaunchMode。 |
| [rtsq_a5.cc:L483](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L483) | `#endif` | 结束前述编译条件控制的实现片段。 |
| [rtsq_a5.cc:L486](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L486) | `if (pendingSqeCnt != PER_LAUNCH_SQE_CNT) {` | 仅当`(pendingSqeCnt != PER_LAUNCH_SQE_CNT)`（本地待提交SQE条数）成立时进入此分支。 |


### RtsqA5::GetCurrSqeBuffer

定义：[rtsq_a5.cc:L451–L459](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L451-L459)。计算当前 SQE 的实际环队列地址并返回本地待提交缓存位置。


本函数没有另外的显式函数调用；按其返回/赋值直接完成包装。


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### RtsqA5::LaunchTask

定义：[rtsq_a5.cc:L288–L379](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L288-L379)。确保空间并复制本地 SQE 到设备 RTSQ，更新硬件 SQ 尾触发执行，按需保存任务缓存。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_a5.cc:L303](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L303)；`MakeSureAvailableSpace();` | MakeSureAvailableSpace → [rtsq_a5.cc:L60–L143](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L60-L143) | 在反压时等待RTSQ可用空间并检查超时/域状态。 |
| [rtsq_a5.cc:L315](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L315)；`PreLaunchSqeForCache(needCacheTask);` | PreLaunchSqeForCache（未展开边界） | 查询本轮是否需要保存SQE缓存；传入/处理是否记录本轮任务缓存。 |
| [rtsq_a5.cc:L318](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L318)；`CopySqeBufToSq(locBuf);` | CopySqeBufToSq → [rtsq_a5.cc:L170–L256](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L170-L256) | 按RTSQ环形槽位复制本地SQE缓存到硬件映射内存；传入/处理本地SQE待提交缓存。 |
| [rtsq_a5.cc:L322](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L322)；`if ((UNLIKELY(GetPlfDebugConfigValue() & PLF_TASK)) \|\| UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {` | GetPlfDebugConfigValue（未展开边界）；HcclCheckLogLevel（未展开边界） | 仅当`((UNLIKELY(GetPlfDebugConfigValue() & PLF_TASK)) \|\| UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO)))`成立时进入此分支；调用GetPlfDebugConfigValue, HcclCheckLogLevel。 |
| [rtsq_a5.cc:L337](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L337)；`ret = hcomm::AicpuTaskUtils::DumpSqeContent(sqePtr);` | hcomm::AicpuTaskUtils::DumpSqeContent（未展开边界） | 设置当前调用状态为/按`hcomm::AicpuTaskUtils::DumpSqeContent(sqePtr)`（待打印的SQE指针）；调用hcomm::AicpuTaskUtils::DumpSqeContent，使用待打印的SQE指针。 |
| [rtsq_a5.cc:L341](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L341)；`THROW<InternalException>(StringFormat("RtsqA5::%s DumpSqeContent failed, ret = %d", __func__, ret));` | StringFormat（未展开边界） | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |
| [rtsq_a5.cc:L356](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L356)；`ConfigSqTail(newTail);` | ConfigSqTail → [rtsq_base.cc:L164–L172](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L164-L172) | 向驱动配置新的SQ尾，触发芯片读取提交任务；传入/处理提交后新的RTSQ尾位置。 |
| [rtsq_a5.cc:L364](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L364)；`PostLaunchSqeForCache();` | PostLaunchSqeForCache（未展开边界） | 在本轮提交后把生成的SQE数组登记到任务缓存。 |
| [rtsq_a5.cc:L377](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L377)；`(void)memset_s(locBuf, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT, 0, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT); // locBuffer清零` | memset_s（未展开边界） | 清零本地待提交SQE缓存；显式丢弃memset_s返回值，已提交队列的执行完成由后续通知/Join观察。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [rtsq_a5.cc:L294](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L294) | `if (pendingSqeCnt == 0) { // 没有SQE ，直接返回` | 仅当`(pendingSqeCnt == 0)`（本地待提交SQE条数）成立时进入此分支。 |
| [rtsq_a5.cc:L306](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L306) | `if (pendingSqeCnt == 0) {` | 仅当`(pendingSqeCnt == 0)`（本地待提交SQE条数）成立时进入此分支。 |
| [rtsq_a5.cc:L322](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L322) | `if ((UNLIKELY(GetPlfDebugConfigValue() & PLF_TASK)) \|\| UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {` | 仅当`((UNLIKELY(GetPlfDebugConfigValue() & PLF_TASK)) \|\| UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO)))`成立时进入此分支；调用GetPlfDebugConfigValue, HcclCheckLogLevel。 |
| [rtsq_a5.cc:L333](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L333) | `for (size_t sqeIdx = 0; sqeIdx < pendingSqeCnt; sqeIdx++) {` | 按`(size_t sqeIdx = 0; sqeIdx < pendingSqeCnt; sqeIdx++)`（当前SQE下标、本地待提交SQE条数）遍历本批条目/分片；各次处理保持数组对应关系。 |
| [rtsq_a5.cc:L339](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L339) | `if (UNLIKELY(ret != HCCL_SUCCESS)) {` | 仅当`(UNLIKELY(ret != HCCL_SUCCESS))`（当前调用状态）成立时进入此分支。 |
| [rtsq_a5.cc:L341](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L341) | `THROW<InternalException>(StringFormat("RtsqA5::%s DumpSqeContent failed, ret = %d", __func__, ret));` | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |
| [rtsq_a5.cc:L362](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L362) | `if (needCacheTask) {` | 仅当`(needCacheTask)`（是否记录本轮任务缓存）成立时进入此分支。 |


### RtsqA5::CopySqeBufToSq

定义：[rtsq_a5.cc:L170–L256](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L170-L256)。按环队列是否回绕一次或两次复制 SQE 缓存到 RTSQ VA。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_a5.cc:L190](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L190)；`int ret = memcpy_sp(sqCurrAddr, pendingSqeCnt * AC_SQE_SIZE, sqeBuf, pendingSqeCnt * RTSQ_SQE_SIZE);` | memcpy_sp（未展开边界） | 设置当前调用状态为/按`memcpy_sp(sqCurrAddr, pendingSqeCnt * AC_SQE_SIZE, sqeBuf, pendingSqeCnt * RTSQ_SQE_SIZE)`（本次RTSQ目标地址、本地待提交SQE条数、待复制的SQE源缓存）；把WQE/SQE数据复制到设备映射队列内存，失败抛异常。 |
| [rtsq_a5.cc:L194](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L194)；`THROW<InternalException>(StringFormat("RtsqA5::%s sqe memcpy_sp failed, ret = %d", __func__, ret));` | StringFormat（未展开边界） | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |
| [rtsq_a5.cc:L209](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L209)；`int ret = memcpy_sp(sqCurrAddr, depthLeft * AC_SQE_SIZE, sqeBuf, depthLeft * RTSQ_SQE_SIZE);` | memcpy_sp（未展开边界） | 设置当前调用状态为/按`memcpy_sp(sqCurrAddr, depthLeft * AC_SQE_SIZE, sqeBuf, depthLeft * RTSQ_SQE_SIZE)`（本次RTSQ目标地址、RTSQ尾部到数组末尾的剩余槽数、待复制的SQE源缓存）；把WQE/SQE数据复制到设备映射队列内存，失败抛异常。 |
| [rtsq_a5.cc:L215](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L215)；`StringFormat("RtsqA5::%s rtsq remaining space memcpy_sp failed, ret = %d", __func__, ret));` | StringFormat（未展开边界） | 为组装带上下文的错误或状态文本补入`StringFormat("RtsqA5::%s rtsq remaining space memcpy_sp failed, ret = %d", __func__, ret))`（当前调用状态）；本行是参数/结构化初始化续行。 |
| [rtsq_a5.cc:L220](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L220)；`ret = memcpy_sp(` | memcpy_sp（未展开边界） | 设置当前调用状态为/按`memcpy_sp(`；把WQE/SQE数据复制到设备映射队列内存，失败抛异常。 |
| [rtsq_a5.cc:L230](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L230)；`StringFormat("RtsqA5::%s remaining sqe memcpy_sp failed, ret = %d", __func__, ret));` | StringFormat（未展开边界） | 为组装带上下文的错误或状态文本补入`StringFormat("RtsqA5::%s remaining sqe memcpy_sp failed, ret = %d", __func__, ret))`（当前调用状态）；本行是参数/结构化初始化续行。 |
| [rtsq_a5.cc:L246](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L246)；`int ret = memcpy_sp(sqCurrAddr, pendingSqeCnt * AC_SQE_SIZE, sqeBuf, pendingSqeCnt * RTSQ_SQE_SIZE);` | memcpy_sp（未展开边界） | 设置当前调用状态为/按`memcpy_sp(sqCurrAddr, pendingSqeCnt * AC_SQE_SIZE, sqeBuf, pendingSqeCnt * RTSQ_SQE_SIZE)`（本次RTSQ目标地址、本地待提交SQE条数、待复制的SQE源缓存）；把WQE/SQE数据复制到设备映射队列内存，失败抛异常。 |
| [rtsq_a5.cc:L250](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L250)；`THROW<InternalException>(StringFormat("RtsqA5::%s sqe memcpy_sp failed, ret = %d", __func__, ret));` | StringFormat（未展开边界） | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [rtsq_a5.cc:L176](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L176) | `if (sqTail_ >= sqHead_) {` | 仅当`(sqTail_ >= sqHead_)`（软件保存的RTSQ尾槽位、软件保存的RTSQ头槽位）成立时进入此分支。 |
| [rtsq_a5.cc:L180](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L180) | `if (pendingSqeCnt <= depthLeft) { // 没有回绕` | 仅当`(pendingSqeCnt <= depthLeft)`（本地待提交SQE条数、RTSQ尾部到数组末尾的剩余槽数）成立时进入此分支。 |
| [rtsq_a5.cc:L192](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L192) | `if (UNLIKELY(ret != 0)) {` | 仅当`(UNLIKELY(ret != 0))`（当前调用状态）成立时进入此分支。 |
| [rtsq_a5.cc:L194](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L194) | `THROW<InternalException>(StringFormat("RtsqA5::%s sqe memcpy_sp failed, ret = %d", __func__, ret));` | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |
| [rtsq_a5.cc:L198](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L198) | `} else {` | 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。 |
| [rtsq_a5.cc:L211](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L211) | `if (ret != 0) {` | 仅当`(ret != 0)`（当前调用状态）成立时进入此分支。 |
| [rtsq_a5.cc:L213](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L213) | `THROW<InternalException>(` | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |
| [rtsq_a5.cc:L226](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L226) | `if (UNLIKELY(ret != 0)) {` | 仅当`(UNLIKELY(ret != 0))`（当前调用状态）成立时进入此分支。 |
| [rtsq_a5.cc:L228](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L228) | `THROW<InternalException>(` | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |
| [rtsq_a5.cc:L236](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L236) | `} else {` | 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。 |
| [rtsq_a5.cc:L248](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L248) | `if (UNLIKELY(ret != 0)) {` | 仅当`(UNLIKELY(ret != 0))`（当前调用状态）成立时进入此分支。 |
| [rtsq_a5.cc:L250](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L250) | `THROW<InternalException>(StringFormat("RtsqA5::%s sqe memcpy_sp failed, ret = %d", __func__, ret));` | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |


### RtsqA5::MakeSureAvailableSpace

定义：[rtsq_a5.cc:L60–L143](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L60-L143)。在 SQ 反压时读取头位置等待空间，检查超时/域状态并尝试其他流发射。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_a5.cc:L64](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L64)；`u32 availableSpace = GetTailToHeadDist();` | GetTailToHeadDist（未展开边界） | 设置RTSQ可用槽位数为/按`GetTailToHeadDist()`；根据RTSQ头尾位置计算可用距离。 |
| [rtsq_a5.cc:L66](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L66)；`auto startTime = std::chrono::steady_clock::now();` | std::chrono::steady_clock::now（未展开边界） | 设置单调时钟开始时刻为/按`std::chrono::steady_clock::now()`；调用std::chrono::steady_clock::now。 |
| [rtsq_a5.cc:L70](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L70)；`sqFullTimeout_ = GetSqFullTimeOut();` | GetSqFullTimeOut（未展开边界） | 设置SQ反压等待超时秒数为/按`GetSqFullTimeOut()`；调用GetSqFullTimeOut。 |
| [rtsq_a5.cc:L74](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L74)；`const std::chrono::seconds printInterval(PRINT_INTERVAL); // 打印间隔30s` | printInterval（未展开边界） | 调用printInterval，使用状态日志打印间隔；对象涉及状态日志打印间隔。 |
| [rtsq_a5.cc:L76](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L76)；`auto lastPrintTime = std::chrono::steady_clock::now() - printInterval;` | std::chrono::steady_clock::now（未展开边界） | 设置上次状态打印时刻为/按`std::chrono::steady_clock::now() - printInterval`（状态日志打印间隔）；调用std::chrono::steady_clock::now，使用状态日志打印间隔。 |
| [rtsq_a5.cc:L87](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L87)；`sqHead_ = QuerySqHead();` | QuerySqHead → [rtsq_base.cc:L123](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L123) | 设置软件保存的RTSQ头槽位为/按`QuerySqHead()`；通过驱动查询SQ头，判断是否追上固定尾位置。 |
| [rtsq_a5.cc:L89](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L89)；`availableSpace = GetTailToHeadDist();` | GetTailToHeadDist（未展开边界） | 设置RTSQ可用槽位数为/按`GetTailToHeadDist()`；根据RTSQ头尾位置计算可用距离。 |
| [rtsq_a5.cc:L98](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L98)；`auto curTime = std::chrono::steady_clock::now();` | std::chrono::steady_clock::now（未展开边界） | 设置当前单调时钟时刻为/按`std::chrono::steady_clock::now()`；调用std::chrono::steady_clock::now。 |
| [rtsq_a5.cc:L115](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L115)；`CheckLaunchTaskStatus(startTime, curTime);` | CheckLaunchTaskStatus（未展开边界） | 检查SQ等待超时与域挂起/不可用状态；传入/处理单调时钟开始时刻、当前单调时钟时刻。 |
| [rtsq_a5.cc:L119](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L119)；`HcclResult ret = HandleDispatchAllStreams();` | HandleDispatchAllStreams（未展开边界） | 设置当前调用状态为/按`HandleDispatchAllStreams()`；在当前流反压时尝试发射上下文内其他执行流。 |
| [rtsq_a5.cc:L125](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L125)；`= StringFormat("RtsqA5::%s HandleDispatchAllStreams failed, ret = %d, sqId:%u, ", __func__, ret, sqId_);` | StringFormat（未展开边界） | 设置当前调用状态为/按`%d, sqId:%u, ", __func__, ret, sqId_)`（硬件SQ编号、当前调用状态）。 |
| [rtsq_a5.cc:L127](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L127)；`HCCL_ERROR("%s", msg.c_str());` | c_str（未展开边界） | 记录RtsqA5::MakeSureAvailableSpace的错误诊断；日志本身不执行传输。 |
| [rtsq_a5.cc:L137](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L137)；`checkOpExecStatusCallback_();` | checkOpExecStatusCallback_（未展开边界） | 调用通信域执行状态检查回调。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [rtsq_a5.cc:L68](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L68) | `#ifdef CCL_KERNEL_AICPU` | 编译条件`ifdef CCL_KERNEL_AICPU`限定后续实现，区分Host/设备或构建能力分支。 |
| [rtsq_a5.cc:L72](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L72) | `#endif` | 结束前述编译条件控制的实现片段。 |
| [rtsq_a5.cc:L85](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L85) | `while (availableSpace <= pendingSqeCnt) {` | 在`(availableSpace <= pendingSqeCnt)`（RTSQ可用槽位数、本地待提交SQE条数）条件下重复执行后续等待或分片处理。 |
| [rtsq_a5.cc:L91](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L91) | `if (availableSpace > pendingSqeCnt) {` | 仅当`(availableSpace > pendingSqeCnt)`（RTSQ可用槽位数、本地待提交SQE条数）成立时进入此分支。 |
| [rtsq_a5.cc:L100](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L100) | `if (UNLIKELY(curTime - lastPrintTime >= printInterval)) {` | 仅当`(UNLIKELY(curTime - lastPrintTime >= printInterval))`（当前单调时钟时刻、上次状态打印时刻、状态日志打印间隔）成立时进入此分支。 |
| [rtsq_a5.cc:L117](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L117) | `#ifdef CCL_KERNEL_AICPU` | 编译条件`ifdef CCL_KERNEL_AICPU`限定后续实现，区分Host/设备或构建能力分支。 |
| [rtsq_a5.cc:L121](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L121) | `if (UNLIKELY(ret != HCCL_SUCCESS)) {` | 仅当`(UNLIKELY(ret != HCCL_SUCCESS))`（当前调用状态）成立时进入此分支。 |
| [rtsq_a5.cc:L129](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L129) | `THROW<InternalException>(msg);` | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |
| [rtsq_a5.cc:L133](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L133) | `#endif` | 结束前述编译条件控制的实现片段。 |
| [rtsq_a5.cc:L135](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L135) | `if (checkOpExecStatusCallback_ != nullptr) {` | 仅当`(checkOpExecStatusCallback_ != nullptr)`成立时进入此分支。 |


### RtsqBase::ConfigSqTail

定义：[rtsq_base.cc:L164–L172](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L164-L172)。把新 SQ 尾通过驱动配置包装写入硬件。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_base.cc:L170](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L170)；`ConfigSqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_TAIL, value);` | ConfigSqStatusByType → [rtsq_base.cc:L130–L161](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L130-L161) | 组装并执行指定SQ属性的驱动配置；传入/处理写给驱动的SQ属性值。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### RtsqBase::ConfigSqStatusByType

定义：[rtsq_base.cc:L130–L161](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L130-L161)。构造驱动 SQ 属性配置请求并调用 halSqCqConfig；失败抛异常。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_base.cc:L149](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L149)；`drvError_t ret = halSqCqConfig(localDevId_, &configInfo);` | halSqCqConfig（未展开边界） | 设置当前调用状态为/按`halSqCqConfig(localDevId_, &configInfo)`（驱动使用的本地设备编号、驱动SQ/CQ配置参数）；外部驱动边界：写SQ/CQ属性，检查驱动错误。 |
| [rtsq_base.cc:L155](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L155)；`= StringFormat("RtsqBase::%s call halSqCqConfig failed, localDevId %u, ret %d", __func__, localDevId_, ret);` | StringFormat（未展开边界） | 组装带上下文的错误或状态文本；传入/处理驱动使用的本地设备编号、当前调用状态。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [rtsq_base.cc:L151](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L151) | `if (UNLIKELY(ret != 0)) {` | 仅当`(UNLIKELY(ret != 0))`（当前调用状态）成立时进入此分支。 |
| [rtsq_base.cc:L157](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L157) | `THROW<DrvApiException>(formatStr);` | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |


### RtsqBase::QuerySqHead

定义：[rtsq_base.cc:L123](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L123)。查询硬件 SQ 当前头位置。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_base.cc:L123](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L123)；`u32 RtsqBase::QuerySqHead() const { return QuerySqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_HEAD); }` | QuerySqStatusByType → [rtsq_base.cc:L66–L100](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L66-L100) | 本行定义并直接执行RtsqBase::QuerySqHead：查询硬件 SQ 当前头位置；组装并执行指定SQ属性的驱动查询。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### RtsqBase::QuerySqTail

定义：[rtsq_base.cc:L125](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L125)。查询硬件 SQ 当前尾位置。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_base.cc:L125](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L125)；`u32 RtsqBase::QuerySqTail() const { return QuerySqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_TAIL); }` | QuerySqStatusByType → [rtsq_base.cc:L66–L100](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L66-L100) | 本行定义并直接执行RtsqBase::QuerySqTail：查询硬件 SQ 当前尾位置；组装并执行指定SQ属性的驱动查询。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### RtsqBase::QuerySqStatusByType

定义：[rtsq_base.cc:L66–L100](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L66-L100)。调用 halSqCqQuery 查询指定硬件 SQ 属性；失败抛异常。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [rtsq_base.cc:L83](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L83)；`drvError_t ret = halSqCqQuery(localDevId_, &queryInfo);` | halSqCqQuery（未展开边界） | 设置当前调用状态为/按`halSqCqQuery(localDevId_, &queryInfo)`（驱动使用的本地设备编号、驱动SQ/CQ查询参数与结果）；外部驱动边界：读SQ/CQ属性，检查驱动错误。 |
| [rtsq_base.cc:L87](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L87)；`std::string formatStr = StringFormat(` | StringFormat（未展开边界） | 设置std::string formatStr为/按`StringFormat(`；组装带上下文的错误或状态文本。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [rtsq_base.cc:L85](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L85) | `if (ret != 0) {` | 仅当`(ret != 0)`（当前调用状态）成立时进入此分支。 |
| [rtsq_base.cc:L93](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc#L93) | `THROW<DrvApiException>(formatStr);` | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |


### IAicpuTsThread::LaunchTask

定义：[aicpu_ts_thread_interface.cc:L70–L88](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L70-L88)。从 StreamLite 取得 RTSQ 并调用具体队列发射。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [aicpu_ts_thread_interface.cc:L74](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L74)；`RtsqBase* rtsqA5 = static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq();` | GetRtsq → [stream_lite.cc:L47](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/stream_lite.cc#L47) | 设置具体RTSQ执行队列为/按`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq()`（接口层保存的StreamLite地址）；返回当前StreamLite持有的具体执行队列。 |
| [aicpu_ts_thread_interface.cc:L81](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L81)；`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId());` | GetId（未展开边界） | 为读取执行流/通知资源的实际ID补入`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId())`（接口层保存的StreamLite地址）；本行是参数/结构化初始化续行。 |
| [aicpu_ts_thread_interface.cc:L84](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L84)；`rtsqA5->LaunchTask();` | LaunchTask → [rtsq_a5.cc:L288–L379](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L288-L379) | 将已生成的任务提交到具体RTSQ队列，不等同于全部任务完成；传入/处理具体RTSQ执行队列的LaunchTask字段。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### IAicpuTsThread::NotifyWait

定义：[aicpu_ts_thread_interface.cc:L109–L128](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L109-L128)。将硬件通知ID与超时交给具体 RTSQ 等待任务。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [aicpu_ts_thread_interface.cc:L113](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L113)；`RtsqBase* rtsqA5 = static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq();` | GetRtsq → [stream_lite.cc:L47](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/stream_lite.cc#L47) | 设置具体RTSQ执行队列为/按`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq()`（接口层保存的StreamLite地址）；返回当前StreamLite持有的具体执行队列。 |
| [aicpu_ts_thread_interface.cc:L120](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L120)；`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), notifyId, timeout);` | GetId（未展开边界） | 为读取执行流/通知资源的实际ID补入`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), notifyId, timeout)`（接口层保存的StreamLite地址、硬件通知ID、超时秒数）；本行是参数/结构化初始化续行。 |
| [aicpu_ts_thread_interface.cc:L123](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L123)；`rtsqA5->NotifyWait(notifyId, timeout);` | NotifyWait → [rtsq_a5.cc:L500–L514](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L500-L514) | 生成指定硬件通知ID的等待SQE；传入/处理具体RTSQ执行队列的NotifyWait字段、硬件通知ID、超时秒数。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### IAicpuTsThread::NotifyRecordLoc

定义：[aicpu_ts_thread_interface.cc:L131–L150](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L131-L150)。从 StreamLite 取得 RTSQ 并生成本地通知记录。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [aicpu_ts_thread_interface.cc:L135](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L135)；`RtsqBase* rtsqA5 = static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq();` | GetRtsq → [stream_lite.cc:L47](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/stream_lite.cc#L47) | 设置具体RTSQ执行队列为/按`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq()`（接口层保存的StreamLite地址）；返回当前StreamLite持有的具体执行队列。 |
| [aicpu_ts_thread_interface.cc:L142](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L142)；`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), notifyId);` | GetId（未展开边界） | 为读取执行流/通知资源的实际ID补入`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), notifyId)`（接口层保存的StreamLite地址、硬件通知ID）；本行是参数/结构化初始化续行。 |
| [aicpu_ts_thread_interface.cc:L145](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L145)；`rtsqA5->NotifyRecordLoc(notifyId);` | NotifyRecordLoc → [rtsq_a5.cc:L517–L527](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L517-L527) | 生成指定硬件ID的本地通知记录SQE；传入/处理具体RTSQ执行队列的NotifyRecordLoc字段、硬件通知ID。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### IAicpuTsThread::SdmaCopy

定义：[aicpu_ts_thread_interface.cc:L153–L188](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L153-L188)。检查单次 SDMA 字节范围并转换为32位长度，转换源/目标顺序交给 RTSQ。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [aicpu_ts_thread_interface.cc:L168](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L168)；`RtsqBase* rtsqA5 = static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq();` | GetRtsq → [stream_lite.cc:L47](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/stream_lite.cc#L47) | 设置具体RTSQ执行队列为/按`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq()`（接口层保存的StreamLite地址）；返回当前StreamLite持有的具体执行队列。 |
| [aicpu_ts_thread_interface.cc:L178](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L178)；`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), static_cast<unsigned long long>(dstAddr),` | GetId（未展开边界） | 为读取执行流/通知资源的实际ID补入`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), static_cast<unsigned long long>(dstAddr),`（接口层保存的StreamLite地址、目标地址整数表示）；本行是参数/结构化初始化续行。 |
| [aicpu_ts_thread_interface.cc:L183](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L183)；`rtsqA5->SdmaCopy(srcAddr, dstAddr, sizeByteNarrowed, 0);` | SdmaCopy → [rtsq_a5.cc:L558–L575](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L558-L575) | 按本端地址/字节长度生成SDMA复制SQE；传入/处理具体RTSQ执行队列的SdmaCopy字段、源地址整数表示、目标地址整数表示。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [aicpu_ts_thread_interface.cc:L159](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L159) | `if (sizeByte > 0x100000000ULL) {` | 仅当`(sizeByte > 0x100000000ULL)`（SDMA字节长度）成立时进入此分支。 |


### IAicpuTsThread::SdmaReduce

定义：[aicpu_ts_thread_interface.cc:L191–L241](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L191-L241)。检查字节长度与归约映射并构造 ReduceIn，交给 RTSQ SDMA归约。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [aicpu_ts_thread_interface.cc:L208](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L208)；`RtsqBase* rtsqA5 = static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq();` | GetRtsq → [stream_lite.cc:L47](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/stream_lite.cc#L47) | 设置具体RTSQ执行队列为/按`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq()`（接口层保存的StreamLite地址）；返回当前StreamLite持有的具体执行队列。 |
| [aicpu_ts_thread_interface.cc:L211](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L211)；`CHK_RET(CheckDataTypeAndReduceOp(dataTypeRaw, reduceOpRaw));` | CheckDataTypeAndReduceOp（未展开边界） | 检查公开归约类型/操作是否存在底层支持映射；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_thread_interface.cc:L213](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L213)；`DataType dataType = mapU32ToDataType.at(dataTypeRaw);` | at（未展开边界） | 设置元素数据类型为/按`mapU32ToDataType.at(dataTypeRaw)`（接口数据类型原始枚举）；调用at，使用接口数据类型原始枚举。 |
| [aicpu_ts_thread_interface.cc:L215](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L215)；`ReduceOp reduceOp = mapU32ToReduceOp.at(reduceOpRaw);` | at（未展开边界） | 设置归约操作为/按`mapU32ToReduceOp.at(reduceOpRaw)`（接口归约操作原始枚举）；调用at，使用接口归约操作原始枚举。 |
| [aicpu_ts_thread_interface.cc:L229](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L229)；`__func__, static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), static_cast<unsigned long long>(dstAddr),` | GetId（未展开边界） | 为读取执行流/通知资源的实际ID补入`__func__, static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), static_cast<unsigned long long>(dstAddr),`（接口层保存的StreamLite地址、目标地址整数表示）；本行是参数/结构化初始化续行。 |
| [aicpu_ts_thread_interface.cc:L231](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L231)；`static_cast<unsigned long long>(srcAddr), sizeByteNarrowed, dataTypeRaw, dataType.Describe().c_str(),` | Describe（未展开边界）；c_str（未展开边界） | 为读取执行流/通知资源的实际ID；取得对象诊断文本用于日志补入`static_cast<unsigned long long>(srcAddr), sizeByteNarrowed, dataTypeRaw, dataType.Describe().c_str(),`（源地址整数表示、接口数据类型原始枚举、元素数据类型的Describe字段）；本行是参数/结构化初始化续行。 |
| [aicpu_ts_thread_interface.cc:L233](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L233)；`reduceOpRaw, reduceOp.Describe().c_str());` | Describe（未展开边界）；c_str（未展开边界） | 为读取执行流/通知资源的实际ID；取得对象诊断文本用于日志补入`reduceOpRaw, reduceOp.Describe().c_str())`（接口归约操作原始枚举、归约操作的Describe字段）；本行是参数/结构化初始化续行。 |
| [aicpu_ts_thread_interface.cc:L236](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L236)；`rtsqA5->SdmaReduce(srcAddr, dstAddr, sizeByteNarrowed, 0, reduceIn);` | SdmaReduce → [rtsq_a5.cc:L592–L628](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc#L592-L628) | 按本端地址/字节长度和归约方式生成SDMA归约SQE；传入/处理具体RTSQ执行队列的SdmaReduce字段、源地址整数表示、目标地址整数表示、底层归约类型/操作描述。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [aicpu_ts_thread_interface.cc:L199](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L199) | `if (sizeByte > 0x100000000ULL) {` | 仅当`(sizeByte > 0x100000000ULL)`（SDMA字节长度）成立时进入此分支。 |
| [aicpu_ts_thread_interface.cc:L211](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc#L211) | `CHK_RET(CheckDataTypeAndReduceOp(dataTypeRaw, reduceOpRaw));` | 检查公开归约类型/操作是否存在底层支持映射；返回非成功时由检查宏立即向上传递。 |


### HcclGetRankId

定义：[coll_comm_rank_graph_a_adpt.cc:L387–L421](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L387-L421)。取得通信域中的本端 Rank ID，按编译路径与新旧域对象适配。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [coll_comm_rank_graph_a_adpt.cc:L396](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L396)；`HCCLV2_FUNC_RUN([&]() -> HcclResult {` | HCCLV2_FUNC_RUN（未展开边界） | 定义逐片处理回调，捕获当前连接/配置上下文；回调参数描述本地与远端同一分片及其首尾位置。 |
| [coll_comm_rank_graph_a_adpt.cc:L400](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L400)；`CHK_RET(GetRankGraphFromComm(comm, &rankGraph));` | GetRankGraphFromComm（未展开边界） | 调用GetRankGraphFromComm，使用通信域句柄；返回非成功时由检查宏立即向上传递。 |
| [coll_comm_rank_graph_a_adpt.cc:L402](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L402)；`CHK_RET(rankGraph->GetRankId(rank));` | GetRankId（未展开边界） | 调用GetRankId，使用Rank编号；返回非成功时由检查宏立即向上传递。 |
| [coll_comm_rank_graph_a_adpt.cc:L412](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L412)；`CHK_RET(hcclComm->GetUserRank(tmpRankId));` | GetUserRank（未展开边界） | 取得当前域中本端Rank编号；返回非成功时由检查宏立即向上传递。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [coll_comm_rank_graph_a_adpt.cc:L392](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L392) | `CHK_PTR_NULL(comm);` | 检查`comm`（通信域句柄）不是空对象；宏命中失败条件时立即返回对应指针错误。 |
| [coll_comm_rank_graph_a_adpt.cc:L394](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L394) | `CHK_PTR_NULL(rank);` | 检查`rank`（Rank编号）不是空对象；宏命中失败条件时立即返回对应指针错误。 |
| [coll_comm_rank_graph_a_adpt.cc:L400](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L400) | `CHK_RET(GetRankGraphFromComm(comm, &rankGraph));` | 调用GetRankGraphFromComm，使用通信域句柄；返回非成功时由检查宏立即向上传递。 |
| [coll_comm_rank_graph_a_adpt.cc:L402](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L402) | `CHK_RET(rankGraph->GetRankId(rank));` | 调用GetRankId，使用Rank编号；返回非成功时由检查宏立即向上传递。 |
| [coll_comm_rank_graph_a_adpt.cc:L412](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L412) | `CHK_RET(hcclComm->GetUserRank(tmpRankId));` | 取得当前域中本端Rank编号；返回非成功时由检查宏立即向上传递。 |


### HcclGetRankSize

定义：[coll_comm_rank_graph_a_adpt.cc:L350–L384](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L350-L384)。取得域内 Rank 总数，输出的是 Rank 条数而非字节容量。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [coll_comm_rank_graph_a_adpt.cc:L359](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L359)；`HCCLV2_FUNC_RUN([&]() -> HcclResult {` | HCCLV2_FUNC_RUN（未展开边界） | 定义逐片处理回调，捕获当前连接/配置上下文；回调参数描述本地与远端同一分片及其首尾位置。 |
| [coll_comm_rank_graph_a_adpt.cc:L363](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L363)；`CHK_RET(GetRankGraphFromComm(comm, &rankGraph));` | GetRankGraphFromComm（未展开边界） | 调用GetRankGraphFromComm，使用通信域句柄；返回非成功时由检查宏立即向上传递。 |
| [coll_comm_rank_graph_a_adpt.cc:L365](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L365)；`CHK_RET(rankGraph->GetRankSize(rankSize));` | GetRankSize（未展开边界） | 取得域Rank总数；返回非成功时由检查宏立即向上传递。 |
| [coll_comm_rank_graph_a_adpt.cc:L375](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L375)；`CHK_RET(hcclComm->GetRankSize(tmpRankSize));` | GetRankSize（未展开边界） | 取得域Rank总数；返回非成功时由检查宏立即向上传递。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [coll_comm_rank_graph_a_adpt.cc:L355](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L355) | `CHK_PTR_NULL(comm);` | 检查`comm`（通信域句柄）不是空对象；宏命中失败条件时立即返回对应指针错误。 |
| [coll_comm_rank_graph_a_adpt.cc:L357](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L357) | `CHK_PTR_NULL(rankSize);` | 检查`rankSize`（域内Rank总数）不是空对象；宏命中失败条件时立即返回对应指针错误。 |
| [coll_comm_rank_graph_a_adpt.cc:L363](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L363) | `CHK_RET(GetRankGraphFromComm(comm, &rankGraph));` | 调用GetRankGraphFromComm，使用通信域句柄；返回非成功时由检查宏立即向上传递。 |
| [coll_comm_rank_graph_a_adpt.cc:L365](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L365) | `CHK_RET(rankGraph->GetRankSize(rankSize));` | 取得域Rank总数；返回非成功时由检查宏立即向上传递。 |
| [coll_comm_rank_graph_a_adpt.cc:L375](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_rank_graph_a_adpt.cc#L375) | `CHK_RET(hcclComm->GetRankSize(tmpRankSize));` | 取得域Rank总数；返回非成功时由检查宏立即向上传递。 |


### AicpuTsUrmaChannel::Init

定义：[aicpu_ts_urma_channel.cc:L265–L304](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L265-L304)。UB_CTP具体通道初始化：端点/监听/Socket/属性/连接/通知与Host UB传输对象。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [aicpu_ts_urma_channel.cc:L276](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L276)；`CHK_RET(ParseInputParam());` | ParseInputParam（未展开边界） | 解析Endpoint与Hcomm描述，取得内存和驱动资源；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L278](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L278)；`CHK_RET(hrtGetDevice(&devLogicId));` | hrtGetDevice（未展开边界） | 读取当前运行时逻辑设备编号；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L280](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L280)；`CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));` | hrtGetDevicePhyIdByIndex（未展开边界） | 把逻辑设备编号转换为物理设备编号；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L282](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L282)；`CHK_RET(StartListen());` | StartListen（未展开边界） | 在具体端点上准备监听；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L284](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L284)；`CHK_RET(BuildSocket());` | BuildSocket（未展开边界） | 准备具体通道的Socket对象；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L286](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L286)；`CHK_RET(BuildAttr());` | BuildAttr（未展开边界） | 形成具体通道的UB资源属性；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L293](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L293)；`CHK_RET(HccpRaGetDevBaseAttr(rdmaHandle_, &devBaseAttr_));` | HccpRaGetDevBaseAttr（未展开边界） | 外部网络适配边界：取得UB单WR最大传输等设备能力；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L295](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L295)；`CHK_RET(BuildConnection());` | BuildConnection（未展开边界） | 按协议与驱动资源创建Host UB连接；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L297](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L297)；`CHK_RET(BuildNotify());` | BuildNotify（未展开边界） | 准备通道本地/远端通知资源描述；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L299](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L299)；`CHK_RET(BuildUbMemTransport());` | BuildUbMemTransport → [aicpu_ts_urma_channel.cc:L174–L212](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L174-L212) | 构造Host UbMemTransport保存连接/通知/内存交换资源；返回非成功时由检查宏立即向上传递。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [aicpu_ts_urma_channel.cc:L276](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L276) | `CHK_RET(ParseInputParam());` | 解析Endpoint与Hcomm描述，取得内存和驱动资源；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L278](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L278) | `CHK_RET(hrtGetDevice(&devLogicId));` | 读取当前运行时逻辑设备编号；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L280](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L280) | `CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));` | 把逻辑设备编号转换为物理设备编号；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L282](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L282) | `CHK_RET(StartListen());` | 在具体端点上准备监听；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L284](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L284) | `CHK_RET(BuildSocket());` | 准备具体通道的Socket对象；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L286](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L286) | `CHK_RET(BuildAttr());` | 形成具体通道的UB资源属性；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L293](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L293) | `CHK_RET(HccpRaGetDevBaseAttr(rdmaHandle_, &devBaseAttr_));` | 外部网络适配边界：取得UB单WR最大传输等设备能力；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L295](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L295) | `CHK_RET(BuildConnection());` | 按协议与驱动资源创建Host UB连接；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L297](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L297) | `CHK_RET(BuildNotify());` | 准备通道本地/远端通知资源描述；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L299](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L299) | `CHK_RET(BuildUbMemTransport());` | 构造Host UbMemTransport保存连接/通知/内存交换资源；返回非成功时由检查宏立即向上传递。 |


### AicpuTsUrmaChannel::BuildUbMemTransport

定义：[aicpu_ts_urma_channel.cc:L174–L212](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L174-L212)。按已建立Socket与端点对描述构造Host UbMemTransport。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [aicpu_ts_urma_channel.cc:L180](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L180)；`locCntNotifyRes.vec.clear();` | clear（未展开边界） | 调用clear。 |
| [aicpu_ts_urma_channel.cc:L182](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L182)；`locCntNotifyRes.desc.clear();` | clear（未展开边界） | 调用clear。 |
| [aicpu_ts_urma_channel.cc:L187](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L187)；`Hccl::LinkData linkData = BuildDefaultLinkData();` | BuildDefaultLinkData（未展开边界） | 设置按端点对构造的链路描述为/按`BuildDefaultLinkData()`；调用BuildDefaultLinkData。 |
| [aicpu_ts_urma_channel.cc:L189](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L189)；`CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));` | EndpointDescPairToLinkData（未展开边界） | 把本端与远端端点描述转换为链路属性；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L192](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L192)；`bool isRecvFirst = socket.GetRole() == Hccl::SocketRole::CLIENT ? true : false;` | GetRole（未展开边界） | 设置按Socket角色决定的描述交换先后顺序为/按`socket.GetRole() == Hccl::SocketRole::CLIENT ? true : false`（已连接的Socket对象的GetRole字段）；调用GetRole，使用已连接的Socket对象的GetRole字段。 |
| [aicpu_ts_urma_channel.cc:L198](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L198)；`memTransport_ = std::make_unique<Hccl::UbMemTransport>(` | std::make_unique<UbMemTransport>（未展开边界） | 为前述多行表达式补入`memTransport_ = std::make_unique<Hccl::UbMemTransport>(`（Host侧UB内存传输对象）；本行是参数/结构化初始化续行。 |
| [aicpu_ts_urma_channel.cc:L208](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L208)；`socket_->Describe().c_str(), linkData.Describe().c_str());` | Describe（未展开边界）；c_str（未展开边界） | 为取得对象诊断文本用于日志补入`socket_->Describe().c_str(), linkData.Describe().c_str())`（按端点对构造的链路描述的Describe字段）；本行是参数/结构化初始化续行。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [aicpu_ts_urma_channel.cc:L189](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L189) | `CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));` | 把本端与远端端点描述转换为链路属性；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L192](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L192) | `bool isRecvFirst = socket.GetRole() == Hccl::SocketRole::CLIENT ? true : false;` | 设置按Socket角色决定的描述交换先后顺序为/按`socket.GetRole() == Hccl::SocketRole::CLIENT ? true : false`（已连接的Socket对象的GetRole字段）；调用GetRole，使用已连接的Socket对象的GetRole字段。 |
| [aicpu_ts_urma_channel.cc:L196](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L196) | `EXCEPTION_CATCH(` | 在异常捕获边界执行后续表达式；异常按后续处理语句转换成HCCL状态或提前返回。 |


### AicpuTsUrmaChannel::PackOpData

定义：[aicpu_ts_urma_channel.cc:L350–L384](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L350-L384)。将Host UB transport唯一标识打包供设备恢复传输对象。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [aicpu_ts_urma_channel.cc:L356](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L356)；`dataVec.resize(Hccl::AicpuResMgrType::__COUNT__);` | resize（未展开边界） | 调用resize，使用分模块的设备资源包数组的resize字段；传入/处理分模块的设备资源包数组的resize字段。 |
| [aicpu_ts_urma_channel.cc:L361](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L361)；`CHK_RET(SetModuleDataName(dataVec[resType], "UbMemTransport"));` | SetModuleDataName（未展开边界） | 为资源模块设置供设备恢复识别的名称；返回非成功时由检查宏立即向上传递。 |
| [aicpu_ts_urma_channel.cc:L368](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L368)；`binaryStream << memTransport_->GetUniqueIdV2();` | GetUniqueIdV2（未展开边界） | 序列化Host transport的设备恢复描述；传入/处理资源标识序列化/反序列化流、Host侧UB内存传输对象的GetUniqueIdV2字段。 |
| [aicpu_ts_urma_channel.cc:L371](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L371)；`binaryStream.Dump(result);` | Dump（未展开边界） | 把序列化流内容输出到字节数组；传入/处理资源标识序列化/反序列化流的Dump字段。 |
| [aicpu_ts_urma_channel.cc:L379](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L379)；`data = helper.GetPackedData(dataVec);` | GetPackedData（未展开边界） | 设置data为/按`helper.GetPackedData(dataVec)`（分模块的设备资源包数组）；将分模块资源内容打包为设备恢复数据。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [aicpu_ts_urma_channel.cc:L361](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc#L361) | `CHK_RET(SetModuleDataName(dataVec[resType], "UbMemTransport"));` | 为资源模块设置供设备恢复识别的名称；返回非成功时由检查宏立即向上传递。 |


### AicpuChannelProcess::ParsePackData

定义：[aicpu_channel_process.cc:L52–L111](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L52-L111)。按打包TransportType创建 UB/RoCE/P2P设备传输对象，UB分支绑定缓存回调并输出对象句柄。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [aicpu_channel_process.cc:L56](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L56)；`HCCL_DEBUG("[HcclCommAicpu][%s] data: ptr[%p], size[%u]", __func__, data.data(), data.size());` | data（未展开边界）；size（未展开边界） | 记录AicpuChannelProcess::ParsePackData的调试诊断；日志本身不执行传输。 |
| [aicpu_channel_process.cc:L58](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L58)；`Hccl::BinaryStream binaryStream(data);` | binaryStream（未展开边界） | 调用binaryStream，使用资源标识序列化/反序列化流；对象涉及资源标识序列化/反序列化流。 |
| [aicpu_channel_process.cc:L66](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L66)；`Hccl::BinaryStream binaryStreamForType(transpUniqueId);` | binaryStreamForType（未展开边界） | 调用binaryStreamForType，使用待恢复的transport序列化标识；对象涉及待恢复的transport序列化标识。 |
| [aicpu_channel_process.cc:L80](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L80)；`(ubTransportLiteImpl = std::make_unique<Hccl::UbTransportLiteImpl>(transpUniqueId)), return HCCL_E_PTR);` | std::make_unique<UbTransportLiteImpl>（未展开边界） | 为前述多行表达式补入`(ubTransportLiteImpl = std::make_unique<Hccl::UbTransportLiteImpl>(transpUniqueId)), return HCCL_E_PTR)`（待恢复的transport序列化标识）；本行是参数/结构化初始化续行。 |
| [aicpu_channel_process.cc:L84](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L84)；`CHK_RET(ubTransportLiteImpl->SetNeedCacheTaskCallback(hcomm::AicpuTaskCacheManager::NeedCacheTask));` | SetNeedCacheTaskCallback（未展开边界） | 绑定设备任务缓存需求判断回调；返回非成功时由检查宏立即向上传递。 |
| [aicpu_channel_process.cc:L86](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L86)；`CHK_RET(ubTransportLiteImpl->SetAddWqeArrayCallback(hcomm::AicpuTaskCacheManager::AddWqeArray));` | SetAddWqeArrayCallback（未展开边界） | 绑定设备任务缓存WQE数组保存回调；返回非成功时由检查宏立即向上传递。 |
| [aicpu_channel_process.cc:L88](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L88)；`handle = ReinterpretAs<uint64_t>(ubTransportLiteImpl.get());` | get（未展开边界） | 把具体UbTransportLiteImpl对象地址编码为设备ChannelHandle；C原语随后以BaseTransportLiteImpl基类指针进行虚派发。 |
| [aicpu_channel_process.cc:L90](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L90)；`transportMap_.insert({handle, std::move(ubTransportLiteImpl)});` | insert（未展开边界）；std::move（未展开边界） | 将UB对象所有权移入transportMap_，以刚输出的设备句柄为键保持其生命周期。 |
| [aicpu_channel_process.cc:L94](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L94)；`CHK_RET(CreateAndInsertTransport<Hccl::RoceTransportLiteImpl>(transpUniqueId, handle, transportMap_));` | CreateAndInsertTransport<RoceTransportLiteImpl>（未展开边界） | 执行本行包裹的资源/任务调用；返回非成功时由检查宏立即向上传递。 |
| [aicpu_channel_process.cc:L98](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L98)；`CHK_RET(CreateAndInsertTransport<Hccl::P2PTransportLiteImpl>(transpUniqueId, handle, transportMap_));` | CreateAndInsertTransport<P2PTransportLiteImpl>（未展开边界） | 执行本行包裹的资源/任务调用；返回非成功时由检查宏立即向上传递。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [aicpu_channel_process.cc:L74](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L74) | `if (transType == Hccl::TransportType::UB \|\| transType == Hccl::TransportType::UBoE) {` | 仅当`(transType == Hccl::TransportType::UB \|\| transType == Hccl::TransportType::UBoE)`（序列化transport类型）成立时进入此分支。 |
| [aicpu_channel_process.cc:L78](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L78) | `EXCEPTION_CATCH(` | 在异常捕获边界执行后续表达式；异常按后续处理语句转换成HCCL状态或提前返回。 |
| [aicpu_channel_process.cc:L82](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L82) | `CHK_SMART_PTR_NULL(ubTransportLiteImpl);` | 检查`ubTransportLiteImpl`不是空对象；宏命中失败条件时立即返回对应指针错误。 |
| [aicpu_channel_process.cc:L84](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L84) | `CHK_RET(ubTransportLiteImpl->SetNeedCacheTaskCallback(hcomm::AicpuTaskCacheManager::NeedCacheTask));` | 绑定设备任务缓存需求判断回调；返回非成功时由检查宏立即向上传递。 |
| [aicpu_channel_process.cc:L86](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L86) | `CHK_RET(ubTransportLiteImpl->SetAddWqeArrayCallback(hcomm::AicpuTaskCacheManager::AddWqeArray));` | 绑定设备任务缓存WQE数组保存回调；返回非成功时由检查宏立即向上传递。 |
| [aicpu_channel_process.cc:L92](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L92) | `} else if (transType == Hccl::TransportType::ROCE) {` | 仅当`(transType == Hccl::TransportType::ROCE)`（序列化transport类型）成立时进入此分支。 |
| [aicpu_channel_process.cc:L94](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L94) | `CHK_RET(CreateAndInsertTransport<Hccl::RoceTransportLiteImpl>(transpUniqueId, handle, transportMap_));` | 执行本行包裹的资源/任务调用；返回非成功时由检查宏立即向上传递。 |
| [aicpu_channel_process.cc:L96](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L96) | `} else if (transType == Hccl::TransportType::P2P) {` | 仅当`(transType == Hccl::TransportType::P2P)`（序列化transport类型）成立时进入此分支。 |
| [aicpu_channel_process.cc:L98](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L98) | `CHK_RET(CreateAndInsertTransport<Hccl::P2PTransportLiteImpl>(transpUniqueId, handle, transportMap_));` | 执行本行包裹的资源/任务调用；返回非成功时由检查宏立即向上传递。 |
| [aicpu_channel_process.cc:L100](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/aicpu_channel_process.cc#L100) | `} else {` | 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。 |


### ParseData

定义：[ub_transport_lite_impl.cc:L1234–L1334](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1234-L1334)。按批传输种类解析本端/远端地址、字节数或归约元素数与通知槽，归约最终换算字节。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [ub_transport_lite_impl.cc:L1270](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1270)；`CHK_RET(ParasReduceData(transferDesc, len, dataType, reduceOp));` | ParasReduceData → [ub_transport_lite_impl.cc:L1207–L1231](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1207-L1231) | 读取批归约count/type/op并检查支持组合；返回非成功时由检查宏立即向上传递。 |
| [ub_transport_lite_impl.cc:L1280](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1280)；`CHK_RET(ParasReduceData(transferDesc, len, dataType, reduceOp));` | ParasReduceData → [ub_transport_lite_impl.cc:L1207–L1231](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1207-L1231) | 读取批归约count/type/op并检查支持组合；返回非成功时由检查宏立即向上传递。 |
| [ub_transport_lite_impl.cc:L1310](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1310)；`CHK_RET(CheckReduceHcommDataTypeAndHcommReduceOp(dataType, reduceOp));` | CheckReduceHcommDataTypeAndHcommReduceOp（未展开边界） | 批归约拒绝缺失映射与RESERVED类型/操作；返回非成功时由检查宏立即向上传递。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [ub_transport_lite_impl.cc:L1242](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1242) | `if (transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE) {` | 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE)`（当前公开批传输描述的transType字段）成立时进入此分支。 |
| [ub_transport_lite_impl.cc:L1252](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1252) | `} else if (transferDesc.transType == HCOMM_TRANSFER_TYPE_READ) {` | 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_READ)`（当前公开批传输描述的transType字段）成立时进入此分支。 |
| [ub_transport_lite_impl.cc:L1262](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1262) | `} else if (transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE_REDUCE) {` | 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE_REDUCE)`（当前公开批传输描述的transType字段）成立时进入此分支。 |
| [ub_transport_lite_impl.cc:L1270](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1270) | `CHK_RET(ParasReduceData(transferDesc, len, dataType, reduceOp));` | 读取批归约count/type/op并检查支持组合；返回非成功时由检查宏立即向上传递。 |
| [ub_transport_lite_impl.cc:L1272](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1272) | `} else if (transferDesc.transType == HCOMM_TRANSFER_TYPE_READ_REDUCE) {` | 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_READ_REDUCE)`（当前公开批传输描述的transType字段）成立时进入此分支。 |
| [ub_transport_lite_impl.cc:L1280](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1280) | `CHK_RET(ParasReduceData(transferDesc, len, dataType, reduceOp));` | 读取批归约count/type/op并检查支持组合；返回非成功时由检查宏立即向上传递。 |
| [ub_transport_lite_impl.cc:L1282](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1282) | `} else if (transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY) {` | 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY)`（当前公开批传输描述的transType字段）成立时进入此分支。 |
| [ub_transport_lite_impl.cc:L1294](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1294) | `} else if (transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY) {` | 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY)`（当前公开批传输描述的transType字段）成立时进入此分支。 |
| [ub_transport_lite_impl.cc:L1310](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1310) | `CHK_RET(CheckReduceHcommDataTypeAndHcommReduceOp(dataType, reduceOp));` | 批归约拒绝缺失映射与RESERVED类型/操作；返回非成功时由检查宏立即向上传递。 |
| [ub_transport_lite_impl.cc:L1312](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1312) | `} else if (transferDesc.transType == HCOMM_TRANSFER_TYPE_NOTIFY_RECORD) {` | 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_NOTIFY_RECORD)`（当前公开批传输描述的transType字段）成立时进入此分支。 |
| [ub_transport_lite_impl.cc:L1318](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1318) | `} else {` | 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。 |
| [ub_transport_lite_impl.cc:L1326](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1326) | `if (reduceOp != HcommReduceOp::HCOMM_REDUCE_RESERVED) { // 对于规约类型, size = count * sizeof(datatype)` | 仅当`(reduceOp != HcommReduceOp::HCOMM_REDUCE_RESERVED)`（归约操作）成立时进入此分支。 |


### ParasReduceData

定义：[ub_transport_lite_impl.cc:L1207–L1231](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1207-L1231)。取得批归约元素数、类型和操作并检查支持映射。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [ub_transport_lite_impl.cc:L1219](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1219)；`auto ret = CheckReduceHcommDataTypeAndHcommReduceOp(dataType, reduceOp);` | CheckReduceHcommDataTypeAndHcommReduceOp（未展开边界） | 设置当前调用状态为/按`CheckReduceHcommDataTypeAndHcommReduceOp(dataType, reduceOp)`（元素数据类型、归约操作）；批归约拒绝缺失映射与RESERVED类型/操作。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [ub_transport_lite_impl.cc:L1221](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc#L1221) | `CHK_PRT_RET(` | 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。 |


### UbConnLite::FillOneSqeWrite

定义：[ub_conn_lite.cc:L324–L350](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L324-L350)。将远端公共字段与本地SGE填入UB数据WQE，零长度时取消SGE。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [ub_conn_lite.cc:L332](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L332)；`HCCL_INFO("[UbConnLite::%s] start, loc size[%llu]", __func__, loc.GetSize());` | GetSize（未展开边界） | 记录UbConnLite::FillOneSqeWrite的状态/性能诊断，字段包含本端缓冲区/地址的GetSize字段；日志本身不执行传输。 |
| [ub_conn_lite.cc:L337](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L337)；`FillCommSqe(&(sqe->comm), rmt, cfg, opCode, slicePos);` | FillCommSqe → [ub_conn_lite.cc:L45–L136](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L45-L136) | 填写远端地址/token、操作码、分片顺序和完成标志；传入/处理当前UB WQE结构的comm字段、远端缓冲区/地址、UB WQE保序/完成配置、UB读写操作码、当前分片首/中/尾位置。 |
| [ub_conn_lite.cc:L339](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L339)；`FillLocalSgeSqe(&(sqe->u.sge), loc);` | FillLocalSgeSqe → [ub_conn_lite.cc:L620–L640](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L620-L640) | 填写本端SGE地址、token及字节长度；传入/处理当前UB WQE结构的u.sge字段、本端缓冲区/地址。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [ub_conn_lite.cc:L341](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L341) | `if (sqe->u.sge.length == 0) {` | 仅当`(sqe->u.sge.length == 0)`（当前UB WQE结构的u.sge.length字段）成立时进入此分支。 |


### UbConnLite::FillCommSqe

定义：[ub_conn_lite.cc:L45–L136](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L45-L136)。按读写操作、切片位置和用户配置填写UB公共WQE字段。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [ub_conn_lite.cc:L99](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L99)；`s32 ret = memcpy_sp(sqe->rmtEid, RMT_EID_BYTE_SIZE, rmtReverseEid_.raw, RAW_SIZE);` | memcpy_sp（未展开边界） | 设置当前调用状态为/按`memcpy_sp(sqe->rmtEid, RMT_EID_BYTE_SIZE, rmtReverseEid_.raw, RAW_SIZE)`（当前UB WQE结构的rmtEid字段）；把WQE/SQE数据复制到设备映射队列内存，失败抛异常。 |
| [ub_conn_lite.cc:L105](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L105)；`THROW<InternalException>(StringFormat("UbConnLite::FillCommSqe memcpy_sp failed, ret = %d", ret));` | StringFormat（未展开边界） | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |
| [ub_conn_lite.cc:L114](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L114)；`sqe->rmtObjId = rmt.GetTokenId();` | GetTokenId（未展开边界） | 设置当前UB WQE结构的rmtObjId字段为/按`rmt.GetTokenId()`（远端缓冲区/地址的GetTokenId字段）；读取当前内存注册token ID。 |
| [ub_conn_lite.cc:L118](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L118)；`sqe->rmtTokenValue = rmt.GetTokenValue();` | GetTokenValue（未展开边界） | 设置当前UB WQE结构的rmtTokenValue字段为/按`rmt.GetTokenValue()`（远端缓冲区/地址的GetTokenValue字段）；读取当前内存注册token value。 |
| [ub_conn_lite.cc:L120](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L120)；`sqe->rmtAddrLow = rmt.GetAddr() & ADDR_BIT_LOW;` | GetAddr（未展开边界） | 设置当前UB WQE结构的rmtAddrLow字段为/按`rmt.GetAddr() & ADDR_BIT_LOW`（远端缓冲区/地址的GetAddr字段）；读取缓冲区起始地址。 |
| [ub_conn_lite.cc:L122](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L122)；`sqe->rmtAddrHigh = rmt.GetAddr() >> ADDR_BIT_OFFSET;` | GetAddr（未展开边界） | 设置当前UB WQE结构的rmtAddrHigh字段为/按`rmt.GetAddr() >> ADDR_BIT_OFFSET`（远端缓冲区/地址的GetAddr字段）；读取缓冲区起始地址。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [ub_conn_lite.cc:L51](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L51) | `u32 cqeEn = (cfg.cqeEn && (slicePos == SlicePosition::LAST \|\| slicePos == SlicePosition::ONLY)) ? 1 : 0;` | 设置u32 cqeEn为/按`(cfg.cqeEn && (slicePos == SlicePosition::LAST \|\| slicePos == SlicePosition::ONLY)) ? 1 : 0`（UB WQE保序/完成配置的cqeEn字段、当前分片首/中/尾位置）。 |
| [ub_conn_lite.cc:L55](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L55) | `sqe->owner = (pi == (sqDepth_ - 1)) ? 1 : 0;` | 设置当前UB WQE结构的owner字段为/按`(pi == (sqDepth_ - 1)) ? 1 : 0`（UB jetty生产指针（16位自然增长）、RTSQ或UB SQ深度）。 |
| [ub_conn_lite.cc:L62](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L62) | `if (cfg.userConfig) {` | 仅当`(cfg.userConfig)`（UB WQE保序/完成配置的userConfig字段）成立时进入此分支。 |
| [ub_conn_lite.cc:L70](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L70) | `} else {` | 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。 |
| [ub_conn_lite.cc:L73](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L73) | `if (slicePos == SlicePosition::ONLY \|\| slicePos == SlicePosition::LAST) {` | 仅当`(slicePos == SlicePosition::ONLY \|\| slicePos == SlicePosition::LAST)`（当前分片首/中/尾位置）成立时进入此分支。 |
| [ub_conn_lite.cc:L81](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L81) | `} else {` | 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。 |
| [ub_conn_lite.cc:L84](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L84) | `sqe->placeOdr = (slicePos == SlicePosition::MIDDLE) ? UB_RELAX_ORDER : cfg.placeOdr;` | 设置当前UB WQE结构的placeOdr字段为/按`(slicePos == SlicePosition::MIDDLE) ? UB_RELAX_ORDER : cfg.placeOdr`（当前分片首/中/尾位置、UB WQE保序/完成配置的placeOdr字段）。 |
| [ub_conn_lite.cc:L86](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L86) | `sqe->compOrder = (slicePos == SlicePosition::MIDDLE) ? 0 : cfg.compOrder;` | 设置当前UB WQE结构的compOrder字段为/按`(slicePos == SlicePosition::MIDDLE) ? 0 : cfg.compOrder`（当前分片首/中/尾位置、UB WQE保序/完成配置的compOrder字段）。 |
| [ub_conn_lite.cc:L88](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L88) | `sqe->fence = (slicePos == SlicePosition::MIDDLE) ? 0 : cfg.fence;` | 设置当前UB WQE结构的fence字段为/按`(slicePos == SlicePosition::MIDDLE) ? 0 : cfg.fence`（当前分片首/中/尾位置、UB WQE保序/完成配置的fence字段）。 |
| [ub_conn_lite.cc:L101](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L101) | `if (UNLIKELY(ret != 0)) {` | 仅当`(UNLIKELY(ret != 0))`（当前调用状态）成立时进入此分支。 |
| [ub_conn_lite.cc:L105](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L105) | `THROW<InternalException>(StringFormat("UbConnLite::FillCommSqe memcpy_sp failed, ret = %d", ret));` | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |


### UbConnLite::FillLocalSgeSqe

定义：[ub_conn_lite.cc:L620–L640](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L620-L640)。填入本端地址、token和字节长度字段。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [ub_conn_lite.cc:L624](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L624)；`sqe->length = loc.GetSize();` | GetSize（未展开边界） | 设置当前UB WQE结构的length字段为/按`loc.GetSize()`（本端缓冲区/地址的GetSize字段）；读取缓冲区字节长度。 |
| [ub_conn_lite.cc:L626](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L626)；`sqe->tokenId = loc.GetTokenId();` | GetTokenId（未展开边界） | 设置当前UB WQE结构的tokenId字段为/按`loc.GetTokenId()`（本端缓冲区/地址的GetTokenId字段）；读取当前内存注册token ID。 |
| [ub_conn_lite.cc:L628](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L628)；`sqe->dataAddrLow = loc.GetAddr() & ADDR_BIT_LOW;` | GetAddr（未展开边界） | 设置当前UB WQE结构的dataAddrLow字段为/按`loc.GetAddr() & ADDR_BIT_LOW`（本端缓冲区/地址的GetAddr字段）；读取缓冲区起始地址。 |
| [ub_conn_lite.cc:L630](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L630)；`sqe->dataAddrHigh = loc.GetAddr() >> ADDR_BIT_OFFSET;` | GetAddr（未展开边界） | 设置当前UB WQE结构的dataAddrHigh字段为/按`loc.GetAddr() >> ADDR_BIT_OFFSET`（本端缓冲区/地址的GetAddr字段）；读取缓冲区起始地址。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### UbConnLite::FillCommSqeReduceInfo

定义：[ub_conn_lite.cc:L139–L178](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L139-L178)。按设备WQE格式填归约类型和操作编码。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [ub_conn_lite.cc:L149](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L149)；`if ((g_ubmaDataOpMap.find(reduceOp) != g_ubmaDataOpMap.end())` | find（未展开边界）；end（未展开边界） | 仅当`((g_ubmaDataOpMap.find(reduceOp) != g_ubmaDataOpMap.end())`（归约操作）成立时进入此分支；调用find, end，使用归约操作。 |
| [ub_conn_lite.cc:L151](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L151)；`&& (g_ubmaDataTypeMap.find(dataType) != g_ubmaDataTypeMap.end())) {` | find（未展开边界）；end（未展开边界） | 补全本分支/循环判断的`&& (g_ubmaDataTypeMap.find(dataType) != g_ubmaDataTypeMap.end()))`（元素数据类型），和前面条件共同决定是否进入后续路径。 |
| [ub_conn_lite.cc:L153](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L153)；`sqeComm.inlinedata.udfData.reduceOp = g_ubmaDataOpMap.at(reduceOp);` | at（未展开边界） | 把底层归约枚举映射成UB WQE操作编码；SUM映射为0xA。 |
| [ub_conn_lite.cc:L155](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L155)；`sqeComm.inlinedata.udfData.reduceType = g_ubmaDataTypeMap.at(dataType);` | at（未展开边界） | 把底层数据类型映射成UB WQE类型编码；FP32映射为0x7。 |
| [ub_conn_lite.cc:L159](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L159)；`THROW<InvalidParamsException>(StringFormat(` | StringFormat（未展开边界） | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |
| [ub_conn_lite.cc:L161](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L161)；`"%s reduceOp[%s] or type[%s] is not supported.", __func__, reduceOp.Describe().c_str(),` | Describe（未展开边界）；c_str（未展开边界） | 为当前UbConnLite::FillCommSqeReduceInfo诊断/异常表达式提供格式文本，将报告归约操作的Describe字段；这一物理行没有数据搬运副作用。 |
| [ub_conn_lite.cc:L163](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L163)；`dataType.Describe().c_str()));` | Describe（未展开边界）；c_str（未展开边界） | 为组装带上下文的错误或状态文本；取得对象诊断文本用于日志补入`dataType.Describe().c_str()))`（元素数据类型的Describe字段）；本行是参数/结构化初始化续行。 |
| [ub_conn_lite.cc:L174](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L174)；`"[UbConnLite::%s] end, reduceOp[%s], reduceType[%s]", __func__, reduceOp.Describe().c_str(),` | Describe（未展开边界）；c_str（未展开边界） | 为当前UbConnLite::FillCommSqeReduceInfo诊断/异常表达式提供格式文本，将报告归约操作的Describe字段；这一物理行没有数据搬运副作用。 |
| [ub_conn_lite.cc:L176](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L176)；`dataType.Describe().c_str());` | Describe（未展开边界）；c_str（未展开边界） | 为取得对象诊断文本用于日志补入`dataType.Describe().c_str())`（元素数据类型的Describe字段）；本行是参数/结构化初始化续行。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [ub_conn_lite.cc:L149](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L149) | `if ((g_ubmaDataOpMap.find(reduceOp) != g_ubmaDataOpMap.end())` | 仅当`((g_ubmaDataOpMap.find(reduceOp) != g_ubmaDataOpMap.end())`（归约操作）成立时进入此分支；调用find, end，使用归约操作。 |
| [ub_conn_lite.cc:L157](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L157) | `} else {` | 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。 |
| [ub_conn_lite.cc:L159](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L159) | `THROW<InvalidParamsException>(StringFormat(` | 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。 |


### HcclThreadAcquireWithConfig

定义：[thread_c_adpt.cc:L226–L328](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L226-L328)。校验显式ThreadConfig后申请执行Thread与通知槽，V2和兼容域分别管理资源。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [thread_c_adpt.cc:L239](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L239)；`CHK_RET(ValidateThreadAcquireParams(engine, type, config, threadNum));` | ValidateThreadAcquireParams（未展开边界） | 调用ValidateThreadAcquireParams，使用请求的通信引擎、Thread类型、Thread配置数组、执行Thread条数；返回非成功时由检查宏立即向上传递。 |
| [thread_c_adpt.cc:L242](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L242)；`u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();` | Hccl::DfxDlProfFunction::GetInstance（未展开边界）；dlMsprofSysCycleTime（未展开边界） | 设置性能观测起始时间戳为/按`Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime()`；取得该管理器单例。 |
| [thread_c_adpt.cc:L246](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L246)；`std::string commId = hcclComm->GetIdentifier();` | GetIdentifier（未展开边界） | 设置通信域名称为/按`hcclComm->GetIdentifier()`（域的兼容外层对象的GetIdentifier字段）；取得通信域标识字符串。 |
| [thread_c_adpt.cc:L250](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L250)；`"Entry-%s:comm[%s] engine[%s] ThreadNum[%u].", __func__, commId.c_str(),` | c_str（未展开边界） | 为当前HcclThreadAcquireWithConfig诊断/异常表达式提供格式文本，将报告通信域名称的c_str字段；这一物理行没有数据搬运副作用。 |
| [thread_c_adpt.cc:L252](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L252)；`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum);` | GetEnumToString（未展开边界）；GetCommEngineStatusStrMap（未展开边界）；c_str（未展开边界） | 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum)`（请求的通信引擎、执行Thread条数）；本行是参数/结构化初始化续行。 |
| [thread_c_adpt.cc:L259](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L259)；`if (hcclComm->IsCommunicatorV2()) {` | IsCommunicatorV2（未展开边界） | 仅当`(hcclComm->IsCommunicatorV2())`（域的兼容外层对象的IsCommunicatorV2字段）成立时进入此分支；判断通信域是否使用V2对象实现。 |
| [thread_c_adpt.cc:L261](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L261)；`hccl::CollComm* collComm = hcclComm->GetCollComm();` | GetCollComm（未展开边界） | 设置V2通信域对象为/按`hcclComm->GetCollComm()`（域的兼容外层对象的GetCollComm字段）；从域外层对象取得V2通信域。 |
| [thread_c_adpt.cc:L265](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L265)；`CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();` | GetCommEngineResMgr（未展开边界） | 设置CommEngineResMgr* engineResMgr为/按`collComm->GetCommEngineResMgr()`（V2通信域对象的GetCommEngineResMgr字段）；取得域引擎资源管理器。 |
| [thread_c_adpt.cc:L270](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L270)；`ret = engineResMgr->HcclThreadAcquireV2(engine, threadNum, type, config, threads, threadId);` | HcclThreadAcquireV2（未展开边界） | 设置当前调用状态为/按`engineResMgr->HcclThreadAcquireV2(engine, threadNum, type, config, threads, threadId)`（请求的通信引擎、执行Thread条数、Thread类型、Thread配置数组、执行Thread句柄数组、实际执行流ID数组）；按CPU/AICPU引擎、TS类型及配置申请域内Thread/通知资源。 |
| [thread_c_adpt.cc:L278](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L278)；`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, ret);` | GetEnumToString（未展开边界）；GetCommEngineStatusStrMap（未展开边界）；c_str（未展开边界） | 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, ret)`（请求的通信引擎、执行Thread条数、当前调用状态）；本行是参数/结构化初始化续行。 |
| [thread_c_adpt.cc:L284](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L284)；`CHK_RET(HcclThreadAcquireWithConfigDfx(collComm, commId, engine, beginTime, threadNum, threads, threadId));` | HcclThreadAcquireWithConfigDfx → [thread_c_adpt.cc:L114–L184](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L114-L184) | 注册新申请Thread的域/流观测信息或回调；返回非成功时由检查宏立即向上传递。 |
| [thread_c_adpt.cc:L290](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L290)；`auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();` | GetIndependentOp（未展开边界）；GetCommEngineResMgr（未展开边界） | 设置auto& engineResMgr为/按`hcclComm->GetIndependentOp().GetCommEngineResMgr()`（域的兼容外层对象的GetIndependentOp字段）；取得旧兼容域的独立算子资源入口；取得域引擎资源管理器。 |
| [thread_c_adpt.cc:L292](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L292)；`ret = engineResMgr.HcclThreadAcquire(engine, threadNum, type, config, threads, threadId);` | HcclThreadAcquire（未展开边界） | 设置当前调用状态为/按`engineResMgr.HcclThreadAcquire(engine, threadNum, type, config, threads, threadId)`（请求的通信引擎、执行Thread条数、Thread类型、Thread配置数组、执行Thread句柄数组、实际执行流ID数组）；进入兼容资源管理器申请Thread配置。 |
| [thread_c_adpt.cc:L297](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L297)；`if (threadNum != threadId.size()) {` | size（未展开边界） | 仅当`(threadNum != threadId.size())`（执行Thread条数、实际执行流ID数组的size字段）成立时进入此分支；读取容器登记项数。 |
| [thread_c_adpt.cc:L299](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L299)；`HCCL_ERROR("[%s] threadNum [%u] != threadId.size[%zu]", __func__, threadNum, threadId.size());` | size（未展开边界） | 记录HcclThreadAcquireWithConfig的错误诊断，字段包含执行Thread条数、实际执行流ID数组的size字段；日志本身不执行传输。 |
| [thread_c_adpt.cc:L305](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L305)；`CHK_RET(HcclStreamProfilingReport(comm, threadNum, threadId.data()));` | HcclStreamProfilingReport（未展开边界）；data（未展开边界） | 兼容AICPU申请后上报实际流ID；返回非成功时由检查宏立即向上传递。 |
| [thread_c_adpt.cc:L317](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L317)；`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, ret);` | GetEnumToString（未展开边界）；GetCommEngineStatusStrMap（未展开边界）；c_str（未展开边界） | 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, ret)`（请求的通信引擎、执行Thread条数、当前调用状态）；本行是参数/结构化初始化续行。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [thread_c_adpt.cc:L234](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L234) | `CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);` | 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。 |
| [thread_c_adpt.cc:L236](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L236) | `CHK_PRT_RET(threads == nullptr, HCCL_ERROR("[%s] threads is null", __func__), HCCL_E_PTR);` | 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。 |
| [thread_c_adpt.cc:L239](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L239) | `CHK_RET(ValidateThreadAcquireParams(engine, type, config, threadNum));` | 调用ValidateThreadAcquireParams，使用请求的通信引擎、Thread类型、Thread配置数组、执行Thread条数；返回非成功时由检查宏立即向上传递。 |
| [thread_c_adpt.cc:L259](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L259) | `if (hcclComm->IsCommunicatorV2()) {` | 仅当`(hcclComm->IsCommunicatorV2())`（域的兼容外层对象的IsCommunicatorV2字段）成立时进入此分支；判断通信域是否使用V2对象实现。 |
| [thread_c_adpt.cc:L263](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L263) | `CHK_PTR_NULL(collComm);` | 检查`collComm`（V2通信域对象）不是空对象；宏命中失败条件时立即返回对应指针错误。 |
| [thread_c_adpt.cc:L267](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L267) | `CHK_PTR_NULL(engineResMgr);` | 检查`engineResMgr`不是空对象；宏命中失败条件时立即返回对应指针错误。 |
| [thread_c_adpt.cc:L272](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L272) | `if (ret != HCCL_SUCCESS) {` | 仅当`(ret != HCCL_SUCCESS)`（当前调用状态）成立时进入此分支。 |
| [thread_c_adpt.cc:L284](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L284) | `CHK_RET(HcclThreadAcquireWithConfigDfx(collComm, commId, engine, beginTime, threadNum, threads, threadId));` | 注册新申请Thread的域/流观测信息或回调；返回非成功时由检查宏立即向上传递。 |
| [thread_c_adpt.cc:L288](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L288) | `} else {` | 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。 |
| [thread_c_adpt.cc:L294](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L294) | `if (engine == CommEngine::COMM_ENGINE_AICPU) {` | 仅当`(engine == CommEngine::COMM_ENGINE_AICPU)`（请求的通信引擎）成立时进入此分支。 |
| [thread_c_adpt.cc:L297](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L297) | `if (threadNum != threadId.size()) {` | 仅当`(threadNum != threadId.size())`（执行Thread条数、实际执行流ID数组的size字段）成立时进入此分支；读取容器登记项数。 |
| [thread_c_adpt.cc:L305](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L305) | `CHK_RET(HcclStreamProfilingReport(comm, threadNum, threadId.data()));` | 兼容AICPU申请后上报实际流ID；返回非成功时由检查宏立即向上传递。 |
| [thread_c_adpt.cc:L311](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc#L311) | `if (ret != HCCL_SUCCESS) {` | 仅当`(ret != HCCL_SUCCESS)`（当前调用状态）成立时进入此分支。 |


### AddThread

定义：[launch_context.h:L49–L65](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/primitives/launch_context.h#L49-L65)。仅BATCH模式登记参与Thread并按句柄去重；批结束提交列表。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [launch_context.h:L59](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/primitives/launch_context.h#L59)；`if (std::find(threadVec_.begin(), threadVec_.end(), thread) == threadVec_.end()) {` | std::find（未展开边界）；begin（未展开边界）；end（未展开边界） | 仅当当前Thread句柄尚未存在于threadVec_时追加，避免同一批重复提交同一Thread。 |
| [launch_context.h:L61](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/primitives/launch_context.h#L61)；`threadVec_.push_back(thread);` | push_back（未展开边界） | 把当前Thread加入本轮BATCH参与列表；批结束HandleEagerMode将把该列表交CommTaskLaunch。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [launch_context.h:L53](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/primitives/launch_context.h#L53) | `if (UNLIKELY(mode_ != HCOMM_LAUNCH_MODE_BATCH)) {` | 仅当`(UNLIKELY(mode_ != HCOMM_LAUNCH_MODE_BATCH))`（当前提交模式）成立时进入此分支。 |
| [launch_context.h:L59](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/primitives/launch_context.h#L59) | `if (std::find(threadVec_.begin(), threadVec_.end(), thread) == threadVec_.end()) {` | 仅当当前Thread句柄尚未存在于threadVec_时追加，避免同一批重复提交同一Thread。 |


### AddThreadWithTag

定义：[launch_context.h:L32–L46](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/primitives/launch_context.h#L32-L46)。仅BATCH模式按当前tag保存Thread集合。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [launch_context.h:L44](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/primitives/launch_context.h#L44)；`threadSet.insert(thread);` | insert（未展开边界） | 把键/对象写入对应映射或集合；传入/处理当前执行Thread句柄。 |


| 条件/检查位置 | 源码条件/边界 | 进入/退出意义 |
| --- | --- | --- |
| [launch_context.h:L36](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/base_comm/primitives/launch_context.h#L36) | `if (mode_ != HCOMM_LAUNCH_MODE_BATCH) {` | 仅当`(mode_ != HCOMM_LAUNCH_MODE_BATCH)`（当前提交模式）成立时进入此分支。 |


### ProcessOneWqe

定义：[ub_conn_lite.h:L156–L164](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.h#L156-L164)。忽略stream参数并将已构造数据WQE交LaunchOneWqe写入UB SQ。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [ub_conn_lite.h:L162](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.h#L162)；`LaunchOneWqe(sqe, opCode);` | LaunchOneWqe → [ub_conn_lite.cc:L353–L395](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc#L353-L395) | 维护UB PI并复制WQE到对应UB SQ环槽；传入/处理当前UB WQE结构、UB读写操作码。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### SetSqeHeaderTaskFields

定义：[sqe_build_a5.h:L28–L38](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L28-L38)。把32位taskId拆低/高16位写入SQE头两个任务标识字段。


本函数没有另外的显式函数调用；按其返回/赋值直接完成包装。


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### BuildA5SqeNotifyWait

定义：[sqe_build_a5.h:L60–L90](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L60-L90)。填写本端硬件通知等待SQE，超时原值写入timeout并开启消费清除。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [sqe_build_a5.h:L68](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L68)；`SetSqeHeaderTaskFields(sqe, taskId);` | SetSqeHeaderTaskFields → [sqe_build_a5.h:L28–L38](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L28-L38) | 把taskId低16位/高16位写入SQE头rtStreamId/taskId；传入/处理当前UB WQE结构、当前任务编号。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### BuildA5SqeNotifyRecord

定义：[sqe_build_a5.h:L93–L117](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L93-L117)。填写本端硬件通知记录SQE，设置通知ID与单通知子类型。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [sqe_build_a5.h:L101](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L101)；`SetSqeHeaderTaskFields(sqe, taskId);` | SetSqeHeaderTaskFields → [sqe_build_a5.h:L28–L38](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L28-L38) | 把taskId低16位/高16位写入SQE头rtStreamId/taskId；传入/处理当前UB WQE结构、当前任务编号。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### BuildA5SqeSdmaCopy

定义：[sqe_build_a5.h:L196–L241](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L196-L241)。按字节长度/地址/归约opcode填写本地SDMA SQE。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [sqe_build_a5.h:L206](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L206)；`SetSqeHeaderTaskFields(sqe, taskId);` | SetSqeHeaderTaskFields → [sqe_build_a5.h:L28–L38](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L28-L38) | 把taskId低16位/高16位写入SQE头rtStreamId/taskId；传入/处理当前UB WQE结构、当前任务编号。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。


### BuildA5SqeUbDbSend

定义：[sqe_build_a5.h:L244–L276](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L244-L276)。把jetty/die/function标识与16位UB PI编码到RTSQ Doorbell SQE。


| caller调用点与原实参 | callee定义/边界 | 该行功能 |
| --- | --- | --- |
| [sqe_build_a5.h:L254](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L254)；`SetSqeHeaderTaskFields(sqe, taskId);` | SetSqeHeaderTaskFields → [sqe_build_a5.h:L28–L38](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L28-L38) | 把taskId低16位/高16位写入SQE头rtStreamId/taskId；传入/处理当前UB WQE结构、当前任务编号。 |
| [sqe_build_a5.h:L266](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L266)；`sqe->jettyId1 = jettyLiteId.GetJettyId();` | GetJettyId（未展开边界） | 设置当前UB WQE结构的jettyId1字段为/按`jettyLiteId.GetJettyId()`（UB jetty的die/function/jetty标识的GetJettyId字段）；读取UB jetty编号以填写Doorbell目标。 |
| [sqe_build_a5.h:L268](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L268)；`sqe->funcId1 = jettyLiteId.GetFuncId();` | GetFuncId（未展开边界） | 设置当前UB WQE结构的funcId1字段为/按`jettyLiteId.GetFuncId()`（UB jetty的die/function/jetty标识的GetFuncId字段）；读取UB功能编号以填写Doorbell目标。 |
| [sqe_build_a5.h:L272](https://github.com/zstar1003/hccl-hcomm-annotated/blob/824a8a80731bd66ef6eb78891b9aec0acfca281c/hcomm/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.h#L272)；`sqe->dieId1 = jettyLiteId.GetDieId();` | GetDieId（未展开边界） | 设置当前UB WQE结构的dieId1字段为/按`jettyLiteId.GetDieId()`（UB jetty的die/function/jetty标识的GetDieId字段）；读取UB die编号以填写Doorbell目标。 |


函数内无显式条件分流；构造/调用失败是否抛异常仍遵循被调用实现。
