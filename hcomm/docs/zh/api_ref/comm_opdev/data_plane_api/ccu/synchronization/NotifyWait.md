# NotifyWait

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

在CCU kernel内阻塞等待远端die通过channel发来的同步信号的消费者侧接口。硬件执行到此处会阻塞，直到对应notify位被远端置位后继续执行。

适用于跨die场景（同device跨die、同节点跨device、跨节点），对应生产侧为[NotifyRecord](NotifyRecord.md)。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
CcuResult NotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| channel | 输入 | 通信通道句柄，为通过建链流程获取的通道资源。`ChannelHandle`类型的定义可参见[ChannelHandle](../../../datatype_definition/ChannelHandle.md)。channel绑定的die必须与本kernel当前die一致。 |
| localNotifyIdx | 输入 | 本地notify槽位索引，为channel两端协商约定的编号（0-based），必须与生产侧`NotifyRecord`的`remoteNotifyIdx`数值完全一致。 |
| mask | 输入 | 16位事件掩码，指定要等待的bit。默认值为`1`。必须与生产侧`NotifyRecord`的`mask`一致。 |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | 当前不存在处于注册中的kernel（接口在kernel注册阶段之外被调用）。 |

> [!NOTE]说明
> 本接口在调用处不校验channel/`localNotifyIdx`；channel绑定die的一致性由`HcommCcuKernelRegister`统一校验（依据本kernel期间用到的全部channel），不一致时返回`CCU_E_PARA`。

## 约束说明

- `NotifyWait`可在硬件Loop body内调用。
- 本kernel内所有channel必须属于同一die；该一致性由`HcommCcuKernelRegister`统一校验，不一致返回`CCU_E_PARA`。
- `localNotifyIdx`必须与生产侧`NotifyRecord`的`remoteNotifyIdx`使用相同数值，由通信双方在协议层约定，框架不自动配对。单个`ChannelHandle`持有的Notify个数最大为8个，因此该参数的取值范围为0~7。
- 调用前必须已存在对应的`NotifyRecord`在生产侧的先序位置，否则将永久阻塞导致硬件级死锁。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景：die B 等待die A 完成写操作后再读取数据
// die B 的CCU kernel函数体内
CcuResult DieReceiverKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg 为void*，先转型为用户入参结构体
    // 等待来自die A 的信号，本地notify槽位3的bit0
    // 对应die A 侧NotifyRecord(channel, /*remoteNotifyIdx=*/3, 0x1)
    NotifyWait(params->channel, /*localNotifyIdx=*/3, 0x1);

    // 此后可安全读取die A 写入的数据
    RemoteAddr remote;
    LocalAddr dst;
    Variable len;
    Event evt;
    Read(params->channel, dst, remote, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```
