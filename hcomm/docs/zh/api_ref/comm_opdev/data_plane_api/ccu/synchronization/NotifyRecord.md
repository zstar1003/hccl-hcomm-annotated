# NotifyRecord

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

在CCU kernel内通过channel向远端die发送同步信号的生产者侧接口。运行期通过channel发送die-to-die信号包（同device跨die走HCCS，跨device/跨节点走RDMA），置位远端die对应的notify位。

适用于跨die场景（同device跨die、同节点跨device、跨节点），对应消费侧为[NotifyWait](NotifyWait.md)。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
CcuResult NotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| channel | 输入 | 通信通道句柄，为通过建链流程获取的通道资源。`ChannelHandle`类型的定义可参见[ChannelHandle](../../../datatype_definition/ChannelHandle.md)。channel绑定的die必须与本kernel当前die一致。 |
| remoteNotifyIdx | 输入 | 远端notify槽位索引，为channel两端协商约定的编号（0-based），对应消费侧`NotifyWait`的`localNotifyIdx`，两侧数字必须完全一致。 |
| mask | 输入 | 16位事件掩码，指定要置位的bit。默认值为`1`。必须与消费侧`NotifyWait`的`mask`一致。 |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | 当前不存在处于注册中的kernel（接口在kernel注册阶段之外被调用）。 |

> [!NOTE]说明
> 本接口在调用处返回`CCU_SUCCESS`，不校验channel/`remoteNotifyIdx`；channel绑定die的一致性由`HcommCcuKernelRegister`统一校验（依据本kernel期间用到的全部channel），不一致时返回`CCU_E_PARA`。本接口不拦截硬件Loop body内调用，请勿在Loop body内使用（参见[Loop](../execution_control/Loop.md)）。

## 约束说明

- 本kernel内所有channel必须属于同一die；该一致性由`HcommCcuKernelRegister`统一校验，不一致返回`CCU_E_PARA`。
- `remoteNotifyIdx`与消费侧`NotifyWait`的`localNotifyIdx`必须使用相同数值，由通信双方在协议层约定，框架不自动配对。单个`ChannelHandle`持有的Notify个数最大为8个，因此该参数的取值范围为0~7。
- `NotifyRecord`是生产侧，消费侧`NotifyWait`必须在`NotifyRecord`之后到达（硬件上"先Record 后Wait"，生产先于消费）。若消费侧等不到对应的`NotifyRecord`（未发起、`localNotifyIdx`/`mask`不匹配等），将永久阻塞，导致硬件级死锁。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景：die A 完成写操作后通知die B 可以读取数据
// die A 的CCU kernel函数体内
CcuResult DieSenderKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg 为void*，先转型为用户入参结构体
    // ... 执行写操作，将数据写入对端内存 ...

    // 通知die B 的notify槽位3，bit0 已完成
    NotifyRecord(params->channel, /*remoteNotifyIdx=*/3, 0x1);
    return CCU_SUCCESS;
}

// die B 的CCU kernel函数体内（与die A 的kernel 对应）
CcuResult DieReceiverKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg 为void*，先转型为用户入参结构体
    // 等待来自die A 的信号，本地notify槽位3的bit0
    NotifyWait(params->channel, /*localNotifyIdx=*/3, 0x1);

    // 此后可安全读取die A 写入的数据
    return CCU_SUCCESS;
}
```
