# LocalNotifyWait

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

在CCU kernel内阻塞等待同die内跨kernel同步信号的消费者侧接口，以字符串tag标识配对。硬件执行到此处会阻塞，直到对应bit被置位后继续。

> [!NOTE]说明
> 本接口在C++层实现为`AscendC::ccu::EventWait(const char* notifyTag, uint16_t mask)`重载，与`EventWait(Event, mask)`共用同一函数名，通过入参类型区分。适用于同一die内不同kernel之间的顺序同步，不适用于跨die或跨rank场景，跨die场景请使用[NotifyWait](NotifyWait.md)。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
// LocalNotifyWait 对应此重载
CcuResult EventWait(const char *notifyTag, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| notifyTag | 输入 | 字符串tag，必须与生产侧`LocalNotifyRecord`的`notifyTag`完全一致以完成配对。不可为空指针。约束同生产侧，需为字符串字面量或kernel内的`static`存储。 |
| mask | 输入 | 16位事件掩码，指定要等待的bit。默认值为`1`。必须与生产侧`LocalNotifyRecord`的`mask`一致。 |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | `notifyTag`为空指针，或当前不存在处于注册中的kernel。 |

## 约束说明

- `LocalNotifyWait`可在硬件Loop body 内调用。
- `notifyTag`必须在kernel注册期间保持有效，约束与`LocalNotifyRecord`一致。
- 调用前必须已存在对应的`LocalNotifyRecord`在某个先序位置，否则将永久阻塞导致硬件级死锁。
- 等待的`mask`必须与生产侧`LocalNotifyRecord`传入的`mask`一致。

## 调用示例

> 下面示例中调用的`EventWait("phase1_done", 0x1)`即本文档所述的`LocalNotifyWait`——在C++层它与[EventWait](EventWait.md)（`Event`对象重载）共用同一函数名，通过入参类型（`const char *` vs `Event`）区分。

```cpp
using namespace AscendC::ccu;

// 场景：同die内，PhaseConsumerKernel等待PhaseProducerKernel完成阶段1后才开始阶段2
// 注册到 core 1 的CCU kernel函数体内
CcuResult PhaseConsumerKernel(CcuKernelArg arg) {
    // 等待PhaseProducerKernel发出"phase1_done"信号的bit0（即LocalNotifyWait语义）
    EventWait("phase1_done", 0x1);

    // 此后可安全使用PhaseProducerKernel阶段1的结果
    // ... 执行阶段2操作 ...
    return CCU_SUCCESS;
}
```
