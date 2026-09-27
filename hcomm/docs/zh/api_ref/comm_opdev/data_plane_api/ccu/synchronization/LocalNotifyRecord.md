# LocalNotifyRecord

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

在CCU kernel内发出同die内跨kernel同步信号的生产者侧接口，以字符串tag标识配对：相同tag的生产/消费两侧自动绑定到同一同步资源。

> [!NOTE]说明
> 本接口在C++层实现为`AscendC::ccu::EventRecord(const char* notifyTag, uint16_t mask)`重载，与`EventRecord(Event, mask)`共用同一函数名，通过入参类型区分。适用于同一die内不同kernel之间的顺序同步（同device跨kernel），不适用于跨die或跨rank场景，跨die场景请使用[NotifyRecord](NotifyRecord.md)。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
// LocalNotifyRecord 对应此重载
CcuResult EventRecord(const char *notifyTag, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| notifyTag | 输入 | 字符串tag，与消费侧`LocalNotifyWait`的`notifyTag`保持一致即可完成配对，无需提前分配编号。不可为空指针。必须在kernel注册期间保持有效（需为字符串字面量或kernel内的`static`存储），不可使用栈上临时数组。 |
| mask | 输入 | 16位事件掩码，指定要置位的bit。默认值为`1`。同一tag的不同bit相互独立。 |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | `notifyTag`为空指针，或当前不存在处于注册中的kernel。 |
| `CCU_E_NOT_SUPPORT` | 在硬件Loop body 内调用，不支持。 |

## 约束说明

- 不可在硬件Loop body内调用，否则将返回`CCU_E_NOT_SUPPORT`。Loop body会被硬件并行展开为N份，notify记录语义在并行环境中不唯一。
- `notifyTag`必须在kernel注册期间保持有效，应使用字符串字面量（如`"phase1_done"`）或kernel内的`static char[]`，禁止使用栈上临时`char[]`。
- 生产侧`LocalNotifyRecord`与消费侧`LocalNotifyWait`必须使用完全相同的`notifyTag`字符串内容进行配对。
- 同一tag可承载多组配对，由`mask`的不同bit区分；等待侧`mask`必须与此处一致。
- `LocalNotifyRecord`与对应的`LocalNotifyWait`必须成对出现，且`LocalNotifyWait`位于`LocalNotifyRecord`的后序位置。任何无配对的`LocalNotifyWait`将永久阻塞，导致硬件级死锁。

## 调用示例

> 下面示例中调用的`EventRecord("phase1_done", 0x1)`即本文档所述的`LocalNotifyRecord`——在C++层它与[EventRecord](EventRecord.md)（`Event`对象重载）共用同一函数名，通过入参类型（`const char *` vs `Event`）区分。

```cpp
using namespace AscendC::ccu;

// 场景：同die内，PhaseProducerKernel完成阶段1操作后通知PhaseConsumerKernel继续
// 注册到 core 0 的CCU kernel函数体内
CcuResult PhaseProducerKernel(CcuKernelArg arg) {
    // ... 执行阶段1操作 ...

    // 通知同die内的PhaseConsumerKernel，阶段1已完成（即LocalNotifyRecord语义）
    EventRecord("phase1_done", 0x1);
    return CCU_SUCCESS;
}

// 注册到 core 1 的CCU kernel函数体内
CcuResult PhaseConsumerKernel(CcuKernelArg arg) {
    // 等待PhaseProducerKernel发出的阶段1完成信号
    EventWait("phase1_done", 0x1);

    // 此后可安全使用PhaseProducerKernel阶段1的结果
    return CCU_SUCCESS;
}
```
