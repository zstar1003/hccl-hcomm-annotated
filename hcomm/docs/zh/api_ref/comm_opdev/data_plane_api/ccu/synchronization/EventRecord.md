# EventRecord

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

在CCU kernel内显式标记本地Event的指定掩码位完成。

> [!NOTE]说明
> 绝大多数场景下无需显式调用本接口。[数据搬运接口](../data_movement/README.md)（如`LocalCopy`、`Read`、`Write`等）末位的`(event, mask)`参数由硬件在搬运完成时自动置位，等价于隐式的`EventRecord`。仅在需要将控制流的某个分支结束或无搬运操作的位置作为同步点时，才需要显式调用本接口。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
CcuResult EventRecord(Event e, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| e | 输入 | 本地Event对象。`Event`类构造时申请CKE虚拟句柄（物理资源在`HcommCcuKernelRegister`阶段分配），无需手动分配。 |
| mask | 输入 | 16位事件掩码，指定要置位的bit。默认值为`1`（即bit0）。同一个`Event`对象的不同bit相互独立，可承载多组配对。 |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | 当前不存在处于注册中的kernel（接口在kernel注册阶段之外被调用）。 |
| `CCU_E_NOT_FOUND` | 传入的`e`句柄未在当前kernel注册。 |
| `CCU_E_NOT_SUPPORT` | 在硬件Loop body内调用，不支持。 |

> [!NOTE]说明
> 本接口不会返回 `CCU_E_PARA`。

## 约束说明

- 不可在硬件Loop body内调用，否则将返回`CCU_E_NOT_SUPPORT`。Loop body会被硬件并行展开为N 份，`EventRecord`语义在并行环境中不唯一。
- 每次`EventRecord`必须有对应的`EventWait`在其之后出现，否则等待侧将永久阻塞，导致硬件级死锁。
- `mask`指定的bit与`EventWait`的`mask`必须一致，否则`EventWait`永远等不到信号。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景：在没有数据搬运的位置手动插入同步点
// CCU kernel函数体内
CcuResult MyKernel(CcuKernelArg arg) {
    Event evt;

    // 执行某些操作后，手动标记bit0 完成
    EventRecord(evt, 0x1);

    // 下游等待bit0 被置位
    EventWait(evt, 0x1);
    return CCU_SUCCESS;
}
```
