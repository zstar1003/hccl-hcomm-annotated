# EventWait

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

在CCU kernel内阻塞等待本地Event的指定掩码位被置位。硬件执行到此处会阻塞，直到对应Event位为1后继续执行。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
CcuResult EventWait(Event e, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| e | 输入 | 本地Event对象，与生产侧`EventRecord`或数据搬运接口的`(event, mask)`参数对应同一对象。 |
| mask | 输入 | 16位事件掩码，指定要等待的bit。默认值为`1`。支持多bit合并等待，如`mask = 0x3`等待bit0和bit1同时置位。 |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | 当前不存在处于注册中的kernel（接口在kernel注册阶段之外被调用）。 |
| `CCU_E_NOT_FOUND` | 传入的`e`句柄未在当前kernel注册。 |

> [!NOTE]说明
> 本接口不会返回`CCU_E_PARA`，且可在硬件Loop body内调用。

## 约束说明

- `EventWait`可在硬件Loop body内调用。
- 调用前必须已存在对应的`EventRecord`或以`(event, mask)`为末位参数的数据搬运调用，否则将永久阻塞导致硬件级死锁。
- 等待的`mask`必须与生产侧`EventRecord`或搬运接口传入的`mask`一致。
- 同一个`Event`对象可通过不同`mask`承载多组独立的生产-消费配对，各组互不影响。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景：等待本端HBM→HBM拷贝完成后再继续
// CCU kernel函数体内
CcuResult MyKernel(CcuKernelArg arg) {
    LocalAddr src, dst;
    Variable len;
    Event evt;

    // 发起异步拷贝，硬件完成时自动置位evt[0x1]
    LocalCopy(dst, src, len, evt, 0x1);

    // 阻塞等待拷贝完成
    EventWait(evt, 0x1);

    // 此后可安全使用dst 指向的数据
    return CCU_SUCCESS;
}
```
