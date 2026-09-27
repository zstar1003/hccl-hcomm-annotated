# LocalCopy

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

在CCU kernel内发起本地内存拷贝操作（异步），硬件完成搬运时自动将`event`的第`mask`位置1。支持以下三种数据通路：

| 重载 | 源 | 目标 |
| --- | --- | --- |
| 重载1 | 本端HBM（`LocalAddr`） | 本端HBM（`LocalAddr`） |
| 重载2 | 本端HBM（`LocalAddr`） | 本端MS Buffer（`CcuBuffer`） |
| 重载3 | 本端MS Buffer（`CcuBuffer`） | 本端HBM（`LocalAddr`） |

> [!NOTE]说明
> 本接口为异步接口，调用后须通过`EventWait(event, mask)`等待搬运完成，否则目标内存数据不确定。与`EventRecord`不同，硬件搬运完成时自动置位`event[mask]`，无需显式调用`EventRecord`。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
// 重载1：本端HBM→本端HBM
CcuResult LocalCopy(LocalAddr dst, LocalAddr src, Variable len, Event event, uint16_t mask = 1);
// 重载2：本端HBM→本端MS Buffer
CcuResult LocalCopy(CcuBuffer dst, LocalAddr src, Variable len, Event event, uint16_t mask = 1);
// 重载3：本端MS Buffer→本端HBM
CcuResult LocalCopy(LocalAddr dst, CcuBuffer src, Variable len, Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输入/输出 | 目标地址（硬件搬运结果写入此处）。重载1/3为`LocalAddr`（本端HBM地址与token的复合对象）；重载2为`CcuBuffer`（本端MS Buffer切片对象，单片最大4096字节）。 |
| src | 输入 | 源地址。重载1/2为`LocalAddr`；重载3为`CcuBuffer`。 |
| len | 输入 | 拷贝字节数，类型为`Variable`（运行期可变长度）。涉及`CcuBuffer`时不可超过4096字节。 |
| event | 输入 | 完成事件对象。硬件搬运完成时自动置位`event[mask]`，下游调用`EventWait(event, mask)`等待。 |
| mask | 输入 | 16位事件掩码，指定要置位的bit。默认值为`1`（即bit0）。同一`Event`对象的不同bit相互独立，可承载多组配对。 |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | 当前不存在处于注册中的kernel（接口在kernel注册阶段之外被调用）。 |
| `CCU_E_NOT_FOUND` | 传入的`dst`/`src`/`len`/`event`句柄未在当前kernel注册。 |

> [!NOTE]说明
> 本接口不会返回`CCU_E_PARA`。`len`超过`CcuBuffer`单片4096字节时不会报错（详见"约束说明"），但运行期行为未定义。

## 约束说明

- `dst`与`src`的内存区间不可重叠，重叠时行为未定义。
- 涉及`CcuBuffer`时（重载2/3），`len`不可超过`CcuBuffer`单片大小（4096字节）；该上限须由调用方自行保证，超出会导致运行期硬件行为未定义。
- 本接口为异步操作，须通过`EventWait(event, mask)`等待搬运完成后再访问目标内存。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景1：本端HBM→本端HBM拷贝
CcuResult MyKernel(CcuKernelArg arg) {
    LocalAddr src, dst;
    Variable len;
    Event evt;

    LocalCopy(dst, src, len, evt);    // mask默认0x1
    EventWait(evt);
    return CCU_SUCCESS;
}

// 场景2：本端HBM→本端MS Buffer拷贝
CcuResult MyKernel2(CcuKernelArg arg) {
    CcuBuffer buf;
    LocalAddr src;
    Variable len;
    Event evt;

    LocalCopy(buf, src, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}

// 场景3：本端MS Buffer→本端HBM拷贝
CcuResult MyKernel3(CcuKernelArg arg) {
    LocalAddr dst;
    CcuBuffer buf;
    Variable len;
    Event evt;

    LocalCopy(dst, buf, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```
