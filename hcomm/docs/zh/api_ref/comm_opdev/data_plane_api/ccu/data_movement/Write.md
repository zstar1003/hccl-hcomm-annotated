# Write

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

在CCU kernel内发起跨rank写操作（异步），通过已建链的`ChannelHandle`将本端数据写入对端HBM，硬件完成时自动将`event`的第`mask`位置1。支持以下两种源类型：

| 重载 | 源 | 目标 |
| --- | --- | --- |
| 重载1 | 本端HBM（`LocalAddr`） | 对端HBM（`RemoteAddr`） |
| 重载2 | 本端MS Buffer（`CcuBuffer`） | 对端HBM（`RemoteAddr`） |

注意：

- 参数顺序遵循"目的端在前，源端在后"约定：`Write(ch, remote, local, ...)`，即`remote`（目的，对端）在第二位，`local`（源，本端）在第三位。与`Read`中`local`在前的顺序相反，请勿混淆。C++类型系统通过`RemoteAddr`与`LocalAddr`的不同类型在编译期阻止参数顺序错误。
- 同一kernel内使用的所有`ChannelHandle`必须属于同一die。本接口在调用处不做die校验，因此不会因die不一致而失败（调用处仍可能因无注册中kernel返回`CCU_E_PTR`、或句柄无效返回`CCU_E_NOT_FOUND`，详见返回值表）；die一致性由`HcommCcuKernelRegister`统一校验（依据本kernel期间用到的全部channel），不一致时由`HcommCcuKernelRegister`返回`CCU_E_PARA`，而非本接口的返回值。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
// 重载1：本端HBM→对端HBM
CcuResult Write(ChannelHandle ch, RemoteAddr remote, LocalAddr local,
                Variable len, Event event, uint16_t mask = 1);
// 重载2：本端MS Buffer→对端HBM
CcuResult Write(ChannelHandle ch, RemoteAddr remote, CcuBuffer local,
                Variable len, Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ch | 输入 | 跨rank通道句柄（`ChannelHandle`）。channel绑定的die须与本kernel内所有channel属于同一die（在`HcommCcuKernelRegister`内统一校验，详见上文CAUTION）。 |
| remote | 输入 | 对端HBM目标地址（`RemoteAddr`，对端HBM地址与token的复合对象）。 |
| local | 输入 | 本端源地址。重载1为`LocalAddr`（本端HBM地址与token的复合对象）；重载2为`CcuBuffer`（本端MS Buffer切片对象，单片最大4096字节）。 |
| len | 输入 | 写入字节数，类型为`Variable`（运行期可变长度）。重载2时不可超过4096字节。 |
| event | 输入 | 完成事件对象。硬件写入完成时自动置位`event[mask]`，下游调用`EventWait(event, mask)`等待。 |
| mask | 输入 | 16位事件掩码。默认值为`1`（即bit0）。 |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | 当前不存在处于注册中的kernel（接口在kernel注册阶段之外被调用）。 |
| `CCU_E_NOT_FOUND` | 传入的`remote`/`local`/`len`/`event`句柄未在当前kernel注册。 |

> [!NOTE]说明
> 本接口不会返回 `CCU_E_PARA`；channel die不一致由`HcommCcuKernelRegister`返回`CCU_E_PARA`，详见[功能说明](#功能说明)。

## 约束说明

- 参数顺序为对端（目的）在前、本端（源）在后：`Write(ch, remote, local, ...)`。
- 同一kernel内所有`ChannelHandle`须属于同一die，不同die的channel不可混入同一kernel。
- 重载2时，`len`不可超过`CcuBuffer`单片大小（4096字节）；该上限须由调用方自行保证，超出会导致运行期硬件行为未定义。
- 本接口为异步操作，须通过`EventWait(event, mask)`等待写入完成，方可保证对端数据可见。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景1：本端HBM→对端HBM
CcuResult MyKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg 为void*，先转型为用户入参结构体
    ChannelHandle ch = params->channelHandle;
    RemoteAddr remote;
    LocalAddr src;
    Variable len;
    Event evt;

    Write(ch, remote, src, len, evt);    // remote在前，local在后
    EventWait(evt);
    return CCU_SUCCESS;
}

// 场景2：本端MS Buffer→对端HBM
CcuResult MyKernel2(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg 为void*，先转型为用户入参结构体
    ChannelHandle ch = params->channelHandle;
    RemoteAddr remote;
    CcuBuffer buf;
    Variable len;
    Event evt;

    Write(ch, remote, buf, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```
