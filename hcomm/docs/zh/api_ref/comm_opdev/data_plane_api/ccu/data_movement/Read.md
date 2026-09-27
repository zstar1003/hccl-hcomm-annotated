# Read

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

在CCU kernel内发起跨rank读操作（异步），通过已建链的`ChannelHandle`从对端HBM读取数据到本端，硬件完成时自动将`event`的第`mask`位置1。支持以下两种目标类型：

| 重载 | 目标 | 源 |
| --- | --- | --- |
| 重载1 | 本端HBM（`LocalAddr`） | 对端HBM（`RemoteAddr`） |
| 重载2 | 本端MS Buffer（`CcuBuffer`） | 对端HBM（`RemoteAddr`） |

注意：

- 参数顺序遵循"目的端在前，源端在后"约定：`Read(ch, local, remote, ...)`，即`local`（目的）在第二位，`remote`（源）在第三位。与`Write`中`remote`在前的顺序相反，请勿混淆。C++类型系统通过`LocalAddr`与`RemoteAddr`的不同类型在编译期阻止参数顺序错误。
- 同一kernel内使用的所有`ChannelHandle`必须属于同一die。本接口在调用处不做die校验，因此不会因die不一致而失败（调用处仍可能因无注册中kernel返回`CCU_E_PTR`、或句柄无效返回`CCU_E_NOT_FOUND`，详见返回值表）；die一致性由`HcommCcuKernelRegister`统一校验（依据本kernel期间用到的全部channel），不一致时由`HcommCcuKernelRegister`返回`CCU_E_PARA`，而非本接口的返回值。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
// 重载1：对端HBM→本端HBM
CcuResult Read(ChannelHandle ch, LocalAddr local, RemoteAddr remote,
               Variable len, Event event, uint16_t mask = 1);
// 重载2：对端HBM→本端MS Buffer
CcuResult Read(ChannelHandle ch, CcuBuffer local, RemoteAddr remote,
               Variable len, Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ch | 输入 | 跨rank通道句柄（`ChannelHandle`）。channel绑定的die须与本kernel内所有channel属于同一die（在`HcommCcuKernelRegister`内统一校验，详见[功能说明](#功能说明)）。 |
| local | 输入/输出 | 本端目标地址。重载1为`LocalAddr`（本端HBM地址与token的复合对象）；重载2为`CcuBuffer`（本端MS Buffer切片对象，单片最大4096字节）。 |
| remote | 输入 | 对端源地址（`RemoteAddr`，对端HBM地址与token的复合对象）。 |
| len | 输入 | 读取字节数，类型为`Variable`（运行期可变长度）。重载2时不可超过4096字节。 |
| event | 输入 | 完成事件对象。硬件读取完成时自动置位`event[mask]`，下游调用`EventWait(event, mask)`等待。 |
| mask | 输入 | 16位事件掩码。默认值为`1`（即bit0）。 |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | 当前不存在处于注册中的kernel（接口在kernel注册阶段之外被调用）。 |
| `CCU_E_NOT_FOUND` | 传入的`local`/`remote`/`len`/`event`句柄未在当前kernel注册。 |

> [!NOTE]说明
> 本接口不会返回 `CCU_E_PARA`；channel die不一致由`HcommCcuKernelRegister`返回`CCU_E_PARA`，详见[功能说明](#功能说明)。

## 约束说明

- 参数顺序为本端（目的）在前、对端（源）在后：`Read(ch, local, remote, ...)`。
- 同一kernel内所有`ChannelHandle`须属于同一die，不同die的channel不可混入同一kernel。
- 重载2时，`len`不可超过`CcuBuffer`单片大小（4096字节）；该上限须由调用方自行保证，超出会导致运行期硬件行为未定义。
- 本接口为异步操作，须通过`EventWait(event, mask)`等待读取完成后再访问目标内存。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景1：从对端HBM读取到本端HBM
CcuResult MyKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg 为void*，先转型为用户入参结构体
    ChannelHandle ch = params->channelHandle;
    LocalAddr dst;
    RemoteAddr remote;
    Variable len;
    Event evt;

    Read(ch, dst, remote, len, evt);    // local在前，remote在后
    EventWait(evt);
    return CCU_SUCCESS;
}

// 场景2：从对端HBM读取到本端MS Buffer
CcuResult MyKernel2(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg 为void*，先转型为用户入参结构体
    ChannelHandle ch = params->channelHandle;
    CcuBuffer buf;
    RemoteAddr remote;
    Variable len;
    Event evt;

    Read(ch, buf, remote, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```
