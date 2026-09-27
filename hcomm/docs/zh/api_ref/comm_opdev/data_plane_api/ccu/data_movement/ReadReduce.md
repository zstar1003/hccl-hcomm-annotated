# ReadReduce

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

在CCU kernel内发起跨rank读并原地归约操作，通过已建链的`ChannelHandle`从对端HBM读取数据，并与本端HBM现有数据按指定算子合并（`local = reduce(local, fetch_from_remote, opType)`），硬件完成时自动将`event`的第`mask`位置1。本接口为异步接口。

> [!NOTE]说明
> 参数顺序遵循"目的端在前，源端在后"约定：`ReadReduce(ch, local, remote, ...)`，即`local`（目的，同时也是归约的累加端）在第二位，`remote`（源）在第三位。

<!-- -->

> [!CAUTION]注意
> 同一kernel内使用的所有`ChannelHandle`必须属于同一die。本接口在调用处不做die校验，因此不会因die 不一致而失败（调用处仍可能因无注册中kernel返回`CCU_E_PTR`、或句柄无效返回`CCU_E_NOT_FOUND`，详见返回值表）；die一致性由`HcommCcuKernelRegister`统一校验（依据本kernel 期间用到的全部channel），不一致时由`HcommCcuKernelRegister`返回`CCU_E_PARA`，而非本接口的返回值。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
CcuResult ReadReduce(ChannelHandle ch, LocalAddr local, RemoteAddr remote,
                     Variable len, HcclDataType dataType, HcclReduceOp opType,
                     Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ch | 输入 | 跨rank通道句柄（`ChannelHandle`）。channel绑定的die须与本kernel内所有channel属于同一die（在`HcommCcuKernelRegister`内统一校验，详见上文CAUTION）。 |
| local | 输入/输出 | 本端HBM目标地址（`LocalAddr`）。调用前须写入有效初值；硬件完成后更新为归约结果。 |
| remote | 输入 | 对端HBM源地址（`RemoteAddr`）。 |
| len | 输入 | 操作字节数，类型为`Variable`（运行期可变长度）。 |
| dataType | 输入 | 数据类型，取值见`HcclDataType`枚举。仅支持以下6种：`HCCL_DATA_TYPE_UINT8`、`HCCL_DATA_TYPE_INT16`、`HCCL_DATA_TYPE_INT32`、`HCCL_DATA_TYPE_FP16`、`HCCL_DATA_TYPE_FP32`、`HCCL_DATA_TYPE_BFP16`；其他取值会被拒绝并抛出异常（携带错误码）。 |
| opType | 输入 | 归约算子，取值见`HcclReduceOp`枚举。仅支持`HCCL_REDUCE_SUM`（求和）、`HCCL_REDUCE_MAX`（最大值）、`HCCL_REDUCE_MIN`（最小值）；`HCCL_REDUCE_PROD`不支持，传入会被拒绝并抛出异常（携带错误码）。 当采用SUM操作时低精度输入数据的求和结果会先进行精度上升然后再进行精度调整为与输入数据精度相同。 |
| event | 输入 | 完成事件对象。硬件归约读完成时自动置位`event[mask]`，下游调用`EventWait(event, mask)`等待。 |
| mask | 输入 | 16位事件掩码。默认值为`1`（即bit0）。 |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | 当前不存在处于注册中的kernel（接口在kernel注册阶段之外被调用）。 |
| `CCU_E_NOT_FOUND` | 传入的`local`/`remote`/`len`/`event`句柄未在当前kernel注册。 |

> [!NOTE]说明
> 本接口不会返回`CCU_E_PARA`；channel die不一致由`HcommCcuKernelRegister`返回`CCU_E_PARA`。若`dataType`/`opType`不在支持范围内（见参数说明），运行时将抛出异常（携带错误码）。

## 约束说明

- `local`在调用前须已写入有效初值（如0或负无穷），否则归约结果未定义。
- 参数顺序为本端（目的）在前、对端（源）在后：`ReadReduce(ch, local, remote, ...)`。
- 同一kernel内所有`ChannelHandle`须属于同一die，不同die的channel不可混入同一kernel。
- 本接口为异步操作，须通过`EventWait(event, mask)`等待归约完成后再读取结果。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景：从对端HBM读取并与本端数据做FP16 SUM归约
CcuResult MyKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg 为void*，先转型为用户入参结构体
    ChannelHandle ch = params->channelHandle;
    LocalAddr dst;      // dst须预先写入初值（如0.0）
    RemoteAddr remote;
    Variable len;
    Event evt;

    ReadReduce(ch, dst, remote, len, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```
