# LocalReduce

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

在CCU kernel内发起本地归约操作（异步），将源数据与目标内存或多个MS Buffer按指定算子合并，硬件完成时自动将`event`的第`mask`位置1。支持以下两种通路：

| 重载  | 数据通路                                      | 归约方式                                                                                                                                                                                                                                                     |
| --- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 重载1 | 本端HBM（`src`）→本端HBM（`dst`）                 | `dst = reduce(dst, src, opType)`，输入输出类型相同                                                                                                                                                                                                                |
| 重载2 | N 个本端MS Buffer → `buffers[0]`（2 ≤ N ≤ 8） | `buffers[0] = reduce(buffers[0..N-1], opType)`，硬件一次性归约N 路MS 到 `buffers[0]`。输入数据类型可以为（`HCCL_DATA_TYPE_UINT8`/`HCCL_DATA_TYPE_INT16`/`HCCL_DATA_TYPE_INT32`/`HCCL_DATA_TYPE_FP16`/`HCCL_DATA_TYPE_BFP16`/`HCCL_DATA_TYPE_FP32`），输出数据类型支持与输入同精度，或在 `HCCL_REDUCE_SUM` 下升精度输出（详见 `outputDataType` 参数说明）。 |

> [!NOTE]说明
> 本接口为异步接口，调用后须通过`EventWait(event, mask)`等待归约完成，否则目标内存数据不确定。归约为原地操作，调用前`dst`（重载1）或`buffers[0]`（重载2）须已写入有效初值（如0或负无穷）。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
// 重载1：本端HBM→本端HBM归约
CcuResult LocalReduce(LocalAddr dst, LocalAddr src, Variable len,
                      HcclDataType dataType, HcclReduceOp opType,
                      Event event, uint16_t mask = 1);
// 重载2：N 个本端MS Buffer 归约到buffers[0]（2 ≤ count ≤ 8）
CcuResult LocalReduce(CcuBuffer* buffers, uint32_t count,
                      HcclDataType dataType, HcclDataType outputDataType,
                      HcclReduceOp opType,
                      Variable len, Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

### 重载1参数

| 参数名            | 输入/输出 | 描述                                                                                                                                                                                            |
| -------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| dst            | 输入/输出 | 目标HBM地址（`LocalAddr`）。调用前须写入有效初值；硬件完成后更新为归约结果。                                                                                                                                                 |
| src            | 输入    | 源HBM地址（`LocalAddr`）。                                                                                                                                                                          |
| len            | 输入    | 操作字节数，类型为`Variable`（运行期可变长度）。                                                                                                                                                                 |
| dataType       | 输入    | 数据类型，取值见`HcclDataType`枚举。仅支持以下6种：`HCCL_DATA_TYPE_UINT8`、`HCCL_DATA_TYPE_INT16`、`HCCL_DATA_TYPE_INT32`、`HCCL_DATA_TYPE_FP16`、`HCCL_DATA_TYPE_FP32`、`HCCL_DATA_TYPE_BFP16`；其他取值会被拒绝并抛出异常（携带错误码）。 |
| opType         | 输入    | 归约算子，取值见`HcclReduceOp`枚举。仅支持`HCCL_REDUCE_SUM`（求和）、`HCCL_REDUCE_MAX`（最大值）、`HCCL_REDUCE_MIN`（最小值）；`HCCL_REDUCE_PROD`不支持，传入会被拒绝并抛出异常（携带错误码）。 当采用SUM操作时低精度输入数据的求和结果会先进行精度上升然后再进行精度调整为与输入数据精度相同。 |
| event          | 输入    | 完成事件对象。硬件归约完成时自动置位`event[mask]`。                                                                                                                                                              |
| mask           | 输入    | 16位事件掩码。默认值为`1`（即bit0）。                                                                                                                                                                       |

### 重载2参数

| 参数名            | 输入/输出 | 描述                                                                                                                                                                                                                                                                                                       |
| -------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| buffers        | 输入/输出 | MS Buffer数组首地址（`CcuBuffer*`），不可为`nullptr`。推荐通过`ccu::Array<CcuBuffer>`申请以保证物理连续。`buffers[0]`为归约结果的输出位置，调用前须写入有效初值。精度膨胀场景（低精度输入 + SUM 升精度输出，如INT8→FP32时输出元素4×于输入）下，传入的MS 数组须同时覆盖输入与膨胀后的输出占用（例如2路INT8输入升FP32输出时须预留4个MS，而非2个）；预留不足会导致硬件读写越界、行为未定义。                                                          |
| count          | 输入    | Buffer数量。取值范围为`[2, 8]`（超过上限会抛出异常并携带错误码）。`count == 0` 会被直接拒绝并返回`CCU_E_PARA`；`count == 1`不会被拒绝但硬件行为未定义，单Buffer 场景请用重载1。`count`须等于`buffers`数组实际长度。                                                                                                                                                         |
| dataType       | 输入    | 输入数据类型，取值见`HcclDataType`枚举。仅支持以下6种：`HCCL_DATA_TYPE_UINT8`、`HCCL_DATA_TYPE_INT16`、`HCCL_DATA_TYPE_INT32`、`HCCL_DATA_TYPE_FP16`、`HCCL_DATA_TYPE_FP32`、`HCCL_DATA_TYPE_BFP16`。 其他取值会被拒绝并抛出异常（携带错误码）。 |
| outputDataType | 输入    | 输出数据类型，取值见`HcclDataType`枚举。支持两种组合：①同精度——取值与`dataType`相同；②升精度——仅在`opType == HCCL_REDUCE_SUM`时支持，将低精度输入归约后升至高精度输出（如`HCCL_DATA_TYPE_INT8`→`HCCL_DATA_TYPE_FP32`，见"调用示例 - 场景2"）。其他`dataType`/`outputDataType`组合返回`CCU_E_NOT_SUPPORT`（见返回值表）。升精度场景下`buffers`须按精度膨胀比例预留 MS，详见`buffers`参数说明。 |
| opType         | 输入    | 归约算子，取值见`HcclReduceOp`枚举。仅支持`HCCL_REDUCE_SUM`、`HCCL_REDUCE_MAX`、`HCCL_REDUCE_MIN`；`HCCL_REDUCE_PROD`不支持，传入会被拒绝并抛出异常（携带错误码）。                                                                                                                                                                              |
| len            | 输入    | 每个Buffer参与归约的字节数，类型为`Variable`，不可超过单片大小（4096字节）。                                                                                                                                                                                                                                                         |
| event          | 输入    | 完成事件对象。                                                                                                                                                                                                                                                                                                  |
| mask           | 输入    | 16位事件掩码。默认值为`1`（即bit0）。                                                                                                                                                                                                                                                                                  |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值                 | 说明                                                                    |
| ------------------- | --------------------------------------------------------------------- |
| `CCU_SUCCESS`       | 操作成功。                                                                 |
| `CCU_E_PARA`        | 参数错误，包括`buffers`为`nullptr`或`count`为0。                                 |
| `CCU_E_NOT_SUPPORT` | 仅重载2：`dataType`/`outputDataType`组合不满足升精度/同精度约束（详见`outputDataType`说明）。 |

> [!NOTE]说明
> 若`dataType`/`opType`取值不在支持范围内（详见参数说明），运行时将抛出异常（携带错误码），不会通过返回值反馈。

## 约束说明

- 归约为原地操作，调用前`dst`（重载1）或`buffers[0]`（重载2）须已写入有效初值，否则归约结果未定义。
- 重载2的`buffers`须指向物理连续的`CcuBuffer`数组，推荐使用`ccu::Array<CcuBuffer>`申请。
- 重载2的`count`取值范围为 `[2, 8]`；`count > 8` 时抛出异常（携带错误码），`count == 0` 直接返回 `CCU_E_PARA`，`count == 1` 不被拒但硬件行为未定义（请改用重载1）。
- 重载2在精度膨胀场景（低精度输入 + SUM 升精度输出）下，`buffers` 须按膨胀比例额外预留MS 以容纳膨胀后的输出，否则硬件读写越界、行为未定义。
- 涉及`CcuBuffer`时（重载2），`len`不可超过`CcuBuffer`单片大小（4096字节）；该上限须由调用方自行保证，超出会导致运行期硬件行为未定义。
- 本接口为异步操作，须通过`EventWait(event, mask)`等待归约完成后再读取结果。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景1：本端HBM→本端HBM归约（FP16 SUM）
CcuResult MyKernel(CcuKernelArg arg) {
    LocalAddr dst, src;
    Variable len;
    Event evt;

    // dst须预先写入初值（如0.0）
    LocalReduce(dst, src, len, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}

// 场景2：4 个MS Buffer 归约到buffers[0]，输入INT8 升精度输出FP32（合法的"输入≠输出"组合；
//        INT8→FP32 = 4× 膨胀，故buffers 数组按膨胀比例预留MS 以容纳膨胀后的输出）
CcuResult MyKernel2(CcuKernelArg arg) {
    Array<CcuBuffer> bufs(4);    // 4个物理连续Buffer
    Variable len;
    Event evt;

    // bufs[0]须预先写入初值
    LocalReduce(bufs.data(), 4,
                HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_FP32,
                HCCL_REDUCE_SUM, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}

// 场景3：输入输出同类型的常规归约（FP16 SUM）
CcuResult MyKernel3(CcuKernelArg arg) {
    Array<CcuBuffer> bufs(4);
    Variable len;
    Event evt;

    LocalReduce(bufs.data(), 4,
                HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP16,
                HCCL_REDUCE_SUM, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```
