# HcommCcuKernelQueryResReq

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

在创建CCU实例前查询指定CCU Kernel的资源诉求。接口在主机侧对`kernelFunc`执行一次dry-run，将统计得到的各类资源数量写入调用方预先创建的资源描述符`resDesc`。

查询过程会执行Kernel函数及其既有的Channel检查和选die逻辑，但不会注册Kernel、申请CCU实例资源、生成Kernel句柄、翻译指令或下发任务。

## 函数原型

```c
CcuResult HcommCcuKernelQueryResReq(const void *kernelFunc,
    const void **kernelArgs, uint32_t argNum, HcommCcuResDescHandle resDesc)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| kernelFunc | 输入 | CCU Kernel函数指针，不能为空指针。`argNum`为`0`时，函数签名须与无入参Kernel一致；`argNum`为`1`时，函数签名须与单入参Kernel一致。 |
| kernelArgs | 输入 | Kernel函数入参指针数组。`argNum`为`0`时该参数被忽略，可传入空指针；`argNum`为`1`时不能为空指针，且`kernelArgs[0]`也不能为空指针。 |
| argNum | 输入 | Kernel函数入参个数。当前仅支持`0`或`1`。 |
| resDesc | 输入/输出 | CCU资源描述符句柄。须通过`HcommCcuInsResDescCreate`接口根据kernel运行的dieId预先创建，不能为`0`。查询成功后，接口将本次统计的资源数量写入该描述符，并保留创建描述符时设置的die ID。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 查询成功，Kernel资源诉求已写入`resDesc`。 |
| `CCU_E_PTR` | `kernelFunc`为空指针；或`argNum`为`1`时，`kernelArgs`或`kernelArgs[0]`为空指针。 |
| `CCU_E_PARA` | `argNum`大于`1`、`resDesc`为`0`、描述符中的die ID非法，或Kernel根据Channel选择的die与描述符中的die ID不一致。 |
| `CCU_E_NOT_FOUND` | 未找到`resDesc`对应的资源描述符，或Kernel使用的Channel不存在。 |
| `CCU_E_INTERNAL` | Kernel dry-run或资源数量写入过程中发生内部错误。 |
| 其他错误码 | Kernel执行、Channel检查或选die过程中产生的其他错误。 |

## 约束说明

- 调用本接口前，须通过`HcommCcuInsResDescCreate`创建`resDesc`。调用结束后，调用方负责通过`HcommCcuInsResDescDestroy`销毁该描述符。
- 如果Kernel使用Channel，须提前通过[HcclChannelAcquire](../comms_domain_resource_mgmt/HcclChannelAcquire.md)获取`COMM_ENGINE_CCU`类型的Channel，并保证Channel在查询期间有效。
- 调用`HcommCcuInsResDescCreate`创建`resDesc`与调用本接口时，当前线程须绑定同一NPU Device，即两次调用获取到的`deviceLogicId`必须一致。调用方可在相应线程中调用AscendCL接口`aclrtSetDevice(int32_t deviceId)`指定该线程使用的Device。本接口从当前线程获取`deviceLogicId`，不从`resDesc`推导设备信息，不支持跨Device使用资源描述符。Kernel使用的Channel必须位于同一die，且选出的die须与创建`resDesc`时指定的die ID一致。
- 本接口不支持多线程并发调用。调用方须保证不同线程对本接口的调用串行执行。
- 查询成功时，本次统计结果覆盖`resDesc`中已有的资源数量，不修改其die ID。
- Kernel dry-run失败时不会写入资源数量。写入资源数量的过程中如果发生错误，接口立即返回，已经写入的资源类型不会回滚。
- 本接口只能在主机侧调用，不能在Kernel函数体内调用。

## 调用示例

```c
typedef struct {
    ChannelHandle channel;
    uint32_t loopCount;
} MyKernelArg;

CcuResult MyKernel(CcuKernelArg arg)
{
    MyKernelArg *kernelArg = (MyKernelArg *)arg;
    // 使用kernelArg->channel和CCU数据面接口描述Kernel操作序列。
    // ...
    return CCU_SUCCESS;
}

uint32_t dieId = 0;
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(dieId, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

// channel须提前通过HcclChannelAcquire获取，并与dieId对应。
ChannelHandle channel = 0;
// ... 此处省略HcclChannelAcquire调用 ...

MyKernelArg kernelArg = { .channel = channel, .loopCount = 10 };
const void *kernelArgs[] = { &kernelArg };
ret = HcommCcuKernelQueryResReq(
    (const void *)MyKernel, kernelArgs, 1, resDesc);
if (ret != CCU_SUCCESS) {
    HcommCcuInsResDescDestroy(resDesc);
    return ret;
}

uint32_t instructionNum = 0;
ret = HcommCcuInsResDescQueryNum(
    resDesc, HCOMM_CCU_RES_TYPE_INSTRUCTION, &instructionNum);

HcommCcuInsResDescDestroy(resDesc);
return ret;
```
