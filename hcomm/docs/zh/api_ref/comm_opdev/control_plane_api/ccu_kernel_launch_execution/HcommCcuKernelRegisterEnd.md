# HcommCcuKernelRegisterEnd

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

结束一轮Kernel注册，将本轮注册过程中产生的所有Kernel一次性翻译为CCU设备指令并下发到设备内存。翻译完成后，本轮注册的所有Kernel均可通过[HcommCcuKernelLaunch](HcommCcuKernelLaunch.md)启动执行。

## 函数原型

```c
CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle insHandle)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| insHandle | 输入 | CCU实例句柄，从Hccl通信域中获取。有效句柄从1开始；传入0或未注册的句柄将返回`CCU_E_PTR`。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | `insHandle`为0或无效，未找到对应实例。 |
| `CCU_E_INTERNAL` | 内部错误，翻译失败或设备内存拷贝失败；或时序错误：未先调用[HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md)就调用本接口。 |
| `CCU_E_UNAVAIL` | 硬件资源不足，无法完成本轮Kernel的资源分配与翻译。 |

## 约束说明

- 应在[HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md)和至少一次[HcommCcuKernelRegister](HcommCcuKernelRegister.md)之后调用，且应先于[HcommCcuKernelLaunch](HcommCcuKernelLaunch.md)；若未先调用[HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md)，本接口返回`CCU_E_INTERNAL`。

> [!NOTE]说明
> 须按 [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) → [HcommCcuKernelRegister](HcommCcuKernelRegister.md) → 本接口的顺序调用。未先调用[HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md)就调用本接口会返回`CCU_E_INTERNAL`。

- 本接口调用成功后，本轮注册的Kernel即可独立启动。若需注册新一轮Kernel，须重新调用[HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md)开始新一轮流程。
- 本接口只能在Host侧调用，不能在Kernel函数体内调用。

## 调用示例

```c
// insHandle 从通信域中获取，且已完成RegisterStart 与Register
CcuInsHandle insHandle = 0;
// ... 此处省略HcommCcuKernelRegisterStart、HcommCcuKernelRegister 调用 ...

// 结束注册，翻译为设备指令并下发
CcuResult ret = HcommCcuKernelRegisterEnd(insHandle);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelRegisterEnd failed, ret = %d\n", ret);
    return ret;
}
// 本轮注册的所有Kernel 均已就绪，可调用HcommCcuKernelLaunch 启动
```
