# HcommCcuGetMemToken

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

将主机进程的虚拟地址（VA）转换为CCU硬件可访问的内存Token。CCU硬件不直接使用进程虚拟地址访问HBM，须先通过本接口将`(srcVa, size)`对应的内存区域转换为驱动颁发的64位Token，再将Token通过`kernelArg`或`taskArgs`传入Kernel函数，由Kernel内部赋值给`LocalAddr.token`或`RemoteAddr.token`。

本接口与CCU实例生命周期主流程无依赖关系，可在创建实例后的任意时机调用，通常在调用[HcommCcuKernelRegister](HcommCcuKernelRegister.md)之前获取Token并填充到`kernelArg`中。

## 函数原型

```c
CcuResult HcommCcuGetMemToken(uint64_t srcVa, uint64_t size, uint64_t *tokenInfo)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| srcVa | 输入 | 内存区域的起始虚拟地址，须为已向驱动注册的合法地址，不能为0。 |
| size | 输入 | 内存区域的长度，单位为字节，须大于0，且不能超过该虚拟地址对应映射的实际内存大小。 |
| tokenInfo | 输出 | 出参，指向单个`uint64_t`的指针，成功时写入编码后的Token值。不能为空指针。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | 空指针错误，`tokenInfo`为空指针。 |
| `CCU_E_PARA` | 参数错误，`srcVa`为0或`size`为0。 |
| `CCU_E_RUNTIME` | 运行时错误，`srcVa`未分配或者参数`size`大于分配的大小。 |
| `CCU_E_DRV_*` | 驱动层错误，当前可能返回4097（`CCU_E_DRV_INIT_FAILED`）或4098（`CCU_E_DRV_BUSY`）；驱动层错误码段定义为 [`CCU_E_DRV_START`=4096, `CCU_E_DRV_END`=4224]，详见[CcuResult](../../datatype_definition/CcuResult.md)。 |

## 约束说明

- Token属于安全信息，禁止将Token值打印到主机日志或设备日志。
- `(srcVa, size)`须落在同一段已注册的连续内存区域内，跨区域或覆盖未注册内存区域会导致驱动返回错误。
- Token的有效生命周期与对应内存的注册生命周期绑定，内存注销后Token失效。
- 在跨rank通信场景中，本端Token须通过带外通道（非CCU数据面）传递给对端，由对端组装`RemoteAddr.token`，不能通过CCU数据面明文传输Token值。
- 本接口只能在Host侧调用，不能在Kernel函数体内调用。

## 调用示例

```c
// srcVa为已注册的HBM内存起始地址，size为该内存区域的字节大小
uint64_t srcVa = /* 已注册内存的虚拟地址 */;
uint64_t size  = /* 内存区域字节大小 */;

uint64_t tokenInfo = 0;
CcuResult ret = HcommCcuGetMemToken(srcVa, size, &tokenInfo);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuGetMemToken failed, ret = %d\n", ret);
    return ret;
}

// 将tokenInfo通过kernelArg或taskArgs传入Kernel
// Kernel内部将其赋值给LocalAddr.token或RemoteAddr.token
```
