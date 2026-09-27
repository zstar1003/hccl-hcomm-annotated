# HcommCcuInsResDescCreate

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

创建空白CCU资源描述符。资源描述符用于描述CCU实例所需的各类资源（loop、buffer、variable、address、event、thread、instruction）的数量规格，不包含实际资源。

一个资源描述符对应一个IO Die。创建后其内部各资源类型的数量初始化为0，后续可通过`HcommCcuInsResDescSetNum`逐类型设置。

## 函数原型

```c
CcuResult HcommCcuInsResDescCreate(uint32_t dieId, HcommCcuResDescHandle *resDesc)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dieId | 输入 | 资源描述符归属的 IO Die 编号，取值范围：`[0, CCU_MAX_IODIE_NUM)`。 |
| resDesc | 输出 | 创建成功后返回的资源描述符句柄，不能为空指针。调用方负责通过`HcommCcuInsResDescDestroy`销毁。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 创建成功，`*resDesc`为有效的资源描述符句柄。 |
| `CCU_E_PARA` | `dieId`超出合法范围（`>= CCU_MAX_IODIE_NUM`）。 |
| `CCU_E_PTR` | `resDesc`为空指针。 |
| `CCU_E_INTERNAL` | 内存分配或内部注册失败。 |

## 约束说明

- 调用本接口前须通过 AscendCL 接口`aclrtSetDevice(int32_t deviceId)`绑定当前线程到目标 NPU Device，本接口从当前线程获取设备上下文。
- 创建成功后，调用方须在不再使用时通过`HcommCcuInsResDescDestroy`销毁该描述符，避免资源泄漏。
- 资源描述符不能跨线程共享。调用方须保证同一描述符的访问串行执行。
- 本接口只能在主机侧调用。

## 依赖关系

- 依赖`HcommCcuInsResDescDestroy`销毁描述符。
- 依赖`HcommCcuInsResDescSetNum`设置资源数量。
- 依赖`HcommCcuInsResDescQueryNum`、`HcommCcuInsResDescQueryDieId`、`HcommCcuQueryRemainResDesc` 查询描述符信息。

## 调用示例

```c
uint32_t dieId = 0;
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(dieId, &resDesc);
if (ret != CCU_SUCCESS) {
    // 错误处理
    return ret;
}

// 使用 resDesc 设置资源规格、查询剩余资源等 ...
// ...

HcommCcuInsResDescDestroy(resDesc);
return CCU_SUCCESS;
```
