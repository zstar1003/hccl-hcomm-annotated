# HcclCommActivateCommMemory

> [!NOTE]说明
> 本接口为试用接口，后续可能存在变更，暂不支持应用于生产环境。

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：不支持
<!-- end id5 -->

## 功能说明

激活预留的虚拟内存，只有使用激活后的内存作为通信算子的输入、输出才可开启零拷贝功能。

## 函数原型

```c
HcclResult HcclCommActivateCommMemory(HcclComm comm, void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | HCCL通信域，建议使用Server内最大的通信域，即覆盖最大卡数的通信域。 |
| virPtr | 输入 | 需要激活的虚拟内存地址，即用户调用aclrtMapMem接口进行物理内存与虚拟内存映射时，传入的待映射的虚拟内存地址。 |
| size | 输入 | 需要激活的内存大小，单位：Byte。 |
| offset | 输入 | 预留字段。<br>当前仅支持配置为“0”。 |
| handle | 输入 | 申请的物理内存信息handle，即用户调用aclrtMallocPhysical接口申请的Device物理内存信息handle。 |
| flags | 输入 | 预留字段。<br>当前仅支持配置为“0”。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 待激活的虚拟内存地址必须在[HcclCommSetMemoryRange](HcclCommSetMemoryRange.md)设置的地址范围内。
- 该虚拟内存地址不能与已经激活的虚拟内存地址有重叠、交叠。

## 调用示例

```c
// 假设已通过HcclCommSetMemoryRange设置了虚拟内存范围
// baseVirPtr为虚拟内存基地址

// 申请物理内存
aclrtDrvMemHandle handle;
aclrtMallocPhysical(&handle, size, NULL, 0);

// 将物理内存映射到虚拟内存
void *virPtr = baseVirPtr;
aclrtMapMem(virPtr, size, 0, handle, 0);

// 激活预留的虚拟内存
HcclCommActivateCommMemory(hcclComm, virPtr, size, 0, handle, 0);

// 后续可使用激活后的内存进行零拷贝通信
// ...

// 反激活内存
HcclCommDeactivateCommMemory(hcclComm, virPtr);
```
