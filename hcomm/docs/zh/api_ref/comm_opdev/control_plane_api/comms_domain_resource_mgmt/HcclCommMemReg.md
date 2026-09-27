# HcclCommMemReg

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

向通信域注册已申请的内存，并获取对应的注册句柄。

## 函数原型

```c
HcclResult HcclCommMemReg(HcclComm comm, const char *memTag, const CommMem *mem, HcclMemHandle *memHandle)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| memTag | 输入 | 内存字符串标签，最大字符长度为HCCL_RES_TAG_MAX_LEN。<br>const uint32_t HCCL_RES_TAG_MAX_LEN = 255; |
| mem | 输入 | 内存信息，CommMem类型的定义可参见[CommMem](../../datatype_definition/CommMem.md)。 |
| memHandle | 输出 | 内存句柄。<br>HcclMemHandle类型的定义如下：<br>typedef void *HcclMemHandle; |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 一个通信域内，同一个memTag，只允许注册一块内存。
- 一个通信域内，相同memTag重复注册会返回HCCL_E_PARA报错，不会复用已有的注册内存句柄。
- 一个通信域内，不同memTag可映射到重叠或相同的内存区域。
- 当`mem->type`为`COMM_MEM_TYPE_CCU`时，表示注册CCU资源空间内存，仅Ascend 950PR/Ascend 950DT支持。CCU类型内存的注册流程与DEVICE一致。详见[CommMemType](../../datatype_definition/CommMemType.md)。

## 调用示例

```c
HcclComm comm; // 通信域句柄，省略获取
const char* memTag = "memTag"; // 内存标签
void *deviceBuffer = nullptr; // 已申请的设备内存地址（需通过aclrtMalloc等接口申请）
CommMem memInfo; // 内存信息
memInfo.addr = deviceBuffer; // 配置已申请的内存地址
memInfo.size = 1024; // 配置已申请的内存大小
memInfo.type = COMM_MEM_TYPE_DEVICE; // 配置已申请的内存类型
HcclMemHandle memHandle; // 内存句柄
HcclCommMemReg(comm, memTag, &memInfo, &memHandle);
```
