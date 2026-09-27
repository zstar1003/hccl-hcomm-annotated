# HcclGetHcclBufferCleared

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

获取通信域中本端rank的HCCL通信内存，且获取到的通信内存已做内存清零，首次调用时将在Device侧进行内存初始化分配，后续调用将复用已分配内存，不会重复初始化。

> [!NOTE]注意
> 返回的HCCL通信内存由hcomm内部管理，调用者严禁释放。

## 函数原型

```c
HcclResult HcclGetHcclBufferCleared(HcclComm comm, void **buffer, uint64_t *size)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| buffer | 输出 | HCCL通信内存地址。 |
| size | 输出 | HCCL通信内存大小。内存大小为通信域初始化配置或环境变量HCCL_BUFFSIZE配置值的2倍，默认为400MB。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

如果通信域仅包含一个rank则不会申请HCCL通信内存，出参`buffer`为空指针，`size`为0。

## 调用示例

```c
HcclComm comm; // 已初始化
void *hcclBuffer = nullptr;
uint64_t hcclBufferSize = 1 * 1024;   // 1KB
HcclResult result = HcclGetHcclBufferCleared(comm, &hcclBuffer, &hcclBufferSize);
```
