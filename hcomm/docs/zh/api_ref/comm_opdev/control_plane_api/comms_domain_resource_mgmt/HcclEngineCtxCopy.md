# HcclEngineCtxCopy

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

指定通信域、通信引擎与通信引擎上下文标签，将Host侧内存数据拷贝至对应的通信引擎上下文中。

## 函数原型

```c
HcclResult HcclEngineCtxCopy(HcclComm comm, CommEngine engine, const char *ctxTag, const void *srcCtx, uint64_t size, uint64_t dstCtxOffset)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| engine | 输入 | 通信引擎类型。 |
| ctxTag | 输入 | 通信引擎上下文标签（最大字符长度为HCCL_RES_TAG_MAX_LEN）。 |
| srcCtx | 输入 | 源内存地址。 |
| size | 输入 | 源内存大小。 |
| dstCtxOffset | 输入 | 拷贝至通信引擎上下文中的地址偏移。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无

## 调用示例

```c
// 通信域句柄
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;
const char *ctxTag = "ctxTag";
void *resCtx = nullptr;  // 有效的源ctx，需指向已分配的内存
uint64_t size = 16; // 需要拷贝的实际大小
uint64_t dstCtxOffset = 0; // 全部拷贝情况下，偏移传0
HcclResult ret = HcclEngineCtxCopy(comm, engine, ctxTag, resCtx, size, dstCtxOffset);
if (ret != HCCL_SUCCESS) {
    // 错误处理
}
```
