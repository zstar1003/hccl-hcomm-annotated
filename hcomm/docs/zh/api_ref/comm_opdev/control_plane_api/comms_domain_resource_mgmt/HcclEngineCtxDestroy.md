# HcclEngineCtxDestroy

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

指定通信域与通信引擎，使用特定标签销毁对应的通信引擎上下文。

## 函数原型

```c
HcclResult HcclEngineCtxDestroy(HcclComm comm, const char *ctxTag, CommEngine engine)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| ctxTag | 输入 | 通信引擎上下文标签（最大字符长度为HCCL_RES_TAG_MAX_LEN）。 |
| engine | 输入 | 通信引擎类型。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无

## 调用示例

```c
HcclComm comm;
uint64_t size = 16;
void *ctx = nullptr;
string ctxTag = "ctxTag";
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
HcclResult ret = HcclEngineCtxCreate(comm, ctxTag, engine, size, &ctx);
ret = HcclEngineCtxDestroy(comm, ctxTag, engine);
```
