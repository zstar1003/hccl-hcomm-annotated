# HcclEngineCtxCreate

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

指定通信域与通信引擎，使用特定标签创建对应的通信引擎上下文。

通信引擎上下文是该通信引擎数据面可以使用的一块内存，用于存放执行算子时所需的资源句柄或参数等信息，创建一次后可重复获取使用。指定通信域和通信引擎类型，一个通信引擎标签可以索引一个通信引擎上下文。

## 函数原型

```c
HcclResult HcclEngineCtxCreate(HcclComm comm, const char *ctxTag, CommEngine engine, uint64_t size, void **ctx)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义可参见[HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md)。 |
| ctxTag | 输入 | 通信引擎上下文标签，最大字符长度为HCCL_RES_TAG_MAX_LEN。<br>const uint32_t HCCL_RES_TAG_MAX_LEN = 255; |
| engine | 输入 | 通信引擎类型。<br>CommEngine的定义可参见[CommEngine](../../datatype_definition/CommEngine.md)。 |
| size | 输入 | ctx内存大小，单位Byte。<br>size不能为0。 |
| ctx | 输出 | 通信引擎上下文。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无。

## 调用示例

```c
// 通信域句柄
HcclComm comm;
// 创建 16B 大小的通信引擎上下文
uint64_t size = 16;
void *ctx = nullptr;
string ctxTag = "ctxTag";
CommEngine engine = COMM_ENGINE_CPU_TS;
HcclResult ret = HcclEngineCtxCreate(comm, ctxTag, engine, size, &ctx);
if (ret != HCCL_SUCCESS) {
    // 错误处理
}
```
