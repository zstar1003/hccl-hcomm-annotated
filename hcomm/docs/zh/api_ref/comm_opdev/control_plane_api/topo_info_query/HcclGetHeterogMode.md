# HcclGetHeterogMode

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：不支持
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

给定通信域，获取通信域的异构组网模式。

## 函数原型

```c
HcclResult HcclGetHeterogMode(HcclComm comm, HcclHeterogMode *mode)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 集合通信操作所在的通信域。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| mode | 输出 | 异构模式<br>HcclHeterogMode的类型可以参见[HcclHeterogMode](../../datatype_definition/HcclHeterogMode.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无

## 调用示例

```c
HcclHeterogMode mode;
HcclResult ret = HcclGetHeterogMode(comm, &mode);
if (ret == HCCL_SUCCESS) {
    switch (mode) {
        case HCCL_HETEROG_MODE_HOMOGENEOUS:
            printf("同构组网\n");
            break;
        case HCCL_HETEROG_MODE_MIX_A2_A3:
            printf("A2/A3异构组网\n");
            break;
        default:
            printf("未知组网模式\n");
            break;
    }
}
```
