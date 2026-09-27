# HCCL_INTER_HCCS_DISABLE

## 功能描述

此环境变量用于配置超节点模式组网中超节点内的通信链路类型，支持如下取值：

- TRUE：代表超节点内的AI节点间使用RoCE进行RDMA通信。
- FALSE：代表超节点内的AI节点间使用HCCS通信链路进行SDMA通信。

默认值为“FALSE”。

## 配置示例

```bash
export HCCL_INTER_HCCS_DISABLE=FALSE
```

## 使用约束

无

## 产品支持情况

<!-- npu="950" id2 -->
- Ascend 950PR/Ascend 950DT：不支持
<!-- end id2 -->
<!-- npu="A3" id1 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->
