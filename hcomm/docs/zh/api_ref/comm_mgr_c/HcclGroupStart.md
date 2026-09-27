# HcclGroupStart

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
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：不支持
<!-- end id5 -->

## 功能说明

开始一个组调用。

在HcclGroupStart和HcclGroupEnd之间调用多个函数，作为一个整体执行，组调用支持以下三种场景：

- 单进程多线程管理NPU：支持调用通信域管理接口[HcclCommInitClusterInfo](HcclCommInitClusterInfo.md)、[HcclCommInitClusterInfoConfig](HcclCommInitClusterInfoConfig.md)、[HcclCommInitRootInfo](HcclCommInitRootInfo.md)、[HcclCommInitRootInfoConfig](HcclCommInitRootInfoConfig.md)  、[HcclCommDestroy](HcclCommDestroy.md)。
- 合并多个集合通信（Ascend 950PR/Ascend 950DT暂不支持）。
- 合并多个点对点通信。

## 函数原型

```c
HcclResult HcclGroupStart()
```

## 参数说明

无

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 仅支持在单机环境中使用组调用接口进行通信域管理。
- 在一个组调用中，通信域管理、集合通信、点对点通信类型的接口不可混用。
- 合并多个点对点通信时，不支持调用[HcclBatchSendRecv](https://gitcode.com/cann/hccl/blob/master/docs/zh/api_ref/comm_op_interface/HcclBatchSendRecv.md)接口。
- 合并多个集合通信时，须确保各集合通信间无依赖关系。
- HcclGroupStart必须和HcclGroupEnd配套使用，HcclGroupStart在前，HcclGroupEnd在后。

## 调用示例

- 示例一：单进程多线程管理NPU

    ```c
    HcclComm hccl_comms[devCount];
    HcclGroupStart();
    for(int i = 0; i &lt; ndev; i++){
        //
            aclrtSetDevice(i);
            HcclCommInitRootInfo(devCount, &rootInfo, global_rank, &(hccl_comms[i]));
        }
    HcclGroupEnd();
    ```

- 示例二：合并多个集合通信操作

    ```c
    HcclGroupStart();
        HCCLCHECK(HcclReduceScatter(sendBuf, recvBuf, 1, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream));
        HCCLCHECK(HcclAllGather(recvBuf, sendBuf, 1, HCCL_DATA_TYPE_FP32, hcclComm, stream));
    HcclGroupEnd();
    ```

- 示例三：合并多个点对点通信

    ```c
    HcclGroupStart();
    for(int i = 0; i &lt; devCount; i++){
        HCCLCHECK(HcclSend(sendBuf[i], count, HCCL_DATA_TYPE_FP32, i, hcclComm, stream));
        }
    for(int i = 0; i &lt; devCount; i++){
        HCCLCHECK(HcclRecv(recvBuf[i], count, HCCL_DATA_TYPE_FP32, i, hcclComm, stream));
        }
    HcclGroupEnd();
    ```
