# HcclGetRootInfo

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
- Atlas 推理系列产品：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：支持
<!-- end id5 -->

## 功能说明

此接口需要在HCCL初始化接口[HcclCommInitRootInfo](HcclCommInitRootInfo.md)或[HcclCommInitRootInfoConfig](HcclCommInitRootInfoConfig.md)前调用，仅需在root节点调用，用于生成root节点的rank标识信息（HcclRootInfo）。

- 该接口需要和初始化接口[HcclCommInitRootInfo](HcclCommInitRootInfo.md)或[HcclCommInitRootInfoConfig](HcclCommInitRootInfoConfig.md)接口配对使用，不能单独使用。

- 该接口支持单线程循环调用，即开发者可在一个for循环中通过“指定不同的Device + 调用此接口”，从而实现在一个线程中获取不同设备的rootInfo信息。

    假设一个AI Server中有8张卡，8张卡分成4个通信域，每个通信域中的两张卡之间通信，如下图所示。

    ![通信域划分示例](figures/comm_domain_split.png)

    获取rootInfo信息并进行集合通信初始化的流程如下图所示。

    ![单线程循环调用示例](figures/single_thread_loop_call_example.png)

    首先在一个线程中通过切换Device创建4个rootInfo信息，并存入一个长度为4的数组中。rootInfo信息获取完成后，起4个线程，分别调用HcclCommInitRootInfo或者HcclCommInitRootInfoConfig接口（上图中以HcclCommInitRootInfo接口示意）根据不同的rootInfo信息进行通信域初始化。

- 多机集合通信场景，调用HcclGetRootInfo前，可以进行如下操作（非必选）：
  - 配置环境变量[HCCL_IF_IP](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_IF_IP.md)或[HCCL_SOCKET_IFNAME](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_SOCKET_IFNAME.md)，指定HCCL的初始化root网卡IP（环境变量HCCL_IF_IP的优先级高于HCCL_SOCKET_IFNAME，若二者都不配置，默认使用网卡名称的字典序升序选择root网卡）。
  - 配置环境变量[HCCL_WHITELIST_DISABLE](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_WHITELIST_DISABLE.md)开启白名单校验，并通过[HCCL_WHITELIST_FILE](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_WHITELIST_FILE.md)指定通信白名单配置文件（若不配置，默认关闭通信白名单校验）。

## 函数原型

```c
HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| rootInfo | 输出 | 本rank的标识信息，主要包含device ip、device id等信息。此信息需广播至集群内所有rank用来进行HCCL初始化。<br>HcclRootInfo类型的定义可参见[HcclRootInfo](./data_type_definition/HcclRootInfo.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无

## 调用示例

```c
uint32_t rankSize = 8;
uint32_t deviceId = 0;
// 生成root节点的rank标识信息
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);
// 初始化通信域
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);
// 销毁通信域
HcclCommDestroy(hcclComm);
```
