# 简介

本节包含CCU Kernel Host侧生命周期管理接口及内存Token查询接口。

通过这些接口，用户可以完成Kernel的注册、翻译与启动执行，以及将进程虚拟地址转换为CCU访问Token。

## 前置条件

Kernel注册和启动接口需要传入`CcuInsHandle`（CCU实例句柄）。当前仅支持集合通信场景，调用前须完成HCCL通信域初始化、CCU建链和CCU实例绑定。

1. **创建 HcclComm 通信域**：参见 [HcclCommInitClusterInfo](../../../comm_mgr_c/HcclCommInitClusterInfo.md)
   或 [HcclCommInitRootInfo](../../../comm_mgr_c/HcclCommInitRootInfo.md)。

2. **规划资源并创建CCU实例**：

   - 调用[HcommCcuInsResDescCreate](../ccu_resource_mgmt/HcommCcuInsResDescCreate.md)为目标IO Die创建资源描述符。
   - 调用[HcommCcuKernelQueryResReq](../ccu_resource_mgmt/HcommCcuKernelQueryResReq.md)统计Kernel资源诉求。
   - 调用[HcommCcuInsCreate](../ccu_resource_mgmt/HcommCcuInsCreate.md)创建CCU实例。

3. **绑定CCU实例**：

   - 调用[HcclCommAssignCcuIns](../comms_domain_resource_mgmt/HcclCommAssignCcuIns.md)将实例绑定到通信域。绑定成功后，实例所有权和销毁责任转移给通信域。
   - 创建实例的调用方可继续使用`HcommCcuInsCreate`返回的句柄注册和启动Kernel，但不得再自行销毁实例。仅持有通信域句柄时，可调用[HcclCommQueryAssignedCcuIns](../comms_domain_resource_mgmt/HcclCommQueryAssignedCcuIns.md)获取已绑定的实例句柄。

   通过通信域查询已绑定实例句柄的典型用法：

    ```c
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    HcclResult ret = HcclCommQueryAssignedCcuIns(comm, &insHandle, &insNum);
    // 当前一个通信域最多绑定1个CCU实例，查询成功时insNum为1。
    if (ret != HCCL_SUCCESS || insNum != 1) {
        // 错误处理
    }
    ```

> 该接口属于HCCL层（不在`Hcomm*`/`Ccu*`系列内），暂未提供独立API参考页面。
> 完整签名以头文件`include/hccl/hccl_ccu_res.h`为准。
> 如果通信域未绑定CCU实例，`HcclCommQueryAssignedCcuIns`返回`HCCL_E_UNAVAIL`。
> 此外，[HcclCommQueryCcuIns](../comms_domain_resource_mgmt/HcclCommQueryCcuIns.md)用于查询通信域自有的CCU实例（未创建时创建CCU实例），与本接口查询的绑定实例相互独立。

## 接口调用顺序

Kernel注册与启动接口的标准顺序如下：

1. [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md)：开始一轮Kernel注册。
2. [HcommCcuKernelRegister](HcommCcuKernelRegister.md)：注册Kernel函数，记录其操作序列。
3. [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md)：结束本轮注册，翻译为设备指令并下发。
4. [HcommCcuKernelLaunch](HcommCcuKernelLaunch.md)：启动Kernel执行（可重复调用）。

旁路接口：

- [HcommCcuGetMemToken](HcommCcuGetMemToken.md)：将进程虚拟地址转换为CCU可用的内存Token，与主流程无依赖关系，可在任意时机调用。

## 参见

- [CCU 快速上手（含 AllGather 完整流程）](../../../../comm_op_dev_guide/ccu_quick_start.md)
- [CCU 通信算子开发指南（分步详解）](../../../../comm_op_dev_guide/ccu_comm_op_dev/README.md)
