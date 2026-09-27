# HcommMemFree

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：不支持
<!-- end id5 -->

## 功能说明

释放通过 [HcommMemAlloc](./HcommMemAlloc.md) 申请的内存，依次完成解映射、释放物理内存与释放虚拟地址空间。

该接口封装了如下ACL运行时接口的调用流程：

- `aclrtMemRetainAllocationHandle`：根据虚拟地址反查物理内存句柄。
- `aclrtUnmapMem`：解除虚拟地址到物理内存的映射。
- `aclrtFreePhysical`：释放物理内存。
- `aclrtReleaseMemAddress`：释放预留的虚拟地址空间。

## 函数原型

```c
HcommResult HcommMemFree(void *ptr)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ptr | 输入 | 待释放的虚拟地址指针，需为 [HcommMemAlloc](./HcommMemAlloc.md) 返回的地址。传入 `nullptr` 时接口直接返回成功，不执行任何操作。 |

## 返回值

[HcommResult](./data_type_definition/HcclResult.md)：返回 `HCCL_SUCCESS` 表示接口调用成功；返回 `HCCL_E_RUNTIME` 表示ACL运行时接口调用失败（反查句柄、解映射、释放物理内存或释放虚拟地址失败）。

## 约束说明

- `ptr` 必须为 [HcommMemAlloc](./HcommMemAlloc.md) 返回的虚拟地址，不能传入其他途径申请的内存地址。
- 接口通过 `aclrtMemRetainAllocationHandle` 反查物理内存句柄，若 `ptr` 为非法值，则会触发运行时错误。
- 释放后不应再访问 `ptr` 指向的内存。
- 传入 `nullptr` 是安全的，接口会直接返回成功。

## 调用示例

以下示例展示申请内存、注册为对称内存窗口、解注册并释放的流程，更多对称内存使用约束请参见[HcclCommSymWinRegister](./HcclCommSymWinRegister.md)。

```c
// 返回值检查宏
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define HCOMMCHECK(cmd) do { HcommResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

// 创建并初始化通信域配置项
HcclCommConfig config;
HcclCommConfigInit(&config);
config.hcclSymWinMaxMemSizePerRank = 2; //单位GB

// 获取通信域参数
uint32_t rankSize = 4;
uint32_t rankId = 0;
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));

// 初始化集合通信域
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, rankId, &config, &hcclComm));

// 创建任务流
aclrtStream stream;
ACLCHECK(aclrtCreateStream(&stream));

// 申请内存并注册为对称内存窗口
size_t sendBytes = 1024;
size_t recvBytes = rankSize * sendBytes;
size_t memSize = sendBytes + recvBytes;
void *devPtr = nullptr;
HCOMMCHECK(HcommMemAlloc(&devPtr, memSize));

HcclCommSymWindow symWin;
HCCLCHECK(HcclCommSymWinRegister(hcclComm, devPtr, memSize, &symWin, 1));

// 调用集合通信算子
void *sendBuff = devPtr;
void *recvBuff = static_cast<char*>(devPtr) + sendBytes;
HCCLCHECK(HcclAllGather(sendBuff, recvBuff, sendBytes, HCCL_DATA_TYPE_INT8, hcclComm, stream));

// 阻塞等待任务流中的集合通信任务执行完成
ACLCHECK(aclrtSynchronizeStream(stream));

// 解注册对称内存
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// 释放HcommMemAlloc申请的内存
HCOMMCHECK(HcommMemFree(devPtr));

// 销毁任务流
ACLCHECK(aclrtDestroyStream(stream));

// 销毁通信域
HCCLCHECK(HcclCommDestroy(hcclComm));
```
