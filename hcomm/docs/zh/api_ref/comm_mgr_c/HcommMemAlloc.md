# HcommMemAlloc

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

申请Device物理内存并映射到虚拟地址空间，一步完成虚拟地址预留、物理内存申请与映射。

该接口封装了如下ACL运行时接口的调用流程，便于通信库开发者使用：

- `aclrtReserveMemAddress`：在进程虚拟地址空间预留一段区间。
- `aclrtMallocPhysical`：申请Device物理内存。
- `aclrtMapMem`：将物理内存映射到预留的虚拟地址。

申请到的虚拟地址需使用[HcommMemFree](./HcommMemFree.md) 接口释放。当前该接口主要服务于对称内存注册场景，配合[HcclCommSymWinRegister](./HcclCommSymWinRegister.md) 使用。

## 函数原型

```c
HcommResult HcommMemAlloc(void **ptr, size_t size)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ptr | 输出 | 返回的虚拟地址指针的地址。接口调用成功后，`*ptr` 指向已映射物理内存的虚拟地址起始位置。 |
| size | 输入 | 申请内存的大小，单位为字节，必须大于 0。接口内部会按设备内存分配对齐粒度（通常为 2MB）向上对齐。 |

## 返回值

[HcommResult](./data_type_definition/HcclResult.md)：接口成功返回 `HCCL_SUCCESS`，其他值表示失败。

- `HCCL_E_PARA`：参数非法（`ptr` 为空指针或 `size` 为 0）。
- `HCCL_E_RUNTIME`：ACL运行时接口调用失败（获取设备、预留虚拟地址、申请物理内存或映射失败）。

## 约束说明

- 调用前需通过 `aclrtSetDevice` 设置当前Device，接口内部通过 `aclrtGetDevice` 获取当前设备ID。
- 申请的物理内存属性固定为：`ACL_HBM_MEM_HUGE`、`ACL_MEM_ALLOCATION_TYPE_PINNED`、`ACL_MEM_HANDLE_TYPE_NONE`。
- `size` 会按照 `aclrtMemGetAllocationGranularity` 返回的对齐粒度进行向上对齐，因此实际申请的物理内存大小可能大于 `size`。
- 申请到的虚拟地址需通过[HcommMemFree](./HcommMemFree.md) 释放，不能直接使用 `aclrtFree`、`aclrtMalloc` 系列接口混用管理。
- 接口内部在申请物理内存或映射失败时，会自动回滚已预留的虚拟地址或已申请的物理内存。

## 调用示例

以下示例展示使用 `HcommMemAlloc` 申请内存并注册为对称内存窗口的流程，更多对称内存使用约束请参见[HcclCommSymWinRegister](./HcclCommSymWinRegister.md)。

```c
// 返回值检查宏
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define HCOMMCHECK(cmd) do { HcommResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

// 创建并初始化通信域配置项
HcclCommConfig config;
HcclCommConfigInit(&config);
config.hcclSymWinMaxMemSizePerRank = 2; //单位GB，预留每rank对称内存VA大小

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

// 申请对称内存所需的Device内存
size_t sendBytes = 1024;
size_t recvBytes = rankSize * sendBytes;
size_t memSize = sendBytes + recvBytes;
void *devPtr = nullptr;
HCOMMCHECK(HcommMemAlloc(&devPtr, memSize));

// 注册为对称内存窗口
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

// 释放内存
HCOMMCHECK(HcommMemFree(devPtr));

// 销毁任务流
ACLCHECK(aclrtDestroyStream(stream));

// 销毁通信域
HCCLCHECK(HcclCommDestroy(hcclComm));
```
