# HcclCommSymWinRegister

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
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

用户可调用此接口将业务内存注册为对称内存窗口，使HCCL在支持的集合通信算子中直接使用该内存。

对称内存是一种内存管理模型，允许并行处理单元（例如每个rank）以一种“全局可见”但“无需显式地址交换”的方式访问彼此的内存。

当前对称内存支持如下场景：

<!-- npu="950" id6 -->
- Ascend 950PR/Ascend 950DT的URMA场景：用户将已申请的Device内存注册为对称内存窗口。该场景下，HcclCommSymWinRegister仅完成本端对称内存窗口登记；跨rank的内存注册、memHandle交换和远端内存信息更新在相关UB/URMA通信通道创建时完成。使用集合通信接口时，该流程由集合通信算子内部触发。在远端内存信息更新完成前，不应调用[HcclSymWinGetRemoteAddr](HcclSymWinGetRemoteAddr.md)获取远端地址。
- Ascend 950PR/Ascend 950DT的UB Memory场景：用户在申请虚拟内存和物理内存并完成映射后，将虚拟内存注册为对称内存。底层以addr所属的完整allocation建立共享映射，对外窗口范围仍为用户传入的[addr, addr+size)，并通过提前预留相同大小、相同布局的虚拟地址来实现对称内存。
<!-- end id6 -->
<!-- npu="A3" id7 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品的HCCS场景：用户在申请虚拟内存和物理内存并完成映射后，将虚拟内存注册为对称内存。该场景通过提前预留相同大小、相同布局的虚拟地址来实现对称内存。
<!-- end id7 -->

<!-- npu="A3" id8 -->
Atlas A3 训练系列产品/Atlas A3 推理系列产品HCCS场景的对称内存基本实现模型如下图所示。

![对称内存实现模型](./figures/symmetric_memory.png)

- 为每个rank申请虚拟内存，假设每个rank对应的虚拟内存大小为heap_size，通信域中的rank数量为rank_size，则通信域中总虚拟内存大小为heap_size\*rank_size。
- 每个rank的虚拟地址布局都是相同的。
- 不同rank的物理地址映射到每个rank对应位置的虚拟地址，实现对其他rank内存的访问。
<!-- end id8 -->

对称内存功能使得HCCL可以直接对业务传入的内存进行操作，无需经过中间缓冲区（HCCL buffer），从而减少内存拷贝开销。

## 函数原型

```c
HcclResult HcclCommSymWinRegister(HcclComm comm, void *addr, uint64_t size, HcclCommSymWindow *winHandle, uint32_t flag)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | HCCL通信域。<br>不同产品型号下此参数的配置说明请参见[comm说明](#comm说明)。|
| addr | 输入 | 对称内存窗口的起始地址。<br>不同产品型号下此参数的配置说明请参见[addr说明](#addr说明)。|
| size | 输入 | 对称内存窗口的大小。<br>不同产品型号下此参数的配置说明请参见[size说明](#size说明)。|
| winHandle | 输出 | 存放“对称内存窗口资源句柄“的指针。<br>HcclCommSymWindow类型的定义请参见[HcclCommSymWindow](./data_type_definition/HcclCommSymWindow.md)。 |
| flag | 输入 | 是否启用对称内存，当前仅支持传入1。 |

### comm说明

<!-- npu="950" id13 -->
- Ascend 950PR/Ascend 950DT的URMA场景下，无需通过hcclSymWinMaxMemSizePerRank配置预留的对称内存大小。
<!-- end id13 -->
<!-- npu="A3" id14 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品的HCCS场景下，建议使用超节点内最大的通信域，即覆盖最大卡数的通信域。初始化通信域时可以通过[HcclCommConfig](./data_type_definition/HcclCommConfig.md)的hcclSymWinMaxMemSizePerRank参数设置每个rank预留的对称内存大小，若不设置hcclSymWinMaxMemSizePerRank，使用默认值16GB。当前通信域预留的总虚拟对称内存大小为：rankSize * HcclCommConfig.hcclSymWinMaxMemSizePerRank。
<!-- end id14 -->

### addr说明

<!-- npu="950" id15 -->
- Ascend 950PR/Ascend 950DT的URMA场景下，该地址为已申请的Device内存地址，内存需要在调用[HcclCommSymWinDeregister](HcclCommSymWinDeregister.md)解注册前保持有效。
- Ascend 950PR/Ascend 950DT的UB Memory场景下，该地址为预留并完成物理内存映射的虚拟地址。
<!-- end id15 -->
<!-- npu="A3" id16 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品的HCCS场景下，该地址为预留的虚拟内存地址，虚拟内存需要调用aclrtReserveMemAddress接口预留。
<!-- end id16 -->

### size说明

<!-- npu="950" id17 -->
- Ascend 950PR/Ascend 950DT的URMA场景下，size需要大于0，且所有rank调用该接口时输入的size需要保持一致。
- Ascend 950PR/Ascend 950DT的UB Memory场景下，size需要大于0，且size不能超过addr所属allocation的大小。实际注册的对称内存窗口大小等于addr所属allocation的大小。
<!-- end id17 -->
<!-- npu="A3" id18 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品的HCCS场景下，0 < size <= HcclCommConfig.hcclSymWinMaxMemSizePerRank，并且size不能超过“与addr做映射的物理内存”大小（即调用aclrtMallocPhysical接口申请的Device物理内存）。对称内存注册按物理内存的大小对齐，实际注册的对称内存窗口大小等于“与addr做映射的物理内存”的大小。
<!-- end id18 -->

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

<!-- npu="950" id11 -->
- 针对Ascend 950PR/Ascend 950DT：
  - 支持URMA和UB Memory场景。
  - URMA场景仅支持集合通信算子AllGather，依赖集合通信算子内部创建UB/URMA通信通道完成对称内存资源注册和交换，且不要求对称组网。
  - UB Memory场景底层以addr所属的完整allocation建立共享映射，对外窗口范围为[addr, addr+size)。同一通信域内所有LSA WorldTeam成员调用本接口时，各成员的注册调用次序及每次注册对应的allocation大小必须保持一致（例如所有成员的第1次调用注册相同大小的allocation、第2次调用同样注册相同大小的allocation，依此类推），否则注册失败。
  - UB Memory场景的注册包含LSA WorldTeam成员间的集合操作，若集合操作完成后成员本地执行失败（如本地映射失败、资源不足），本通信域的UB Memory对称内存将进入不可用状态，后续注册直接返回错误；此时需解注册已注册的窗口并销毁重建通信域。
<!-- end id11 -->
<!-- npu="A3" id12 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品：
  - 仅支持HCCS链路通信场景。
  - 仅支持对称组网，即每个Server内卡数相同的场景。
  - 仅支持超节点内AI Server间使用HCCS链路进行SDMA通信的场景，不支持使用RoCE进行RDMA通信的场景（即不支持设置环境变量HCCL_INTER_HCCS_DISABLE为"TRUE"，单机场景该环境变量无效）。
  - 仅支持集合通信算子AllGather、ReduceScatter、AllReduce、AllToAll。
  - 所有rank的输入地址映射的物理内存大小一致（对称内存注册按物理内存的大小对齐）。
<!-- end id12 -->
- 该接口仅支持通信算子展开模式为AI CPU的场景。
- 需确保通信域中的所有rank同时调用该注册接口。
- 所有rank调用该接口时，URMA场景下输入的size参数需要保持一致；UB Memory场景下各rank对应次序注册的allocation大小需要保持一致。
- 使用对称内存功能时，算子的输入、输出内存必须调用此接口注册为对称内存。
- 调用该接口注册的内存需要使用[HcclCommSymWinDeregister](HcclCommSymWinDeregister.md)接口解注册。

## 调用示例

<!-- npu="950" id9 -->
### Ascend 950PR/Ascend 950DT URMA场景

```c
// 创建并初始化通信域配置项
HcclCommConfig config;
HcclCommConfigInit(&config);

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

size_t sendBytes = 1024;
size_t recvBytes = rankSize * sendBytes;
size_t memSize = sendBytes + recvBytes;

// 申请Device内存
void *devPtr = nullptr;
ACLCHECK(aclrtMalloc(&devPtr, memSize, ACL_MEM_MALLOC_HUGE_FIRST));

HcclCommSymWindow symWin;
// 注册对称内存
HCCLCHECK(HcclCommSymWinRegister(hcclComm, devPtr, memSize, &symWin, 1));

// 使用对称内存
void *sendBuff = devPtr;
void *recvBuff = static_cast<char*>(sendBuff) + sendBytes;

// 调用集合通信算子。
// Ascend 950PR/Ascend 950DT的URMA场景下，HcclCommSymWinRegister仅完成对称内存窗口登记；
// HcclAllGather内部创建UB/URMA通信通道时，会通过HcclChannelAcquire相关流程完成内存注册、memHandle交换和远端内存信息更新。
HCCLCHECK(HcclAllGather(sendBuff, recvBuff, sendBytes, HCCL_DATA_TYPE_INT8, hcclComm, stream));

// 阻塞等待任务流中的集合通信任务执行完成
ACLCHECK(aclrtSynchronizeStream(stream));

// 解注册对称内存
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// 释放内存
ACLCHECK(aclrtFree(devPtr));

// 销毁任务流
ACLCHECK(aclrtDestroyStream(stream));

// 销毁通信域
HCCLCHECK(HcclCommDestroy(hcclComm));
```
<!-- end id9 -->

<!-- npu="A3" id10 -->
### Atlas A3 训练系列产品/Atlas A3 推理系列产品HCCS场景

```c
// 创建并初始化通信域配置项
HcclCommConfig config;
HcclCommConfigInit(&config);
// 按需修改通信域配置
config.hcclSymWinMaxMemSizePerRank = 10; //单位GB, 默认值为16。设置当前通信域预留的总虚拟对称内存大小 = rankSize * config.hcclSymWinMaxMemSizePerRank;

// 获取通信域参数
uint32_t rankSize = 4;
uint32_t rankId = 0;
int32_t deviceId;
ACLCHECK(aclrtGetDevice(&deviceId));
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));

// 初始化集合通信域
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, rankId, &config, &hcclComm));

// 创建任务流
aclrtStream stream;
ACLCHECK(aclrtCreateStream(&stream));

// 物理内存属性配置
aclrtPhysicalMemProp prop;
prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
prop.memAttr = ACL_HBM_MEM_HUGE;
prop.location.id = deviceId;
prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
prop.reserve = 0;

// 获取对齐粒度，通常为2MB
size_t granularity;
ACLCHECK(aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity));

// size按粒度对齐
size_t size = 2 * 1024 * 1024;
size_t allocSize = (size + granularity - 1) / granularity * granularity;

// 预留虚拟内存
void *virPtr;
ACLCHECK(aclrtReserveMemAddress(&virPtr, allocSize, 0, nullptr, 1));

// 申请物理内存
aclrtDrvMemHandle memHandle;
ACLCHECK(aclrtMallocPhysical(&memHandle, allocSize, &prop, 0));

// 建立物理到虚拟的映射
ACLCHECK(aclrtMapMem(virPtr, allocSize, 0, memHandle, 0));

size_t sendBytes = 1024;
size_t recvBytes = rankSize * sendBytes;
HcclCommSymWindow symWin;
// 注册对称内存
HCCLCHECK(HcclCommSymWinRegister(hcclComm, virPtr, sendBytes + recvBytes, &symWin, 1));

// 使用对称内存
void *sendBuff = virPtr;
void *recvBuff = static_cast<char*>(sendBuff) + sendBytes;

// 调用集合通信算子
HCCLCHECK(HcclAllGather(sendBuff, recvBuff, sendBytes, HCCL_DATA_TYPE_INT8, hcclComm, stream));

// 阻塞等待任务流中的集合通信任务执行完成
ACLCHECK(aclrtSynchronizeStream(stream));

// 解注册对称内存
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// 释放内存
ACLCHECK(aclrtUnmapMem(virPtr));
ACLCHECK(aclrtFreePhysical(memHandle));
ACLCHECK(aclrtReleaseMemAddress(virPtr));

// 销毁任务流
ACLCHECK(aclrtDestroyStream(stream));

// 销毁通信域
HCCLCHECK(HcclCommDestroy(hcclComm));
```
<!-- end id10 -->
