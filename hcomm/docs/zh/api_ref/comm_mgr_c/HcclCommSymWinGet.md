# HcclCommSymWinGet

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

根据已注册对称内存的地址指针，返回对应的窗口资源句柄及其在窗口内的偏移量。

<!-- npu="950" id6 -->
- 针对Ascend 950PR/Ascend 950DT，本接口支持URMA和UB Memory场景。查询范围未命中已注册的有效对称内存窗口时，接口返回HCCL_SUCCESS，同时将*winHandle设置为NULL、*offset设置为0。
<!-- end id6 -->
<!-- npu="A3" id7 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，本接口支持HCCS链路通信场景。
<!-- end id7 -->

## 函数原型

```c
HcclResult HcclCommSymWinGet(HcclComm comm, void *ptr, size_t size, HcclCommSymWindow *winHandle, size_t *offset)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | HCCL通信域。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |
| ptr | 输入 | 已注册对称内存的地址指针，该内存需要已使用[HcclCommSymWinRegister](HcclCommSymWinRegister.md)接口进行注册。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品的HCCS场景下，该地址为预留并完成物理内存映射的虚拟地址。<br>Ascend 950PR/Ascend 950DT的URMA场景下，该地址为已注册的Device内存地址；UB Memory场景与Atlas A3的HCCS场景保持一致，该地址为预留并完成物理内存映射的虚拟地址。 |
| size | 输入 | 对称内存窗口大小。<br>假设对称内存窗口大小为symSize，已注册对称内存的地址指针为addr，size需要满足以下条件：<br>  - size > 0<br>  - ptr+size <= addr + symSize |
| winHandle | 输出 | 指向“对称内存窗口资源句柄”的指针。 |
| offset | 输出 | 指向偏移量的指针。<br>假设已注册对称内存的地址指针为addr，则\*offset = ptr - addr。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

<!-- npu="950" id8 -->
- 针对Ascend 950PR/Ascend 950DT，支持URMA和UB Memory场景。查询范围命中窗口时，ptr和ptr+size必须完整位于同一个有效对称内存窗口内。未命中时返回HCCL_SUCCESS，同时将*winHandle设置为NULL、*offset设置为0。
<!-- end id8 -->
<!-- npu="A3" id9 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，仅支持HCCS链路通信场景。
<!-- end id9 -->
- 仅支持通信算子展开模式为AI CPU的场景。

## 调用示例

<!-- npu="950" id10 -->
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

size_t memSize = 2 * 1024 * 1024;

// 申请Device内存
void *devPtr = nullptr;
ACLCHECK(aclrtMalloc(&devPtr, memSize, ACL_MEM_MALLOC_HUGE_FIRST));

HcclCommSymWindow symWin;
// 注册对称内存
HCCLCHECK(HcclCommSymWinRegister(hcclComm, devPtr, memSize, &symWin, 1));

// 使用HcclCommSymWinGet获取对称内存资源
HcclCommSymWindow tempWin;
size_t offset = 0;
HCCLCHECK(HcclCommSymWinGet(hcclComm, devPtr, memSize, &tempWin, &offset));

// 解注册对称内存
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// 释放内存
ACLCHECK(aclrtFree(devPtr));

// 销毁通信域
HCCLCHECK(HcclCommDestroy(hcclComm));
```
<!-- end id10 -->

<!-- npu="A3" id11 -->
### Atlas A3 训练系列产品/Atlas A3 推理系列产品HCCS场景

```c
// 创建并初始化通信域配置项
HcclCommConfig config;
HcclCommConfigInit(&config);
// 按需修改通信域配置
config.hcclSymWinMaxMemSizePerRank = 10; // 单位GB, 默认值为16

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

HcclCommSymWindow symWin;
// 注册对称内存
HCCLCHECK(HcclCommSymWinRegister(hcclComm, virPtr, allocSize, &symWin, 1));

// 使用HcclCommSymWinGet获取对称内存资源
HcclCommSymWindow tempWin;
size_t offset = 0;
HCCLCHECK(HcclCommSymWinGet(hcclComm, virPtr, allocSize, &tempWin, &offset));

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

<!-- end id11 -->
