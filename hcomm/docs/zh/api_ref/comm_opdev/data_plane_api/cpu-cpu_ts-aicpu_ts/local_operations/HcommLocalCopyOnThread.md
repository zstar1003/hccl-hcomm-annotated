# HcommLocalCopyOnThread

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

提供本地内存拷贝功能，将src指向的长度为len的内存数据，拷贝到dst所指向的相同长度的内存中。

## 函数原型

```c
int32_t HcommLocalCopyOnThread(ThreadHandle thread, void *dst, const void *src, uint64_t len)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| thread | 输入 | 通信线程句柄，为通过[HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md)接口获取到的threads。<br>ThreadHandle的类型的定义请参见[ThreadHandle](../../../datatype_definition/ThreadHandle.md)。 |
| dst | 输出 | 目的内存地址，Device内存。 |
| src | 输入 | 源内存地址，Device内存。 |
| len | 输入 | 数据长度（字节）。 |

## 返回值

int32_t：接口成功返回0，其他失败。

## 约束说明

dst、src内存是申请的device内存。

<!-- npu="950" id6 -->
在Ascend 950PR/Ascend 950DT上，仅支持AICPU_TS模式下、在Device侧调用该接口。
<!-- end id6 -->

## 调用示例

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU_TS;
aclrtStream stream;
aclrtCreateStream(&stream);
ThreadHandle thread;
HcclResult result = HcclThreadAcquireWithStream(comm, engine, stream, 2, &thread);

// 申请device内存
uint64_t memSize = 256;
s32 policy = static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH) | static_cast<int>(ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMallocAttrValue moduleIdValue;
moduleIdValue.moduleId = HCCL;
aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};

void* inputMem;
void* outputMem;
aclrtMallocWithCfg(&inputMem, memSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
aclrtMallocWithCfg(&outputMem, memSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
// 执行D2D拷贝
HcommLocalCopyOnThread(thread, outputMem, inputMem, memSize);
```

在Ascend 950PR/Ascend 950DT上，该函数需要编译到Device侧使用：

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;

void *cclBufferAddr = nullptr;
uint64_t cclBufferSize = 0;
HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize);
ThreadHandle thread;
HcclThreadAcquire(comm, engine, 1, 1, &thread);

// 其余资源申请
// 拷贝参数并Launch Kernel

// Device侧算法编排
uint64_t len = 256;
void *src = param.userIn;
void *dst = param.cclBuf;
// 执行D2D拷贝
HcommLocalCopyOnThread(thread, dst, src, len);
```
