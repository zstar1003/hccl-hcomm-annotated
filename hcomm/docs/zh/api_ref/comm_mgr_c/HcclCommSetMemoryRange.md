# HcclCommSetMemoryRange

> [!NOTE]说明
> 本接口为试用接口，后续可能存在变更，暂不支持应用于生产环境。

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

用户通过aclrtReserveMemAddress接口成功申请虚拟内存后，可调用此接口通知HCCL预留的虚拟内存地址。调用此接口后，该虚拟内存对当前进程中的所有通信域可见。

## 函数原型

```c
HcclResult HcclCommSetMemoryRange(HcclComm comm, void *baseVirPtr, size_t size, size_t alignment, uint64_t flags)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | HCCL通信域，建议使用Server内最大的通信域，即覆盖最大卡数的通信域。 |
| baseVirPtr | 输入 | 需要预留的虚拟内存基地址，即aclrtReserveMemAddress接口输出的虚拟内存地址。 |
| size | 输入 | 虚拟内存的大小，单位：Byte。 |
| alignment | 输入 | 预留字段。<br>当前仅支持配置为“0”。 |
| flags | 输入 | 预留字段。<br>当前仅支持配置为“0”。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 该接口在通信域内首次被调用时会进行建链操作，因此用户首次调用该接口时需确保通信域内所有进程均调用该接口，且调用时刻相同，避免建链超时。后续再调用该接口时，无此约束。
- 该接口仅支持在范围是单Server的通信域内调用，否则会报错。
- 多次调用该接口时，输入的内存地址不能重复或存在区间交叠。
- 其他约束请参见[通用约束](./zero_copy_readme.md)。

## 调用示例

```c
// 设备资源初始化
aclInit(NULL);
aclrtSetDevice(devId);

// 创建通信域
HcclComm hcclComm;
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);
HcclCommInitRootInfo(8, &rootInfo, 0, &hcclComm);

// 通过aclrtReserveMemAddress申请虚拟内存
void *baseVirPtr = NULL;
size_t size = 1024 * 1024 * 1024; // 1GB
aclrtReserveMemAddress(&baseVirPtr, size, 0, NULL, 0);

// 通知HCCL预留的虚拟内存地址
HcclCommSetMemoryRange(hcclComm, baseVirPtr, size, 0, 0);

// 后续可调用HcclCommActivateCommMemory激活内存并使用零拷贝功能
// ...

// 销毁通信域
HcclCommDestroy(hcclComm);
aclFinalize();
```
