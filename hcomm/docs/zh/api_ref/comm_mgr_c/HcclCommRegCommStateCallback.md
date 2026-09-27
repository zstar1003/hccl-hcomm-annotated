# HcclCommRegCommStateCallback

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
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

将HCCL定义的类型为 `HcclCommStateCallback` 的回调函数注册到HCOMM，在通信域的不同阶段进行回调函数调用。

## 函数原型

```c
HcclResult HcclCommRegCommStateCallback(const char *regName, HcclCommStateCallback cb, void *args)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| regName | 输入 | 注册的名称。<br>const char*类型，最大长度不超过160字节。 |
| cb | 输入 | 需要注册到HCOMM回调函数的类型，HcclCommStateCallback类型的定义可参见[HcclCommStateCallback](./data_type_definition/HcclCommStateCallback.md)。 |
| args | 输入 | 回调函数传入的用户上下文指针。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

注册的回调函数会在调用[HcclCommResume](./HcclCommResume.md)（恢复阶段）和 [HcclCommDestroy](./HcclCommDestroy.md)（销毁阶段）的前后被调用，具体阶段由[HcclCommStatePhase](./data_type_definition/HcclCommStatePhase.md)枚举值标识。

## 调用示例

```c
// 需要注册的回调函数，可根据实际需求实现
HcclResult myCallback(HcclComm comm, HcclCommStatePhase state, void *args)
{
    (void)comm;
    (void)state;
    (void)args;
    return HCCL_SUCCESS;
}
```

```c
uint32_t rankSize = 8;
uint32_t deviceId = 0;

const char *regName = "my_callback";
void *userPtr = "userContext";

// 生成root节点的rank标识信息
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));
// 初始化通信域
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm));

// 注册通信域恢复和销毁过程中需要进行回调的函数，regName为要注册回调函数的名字，userPtr为回调函数要传入的上下文参数指针
HcclCommRegCommStateCallback(regName, myCallback, userPtr);
// 假设通信域已通过HcclCommSuspend接口或者acl提供的aclrtDeviceTaskAbort接口被挂起，恢复通信域
HCCLCHECK(HcclCommResume(hcclComm));
// 销毁通信域
HCCLCHECK(HcclCommDestroy(hcclComm));
```
