# HcclGetErrorString

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：不支持
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

解析HcclResult类型的错误码。

## 函数原型

```c
const char *HcclGetErrorString(HcclResult code)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| code | 输入 | 需要解析的错误码，[HcclResult](./data_type_definition/HcclResult.md)类型。 |

## 返回值

`const char *` 指针，指向 [HcclResult](./data_type_definition/HcclResult.md) 类型错误码对应的字符串描述。

## 约束说明

无。

## 调用示例

```c
// 设备资源初始化
aclInit(NULL);
uint32_t devId = 0;
aclrtSetDevice(devId);

// 创建通信域
HcclComm hcclComm;
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);
HcclCommInitRootInfo(8, &rootInfo, 0, &hcclComm);

// 查询通信域异步错误并解析错误码
HcclResult asyncError = HCCL_SUCCESS;
HcclGetCommAsyncError(hcclComm, &asyncError);
if (asyncError != HCCL_SUCCESS) {
    const char *errStr = HcclGetErrorString(asyncError);
    printf("comm async error: %s\n", errStr);
}

// 销毁通信域
HcclCommDestroy(hcclComm);
aclFinalize();
```
