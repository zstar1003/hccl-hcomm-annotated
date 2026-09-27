# HcclCommDestroy

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

销毁指定的HCCL通信域。

## 函数原型

```c
HcclResult HcclCommDestroy(HcclComm comm)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 指向需要销毁的通信域的指针。<br>HcclComm类型的定义可参见[HcclComm](./data_type_definition/HcclComm.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 此接口支持跨线程调用：
  - 当通信域状态处于建链卡住或者未被占用状态时，支持跨线程调用此接口销毁通信域，并返回HCCL_SUCCESS。

    通信域销毁成功后，正在执行的通信算子无需等待超时时间会直接报错退出，并打印ERROR级别日志，日志关键字为“Terminating operation due to external request”。

  - 当通信域处于非建链卡住状态，或者其他被占用状态（例如通信域建链过程中、通信算子执行过程中等），跨线程调用此接口时会返回HCCL_E_AGAIN错误，并打印WARNING级别日志，日志关键字为“\[HcclCommDestroy\] comm is in use, please try again later”。

- 在UB Memory场景，建议先调用[HcclCommSymWinDeregister](HcclCommSymWinDeregister.md)解注册对称内存窗口，再调用本接口销毁通信域。
  如果仍有未解注册的UB Memory窗口，本接口会在销毁通信域时兜底释放其本地资源。A3和URMA场景同样不要求先解注册对称内存窗口：销毁通信域时，内部资源管理会遍历所有已注册的对称内存窗口，自动执行解注册并释放对应资源（对称虚拟地址、设备侧窗口副本等），无需用户调用解注册接口，销毁也不会因存在未解注册窗口而失败。

- 多线程场景下，需要确保HCCL接口的调用时序，调用此接口销毁通信域后不再支持调用其他集合通信相关接口。

## 调用示例

```c
// 设备资源初始化
aclInit(NULL);
uint32_t rankSize = 2;
int32_t devices[rankSize] = {0, 1};
HcclComm comms[rankSize];
// 指定集合通信操作使用的设备
for (uint32_t i = 0; i < rankSize; i++) {
    aclrtSetDevice(devices[i]);
}
// 初始化通信域
HcclCommInitAll(rankSize, devices, comms);
// 销毁通信域
for (uint32_t i = 0; i < rankSize; i++) {
    HcclCommDestroy(comms[i]);
}
// 设备资源去初始化
aclFinalize();
```
