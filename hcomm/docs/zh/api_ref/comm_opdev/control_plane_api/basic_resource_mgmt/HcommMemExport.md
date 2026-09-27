# HcommMemExport

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

内存注册后，导出指定内存描述，用于交换。

## 函数原型

```c
HcommResult HcommMemExport(EndpointHandle endpointHandle, HcommMemHandle memHandle, void** memDesc, uint32_t* memDescLen)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| endpointHandle | 输入 | Endpoint句柄。<br>EndpointHandle类型的定义请参见[EndpointHandle](../../datatype_definition/EndpointHandle.md)。 |
| memHandle | 输入 | 注册内存句柄。 |
| memDesc | 输出 | 返回描述信息指针。 |
| memDescLen | 输出 | 返回描述信息长度。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

- NIC插件类型的Endpoint默认不支持本接口：调用时日志中打印not supported告警。

- Endpoint对本接口的支持情况与其位置、protocol及芯片型号有关，具体如下。

   <!-- npu="950" id6 -->
   - 针对Ascend 950PR/Ascend 950DT：
     - 当Endpoint位于HOST侧时，支持通信协议为RoCE、UB_CTP的Endpoint。
     - 当Endpoint位于DEVICE侧时，支持通信协议为UB_CTP、UBoE、UB_RTP的Endpoint。
     - 通信协议为UB_MEM、PCIe的Endpoint（仅DEVICE侧可创建）不支持本接口：调用时不执行实际操作，日志中打印not supported信息。
   <!-- end id6 -->

   <!-- npu="A3" id7 -->
   - 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品：仅支持Endpoint位于DEVICE侧，支持通信协议为RoCE、HCCS的Endpoint。
   <!-- end id7 -->

   <!-- npu="910b" id8 -->
   - 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品：仅支持Endpoint位于DEVICE侧，支持通信协议为RoCE、HCCS的Endpoint。
   <!-- end id8 -->

## 调用示例

```c
const EndpointDesc endpointDesc = {
    .protocol = COMM_PROTOCOL_ROCE,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = {{192, 168, 1, 100}}
    },
    .loc = {
        .locType = ENDPOINT_LOC_TYPE_DEVICE,
        .device = {
            .devPhyId = 0,
            .superDevId = 0,
            .serverIdx = 0,
            .superPodIdx = 0
        }
    },
    .raws = {0}
};
EndpointHandle endpointHandle = nullptr;
HcommResult result = HcommEndpointCreate(&endpointDesc, &endpointHandle);
const char *memTag = "HcclBuffer";
CommMem mem = {
    .type = COMM_MEM_TYPE_DEVICE,
    .addr = reinterpret_cast<void*>(0x1111),
    .size = 100
};
HcommMemHandle memHandle;
result = HcommMemReg(endpointHandle, memTag, &mem, &memHandle);

uint32_t memDescLen = 0;
void* memDesc = nullptr;
result = HcommMemExport(endpointHandle, memHandle, &memDesc, &memDescLen);
```
