# HcommMemReg

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

注册内存到指定EndPoint。

## 函数原型

```c
HcommResult HcommMemReg(EndpointHandle endpointHandle, const char *memTag, const CommMem *mem, HcommMemHandle *memHandle)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| endpointHandle | 输入 | Endpoint句柄。<br>EndpointHandle类型的定义请参见[EndpointHandle](../../datatype_definition/EndpointHandle.md)。 |
| memTag | 输入 | 内存字符串标识，以'\0'结尾，最大长度为256个字符（含'\0'结尾符，即有效字符最多255个），超长时接口返回HCCL_E_PARA。 |
| mem | 输入 | 内存描述信息，包含内存物理位置类型、内存地址、内存区域字节数。<br>CommMem类型的定义请参见[CommMem](../../datatype_definition/CommMem.md)。 |
| memHandle | 输出 | 注册内存句柄 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

- NIC插件类型的Endpoint默认不支持本接口：调用时日志中打印not supported告警。

- Endpoint对本接口的支持情况与其位置、protocol及芯片型号有关，具体如下。

   <!-- npu="950" id6 -->
   - 针对Ascend 950PR/Ascend 950DT：
     - 当Endpoint位于HOST侧时，支持通信协议为RoCE、UB_CTP的Endpoint。
     - 当Endpoint位于DEVICE侧时，支持通信协议为UB_CTP、UB_MEM、PCIe、UBoE、UB_RTP的Endpoint。
     - 当`mem->type`为`COMM_MEM_TYPE_CCU`时，表示注册CCU资源空间内存，仅Ascend 950PR/Ascend 950DT支持。CCU类型内存的注册流程与DEVICE一致。详见[CommMemType](../../datatype_definition/CommMemType.md)。
   <!-- end id6 -->

   <!-- npu="A3" id7 -->
   - 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品：仅支持Endpoint位于DEVICE侧，支持通信协议为RoCE、HCCS的Endpoint。
   <!-- end id7 -->

   <!-- npu="910b" id8 -->
   - 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品：仅支持Endpoint位于DEVICE侧，支持通信协议为RoCE、HCCS的Endpoint。
   <!-- end id8 -->

## 调用示例

```c
struct in_addr ipAddr;
inet_pton(AF_INET, "192.168.1.100", &ipAddr);
const EndpointDesc endpointDesc = {
    .protocol = COMM_PROTOCOL_ROCE,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = ipAddr
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
```
