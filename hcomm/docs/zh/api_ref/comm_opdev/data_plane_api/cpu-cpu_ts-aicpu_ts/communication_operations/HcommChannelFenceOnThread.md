# HcommChannelFenceOnThread

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
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

在指定通信线程和通信通道上插入内存屏障操作，确保屏障前的通道读写操作在屏障后的通道读写操作之前完成。

## 函数原型

```c
int32_t HcommChannelFenceOnThread(ThreadHandle thread, ChannelHandle channel)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| thread | 输入 | 通信线程句柄。AICPU侧调用时，为通过[HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md)接口获取到的threads；Host CPU侧调用时，该参数无作用，传入0即可。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../../datatype_definition/ThreadHandle.md)。 |
| channel | 输入 | 通信通道句柄，为通过[HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md)或[HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)接口获取到的channels。关于channel的约束参见约束说明。<br>ChannelHandle类型的定义可参见[ChannelHandle](../../../datatype_definition/ChannelHandle.md)。 |

## 返回值

int32_t：接口成功返回0，其他失败。

## 约束说明

<!-- npu="950" id6 -->
针对Ascend 950PR/Ascend 950DT，支持AICPU侧和Host CPU侧调用。

- AICPU侧调用时，通信引擎为AICPU_TS，支持通信协议UB_CTP、UBoE。
- Host CPU侧调用时，调用[HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md)或[HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)申请入参channel时，需传入`engine = COMM_ENGINE_CPU`，且`channelDesc.remoteEndpoint.protocol`需为`COMM_PROTOCOL_ROCE`或`COMM_PROTOCOL_UB_CTP`。
- Host CPU侧调用时，`thread`参数无作用，可传入0。
<!-- end id6 -->

## 调用示例

### 集合通信示例

```c
// 申请通信线程资源
CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS; // Ascend 950PR/Ascend 950DT使用
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// 申请通信通道资源
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// 获取本端通信内存信息
void * localBuffer;
uint64_t localBufferSize;
HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize);

// 获取对端通信内存信息
void * remoteBuffer;
uint64_t remoteBufferSize;
HcclChannelGetHcclBuffer(comm, channel, &remoteBuffer, &remoteBufferSize);
uint64_t len = std::min(localBufferSize, remoteBufferSize);

// 将本端内存的内容写到对端内存上
HcommWriteOnThread(thread, channel, remoteBuffer, localBuffer, len);
HcommChannelFenceOnThread(thread, channel);
HcommReadOnThread(thread, channel, localBuffer, remoteBuffer, len);
```

### 基础通信示例

以Host RoCE H2D场景为例。
Client端为写发起端，提供Host源内存；Server端为写入目标端，提供Device目的内存。

#### Server端

```c
// 申请Server端Endpoint资源
const EndpointDesc serverEndpointDesc = {
    .protocol = COMM_PROTOCOL_ROCE,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = {{192, 168, 1, 101}}
    },
    .loc = {
        .locType = ENDPOINT_LOC_TYPE_HOST
    },
    .raws = {0}
};
EndpointHandle serverEndpointHandle = nullptr;
HcommResult result = HcommEndpointCreate(&serverEndpointDesc, &serverEndpointHandle);

// 注册Server端内存
CommMem serverDeviceMem = {
    .type = COMM_MEM_TYPE_DEVICE,
    .addr = serverDeviceAddr,
    .size = dataLen
};
HcommMemHandle serverMemHandle = nullptr;
result = HcommMemReg(serverEndpointHandle, "server_device_mem", &serverDeviceMem, &serverMemHandle);

// 导出Server端内存描述
void *serverMemDesc = nullptr;
uint32_t serverMemDescLen = 0;
result = HcommMemExport(serverEndpointHandle, serverMemHandle, &serverMemDesc, &serverMemDescLen);

// 省略：
// Server端将serverEndpointDesc、serverMemDesc、serverMemDescLen发送给Client端
// Server端获取clientEndpointDesc

// 申请Server端通信通道资源
HcommChannelDesc serverChannelDesc = {};
result = HcommChannelDescInit(&serverChannelDesc, 1);
serverChannelDesc.remoteEndpoint = clientEndpointDesc;
serverChannelDesc.notifyNum = 1;
serverChannelDesc.exchangeAllMems = true;
serverChannelDesc.role = HCOMM_SOCKET_ROLE_SERVER;
serverChannelDesc.roceAttr.queueNum = 1;
ChannelHandle serverChannel = 0;
result = HcommChannelCreate(serverEndpointHandle, COMM_ENGINE_CPU, &serverChannelDesc, 1, &serverChannel);
```

#### Client端

```c
// 申请Client端Endpoint资源
const EndpointDesc clientEndpointDesc = {
    .protocol = COMM_PROTOCOL_ROCE,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = {{192, 168, 1, 100}}
    },
    .loc = {
        .locType = ENDPOINT_LOC_TYPE_HOST
    },
    .raws = {0}
};
EndpointHandle clientEndpointHandle = nullptr;
HcommResult result = HcommEndpointCreate(&clientEndpointDesc, &clientEndpointHandle);

// 注册Client端内存
CommMem clientHostMem = {
    .type = COMM_MEM_TYPE_HOST,
    .addr = clientHostAddr,
    .size = dataLen
};
HcommMemHandle clientMemHandle = nullptr;
result = HcommMemReg(clientEndpointHandle, "client_host_mem", &clientHostMem, &clientMemHandle);

// 省略：
// Client端将clientEndpointDesc发送给Server端
// Client端获取serverEndpointDesc、serverMemDesc、serverMemDescLen

// 导入Server端内存描述
CommMem importedServerDeviceMem = {};
result = HcommMemImport(clientEndpointHandle, serverMemDesc, serverMemDescLen, &importedServerDeviceMem);

// 申请Client端通信通道资源
HcommChannelDesc clientChannelDesc = {};
result = HcommChannelDescInit(&clientChannelDesc, 1);
clientChannelDesc.remoteEndpoint = serverEndpointDesc;
clientChannelDesc.notifyNum = 1;
clientChannelDesc.exchangeAllMems = true;
clientChannelDesc.role = HCOMM_SOCKET_ROLE_CLIENT;
clientChannelDesc.roceAttr.queueNum = 1;
ChannelHandle clientChannel = 0;
result = HcommChannelCreate(clientEndpointHandle, COMM_ENGINE_CPU, &clientChannelDesc, 1, &clientChannel);

// 将Client端内存写入Server端内存
int32_t ret = HcommWriteNbiOnThread(0, clientChannel, importedServerDeviceMem.addr, clientHostMem.addr, dataLen);

// 等待通信通道上已提交的写操作完成
ret = HcommChannelFenceOnThread(0, clientChannel);
```
