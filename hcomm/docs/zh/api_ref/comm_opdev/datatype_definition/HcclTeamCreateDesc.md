# HcclTeamCreateDesc

## 功能说明

team创建描述符，用于[HcclTeamCreate](../control_plane_api/comms_domain_resource_mgmt/HcclTeamCreate.md)接口描述成员、网络层、通信协议、通信引擎、syncMem需求及共享队列等信息。

## 定义原型

```c
typedef struct {
    CommAbiHeader header;
    const uint32_t* rankIds;
    uint32_t rankNum;
    uint32_t selfRankId;
    uint32_t netLayer;
    CommProtocol protocol;
    HcommTeamSyncMemRequirement requirement;
    CommEngine engine;
    uint32_t notifyNum;
    uint32_t channelCnt;
    uint32_t isSharedQueue;
    char sharedQueueTag[HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN];
    uint32_t reserved[8];
} HcclTeamCreateDesc;
```

## 字段说明

| 字段名 | 描述 |
| --- | --- |
| header | ABI头部，由[HcclTeamCreateDescInit](../control_plane_api/comms_domain_resource_mgmt/HcclTeamCreateDescInit.md)初始化。CommAbiHeader类型的定义可参见[CommAbiHeader](CommAbiHeader.md)。 |
| rankIds | rankId数组，长度为rankNum。不可为NULL。 |
| rankNum | 成员数量。不可为0或1，且不可大于通信域的rankSize。 |
| selfRankId | 本rank的实际rankId，必须存在于rankIds中。 |
| netLayer | 希望使用的网络层，只能为0、1或2，0表示默认选择。 |
| protocol | 希望使用的通信协议，不可为COMM_PROTOCOL_RESERVED，1个team仅支持一个协议。CommProtocol类型的定义可参见[CommProtocol](CommProtocol.md)。 |
| requirement | syncMem需求，包含signal/counter/barrier数量。HcommTeamSyncMemRequirement类型的定义可参见[HcommTeamSyncMemRequirement](HcommTeamSyncMemRequirement.md)。 |
| engine | 通信引擎类型，例如COMM_ENGINE_AIV。CommEngine类型的定义可参见[CommEngine](CommEngine.md)。 |
| notifyNum | 每个channel所需的notify数量，取值范围为\[0,64\]。 |
| channelCnt | 每个对端成员的channel个数，不可为0。 |
| isSharedQueue | 是否使用共享队列，0表示不使用，非0表示使用。 |
| sharedQueueTag | 共享队列标签，以'\\0'结尾的字符串，最大长度为HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN（255）。仅当isSharedQueue非0时有效。 |
| reserved[8] | 预留字段。 |
