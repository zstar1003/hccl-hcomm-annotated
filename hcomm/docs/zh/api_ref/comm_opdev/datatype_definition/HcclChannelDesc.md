# HcclChannelDesc

## 功能说明

定义通信通道参数。

## 定义原型

```c
typedef struct {
    CommAbiHeader header;
    uint32_t remoteRank;              /* 远端rankId */
    CommProtocol channelProtocol;     /* 通信协议 */
    EndpointDesc localEndpoint;       /* 本地网络设备端侧描述，仅Ascend 950PR/Ascend 950DT支持 */
    EndpointDesc remoteEndpoint;      /* 远端网络设备端侧描述，仅Ascend 950PR/Ascend 950DT支持 */
    uint32_t notifyNum;               /* channel上使用的同步信号数量，范围为0~64，默认为0 */
    HcclMemHandle *memHandles;        /* 注册到通信域的待交换内存句柄，仅Ascend 950PR/Ascend 950DT的AIV引擎支持 */
    uint32_t memHandleNum;            /* 注册到通信域的待交换内存句柄数量，仅Ascend 950PR/Ascend 950DT的AIV引擎支持 */
    union {
        uint8_t raws[128];            /* 通用缓存 */
        struct {
            uint32_t queueNum;        /* QP数量，当前仅支持一个QP */
            uint32_t retryCnt;        /* 最大重传次数，范围为0~7，默认为7 */
            uint32_t retryInterval;   /* 重传间隔，范围为5~24，默认为20(对应时间4.096*2^20us) */
            uint8_t tc;               /* 流量类别(QoS)，范围为0~255，默认为132, 仅Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品支持 */
            uint8_t sl;               /* 服务等级(QoS)，范围为0~7，默认为4, 仅Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品支持 */
        } roceAttr;
        struct {
            uint8_t pathMode;         /* UB_MEM访问路径模式，取值范围：0、1、2和0xFF(默认值为0，配置为0xFF时按照0处理）。0：自动模式(优先单路径，若无则使用多路径），1：强制单路径模式，2：强制多路径模式 */
        } ubMemAttr;
    };
} HcclChannelDesc;
```
