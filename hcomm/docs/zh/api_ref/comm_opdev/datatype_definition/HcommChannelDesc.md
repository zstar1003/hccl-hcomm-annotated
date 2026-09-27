# HcommChannelDesc

## 功能说明

定义组件间通道参数。

## 定义原型

```c
typedef struct {
    CommAbiHeader header;             /* ABI头部，包含版本等信息 */
    EndpointDesc remoteEndpoint;      /* 远端网络设备端侧描述 */
    uint32_t notifyNum;               /* channel上使用的同步信号数量 */

    // exchangeAllMems = True时不需要配置memHandle
    bool exchangeAllMems;             /* 表示是否交换本地网络设备端注册的内存信息 */
    HcommMemHandle *memHandles;       /* 注册到通信域的待交换内存句柄，exchangeAllMems为True时无效 */
    uint32_t memHandleNum;            /* 注册到通信域的待交换内存句柄数量，exchangeAllMems为True时无效 */
    HcommSocket socket;               /* Socket句柄 */
    HcommSocketRole role;             /* 本端角色(SERVER或CLIENT) */
    uint16_t port;                    /* Socket监听指定端口*/
    union {
        uint8_t raws[128];            /* 通用缓存 */
        struct {
            uint32_t queueNum;        /* QP数量，取值范围：[1,32]，建议配置范围：[1,8] */
            uint32_t retryCnt;        /* 最大重传次数，范围为0~7，默认为7 */
            uint32_t retryInterval;   /* 重传间隔，范围为5~24，默认为20 (对应时间4.096*2^20us) */
            uint8_t tc;               /* 流量类别(QoS)，范围为0~255，默认为132 */
            uint8_t sl;               /* 服务等级(QoS)，范围为0~7，默认为4 */
            uint32_t qpThreshold;     /* 多QP场景下，每个QP最小数据量(B) */
            uint32_t cqAttrFlags;     /* CQ属性标志位，用于配置ibv_cq_init_attr_ex的flags标志位，默认0。
                                         备注：NPU网卡不支持该配置；第三方网卡场景下是否有效，与各自网卡能力相关 */
            uint16_t* srcPortList;    /* QP源端口号列表，用于RoCE网络哈希分流。NULL表示不配置。
                                         ABI v4新增字段，低版本调用方未设置时HCOMM置NULL
                                         数组长度需等于queueNum，第i个QP使用srcPortList[i % 数组长度]作为UDP源端口号。
                                         当通过HcclChannelAcquire创建通道时，HCOMM根据环境变量
                                         HCCL_RDMA_QP_PORT_CONFIG_PATH指向的MultiQpSrcPort.cfg配置文件自动填充此字段。
                                         exchangeAllMems为true时（即单边通信场景）此字段不生效（udpSport置0）。 */
        } roceAttr;
        struct {
            uint32_t qos;             /* HCCS QoS */
        } hccsAttr;
        struct {
            uint32_t sqDepth;         /* UB队列深度，0和0xffffffff表示使用默认值 */
        } ubAttr;
        struct {
            uint8_t pathMode;         /* UB_MEM访问路径模式，取值范围：0、1、2和0xFF(默认值为0，配置为0xFF时按照0处理）。0：自动模式(优先单路径，若无则使用多路径），1：强制单路径模式，2：强制多路径模式 */
        } ubMemAttr;
    };
    uint32_t qos;             /* 通道 QoS（与协议解耦；集合通信场景通常来自通信域 hcclQos）；取值范围：[0, 7]，或 0xFFFFFFFF 表示未配置。
                                 未配置时：RoCE 不改写 roceAttr.sl/tc；UB 类协议由内部默认 QoS（4）决定。 */
    const char *channelName;  /* channel业务匹配标识，两端需相同；NULL表示匿名channel */
} HcommChannelDesc;
```
