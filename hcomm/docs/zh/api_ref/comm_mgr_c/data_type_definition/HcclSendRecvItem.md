# HcclSendRecvItem

## 功能说明

用于批量点对点通信操作，定义每一个通信任务的基本信息，包含任务类型、数据地址、数据个数、数据类型、对端rank编号等。

## 定义原型

```c
typedef struct HcclSendRecvItemDef {
    HcclSendRecvType sendRecvType;    /* 标识当前任务类型是发送还是接收 */
    void *buf;        /* 数据发送/接收的buffer地址 */
    uint64_t count;   /* 发送/接收数据的个数 */
    HcclDataType dataType;   /* 发送/接收数据的数据类型 */
    uint32_t remoteRank;     /* 通信域内数据接收/发送端的rank编号 */
} HcclSendRecvItem;
```
