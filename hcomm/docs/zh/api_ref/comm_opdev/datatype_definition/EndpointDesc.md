# EndpointDesc

## 功能说明

定义Endpoint描述类型结构体。

## 定义原型

```c
typedef struct {
    CommProtocol protocol;  /* 通信协议 */
    CommAddr commAddr;      /* 通信地址 */
    EndpointLoc loc;        /* Endpoint的位置信息 */
    union {
        uint8_t raws[52];   /* 通用数据 */
    };
} EndpointDesc;
```
