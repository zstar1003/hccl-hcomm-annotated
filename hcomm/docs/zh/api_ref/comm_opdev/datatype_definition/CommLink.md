# CommLink

## 功能说明

通信连接信息，包含用于创建通信Channel的协议、地址等信息。

## 定义原型

```c
typedef struct {
    CommAbiHeader header; /* 兼容Abi字段 */
    EndpointDesc srcEndpointDesc; /* 源Endpoint描述类型 */
    EndpointDesc dstEndpointDesc; /* 目的Endpoint描述类型 */
    union {
        uint8_t raws[128];
        struct {
            CommProtocol linkProtocol; /* 通信协议类型 */
            uint8_t hop; /* 链路跳数 */
        };
    } linkAttr;
} CommLink;
```
