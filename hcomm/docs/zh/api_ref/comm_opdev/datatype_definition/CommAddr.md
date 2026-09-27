# CommAddr

## 功能说明

通信设备地址描述结构体。

## 定义原型

```c
static const uint32_t COMM_ADDR_EID_LEN = 16U;
typedef struct {
    CommAddrType type;         /* 通信地址类别 */
    union {
        uint8_t raws[36];      /* 通用数据 */
        struct in_addr addr;   /* IPv4地址结构 */
        struct in6_addr addr6; /* IPv6地址结构 */
        uint32_t id;           /* 标识 */
        uint8_t eid[COMM_ADDR_EID_LEN];  /* EID地址类型 */
    };
} CommAddr;
```
