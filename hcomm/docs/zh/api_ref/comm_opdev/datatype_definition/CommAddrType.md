# CommAddrType

## 功能说明

通信设备地址类别。

## 定义原型

```c
typedef enum {
    COMM_ADDR_TYPE_RESERVED = -1, /* 保留地址类型 */
    COMM_ADDR_TYPE_IP_V4 = 0,     /* IPv4地址类型 */
    COMM_ADDR_TYPE_IP_V6 = 1,     /* IPv6地址类型 */
    COMM_ADDR_TYPE_ID = 2,        /* ID地址类型 */
    COMM_ADDR_TYPE_EID = 3,       /* EID地址类型 */
} CommAddrType;
```
