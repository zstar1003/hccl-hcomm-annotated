# EndpointLocType

## 功能说明

定义通信设备Endpoint的位置。

## 定义原型

```c
typedef enum {
    ENDPOINT_LOC_TYPE_RESERVED = -1,  /* 保留的Endpoint位置 */
    ENDPOINT_LOC_TYPE_DEVICE = 0,     /* Endpoint在Device上 */
    ENDPOINT_LOC_TYPE_HOST = 1,       /* Endpoint在Host上 */
} EndpointLocType;
```
