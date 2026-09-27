# HcommSocketRole

## 功能说明

定义套接字角色类型。

## 定义原型

```c
typedef enum {
    HCOMM_SOCKET_ROLE_RESERVED = -1, /* 保留的套接字角色 */
    HCOMM_SOCKET_ROLE_CLIENT = 0,    /* 客户端角色，用于发起连接 */
    HCOMM_SOCKET_ROLE_SERVER = 1,    /* 服务器角色，用于监听连接 */
} HcommSocketRole;
```
