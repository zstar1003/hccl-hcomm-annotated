# EI0020 Communication_Error_Bind_IP_Port

## 错误信息

报错格式如下，占位符%s表示报错原因：

```text
Failed to enable listening for the NPU network adapter socket. Reason: %s
```

报错示例如下：

```text
Failed to enable listening for the NPU network adapter socket. Reason: The IP address 192.1.3.198 add port 16666 have already been bound.
```

## 解决方法

请确认是否为单卡多进程场景，如果是的话请通过环境变量HCCL_NPU_SOCKET_PORT_RANGE配置对应的端口号。
