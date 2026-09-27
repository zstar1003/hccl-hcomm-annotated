# EI0019 Communication_Error_Bind_IP_Port

## 错误信息

报错格式如下，占位符%s表示报错原因：

```text
Failed to enable listening for the host network adapter socket. Reason: %s
```

报错示例如下：

```text
Failed to enable listening for the host network adapter socket. Reason: The IP address 10.23.146.197 add port 50001 have already been bound.
```

## 解决方法

1. 请确认是否已经有其他进程占用此端口，若已被占用可以通过环境变量HCCL_IF_BASE_PORT进行调整，并通过`sysctl -w net.ipv4.ip_local_reserved_ports=****-****`调整预留端口范围。

2. 请确认是否可能存在本次业务中一个device被多次拉起业务进程的情况。
