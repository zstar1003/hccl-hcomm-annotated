# EI0013 Execution_Error_ROCE_CQE

## 错误信息

报错格式如下，占位符%s的含义依次为本端Server ID、Device ID、Device IP以及对端Server ID、Device ID、Device IP：

```text
An error CQE occurred during operator execution. Local information: server %s, device ID %s, device IP %s. Peer information: server %s, device ID %s, device IP %s.
```

报错示例如下：

```text
An error CQE occurred during operator execution. Local information: server 10.78.106.107, device ID 0, device IP 192.168.200.100. Peer information: server 10.78.106.111, device ID 0, device IP 192.168.200.101.
```

## 可能原因

1. 两个Device之间存在网络问题，如网口异常闪断。
2. 对端进程提前异常退出，导致本端收不到对端的回复。

## 解决方法

1. 请检查两端之间的网络设备是否有异常。
2. 请检查对端进程是否先退出，如果有，则需排查该进程先退出的原因。
