# EI0018 Execution_Error_UB_CQE

## 错误信息

报错格式如下，占位符%s的含义依次为本端Server ID、本端Device ID、本端Device IP、对端Server ID、对端Device ID、对端Device IP：

```text
An error CQE occurred during operator execution. Local information: server %s, device ID %s, device IP %s. Peer information: server %s, device ID %s, device IP %s.
```

报错示例如下：

```text
An error CQE occurred during operator execution. Local information: server az0-rack0, device ID 1, device IP 0000:0000:0000:0000:0000:0000:df00:000a. Peer information: server az0-rack0, device ID 0, device IP 0000:0000:0000:0000:0000:0000:df00:001c.
```

## 可能原因

1. 两个设备之间的网络异常。例如，网口出现间歇性断开。

2. 对端进程提前异常退出，导致本端无法接收到对端的响应。

3. 两个设备中的任意一个设备的HBM或UB芯片处理模块发生了硬件异常。

## 解决方法

1. 检查两端之间的网络设备是否异常，通常原因为端口闪断导致的丢包，如果ping测试完全不通，则需要排查端口是否linkdown或者网络配置出了问题。

2. 检查对端进程是否先退出。如果是，检查进程退出的原因。

3. 通过RAS故障检查两台设备中任一设备的HBM或UB芯片处理模块是否发生硬件异常, 硬件错误请联系华为工程师处理。
