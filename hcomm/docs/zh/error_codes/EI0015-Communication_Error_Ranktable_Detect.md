# EI0015 Communication_Error_Ranktable_Detect

## 错误信息

报错格式如下，占位符%s表示报错原因：

```text
Failed to collect cluster information of the communicator based on rootInfo detection. Reason: %s.
```

报错示例如下：

```text
Failed to collect cluster information of the communicator based on rootInfo detection. Reason: No rank in the communicator can connect to the root node within the timeout period. List of unconnected ranks: "[1,]".
```

## 解决方法

1. 检查通信域内是否所有rank都下发了通信域创建接口。
2. 检查所有节点与server节点的host侧网络的连通性。
3. 检查所有节点的HCCL_SOCKET_IFNAME环境变量配置是否正确。
4. 通过配置HCCL_CONNECT_TIMEOUT环境变量增加超时等待时间。
