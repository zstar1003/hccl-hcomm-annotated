# EI0006 Communication_Error_Get_Socket

## 错误信息

报错格式如下，占位符%s表示报错原因：

```text
Getting socket times out. Reason: %s
```

报错示例如下：

```text
Getting socket times out. Reason: 
This node (server 192.168.200.100, device ID 1) detects that srcRank (server 192.168.200.100, device ID 1) fails to connect to dstRank (server 192.168.200.100, device ID 0). Continue to analyze the fault based on the logs of srcRank and dstRank.1. If the link setup timeout is reported on both ends, check the network connectivity between the two ends.2. If dstRank reports other exceptions, locate the cause based on the exception information of dstRank.3. If dstRank does not report any error, the possible cause is that the service process is suspended or exits in advance.
```

## 解决方法

需按照Reason中的提示定位问题。
