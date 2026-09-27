# EI0010 Communication_Error_P2P

## 错误信息

报错格式如下，占位符%s表示报错原因：

```text
P2P communication failed. Reason: %s
```

报错示例如下：

```text
P2P communication failed. Reason: Device ID 0 in module 0 and device ID 9 in module 1 are not on the same plane.
```

## 解决方法

确保NPU卡正常，并设置环境变量 'export HCCL_INTRA_ROCE_ENABLE=1'。
