# EI0007 Resource_Error

## 错误信息

报错格式如下，占位符%s的含义依次为内存所在位置、内存大小：

```text
Failed to allocate resource %s with info %s. Reason: Resources are exhausted.
```

报错示例如下：

```text
Failed to allocate resource HostMemory with info size:8928 bytes. Reason: Resources are exhausted.
```

## 解决方法

资源不足，请根据报错提示调整代码。
