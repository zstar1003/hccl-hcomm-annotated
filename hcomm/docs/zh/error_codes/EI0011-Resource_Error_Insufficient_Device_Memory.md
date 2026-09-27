# EI0011 Resource_Error_Insufficient_Device_Memory

## 错误信息

报错格式如下，占位符%s表示内存大小：

```text
Failed to allocate %s bytes of NPU memory.
```

报错示例如下：

```text
Failed to allocate 262144~3145728 bytes of NPU memory.
```

## 可能原因

NPU内存不足。

## 解决方法

需停止不必要的进程，并确保有所需的内存可用。
