# HcclKernelLaunchCfg

## 功能说明

用于描述AICPU运行核函数的配置信息，包含超时时间等配置参数。

## 定义原型

```c
typedef struct {
    CommAbiHeader header;
    uint64_t timeOut;
    uint8_t reserved[104];
} HcclKernelLaunchCfg;
```

## 参数说明

- **header**：ABI头部，包含版本等信息。类型定义请参见[CommAbiHeader](../../comm_opdev/datatype_definition/CommAbiHeader.md) 。
- **timeOut**：任务调度器等待任务执行的超时时间，单位为秒。
- **reserved**：预留字段，长度104字节，用于未来扩展。
