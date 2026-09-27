# ThreadConfig

## 功能说明

线程配置结构体，用于逐线程配置同步资源数量。

## 定义原型

```c
typedef struct {
    CommAbiHeader header;              /* ABI头部，包含版本等信息 */
    uint16_t notifyNumPerThread;       /* 每个通信线程中的同步资源（Notify）数量 */
    uint8_t reserved[14];              /* 预留字段 */
} ThreadConfig;
```

## 成员说明

| 成员 | 描述 |
| --- | --- |
| header | ABI头部，包含版本等信息。CommAbiHeader类型的定义可参见[CommAbiHeader](CommAbiHeader.md)。 |
| notifyNumPerThread | 每个通信线程中的同步资源（Notify）数量，取值范围0~65535。受底层Notify资源池限制，一个通信域内所有线程的同步资源总量不超过65536个。 |
| reserved | 预留字段。 |
