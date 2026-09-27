# CommMem

## 功能说明

内存段元数据描述结构体。

## 定义原型

```c
typedef struct {
    CommMemType type; /* 内存物理位置类型 */
    void *addr;       /* 内存地址 */
    uint64_t size;    /* 内存区域字节数 */
} CommMem;
```
