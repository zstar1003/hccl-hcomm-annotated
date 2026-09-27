# CommAbiHeader

## 功能说明

兼容Abi字段结构体。

## 定义原型

```c
typedef struct {
    uint32_t version;
    uint32_t magicWord;
    uint32_t size;
    uint32_t reserved;
} CommAbiHeader;
```
