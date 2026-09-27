# HcommBatchTransferDesc

## 功能说明

批量传输描述符结构体，用于描述单个传输任务的参数。通过transType指定传输类型，由transferInfo联合体提供对应类型的详细参数。

## 定义原型

```c
typedef struct {
    HcommTransferType transType;    /* 传输类型，参见HcommTransferType */
    uint8_t reserved[4];            /* 保留字段 */
    union {
        uint8_t raws[56];           /* 原始字节访问，用于零拷贝序列化等场景 */
        struct {
            uint64_t len;           /* 数据长度（字节） */
            void *dst;              /* 远端目的地址 */
            void *src;              /* 本地源地址 */
        } write;
        struct {
            uint64_t len;           /* 数据长度（字节） */
            void *dst;              /* 本地目的地址 */
            void *src;              /* 远端源地址 */
        } read;
        struct {
            uint64_t count;         /* 元素个数 */
            void *dst;              /* 远端目的地址 */
            void *src;              /* 本地源地址 */
            HcommReduceOp reduceOp; /* 归约操作类型 */
            HcommDataType dataType; /* 数据类型 */
        } reduce;
        struct {
            uint32_t notifyIdx;     /* 通知索引 */
        } notifyRecord;
        struct {
            uint64_t len;           /* 数据长度（字节） */
            void *dst;              /* 远端目的地址 */
            void *src;              /* 本地源地址 */
            uint32_t notifyIdx;     /* 写完成后通知的远端通知索引 */
        } writeWithNotify;
        struct {
            uint64_t count;         /* 元素个数 */
            void *dst;              /* 远端目的地址 */
            void *src;              /* 本地源地址 */
            HcommReduceOp reduceOp; /* 归约操作类型 */
            HcommDataType dataType; /* 数据类型 */
            uint32_t notifyIdx;     /* 写归约完成后通知的远端通知索引 */
        } writeReduceWithNotify;
    } transferInfo;
} HcommBatchTransferDesc;
```
