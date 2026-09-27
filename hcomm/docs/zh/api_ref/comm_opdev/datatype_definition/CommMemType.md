# CommMemType

## 功能说明

内存物理位置类型。

## 定义原型

```c
typedef enum {
    COMM_MEM_TYPE_INVALID = -1,   /* 无效的内存类别 */
    COMM_MEM_TYPE_DEVICE = 0,     /* Device侧内存（如NPU等） */
    COMM_MEM_TYPE_HOST = 1,       /* Host侧内存 */
    COMM_MEM_TYPE_CCU = 2,        /* CCU资源空间 */
} CommMemType;
```

## 约束说明

- `COMM_MEM_TYPE_CCU` 表示CCU资源空间内存。该类型仅在支持CCU的产品（Ascend 950PR/Ascend 950DT）上有效。
- CCU类型内存的注册流程与DEVICE一致，均通过标准内存注册路径完成。
