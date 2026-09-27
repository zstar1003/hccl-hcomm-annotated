# CommTopo

## 功能说明

定义通信拓扑类型。

## 定义原型

```c
typedef enum {
    COMM_TOPO_RESERVED = -1,  /* 保留拓扑 */
    COMM_TOPO_CLOS = 0,       /* CLOS互联拓扑 */
    COMM_TOPO_1DMESH = 1,     /* 1DMesh互联拓扑 */
    COMM_TOPO_910_93 = 2,     /* Atlas A3 训练系列产品/Atlas A3 推理系列产品的互联拓扑(带SIO) */
    COMM_TOPO_310P = 3,       /* Atlas 推理系列产品的互联拓扑 */
    COMM_TOPO_A2AXSERVER = 4, /* A2_AX_SERVER */
    COMM_TOPO_CUSTOM = 5      /* 自定义 */
} CommTopo;
```
