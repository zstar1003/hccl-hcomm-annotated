# HcclOpExpansionMode

## 功能说明

定义通信算子的展开模式。

## 定义原型

```c
typedef enum {
    HCCL_OP_EXPANSION_MODE_INVALID  = -1,  /* 无效模式，未初始化或保留。 */
    HCCL_OP_EXPANSION_MODE_AI_CPU   = 0,   /* 在Device侧AI CPU上展开。 */
    HCCL_OP_EXPANSION_MODE_AIV      = 1,   /* 在Device侧Vector Core (AIV) 上展开。 */
    HCCL_OP_EXPANSION_MODE_HOST     = 2,   /* 在Host侧CPU上展开，Device侧根据硬件型号自动选择调度器。 */
    HCCL_OP_EXPANSION_MODE_HOST_TS  = 3,   /* 在Host侧CPU上展开，由Host向Device Task Scheduler下发任务，Device侧进行调度执行。 */
    HCCL_OP_EXPANSION_MODE_CCU_MS   = 4,   /* 在Device侧CCU（集合通信加速单元）上展开，使用MS（Memory Slice）模式。 */
    HCCL_OP_EXPANSION_MODE_CCU_SCHED = 5,  /* 在Device侧CCU上展开，使用调度模式（CCU作为调度器向UB引擎调度任务）。 */
    HCCL_OP_EXPANSION_AIV_ONLY      = 6,   /* 仅在Device侧Vector Core (AIV) 上展开，不随数据量变化进行模式切换。 */
} HcclOpExpansionMode;

typedef HcclOpExpansionMode HcclConfigTypeOpExpansionMode;
```
