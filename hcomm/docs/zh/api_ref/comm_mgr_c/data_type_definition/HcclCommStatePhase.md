# HcclCommStatePhase

## 功能说明

通信域的不同阶段

## 定义原型

```c
typedef enum {
    HCCL_COMM_STATE_PHASE_INVALID = -1,
    HCCL_COMM_STATE_PHASE_DESTROY_PRE = 0,   /* 调用通信域销毁HcclCommDestroy前 */
    HCCL_COMM_STATE_PHASE_DESTROY_POST = 1,  /* 调用通信域销毁HcclCommDestroy后 */
    HCCL_COMM_STATE_PHASE_RESUME_PRE = 2,    /* 调用step快恢恢复通信域资源HcclCommResume前 */
    HCCL_COMM_STATE_PHASE_RESUME_POST = 3    /* 调用step快恢恢复通信域资源HcclCommResume后 */
} HcclCommStatePhase;
```
