# HcclCommConfigCapability

## 功能说明

定义通信域初始化时支持的相关配置信息。

## 定义原型

```c
typedef enum {
    HCCL_COMM_CONFIG_BUFFER_SIZE = 0,       /* 共享数据的缓存区大小 */
    HCCL_COMM_CONFIG_DETERMINISTIC = 1,    /* 确定性计算开关 */
    HCCL_COMM_CONFIG_COMM_NAME = 2,        /* 通信域名称 */
    HCCL_COMM_CONFIG_OP_EXPANSION_MODE = 3,     /* 通信算子展开模式 */
    HCCL_COMM_CONFIG_SUPPORT_INIT_BY_ENV = 4,  /* 是否支持以环境变量配置为初始值 */
    HCCL_COMM_CONFIG_WORLD_RANKID = 5,  /* NSLB-DP场景下指定当前进程在AI框架中的全局rank ID，Ascend 950PR/Ascend 950DT不涉及此场景 */
    HCCL_COMM_CONFIG_JOBID = 6,  /* NSLB-DP场景下指定当前分布式业务的唯一标识，由AI框架生成，Ascend 950PR/Ascend 950DT不涉及此场景 */
    HCCL_COMM_CONFIG_ACLGRAPH_ZEROCOPY_ENABLE = 7,  /* 图捕获模式（aclgraph）下用于控制其是否开启零拷贝功能，仅对Reduce类算子生效 */
    HCCL_COMM_CONFIG_EXEC_TIMEOUT = 8,  /* 通信执行超时时间 */ 
    HCCL_COMM_CONFIG_ALGO = 9,  /* 通信算法配置 */
    HCCL_COMM_CONFIG_RETRY = 10,  /* 通信重执行配置，Ascend 950PR/Ascend 950DT暂不支持 */
    HCCL_COMM_CONFIG_BUFFER_NAME = 11,  /* 共享CCLBuffer名称，Ascend 950PR/Ascend 950DT暂不支持 */
    HCCL_COMM_CONFIG_RESERVED              /* 预留字段 */
} HcclCommConfigCapability;
```
