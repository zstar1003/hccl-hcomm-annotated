# HcommTeamSyncMemRequirement

## 功能说明

syncMem需求描述符，用于在创建team时声明所需的signal、counter、barrier同步内存数量。

## 定义原型

```c
typedef struct {
    uint32_t signalCount;
    uint32_t counterCount;
    uint32_t barrierCount;
    uint32_t reserved[5];
} HcommTeamSyncMemRequirement;
```

## 字段说明

| 字段名 | 描述 |
| --- | --- |
| signalCount | 目前暂不支持配置，只支持传0。 |
| counterCount | 目前暂不支持配置，只支持传0。 |
| barrierCount | 所需barrier数量，必须大于等于1。 |
| reserved[5] | 预留字段。 |

## 说明

该结构体内嵌于[HcclTeamCreateDesc](HcclTeamCreateDesc.md)的requirement字段，创建team时由HCOMM层据此计算并输出需要本地申请的syncMem字节数，计算公式为：`(signalCount + counterCount + barrierCount) * sizeof(uint64_t) * memberNum`。
