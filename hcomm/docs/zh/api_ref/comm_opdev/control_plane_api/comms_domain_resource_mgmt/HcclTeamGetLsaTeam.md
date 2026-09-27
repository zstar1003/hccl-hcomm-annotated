# HcclTeamGetLsaTeam

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

获取通信域初始化时预制的UB Memory LSA WorldTeam句柄。该句柄可用于LSA Team成员编号与通信域Rank ID之间的转换，并配合[HcclSymWinGetPeerPointer](../../../comm_mgr_c/HcclSymWinGetPeerPointer.md)使用UB Memory对称内存。

## 函数原型

```c
HcclResult HcclTeamGetLsaTeam(HcclComm comm, HcommTeamHandle *lsaTeam)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 已初始化的通信域句柄，不可为NULL。HcclComm类型的定义可参见[HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md)。 |
| lsaTeam | 输出 | 指向LSA WorldTeam句柄的指针，不可为NULL。返回的句柄由comm管理。HcommTeamHandle类型的定义可参见[HcommTeamHandle](../../datatype_definition/HcommTeamHandle.md)。 |

## 返回值

| 返回值 | 描述 |
| --- | --- |
| HCCL_SUCCESS | 获取LSA WorldTeam成功。 |
| HCCL_E_PTR | comm或lsaTeam为NULL。 |
| HCCL_E_NOT_SUPPORT | comm不是CommunicatorV2通信域。 |
| HCCL_E_NOT_FOUND | 通信域中没有预制LSA WorldTeam。 |
| HCCL_E_INTERNAL | LSA WorldTeam元数据异常。 |

## 约束说明

1. 仅支持CommunicatorV2通信域。
2. 通信域需要在初始化时成功预制UB Memory LSA WorldTeam，否则返回HCCL_E_NOT_FOUND。
3. 返回的LSA WorldTeam由通信域持有，调用方不得通过HcclTeamDestroy单独销毁。通信域销毁后，该句柄失效。

## 调用示例

```c
HcommTeamHandle lsaTeam = NULL;
HcclResult ret = HcclTeamGetLsaTeam(comm, &lsaTeam);
if (ret != HCCL_SUCCESS) {
    return ret;
}
```
