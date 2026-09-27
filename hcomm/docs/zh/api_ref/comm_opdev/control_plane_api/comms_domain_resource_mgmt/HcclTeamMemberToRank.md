# HcclTeamMemberToRank

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

将指定Team中的成员编号memberId转换为对应的通信域Rank ID。对于UB Memory LSA WorldTeam，memberId可作为[HcclSymWinGetPeerPointer](../../../comm_mgr_c/HcclSymWinGetPeerPointer.md)的LSA成员编号。

## 函数原型

```c
HcclResult HcclTeamMemberToRank(
    HcclComm comm, HcommTeamHandle team, uint32_t memberId, uint32_t *rankId)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 已初始化的通信域句柄，不可为NULL。 |
| team | 输入 | 待查询的Team句柄，不可为NULL，且必须属于comm。 |
| memberId | 输入 | Team内的成员编号，取值范围为[0, Team成员数)。 |
| rankId | 输出 | 指向通信域Rank ID的指针，不可为NULL。 |

## 返回值

| 返回值 | 描述 |
| --- | --- |
| HCCL_SUCCESS | 转换成功。 |
| HCCL_E_PTR | comm、team或rankId为NULL。 |
| HCCL_E_NOT_SUPPORT | comm不是CommunicatorV2通信域。 |
| HCCL_E_NOT_FOUND | team未在comm对应的Team管理器中注册。 |
| HCCL_E_PARA | team不属于comm，或memberId超出Team成员范围。 |

## 约束说明

1. 仅支持CommunicatorV2通信域。
2. team必须属于comm，并在调用期间保持有效。

## 调用示例

```c
uint32_t rankId = 0;
HcclResult ret = HcclTeamMemberToRank(comm, lsaTeam, memberId, &rankId);
if (ret != HCCL_SUCCESS) {
    return ret;
}
```
