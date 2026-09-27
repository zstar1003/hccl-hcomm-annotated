# HcclTeamRankToMember

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

将通信域Rank ID转换为指定Team中的成员编号memberId。对于UB Memory LSA WorldTeam，转换得到的memberId可用于数据面LSA访问。

## 函数原型

```c
HcclResult HcclTeamRankToMember(
    HcclComm comm, HcommTeamHandle team, uint32_t rankId, uint32_t *memberId)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 已初始化的通信域句柄，不可为NULL。 |
| team | 输入 | 待查询的Team句柄，不可为NULL，且必须属于comm。 |
| rankId | 输入 | 通信域Rank ID。 |
| memberId | 输出 | 指向Team成员编号的指针，不可为NULL。 |

## 返回值

| 返回值 | 描述 |
| --- | --- |
| HCCL_SUCCESS | 转换成功。 |
| HCCL_E_PTR | comm、team或memberId为NULL。 |
| HCCL_E_NOT_SUPPORT | comm不是CommunicatorV2通信域。 |
| HCCL_E_NOT_FOUND | team未注册，或rankId不属于team。 |
| HCCL_E_PARA | team不属于comm。 |

## 约束说明

1. 仅支持CommunicatorV2通信域。
2. team必须属于comm，并在调用期间保持有效。

## 调用示例

```c
uint32_t memberId = 0;
HcclResult ret = HcclTeamRankToMember(comm, lsaTeam, rankId, &memberId);
if (ret != HCCL_SUCCESS) {
    return ret;
}
```
