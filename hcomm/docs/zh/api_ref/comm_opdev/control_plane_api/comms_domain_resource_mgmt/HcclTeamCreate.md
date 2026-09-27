# HcclTeamCreate

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

创建一个team用于HCCL通信。根据`desc`中指定的`protocol`和`netLayer`，查找通信域初始化时预制的worldTeam，在其上创建team，并内部完成syncMem注册、channel建链、远端内存获取和绑定等操作，创建成功后返回team句柄。

## 函数原型

```c
HcclResult HcclTeamCreate(HcclComm comm, const HcclTeamCreateDesc* desc, HcommTeamHandle* team)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 已初始化的通信域句柄，不可为NULL。HcclComm类型的定义可参见[HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md)。 |
| desc | 输入 | team创建描述符，需先通过[HcclTeamCreateDescInit](HcclTeamCreateDescInit.md)初始化并填充业务字段。HcclTeamCreateDesc类型的定义可参见[HcclTeamCreateDesc](../../datatype_definition/HcclTeamCreateDesc.md)。 |
| team | 输出 | 创建的team句柄，不可为NULL。HcommTeamHandle类型的定义可参见[HcommTeamHandle](../../datatype_definition/HcommTeamHandle.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. worldTeam在通信域初始化时自动创建，用户无需手动创建。HcclTeamCreate根据`desc`中的`protocol`和`netLayer`查找对应的预制worldTeam。

2. `desc`需先通过[HcclTeamCreateDescInit](HcclTeamCreateDescInit.md)初始化。`protocol`不可为COMM_PROTOCOL_RESERVED，当前只支持URMA协议（对应枚举值COMM_PROTOCOL_UB_CTP/UBC_TP/UBOE/UB_RTP）。

3. 建链失败时，HcclTeamCreate会自动回滚已创建的team及其资源，`team`输出为NULL。

## 调用示例

```c
uint32_t rankSize = 2;
uint32_t deviceId = 0;
// 生成root节点的rank标识信息
HcclRootInfo rootInfo;
HcclResult ret = HcclGetRootInfo(&rootInfo);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// 初始化通信域
HcclComm comm;
ret = HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &comm);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// 申请Device内存
void* devPtr = nullptr;
size_t memSize = 1024;
aclError aclRet = aclrtMalloc(&devPtr, memSize, ACL_MEM_MALLOC_HUGE_FIRST);
if (aclRet != ACL_SUCCESS) {
    return HCCL_E_RUNTIME;
}
HcclCommSymWindow symWin;
// 注册对称内存window
ret = HcclCommSymWinRegister(comm, devPtr, memSize, &symWin, 1);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// 初始化描述符
HcclTeamCreateDesc desc;
ret = HcclTeamCreateDescInit(&desc);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// 填充业务字段
uint32_t rankIds[2] = {0, 1};
desc.rankIds = rankIds;
desc.rankNum = 2;
desc.selfRankId = 0;
desc.netLayer = 0;
desc.protocol = COMM_PROTOCOL_UB_CTP;
desc.requirement.barrierCount = 1;
desc.engine = COMM_ENGINE_AIV;
desc.notifyNum = 8;
desc.channelCnt = 1;
// 创建team
HcommTeamHandle team = nullptr;
ret = HcclTeamCreate(comm, &desc, &team);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// 销毁team
ret = HcclTeamDestroy(team);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// 解除对称内存window注册
ret = HcclCommSymWinDeregister(symWin);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// 释放内存
aclRet = aclrtFree(devPtr);
if (aclRet != ACL_SUCCESS) {
    return HCCL_E_RUNTIME;
}
// 销毁通信域
ret = HcclCommDestroy(comm);
if (ret != HCCL_SUCCESS) {
    return ret;
}
```
