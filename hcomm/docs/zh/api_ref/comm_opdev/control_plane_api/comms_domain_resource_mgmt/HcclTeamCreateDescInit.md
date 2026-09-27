# HcclTeamCreateDescInit

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

初始化team创建描述符。该接口会先以0xFF填充整个结构体，再设置ABI头部（version/magicWord/size）与各业务字段的默认值。

## 函数原型

```c
HcclResult HcclTeamCreateDescInit(HcclTeamCreateDesc *desc)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| desc | 输出 | 待初始化的team创建描述符，不可为NULL。HcclTeamCreateDesc类型的定义可参见[HcclTeamCreateDesc](../../datatype_definition/HcclTeamCreateDesc.md)。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，desc为NULL时返回HCCL_E_PTR。

## 约束说明

调用[HcclTeamCreate](HcclTeamCreate.md)前必须通过本接口初始化HcclTeamCreateDesc结构体。初始化后，用户需自行填充rankIds、rankNum、selfRankId、netLayer、protocol、requirement、engine、notifyNum、channelCnt等业务字段。若需使用共享队列，还需填充isSharedQueue和sharedQueueTag字段。

## 调用示例

```c
HcclTeamCreateDesc desc;
HcclResult ret = HcclTeamCreateDescInit(&desc);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// 填充业务字段
uint32_t rankIds[2] = {0, 1};
desc.rankIds = rankIds;
desc.rankNum = 2;
desc.selfRankId = 1;
desc.netLayer = 0;
desc.protocol = COMM_PROTOCOL_UB_CTP;
desc.requirement.signalCount = 0;
desc.requirement.counterCount = 0;
desc.requirement.barrierCount = 1;
desc.engine = COMM_ENGINE_AIV;
desc.notifyNum = 8;
desc.channelCnt = 1;
```
