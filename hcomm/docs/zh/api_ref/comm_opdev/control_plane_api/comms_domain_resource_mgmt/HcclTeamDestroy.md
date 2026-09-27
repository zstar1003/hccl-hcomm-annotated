# HcclTeamDestroy

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

销毁一个team，释放其资源。销毁时会释放该team的syncMem本地内存并注销syncMem的内存句柄。

## 函数原型

```c
HcclResult HcclTeamDestroy(HcommTeamHandle team)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| team | 输入 | 待销毁的team句柄，不可为NULL。可通过[HcclTeamCreate](HcclTeamCreate.md)创建。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

1. 销毁时会释放该team的syncMem本地内存，注销syncMem的CommRegMem内存句柄并清理注册条目。

2. team句柄销毁后不可再使用。

## 调用示例

```c
HcclResult ret = HcclTeamDestroy(team);
if (ret != HCCL_SUCCESS) {
    printf("HcclTeamDestroy failed, ret = %d\n", ret);
}
```
