# HcommCcuEventGetAddr

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

查询指定预约句柄名下第`index`个Event（完成事件单元）已缓存的虚拟地址；预约句柄由[HcommCcuEventAlloc](HcommCcuEventAlloc.md)返回。

该地址在[HcommCcuEventAlloc](HcommCcuEventAlloc.md)阶段已完成映射并缓存，供CCU之外的目标模块使用。本接口在Host侧调用，按`index`返回已缓存的地址，可重复调用，同一`index`多次查询得到相同地址。须按原值交给目标模块。

若Kernel内再用同一预约句柄绑定同一`index`的[Event](../../data_plane_api/ccu/resource_allocation_operation/Event.md)，则可与上述模块指向同一物理完成事件单元。地址映射细节见[HcommCcuEventAlloc](HcommCcuEventAlloc.md)。

`index`选择的是本次预约中的第几个完成事件单元，不是完成事件单元内部的某个bit。Kernel侧通过[EventRecord](../../data_plane_api/ccu/synchronization/EventRecord.md) / [EventWait](../../data_plane_api/ccu/synchronization/EventWait.md)的`mask`选择该完成事件单元内要置位或等待的bit。`index`与`mask`彼此独立：前者选出哪一个完成事件单元，后者选出该完成事件单元内的哪些bit。

## 函数原型

```c
CcuResult HcommCcuEventGetAddr(CcuEventHandle eventHandle, uint32_t index, uint64_t *va)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| eventHandle | 输入 | [HcommCcuEventAlloc](HcommCcuEventAlloc.md)返回的预约句柄。须为Event预约句柄。 |
| index | 输入 | 预约段内序号，取值范围为`[0, num)`，其中`num`是调用[HcommCcuEventAlloc](HcommCcuEventAlloc.md)时指定的个数。 |
| va | 输出 | 接收该Event（完成事件单元）已缓存的虚拟地址，不能为空指针。须按原值交给目标模块。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 查询成功，`*va`为该Event（完成事件单元）已缓存的虚拟地址。 |
| `CCU_E_PTR` | `va`为空指针。 |
| `CCU_E_NOT_FOUND` | 当前Device上不存在`eventHandle`对应的预约记录，或该预约已随CCU实例销毁。 |
| `CCU_E_PARA` | 当前Device上存在该句柄但类型不是Event（例如误传了[HcommCcuVariableAlloc](HcommCcuVariableAlloc.md)的句柄），或`index`超出`[0, num)`。 |

## 约束说明

- 本接口只能在主机侧调用，不能在Kernel函数体内调用。
- 调用本接口的线程须通过`aclrtSetDevice(int32_t deviceId)`接口绑定到发起预约时使用的同一NPU Device。
- 返回地址的有效期与预约句柄一致：[HcommCcuInsDestroy](HcommCcuInsDestroy.md)销毁实例后映射即被解除，不得继续使用先前取得的地址。
- 仅查询地址时不必注册Kernel；若要与Kernel共享同一完成事件单元，须在Kernel内用同一预约句柄绑定对应`index`，完整流程见[HcommCcuEventAlloc](HcommCcuEventAlloc.md)。

## 调用示例

```c
// insHandle为已创建的CCU实例句柄，参见HcommCcuInsCreateDefault/HcommCcuInsCreate
CcuEventHandle acqEventHandle = 0;
const uint32_t eventNum = 4;
CcuResult ret = HcommCcuEventAlloc(insHandle, 0, eventNum, &acqEventHandle);
if (ret != CCU_SUCCESS) {
    return ret;
}

// 逐个取出本次预约的每个Event（完成事件单元）已缓存的虚拟地址
for (uint32_t i = 0; i < eventNum; i++) {
    uint64_t va = 0;
    ret = HcommCcuEventGetAddr(acqEventHandle, i, &va);
    if (ret != CCU_SUCCESS) {
        printf("HcommCcuEventGetAddr failed, index = %u, ret = %d\n", i, ret);
        return ret;
    }
    // 将va按原值交给目标模块，例如：
    //   taskParam.eventVa[i] = va;
}
```
