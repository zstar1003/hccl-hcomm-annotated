# HcommCcuEventAlloc

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

在Host侧从指定CCU实例的资源池中，预约`num`个物理编号连续的完成事件单元（Kernel侧对应[Event](../../data_plane_api/ccu/resource_allocation_operation/Event.md)），并返回预约句柄。预约成功后，这段完成事件单元即从实例资源池中划出，不再参与后续Kernel注册阶段的资源分配；但仍可通过预约句柄在Kernel内绑定使用。

预约完成后有两种用法，彼此独立，可只用其一，也可都用：

- 在Host侧调用[HcommCcuEventGetAddr](HcommCcuEventGetAddr.md)，取出每个完成事件单元已缓存的虚拟地址，再按原值传给CCU之外的目标模块。
- 将预约句柄通过`kernelArgs`传入Kernel，在Kernel内用预约句柄构造[Event](../../data_plane_api/ccu/resource_allocation_operation/Event.md)（如`Event e(acqHandle, index)`）或[Array\<Event\>](../../data_plane_api/ccu/resource_allocation_operation/Array.md)（如`Array<Event> events(acqHandle, count)`），绑定到这段已预约的完成事件单元。

若两条路径使用同一预约句柄、相同的段内序号（即查询地址与Kernel侧构造传入同一个`index`），则Kernel与上述模块指向同一组物理完成事件单元。

与Kernel内默认构造`Event e;`的差异：默认构造在注册阶段只得到虚拟句柄，物理完成事件单元在[HcommCcuKernelRegister](../ccu_kernel_launch_execution/HcommCcuKernelRegister.md)阶段（Kernel函数执行完后）才确定；本接口在Host侧预约时就固定了物理编号及其所属die。

一个Event对应一个完成事件单元。`num`表示预约的完成事件单元个数；后续[HcommCcuEventGetAddr](HcommCcuEventGetAddr.md)的`index`为段内序号，取值`[0, num)`，不是完成事件单元内部的bit编号。同一完成事件单元内的不同bit由Kernel侧[EventRecord](../../data_plane_api/ccu/synchronization/EventRecord.md) / [EventWait](../../data_plane_api/ccu/synchronization/EventWait.md)的`mask`区分；`index`与`mask`彼此独立。

本接口在预约成功路径上会为每个完成事件单元建立地址映射并缓存。若其中任意一个映射失败，接口会解除本次已完成的映射，将整段资源归还资源池，且不会写出有效的预约句柄。

> [!NOTE]说明
> 预约会占用同等数量的可用完成事件单元。后续Kernel若再默认构造`Event`或新申请`Array<Event>`，剩余数量不足时`HcommCcuKernelRegister`返回`CCU_E_UNAVAIL`。

## 函数原型

```c
CcuResult HcommCcuEventAlloc(CcuInsHandle insHandle, uint8_t dieId, uint32_t num, CcuEventHandle *eventHandle)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| insHandle | 输入 | CCU实例句柄，资源从该实例的资源池中分配。须为当前Device上已创建且未销毁的合法实例；传入0或不属于当前Device的句柄返回`CCU_E_PTR`。类型定义可参见[CcuInsHandle](../../datatype_definition/CcuInsHandle.md)。 |
| dieId | 输入 | 资源归属的IO Die编号，取值范围为`[0, CCU_MAX_IODIE_NUM)`，当前`CCU_MAX_IODIE_NUM`为2，即取值为`0`或`1`。该Die的连续完成事件单元资源池中切不出长度为`num`的连续空闲块时，本接口返回`CCU_E_UNAVAIL`而不是`CCU_E_PARA`。 |
| num | 输入 | 预约的连续完成事件单元个数，须大于0。 |
| eventHandle | 输出 | 预约句柄，不能为空指针。预约成功时写入非0句柄；失败时置0。若`eventHandle`本身为空指针，直接返回`CCU_E_PTR`，不写入。类型定义可参见[CcuEventHandle](../../datatype_definition/CcuEventHandle.md)。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 预约成功，`*eventHandle`为有效的预约句柄。 |
| `CCU_E_PTR` | `eventHandle`为空指针，或`insHandle`不是当前Device上已创建且未销毁的合法CCU实例句柄。 |
| `CCU_E_PARA` | `dieId`超出`[0, CCU_MAX_IODIE_NUM)`，或`num`为0。 |
| `CCU_E_UNAVAIL` | 该Die的连续完成事件单元资源池中不存在长度不小于`num`的连续空闲块。总量足够但被切分成多个碎片时同样返回本错误码。 |
| `CCU_E_RUNTIME` | 地址映射失败。失败时解除本次已建立的映射，并将整段资源归还资源池。 |

## 约束说明

- 本接口只能在主机侧调用，不能在Kernel函数体内调用。
- 调用本接口的线程须通过`aclrtSetDevice(int32_t deviceId)`接口绑定到与创建该实例时相同的NPU Device。
- 预约句柄没有单独的释放接口，生命周期与`insHandle`绑定：调用[HcommCcuInsDestroy](HcommCcuInsDestroy.md)销毁实例时，框架解除地址映射，预约句柄随之失效。
- 若要在Kernel内使用预约资源，须先完成本接口预约，再把预约句柄通过`kernelArgs`传给[HcommCcuKernelRegister](../ccu_kernel_launch_execution/HcommCcuKernelRegister.md)。仅查询映射地址时，不必注册Kernel。

## 调用示例

Host侧代码（C++）：

```cpp
CcuInsHandle insHandle = 0;
CcuResult ret = HcommCcuInsCreateDefault(nullptr, 0, &insHandle);
if (ret != CCU_SUCCESS) {
    return ret;
}

// 在die 0上预约4个物理连续的完成事件单元
const uint8_t dieId = 0;
const uint32_t eventNum = 4;
CcuEventHandle acqEventHandle = 0;
ret = HcommCcuEventAlloc(insHandle, dieId, eventNum, &acqEventHandle);
if (ret != CCU_SUCCESS) {
    (void)HcommCcuInsDestroy(insHandle);
    return ret;
}

// 取出每个完成事件单元已缓存的虚拟地址（Host只做传递）
for (uint32_t i = 0; i < eventNum; i++) {
    uint64_t va = 0;
    ret = HcommCcuEventGetAddr(acqEventHandle, i, &va);
    if (ret != CCU_SUCCESS) {
        (void)HcommCcuInsDestroy(insHandle);
        return ret;
    }
    // 将va按原值写入下发给目标模块的任务参数，例如：
    //   taskParam.eventVa[i] = va;
}

// 把预约句柄传入Kernel，Kernel内通过Array<Event>(acqEventHandle, eventNum)绑定同一段完成事件单元
// MyKernelArg与MyKernel的定义见下方Kernel侧代码
MyKernelArg arg = {};
arg.acqEventHandle = acqEventHandle;
arg.eventNum = eventNum;
const void *kernelArgs[] = { &arg };
CcuKernelHandle kernelHandle = 0;

ret = HcommCcuKernelRegisterStart(insHandle);
if (ret == CCU_SUCCESS) {
    // 预约完成事件单元所属die在HcommCcuEventAlloc时已固定。
    // HcommCcuKernelRegister的dieId是预留参数，当前实现未使用，传0即可。
    ret = HcommCcuKernelRegister(insHandle, 0, "MyKernel",
        (const void *)MyKernel, kernelArgs, 1, &kernelHandle);
    // Start成功后须成对调用End，即使本次Register失败
    CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ret == CCU_SUCCESS) {
        ret = endRet;
    }
}

// 销毁实例时，预约句柄失效，映射被解除；没有单独的释放接口
(void)HcommCcuInsDestroy(insHandle);
return ret;
```

Kernel侧代码（C++）：

```cpp
// Kernel入参结构体：把预约句柄和个数传给Kernel
struct MyKernelArg {
    CcuEventHandle acqEventHandle;
    uint32_t eventNum;
};

// Kernel函数：用Array<Event>(acqEventHandle, eventNum)绑定Host侧预约的eventNum个完成事件单元，
// 绑定后的Event用法与普通Event完全相同；更完整的写法见Array文档的调用示例。
// 本例在CCU侧EventRecord（mask默认0x1，置位bit0）；对应的EventWait由目标模块
// 在同一完成事件单元上执行，不在本Kernel内。
CcuResult MyKernel(CcuKernelArg arg)
{
    auto *myArg = static_cast<MyKernelArg *>(arg);
    AscendC::ccu::Array<AscendC::ccu::Event> events(myArg->acqEventHandle, myArg->eventNum);
    for (uint32_t i = 0; i < events.size(); i++) {
        AscendC::ccu::EventRecord(events[i]);
    }
    return CCU_SUCCESS;
}
```
