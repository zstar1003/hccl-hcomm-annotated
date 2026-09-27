# GetResByChannel

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

在CCU kernel内获取指向channel共享Variable槽位的句柄。

该槽位是channel在建链时就已在本端预留好的、与对端编号一一镜像的共享Variable，本接口不消耗新的XN寄存器，只是将已经存在的槽位包装成可操作的Variable对象。

典型场景：对端rank通过[WriteVariableWithNotify](../synchronization/WriteVariableWithNotify.md)将值写入本端channel的共享Variable槽位N，本端kernel用`GetResByChannel<Variable>(ch, N)`取到指向同一槽位的句柄，即可读到对端发来的值。

> [!NOTE]说明
> 当前仅支持`Variable`特化（`GetResByChannel<Variable>`）。对其他类型的模板实例化会在编译期报错。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
template <typename T>
T GetResByChannel(ChannelHandle channel, uint32_t index);

// 当前仅支持Variable特化：
template <>
Variable GetResByChannel<Variable>(ChannelHandle channel, uint32_t varIndex);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| channel | 输入 | 跨rank通道句柄（`ChannelHandle`）。channel须已完成建链，其预分配的Variable池须包含`varIndex`对应的槽位。 |
| varIndex | 输入 | 通道内变量索引（0起），对应channel在建链时预分配的Variable数组中的某个槽位。 |

## 返回值

返回一个`Variable`对象，其`handle`指向channel的第`varIndex`个共享Variable槽位。该Variable不消耗新的XN，其释放权归channel所有，不随返回的Variable析构。

调用失败时抛出异常（携带错误码），常见错误码：

| 情形 | 错误码 |
| --- | --- |
| `channel == nullptr`、channel 类型非法、当前不存在处于注册中的kernel | `CCU_E_PTR` |
| `varIndex` 越界（超过channel 预分配Variable 池大小） | `CCU_E_PARA` |

## 约束说明

- 只能在kernel注册阶段调用，须在`HcommCcuKernelRegister`执行的kernel函数体内。
- 不消耗新的XN资源，与`Variable v;`（普通构造）的分配行为完全不同，不可互换。
- 返回的Variable生命周期归channel管理，在channel销毁前有效。
- 当前仅支持`T = Variable`，对其他类型调用编译期失败。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 端到端场景：对端通过WriteVariableWithNotify向本端channel 的共享Variable 槽位0 写值
// 本端kernel通过GetResByChannel读取该值

CcuResult MyKernel(CcuKernelArg arg) {
    auto* params = static_cast<MyKernelArg*>(arg);
    ChannelHandle ch = params->channelHandle;

    // 获取channel预分配的Variable[0]（不申请新XN）
    Variable syncVar = GetResByChannel<Variable>(ch, 0);

    // 等待对端写入（详见WriteVariableWithNotify）
    NotifyWait(ch, /*localNotifyIdx=*/0);

    // 此时syncVar已持有对端写入的值，可读取参与后续运算
    Variable result;
    result = syncVar;

    return CCU_SUCCESS;
}
```
