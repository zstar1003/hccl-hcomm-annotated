# Array

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

`ccu::Array<T>`是CCU kernel内批量持有物理连续资源的C++模板类。

- `Array(count)`构造即批量分配：一次性申请`count`个物理连续的虚拟句柄。
- `Array(acqHandle, count)`绑定Host侧已预约的前`count`个资源，不申请新的物理资源。
- 析构不释放：析构函数不释放硬件资源；虚拟句柄在翻译完成后失效，物理资源随CCU实例生命周期统一管理、回收。
- 仅可移动：禁止拷贝，允许移动。

物理连续是某些接口的前置条件：`Load(addr, vArr, num)`/`Store`批量加载/存储、`LocalReduce(buffers*, count, ...)`（2 ≤ count ≤ 8，详见 [LocalReduce](../data_movement/LocalReduce.md)）都要求参数物理连续，必须通过`Array<T>`申请，单独声明多个对象不保证物理连续。

当前仅支持以下三种特化类型：

| 特化类型 | 资源描述 | 支持的构造形式 |
| --- | --- | --- |
| `Array<Variable>` | N个物理连续标量寄存器 | `Array(count)`、`Array(acqHandle, count)` |
| `Array<Event>` | N个物理连续完成事件单元 | `Array(count)`、`Array(acqHandle, count)` |
| `Array<CcuBuffer>` | N个物理连续CcuBuffer | 仅`Array(count)` |

对其他类型实例化`Array<T>`将在编译期报错（仅支持上述三种特化类型）。

## 类声明

```cpp
namespace AscendC {
namespace ccu {
template <typename T>  // T 仅支持Variable / Event / CcuBuffer
class Array final {
public:
    explicit Array(uint32_t count);       // 构造即批量虚拟分配，count可为0
    // 绑定Host侧已预约的前count个资源，仅Variable / Event 特化可用
    Array(typename CcuArrayTraits<T>::Handle acqHandle, uint32_t count);
    T& operator[](uint32_t i);            // 下标访问（无边界检查）
    const T& operator[](uint32_t i) const;
    T* data();                             // 获取首元素指针（用于传给需要指针参数的批量接口）
    const T* data() const;
    uint32_t size() const;                 // 返回元素个数
    // 禁止拷贝；允许移动
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    Array(Array&& other) noexcept;
    Array& operator=(Array&& other) noexcept;
};
} // namespace ccu
} // namespace AscendC
```

## 构造函数说明

| 构造形式 | 说明 |
| --- | --- |
| `Array<Variable> vars(n);` | 申请`n`个物理连续标量寄存器句柄。 |
| `Array<Event> evts(n);` | 申请`n`个物理连续完成事件单元句柄。 |
| `Array<CcuBuffer> bufs(n);` | 申请`n`个物理连续MS句柄。 |
| `Array<Variable> vars(acqHandle, n);` | 绑定Host侧预约的前`n`个标量寄存器，不申请新的标量寄存器。 |
| `Array<Event> evts(acqHandle, n);` | 绑定Host侧预约的前`n`个完成事件单元，不申请新的完成事件单元。 |

`count`可以为0，两种构造形式均可（见下方注意）。构造失败时抛出异常。

### 绑定Host侧预约资源的构造

`Array(acqHandle, count)`中的`acqHandle`须与`T`对应：`Array<Variable>`用[HcommCcuVariableAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuVariableAlloc.md)返回的预约句柄，`Array<Event>`用[HcommCcuEventAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuEventAlloc.md)返回的预约句柄。构造时依次绑定预约段内序号`0`到`count - 1`的资源，元素`arr[i]`对应预约段内的第`i`个资源，与[HcommCcuVariableGetAddr](../../../control_plane_api/ccu_resource_mgmt/HcommCcuVariableGetAddr.md)、[HcommCcuEventGetAddr](../../../control_plane_api/ccu_resource_mgmt/HcommCcuEventGetAddr.md)中的`index`一一对应。

该构造只对`Array<Variable>`和`Array<Event>`可用。`Array<CcuBuffer>`没有对应的实现，写成`Array<CcuBuffer> bufs(acqHandle, n);`会在编译期失败。

构造失败时抛出异常，会被[HcommCcuKernelRegister](../../../control_plane_api/ccu_kernel_launch_execution/HcommCcuKernelRegister.md)统一接住，对外返回`CCU_E_INTERNAL`，本轮注册被中止。

> [!CAUTION]注意
> `count`须不大于预约句柄名下的资源个数。`count`为0时直接构造成空Array（`size() == 0`），不绑定资源，也不校验`acqHandle`或当前是否处于kernel注册阶段。

## 成员函数说明

| 函数 | 说明 |
| --- | --- |
| `arr[i]` | 返回第`i`个元素的引用（从0起，无边界检查）。 |
| `arr.data()` | 返回首元素指针，用于传给需要`T*`参数的批量接口（如`LocalReduce(bufs.data(), count, ...)`）。 |
| `arr.size()` | 返回申请时的`count`值。 |

## 约束说明

- `Array(count)`以及`count > 0`的`Array(acqHandle, count)`只能在kernel注册阶段构造。
- 析构不释放硬件资源，不应在kernel之外保存元素的`handle`值，翻译完成后句柄即失效。
- `Array<T>`仅特化`Variable/Event/CcuBuffer`三种类型，对其他类型实例化将在编译期失败。
- 多个单独声明的`Variable`/`Event`/`CcuBuffer`对象不保证物理连续，不可用于需要物理连续资源的接口（如批量`Load`/`Store`、多Buffer`LocalReduce`）。

> [!NOTE]说明
> `Load`/`Store`会对Variable数组做连续性校验（不连续返回`CCU_E_PARA`）；而`LocalReduce`的多Buffer重载不校验CcuBuffer是否物理连续，须由调用方自行保证。用非`Array`申请的多个Buffer时不会立即报错，但运行期行为未定义。

- `Array(count)`只申请虚拟句柄，恒成功；资源池无法凑出N个连续物理资源时，在`HcommCcuKernelRegister`阶段返回`CCU_E_UNAVAIL`，不是在构造时抛出。
- `Array(acqHandle, count)`在`count > 0`时会校验预约句柄与`count`，参数不合法时在构造时抛出异常，不再保证恒成功。

## 调用示例

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    // 批量申请4个物理连续Variable，用于Load批量加载
    Array<Variable> vArr(4);
    Load(0x80000000ULL, vArr, 4);    // 一次加载4个uint64_t

    // 批量申请4个物理连续CcuBuffer，用于多Buffer归约
    Array<CcuBuffer> bufs(4);
    Variable len;
    Event evt;
    len = 4096;
    LocalReduce(bufs.data(), 4,
                HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP16,
                HCCL_REDUCE_SUM, len, evt);
    EventWait(evt);

    // 下标访问单个元素
    vArr[0] = 1024;    // 对第0个Variable赋立即数

    return CCU_SUCCESS;
}
```

绑定Host侧预约资源的用法：

```cpp
using namespace AscendC::ccu;

// Host侧通过HcommCcuVariableAlloc / HcommCcuEventAlloc预约资源后，
// 把预约句柄和个数放进kernelArgs传入kernel
struct MyKernelArg {
    CcuVariableHandle acqHandle;
    uint32_t varNum;
    CcuEventHandle acqEventHandle;
    uint32_t eventNum;
};

CcuResult MyKernel(CcuKernelArg arg) {
    auto* args = static_cast<MyKernelArg*>(arg);

    // 绑定Host侧预约的varNum个标量寄存器，不申请新的标量寄存器；绑定后的Variable用法与
    // 普通Variable完全相同，这里仅以逐个赋值演示下标访问
    Array<Variable> vars(args->acqHandle, args->varNum);
    for (uint32_t i = 0; i < vars.size(); i++) {
        vars[i] = i;
    }

    // 绑定Host侧预约的eventNum个完成事件单元；本例在CCU侧EventRecord
    Array<Event> events(args->acqEventHandle, args->eventNum);
    for (uint32_t i = 0; i < events.size(); i++) {
        EventRecord(events[i]);
    }

    return CCU_SUCCESS;
}
```
