# LoadArg

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

在CCU kernel注册阶段，声明"在运行期将`HcommCcuKernelLaunch`的`taskArgs[argId]`加载到指定Variable"。注册时仅记录该声明，运行期由硬件完成实际加载。

本接口是让同一份已注册kernel运行期可配置的核心机制：注册时只写入模板逻辑，由`LoadArg`声明哪些Variable的值在每次Launch时由host注入，从而避免重复注册。

> [!NOTE]说明
> `LoadArg`在注册阶段并不执行实际加载。实际加载发生在`HcommCcuKernelLaunch`后、CCU kernel开始执行前，由硬件自动从`taskArgs[argId]`取值写入Variable对应的硬件寄存器。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
CcuResult LoadArg(Variable v, uint32_t argId);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| v | 输入 | 目标Variable对象。运行期硬件将`taskArgs[argId]`（`uint64_t`）写入该Variable对应的硬件寄存器。 |
| argId | 输入 | `taskArgs`数组的索引（从0起）。同一kernel内所有`LoadArg`调用的`argId`须全局唯一且从0连续编号。 |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | 当前不存在处于注册中的kernel（接口在kernel注册阶段之外被调用）。 |
| `CCU_E_NOT_FOUND` | 传入的`v`句柄未在当前kernel注册。 |

> [!NOTE]说明
> 本接口不会返回 `CCU_E_PARA`。argId 相关的所有约束错误（重复、跳号、与 `argNum` 不匹配）不在本接口处返回，而是在 `HcommCcuKernelLaunch` 阶段统一校验后返回 `CCU_E_INTERNAL`，详见下节"约束说明"。

## 约束说明

> [!CAUTION]注意
> 同一kernel内所有 `LoadArg` 调用使用的不同 `argId` 总数，必须与 `HcommCcuKernelLaunch` 的 `argNum` 参数严格相等；且 `argId` 必须覆盖 `0..argNum-1` 全部值。两条都在 `HcommCcuKernelLaunch` 阶段统一校验，违反时返回 `CCU_E_INTERNAL`。

- `argId` 须从0 连续编号（0、1、2、…），不可跳过或乱序——Launch 阶段强校验。
- 同一kernel 内不同 `LoadArg` 调用的 `argId` 不可重复——本接口处不会拦截重复，但重复会使有效 `argId` 总数小于 `argNum`，在 `HcommCcuKernelLaunch` 阶段报 `CCU_E_INTERNAL`。两个Variable 绑同一 `argId` 是危险错误，请自行避免。
- `argNum` 的单位是 `uint64_t` 元素个数，而非字节数。例如有3 个 `LoadArg` 调用（argId 为0、1、2），则 `HcommCcuKernelLaunch` 的 `argNum` 须传 `3`，`taskArgs` 须有至少3 个元素。
- `LoadArg` 不在调用处生效：翻译时所有 `LoadArg` 会被提到kernel 最前面执行，只负责给 `v` 设"进入body 时的初值"。因此若body 里又对同一个 `v` 用立即数（`v = 1024;`）或 `Load` 赋值，这些赋值都排在 `LoadArg` 之后，会覆盖掉注入的初值。混用同一个 `v` 时须先把注入值读走、再重新赋值，否则注入值还没用上就被覆盖，`LoadArg` 等于没生效（作用在不同Variable 上，如 `LoadArg(addrVar, 0); Load(addrVar, v);`，不受此影响）。
- `LoadArg` 的 `argId` 必须小于13，即0，1, ..., 12，因为CCU下发算子时最大只能携带13个参数。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景：注册时声明循环次数n和起始偏移offset由host在Launch时传入
CcuResult MyKernel(CcuKernelArg arg) {
    Variable n, offset;

    LoadArg(n, 0);        // 运行期taskArgs[0] → n
    LoadArg(offset, 1);   // 运行期taskArgs[1] → offset

    // 后续使用n和offset进行循环和地址计算
    // ...
    return CCU_SUCCESS;
}

// Host侧对应的Launch调用（示意）：
// uint64_t taskArgs[] = {100, 4096};   // n=100轮，offset=4096字节
// HcommCcuKernelLaunch(..., taskArgs, /*argNum=*/2);
```
