# CallFunc

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

`ccu::CallFunc<Obj>(args...)`是FuncBlock机制的调用点接口，与[Func](Func.md)配合使用：首次调用时合成`Obj`对应的FuncBlock段并追加一条`Call`指令；后续对同一`Obj`的调用直接追加`Call`指令，复用已有的FuncBlock段，从而将一段重复使用的逻辑只生成一份指令，多处调用均通过`Call`指令跳转，节省SRAM。

FuncBlock定义（`ccu::Func`）的说明请参见[Func](Func.md)。

## 模板函数

```cpp
namespace AscendC {
namespace ccu {

// Obj 必须为global 或static 存储的ccu::Func 对象（C++ NTTP 要求）
// Args 中每个参数必须为ccu::Variable
template <Func &Obj, typename... Args>
CcuResult CallFunc(Args... args);

} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 描述 |
| --- | --- |
| Obj | 模板非类型参数（NTTP），必须为`global`或`static`存储的`ccu::Func`对象的引用。 |
| args... | 实参列表，个数必须与`Obj`的lambda形参个数一致，每个参数类型必须为`ccu::Variable`。 |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PARA` | 实参个数与`Obj.NumIn()`不一致（该校验在最前，直接返回）。 |
| `CCU_E_PTR` | 当前不存在处于注册中的kernel（接口在kernel注册阶段之外被调用）。 |

> [!NOTE]说明
> 其余失败情形（如在硬件Loop body 内调用、嵌套CallFunc 等）会抛出异常（携带错误码），不通过返回值反馈，参见[异常](#异常)描述。

## 异常

`CallFunc`调用失败时会抛出异常（携带[CcuResult](../../../datatype_definition/CcuResult.md)错误码），常见情形：

| 情形 | 异常携带的错误码 |
| --- | --- |
| 在硬件Loop body 内调用`CallFunc`，或在`Func`的lambda body内嵌套调用`CallFunc`（仅允许在顶层调用） | `CCU_E_INTERNAL` |
| FuncBlock合成过程中的其他底层失败（如形参`Variable`资源申请失败） | 对应错误码 |

> [!NOTE]说明
> `CallFunc`抛出的异常不会穿透到用户kernel之外：kernel在`HcommCcuKernelRegister`阶段被执行，该入口对kernel body统一`try/catch`。异常被捕获后按标准异常路径归一，注册接口最终返回`CCU_E_INTERNAL`（其枚举值与`CCU_SUCCESS`/`CCU_E_PARA`等同属`CcuResult`，等于`HCCL_E_INTERNAL`）。因此调用方无需自行`try/catch`，只需检查`HcommCcuKernelRegister`的返回值即可感知上述失败。

## 约束说明

- `CallFunc<Obj>`的`Obj`必须为`global`或`static`存储的`ccu::Func`对象（不能是函数局部变量），这是`template <Func& Obj, ...>`引用类型NTTP的C++语言要求，违反时编译期直接报错。`Func`对象的链接性约束详见[Func](Func.md)的约束说明。
- `CallFunc<F>`不可在硬件Loop body内调用，否则抛出异常（错误码`CCU_E_INTERNAL`）。
- `CallFunc<F>`不可嵌套：`Func`的lambda内不可再调用其他`CallFunc`。此时内层`CallFunc`在最开始的`FuncBlockLookup`阶段即检测到当前处于`Func` body内（`inFuncBody_`）并抛出异常（错误码`CCU_E_INTERNAL`）；该异常直接穿透外层`CallFunc`（外层`RunBody`未包裹try/catch），外层的`FuncBlockEnd`不会执行。
- 首次`CallFunc`时合成FuncBlock，对同一`Obj`的后续`CallFunc`仅追加`Call`指令，复用已有FuncBlock。不同的`ccu::Func`对象即使lambda内容相同，也各自合成独立的FuncBlock（Key是`&Obj`，不是lambda内容的hash）。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 声明为static，满足CallFunc NTTP 要求
static Func MyAdd([](Variable x) {
    Variable tmp;
    tmp = x + x;   // 对x 翻倍
});

CcuResult MyKernel(CcuKernelArg arg) {
    Variable a, b;
    a = 2;
    b = 5;

    // 首次调用：合成FuncBlock 段 + 追加Call 指令
    CcuResult ret = CallFunc<MyAdd>(a);
    if (ret != CCU_SUCCESS) { return ret; }

    // 第二次调用：仅追加Call 指令，复用已有FuncBlock
    return CallFunc<MyAdd>(b);
}
```
