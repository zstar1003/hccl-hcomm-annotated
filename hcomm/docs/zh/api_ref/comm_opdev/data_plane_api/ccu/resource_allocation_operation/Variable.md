# Variable

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

`ccu::Variable`是CCU kernel内标量寄存器的C++包装类。

- 构造即分配：默认构造函数自动申请1个标量寄存器虚拟句柄。
- 析构不释放：析构函数不释放硬件资源；虚拟句柄在翻译完成后失效，物理资源随CCU实例生命周期统一管理、回收。
- 运算符即device操作：Variable上的赋值与算术运算符描述的是device端（硬件）执行的操作，运行期操作对应的标量寄存器，而非在host端立即计算。

> [!NOTE]说明
> CCU资源分配采用"先虚后实"两阶段模型：注册阶段的`Variable()`构造仅产生虚拟句柄，真正的物理标量寄存器分配在 `HcommCcuKernelRegister` 阶段（kernel 函数执行完后）完成。

## 类声明

```cpp
namespace AscendC {
namespace ccu {
class Variable final {
public:
    Variable();                                                   // 构造即Alloc
    // 绑定Host侧已预约的第index个标量寄存器，不申请新的标量寄存器
    explicit Variable(CcuVariableHandle varHandle, uint32_t index = 0);
    void operator=(uint64_t immediate) const;                    // 赋立即数
    void operator=(const Variable& other) const;                 // Variable间赋值
    void operator+=(const Variable& other) const;               // 就地加法
    /*内部表达式对象*/ operator+(const Variable& that) const;    // 加法（表达式模板）
    CondExpr operator==(uint64_t immediate);                     // 产出CondExpr
    CondExpr operator!=(uint64_t immediate);                     // 产出CondExpr
    CcuVariableHandle handle{0};                                 // 虚拟句柄
};
} // namespace ccu
} // namespace AscendC
```

## 构造函数说明

| 构造形式 | 说明 |
| --- | --- |
| `Variable v;` | 申请1个标量寄存器虚拟句柄。只能在kernel注册阶段（`HcommCcuKernelRegister`执行的kernel函数体内）调用。 |
| `Variable v(varHandle, index);` | 绑定Host侧通过[HcommCcuVariableAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuVariableAlloc.md)预约的第`index`个标量寄存器，不申请新的标量寄存器。`index`默认值为`0`。 |

### 绑定Host侧预约资源的构造

`explicit Variable(CcuVariableHandle varHandle, uint32_t index = 0)`中的`varHandle`是Host侧[HcommCcuVariableAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuVariableAlloc.md)返回的**预约句柄**，不是另一个`Variable`的`handle`。典型用法是把预约句柄通过`kernelArgs`传入kernel，再在kernel内绑定其中某个标量寄存器。

对同一预约句柄的同一`index`多次调用本构造函数，每次得到新的kernel内`handle`，指向同一个物理标量寄存器。

构造失败时抛出异常，会被[HcommCcuKernelRegister](../../../control_plane_api/ccu_kernel_launch_execution/HcommCcuKernelRegister.md)统一接住，对外返回`CCU_E_INTERNAL`，本轮注册被中止。

> [!CAUTION]注意
> copy/move构造函数只拷贝`handle`字段、不申请新的标量寄存器。`Variable v2 = v1;`后两个对象是同一个kernel内句柄。如需独立的Variable，使用默认构造或[`Array<Variable>`](Array.md)。

## 运算符说明

### 赋值运算符

| 表达式写法 | 硬件语义 |
| --- | --- |
| `v = imm;`（`imm`为`uint64_t`） | `v ← imm`。立即数在注册阶段确定，运行期不可变。 |
| `d = s;`（`s`为`Variable`） | `d ← s`。device端寄存器赋值，而非host端handle拷贝。 |

### 算术运算符

| 表达式写法 | 硬件语义 |
| --- | --- |
| `r = a + b;` | `r ← a + b`（单条双源加法指令）。`operator+`返回表达式模板对象，被`operator=`消费时生成一条device加法，不产生临时Variable。 |
| `r += b;` | `r ← r + b`。与`r = r + b`语义相同，无临时对象。 |

> [!CAUTION]注意
> `r = a + b`使用表达式模板（内部类型）以避免产生临时Variable占用额外标量寄存器。不要把`a + b`的结果存入普通C++变量，否则不会生成对应的device操作。

### 条件运算符

| 表达式写法 | 返回类型 | 说明 |
| --- | --- | --- |
| `n == imm` | `CondExpr` | 产出条件表达式对象，不产生任何device操作，专供`CCU_IF`/`CCU_WHILE`/`CCU_DO...CCU_WHILE()`宏消费。 |
| `n != imm` | `CondExpr` | 同上。 |

> [!CAUTION]注意
> `CondExpr`只能被控制流宏消费，不能用作普通C++布尔表达式（如`if (n == 0)`）。塞入普通`if`不会产生任何CCU控制流，表达式被直接丢弃。

## 约束说明

- 只能在kernel注册阶段构造Variable。
- 析构不释放硬件资源，不应在kernel之外保存或比较`handle`值，翻译完成后句柄即失效。
- 当前仅支持加法运算，不支持减法、乘法、除法。
- 立即数不可直接参与算术（`v + 1`不合法），须先赋给一个Variable（`one = 1;`）后再参与运算（`v = v + one;`）。
- 默认构造仅申请虚拟句柄（不消耗物理标量寄存器），**在kernel注册阶段内构造恒成功**；若在注册阶段之外构造（不在`HcommCcuKernelRegister`调用的kernel函数体内），底层`CcuVariableAlloc`找不到当前kernel，会抛出携带`CCU_E_PTR`的`CcuException`。标量寄存器物理资源不足由`HcommCcuKernelRegister`阶段返回`CCU_E_UNAVAIL`，不在构造时触发。
- 预约句柄构造`Variable v(varHandle, index);`同样只能在kernel注册阶段调用，会校验预约句柄与`index`，参数不合法时抛出异常，不再保证恒成功。

## 调用示例

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    Variable n, i, one, step;

    // 赋立即数（注册阶段确定）
    n = 100;
    i = 0;
    one = 1;
    step = 8;

    // Variable间赋值
    Variable cursor;
    cursor = i;           // cursor ← i

    // 算术：表达式模板写法（r = a + b），仅生成一条device加法
    Variable sum;
    sum = i + step;       // sum ← i + step

    // 就地加法
    i += one;             // i ← i + one

    // 条件运算（供CCU_WHILE宏消费）
    CCU_WHILE(n != 0) {
        // ...
    }

    return CCU_SUCCESS;
}
```
