# Address

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

`ccu::Address`是CCU kernel内地址寄存器（GSA）的C++包装类。

- 构造即分配：默认构造函数自动申请1个GSA虚拟句柄。
- 析构不释放：析构函数不释放硬件资源；虚拟句柄在翻译完成后失效，物理资源随CCU 实例生命周期统一管理、回收。
- 运算符即device操作：Address上的赋值与算术运算符描述的是device端（硬件）执行的操作，而非host端立即计算，运行期操作对应的GSA寄存器。

与`Variable`（标量值）不同，`Address`专门承载地址值（HBM物理地址）。典型用法是将基地址与偏移量相加得到目标地址，供数据搬运接口使用。

> [!NOTE]说明
> `Address`承载device端地址值，不能在host端直接解引用。Address不能被赋值到Variable（不提供反向赋值API），但Variable可以赋值到Address（`addr = var;`）。

## 类声明

```cpp
namespace AscendC {
namespace ccu {
class Address final {
public:
    Address();                                                   // 构造即Alloc
    void operator=(uint64_t immediate) const;                   // 赋立即数地址
    void operator=(const Variable& var) const;                  // Variable值赋给Address
    void operator=(const Address& other) const;                 // Address间赋值
    void operator+=(const Variable& var) const;                 // Address就地加Variable
    void operator+=(const Address& other) const;               // Address就地加Address
    /*内部表达式对象*/ operator+(const Address& that) const;    // Address+Address
    /*内部表达式对象*/ operator+(const Variable& var) const;    // Address+Variable
    CcuAddressHandle handle{0};                                 // 虚拟句柄
};
// Variable + Address（交换律，全局函数）
/*内部表达式对象*/ operator+(const Variable& var, const Address& addr);
} // namespace ccu
} // namespace AscendC
```

## 构造函数说明

| 构造形式 | 说明 |
| --- | --- |
| `Address a;` | 申请1个GSA虚拟句柄。只能在kernel注册阶段（`HcommCcuKernelRegister`执行的kernel函数体内）调用。 |

> [!CAUTION]注意
> `Address`类有copy/move构造函数，它们只拷贝`handle`字段、不申请新的GSA——`Address a2 = a1;` 后`a1`和`a2`指向同一个地址寄存器。如需独立的Address，必须显式`Address a2;`默认构造。`operator=(const Address&)`则会发出device端寄存器赋值指令，操作的是两个独立寄存器之间的值搬移。

构造失败时抛出异常（携带[CcuResult](../../../datatype_definition/CcuResult.md)错误码）。

## 运算符说明

### 赋值运算符

| 表达式写法 | 硬件语义 |
| --- | --- |
| `addr = imm;`（`imm`为`uint64_t`） | `GSA_addr ← imm`。地址立即数在注册阶段确定，运行期不可变。 |
| `addr = var;`（`var`为`Variable`） | `GSA_addr ← XN_var`。将Variable的运行期值写入地址寄存器，适用于运行期动态地址。 |
| `dst = src;`（`src`为`Address`） | `GSA_dst ← GSA_src`。Address间寄存器赋值。 |

### 算术运算符

| 表达式写法 | 硬件语义 |
| --- | --- |
| `r = a + b;`（`a/b`均为`Address`） | `GSA_r ← GSA_a + GSA_b`。 |
| `r = addr + var;` / `r = var + addr;` | `GSA_r ← GSA_addr + XN_var`。两种写法语义相同（交换律）。 |
| `addr += var;`（`var`为`Variable`） | `GSA_addr ← GSA_addr + XN_var`（就地操作，比`addr = addr + var`节省一条指令）。 |
| `addr += other;`（`other`为`Address`） | `GSA_addr ← GSA_addr + GSA_other`。 |

> [!NOTE]说明
> `r = addr + var`与`r = var + addr`均合法（后者使用全局`operator+(Variable, Address)`友元函数），两种写法语义相同（交换律），参数顺序由运算符重载内部统一处理。

## 约束说明

- 只能在kernel注册阶段构造Address。
- 析构不释放硬件资源，不应在kernel之外保存或比较`handle`值，翻译完成后句柄即失效。
- `Address`不可被赋值到`Variable`，即不提供`Variable = Address`操作。
- 当前仅支持加法运算，不支持减法、乘法、除法。
- 立即数不可直接参与算术（`addr + 0x100`不合法），须先将偏移量赋给一个Variable后再参与运算。
- C++ 构造只申请虚拟句柄，恒成功不抛异常；`GSA` 物理资源不足时，由 `HcommCcuKernelRegister` 阶段返回 `CCU_E_UNAVAIL`，不是在构造时抛出。

## 调用示例

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    Address base, dst;
    Variable offset, stride;

    // 赋立即数地址（注册阶段固化）
    base = 0x80000000ULL;

    // 赋Variable值到Address（运行期动态地址）
    // offset通过LoadArg在Launch时注入
    LoadArg(offset, 0);
    dst = offset;         // GSA_dst ← XN_offset（运行期确定）

    // Address + Variable 偏移（两种等价写法）
    stride = 4096;
    dst = base + stride;  // r = addr + var
    dst = stride + base;  // r = var + addr（等价）

    // Address就地偏移
    base += stride;        // base += 4096

    // Address + Address
    Address result;
    result = base + dst;

    return CCU_SUCCESS;
}
```
