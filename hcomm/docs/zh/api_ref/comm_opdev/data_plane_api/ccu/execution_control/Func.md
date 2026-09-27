# Func

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

`ccu::Func`用于封装一个lambda为可复用的FuncBlock定义，与调用点接口[CallFunc](CallFunc.md)共同构成CCU kernel内的FuncBlock机制，用于在kernel内将一段重复使用的逻辑只生成一份指令，多处调用均通过`Call`指令跳转，节省SRAM。

`ccu::Func`的lambda形参为零个或多个`ccu::Variable`，返回`void`。定义好的`Func`对象通过[CallFunc](CallFunc.md)在kernel内调用。

## 类定义

```cpp
namespace AscendC {
namespace ccu {

class Func {
public:
    // Lambda 必须返回void，形参必须全部为ccu::Variable
    template <typename Lambda>
    explicit Func(Lambda body);

    Func(const Func &) = delete;             // 不可复制
    Func &operator=(const Func &) = delete;  // 不可赋值
};

} // namespace ccu
} // namespace AscendC
```

## 参数说明

| 参数名 | 描述 |
| --- | --- |
| body | lambda表达式，形参为零个或多个`ccu::Variable`，返回`void`。形参个数在编译期由lambda签名自动推导。 |

## 约束说明

- `ccu::Func`对象必须具有"链接性"：只能定义在namespace作用域（含`static`）或作为类的静态数据成员，不能是函数局部变量（即使加`static`也不行）。这是`template <Func& Obj, ...>`引用类型NTTP的C++语言要求，违反时编译期直接报错，与运行时无关。[CallFunc](CallFunc.md)内部还会用`&Obj`作为标识来复用同一个FuncBlock，这是该约束的副作用收益，不是约束的成因。
- `ccu::Func`不可复制（拷贝构造和赋值运算符已`= delete`），每个Func对象有唯一身份。
- `ccu::Func`lambda形参类型必须能从`ccu::Variable&`传入，实际可写为`ccu::Variable` / `ccu::Variable&` / `const ccu::Variable&`三者之一，其他类型（如`int`、`ccu::Address`）会因模板替换失败而编译报错（注：当前未对形参类型加`static_assert`，错误信息以深层模板诊断形式出现，可读性不如返回值类型那条）。
- `ccu::Func`lambda必须返回`void`，否则编译期报错（由实现内显式`static_assert`给出可读诊断）。
- 不同的`ccu::Func`对象即使lambda内容相同，也各自合成独立的FuncBlock（Key是`&Obj`，不是lambda内容的hash）。

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

    // 通过CallFunc 在kernel 内调用已定义的Func
    CcuResult ret = CallFunc<MyAdd>(a);
    if (ret != CCU_SUCCESS) { return ret; }

    return CallFunc<MyAdd>(b);
}
```

`CallFunc`的调用语义、返回值与异常详见[CallFunc](CallFunc.md)。
