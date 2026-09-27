# CCU_IF

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

在CCU kernel内开始一个软件条件分支块。条件在运行期由CCU硬件计算，条件不满足时跳转至`CCU_ELSE`分支或if块出口。

`CCU_IF`是一个预处理器宏，可单独使用，也可紧跟一个可选的`CCU_ELSE`分支。`CCU_IF`与`CCU_ELSE`通过宏内部的状态管理自动配对，用户只需按宏语法书写。

## 宏语法

```cpp
CCU_IF(condExpr) {
    // then 分支 body
}

// 可选：紧跟 CCU_ELSE
CCU_IF(condExpr) {
    // then 分支 body
} CCU_ELSE {
    // else 分支 body
}
```

## 参数说明

| 参数名 | 描述 |
| --- | --- |
| condExpr | 条件表达式，类型为`AscendC::ccu::CondExpr`。通过`ccu::Variable`的`operator==(uint64_t)`或`operator!=(uint64_t)`产生，例如`v == 0`或`v != limit`。条件在运行期由CCU硬件计算。 |

## 返回值

`CCU_IF`为预处理器宏，本身不返回[CcuResult](../../../datatype_definition/CcuResult.md)。宏展开后若发生内部错误（如label重复、句柄无效），可能产生`CCU_E_PARA`/`CCU_E_NOT_FOUND`，此时宏不抛异常、也不阻断注册：失败时body会被跳过，但注册仍然成功。这与"`CCU_ELSE`无法配对时body被跳过"属于同一类情况——错误不通过返回值反馈，调试时需留意。

## 约束说明

- `CCU_IF（condExpr）`后必须紧跟{}包裹的then代码块
- 针对Ascend 950PR/Ascend 950DT，`CCU_IF`当前的`condExpr`只支持==和!=两种判断模式
- `CCU_IF`可以单独使用（无`CCU_ELSE`），也可以配合`CCU_ELSE`使用，两种写法均合法。
- 支持嵌套：`CCU_IF`内部可以再嵌套`CCU_IF`。
- `CCU_IF`不建议在硬件Loop（`ccu::Loop`）的body lambda内部使用——硬件Loop body内不支持软件分支；框架不强制校验，但行为未定义。
- `condExpr`中比较的立即数（`imm`）必须为`uint64_t`类型，Variable与Variable之间的比较不支持直接用于`CCU_IF`。

### 为什么`CCU_IF`与`CCU_ELSE`之间不能插入其他调用

`CCU_IF`的关闭并不在`CCU_IF { body }`的闭合花括号处立即发生，而是延迟到下面三种时机之一：

1. 紧跟的`CCU_ELSE`；
2. 其后第一个CCU API调用；
3. `HcommCcuKernelRegister` 结束时兜底关闭。

> 副作用：在`CCU_IF { body }`之后、`CCU_ELSE`之前如果插入任何CCU API调用，`CCU_IF`会被提前关闭，随后到达的`CCU_ELSE`找不到可配对的`CCU_IF`，其body会被跳过（注册仍然成功，不报错）。因此务必保持二者之间没有任何其他CCU API调用，详见[CCU_ELSE](CCU_ELSE.md)的约束说明。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景：根据运行期变量值选择不同的数据搬运路径
CcuResult MyKernel(CcuKernelArg arg) {
    Variable flag;
    LoadArg(flag, 0);   // 从taskArgs[0] 加载运行期标志

    CCU_IF(flag == 0) {
        // flag 为0 时执行此分支
        Variable a, b;
        a = 10;
        b = 20;
    } CCU_ELSE {
        // flag 非0 时执行此分支
        Variable c;
        c = 99;
    }

    return CCU_SUCCESS;
}
```
