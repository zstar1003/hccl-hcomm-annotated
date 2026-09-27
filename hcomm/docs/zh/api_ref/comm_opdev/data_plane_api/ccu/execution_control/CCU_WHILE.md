# CCU_WHILE

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

`CCU_WHILE`是一个预处理器宏，承担两种语义，由其前面是否有`CCU_DO`自动区分：

1. 独立while循环：`CCU_WHILE(condExpr) { body }`——当`CCU_WHILE`前没有`CCU_DO`时，作为普通while循环：每轮先判断条件，条件满足才执行body。
2. do-while收尾：`CCU_DO { body } CCU_WHILE(condExpr);`——当`CCU_WHILE`前有`CCU_DO`时，作为do-while循环的收尾：执行完body后判断条件，条件满足则回到循环开头。

两种语义对用户完全透明，写法与C++标准while/do-while一致。do-while用法请参见[CCU_DO](CCU_DO.md)。

## 宏语法

```cpp
// 语义1：独立while 循环
CCU_WHILE(condExpr) {
    // body
}

// 语义2：do-while 收尾（必须紧跟CCU_DO { } 之后）
CCU_DO {
    // body
} CCU_WHILE(condExpr);
```

## 参数说明

| 参数名 | 描述 |
| --- | --- |
| condExpr | 条件表达式，类型为`AscendC::ccu::CondExpr`。通过`ccu::Variable`的`operator==(uint64_t)`或`operator!=(uint64_t)`产生。条件在运行期由CCU硬件计算。 |

`CondExpr`的定义与生成方式请参见[CCU_IF](CCU_IF.md#参数说明)。

## 返回值

`CCU_WHILE`为预处理器宏，本身不返回[CcuResult](../../../datatype_definition/CcuResult.md)。宏展开后若发生内部错误（如label重复、句柄无效），可能产生`CCU_E_PARA`/`CCU_E_NOT_FOUND`，此时宏不抛异常、也不阻断注册：失败时body会被跳过，但注册仍然成功，错误不通过返回值反馈，调试时需留意。

## 约束说明

- `CCU_WHILE（condExpr）`后必须紧跟{}包裹的循环代码块。
- 针对Ascend 950PR/Ascend 950DT，`CCU_WHILE`当前的`condExpr`只支持==和!=两种判断模式。
- `condExpr`中比较的立即数（`imm`）必须为`uint64_t`类型。
- 作为独立while循环使用时，`CCU_WHILE`可以嵌套，内层`CCU_WHILE`可在外层`CCU_WHILE`的body内使用。
- `CCU_WHILE`（独立while）可以与`CCU_IF`组合嵌套。
- `CCU_WHILE`（无论哪种语义）不可在`ccu::Loop`的body lambda内部使用，硬件Loop body内不支持软件分支/循环。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景：独立while循环，对计数器循环10次
CcuResult MyKernel(CcuKernelArg arg) {
    Variable counter, limit, one;
    counter = 0;
    limit = 10;
    one = 1;

    CCU_WHILE(counter != limit) {
        // 循环体
        counter = counter + one;
    }

    return CCU_SUCCESS;
}
```
