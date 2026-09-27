# CCU_DO

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

在CCU kernel内开始一个do-while循环块：body至少执行一次，然后由配套的`CCU_WHILE`判断是否继续。

`CCU_DO`是一个预处理器宏，必须配合[CCU_WHILE](CCU_WHILE.md)收尾，构成完整的do-while循环：

```cpp
CCU_DO { body } CCU_WHILE(condExpr);
```

> [!CAUTION]注意
> `CCU_DO`必须配`CCU_WHILE(cond);`收尾。漏写收尾不会产生编译错误，但运行期行为错误：body仅执行一次后顺序向下，且会影响后续的`CCU_WHILE`，导致循环跳转错乱，极难调试。

## 宏语法

```cpp
CCU_DO {
    // body（至少执行一次）
} CCU_WHILE(condExpr);
```

## 参数说明

`CCU_DO`无参数。`CCU_WHILE(condExpr)`的`condExpr`定义与生成方式请参见[CCU_IF](CCU_IF.md#参数说明)。

## 返回值

`CCU_DO`为预处理器宏，本身不返回[CcuResult](../../../datatype_definition/CcuResult.md)；在正常用法下不会失败。

## 约束说明

- `CCU_DO`必须紧跟`CCU_WHILE(cond);`收尾，缺少收尾时运行期行为错误，且不会产生任何编译期或注册期错误。
- `CCU_DO { body }`与配套的`CCU_WHILE(cond);`之间不允许插入其他CCU API调用，否则do-while无法正确配对。
- `CCU_DO ... CCU_WHILE`不可在`ccu::Loop`的body lambda内部使用，硬件Loop body内不支持软件循环。
- `condExpr`中比较的立即数（`imm`）必须为`uint64_t`类型。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景：do-while循环，body至少执行一次，然后根据条件决定是否继续
CcuResult MyKernel(CcuKernelArg arg) {
    Variable inner, innerLimit, one;
    inner = 0;
    innerLimit = 5;
    one = 1;

    // do-while：body先执行一次，再判断条件
    CCU_DO {
        inner = inner + one;
    } CCU_WHILE(inner != innerLimit);

    return CCU_SUCCESS;
}
```
