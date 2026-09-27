# Store

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

在CCU kernel内将一个或多个Variable的`uint64_t`值写入HBM地址。

为`Load`的反向操作，支持相同的两种地址类型，通过第一参数的类型自动选择：

| 重载组 | 地址类型 | 地址何时确定 |
| --- | --- | --- |
| 重载1/2 | 立即数地址（`uint64_t`） | 注册阶段确定 |
| 重载3/4 | Variable地址（`Variable`） | 运行期从硬件寄存器读出 |

每组内部按存储数量分为单个Variable（num=1）和批量`Array<Variable>`（num>1）两个重载。

## 函数原型

```cpp
namespace AscendC {
namespace ccu {
// 重载1：将1个Variable存储到立即数地址
CcuResult Store(uint64_t addr, Variable v);
// 重载2：将num个Variable批量存储到立即数地址
CcuResult Store(uint64_t addr, Array<Variable>& vArr, uint32_t num);
// 重载3：将1个Variable存储到Variable地址（间接寻址）
CcuResult Store(Variable addrVar, Variable v);
// 重载4：将num个Variable批量存储到Variable地址（间接寻址）
CcuResult Store(Variable addrVar, Array<Variable>& vArr, uint32_t num);
} // namespace ccu
} // namespace AscendC
```

## 参数说明

### 重载1/2参数（立即数地址）

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| addr | 输入 | 立即数HBM目标地址（`uint64_t`）。须为CCU可访问的物理地址或经token化的VA，注册阶段确定，运行期不可变。 |
| v | 输入 | 源Variable（重载1，num=1）。运行期将该Variable的值写入`HBM[addr]`。 |
| vArr | 输入 | 源Variable数组首元素（重载2，num>1）。须通过`ccu::Array<Variable>`申请以保证物理连续。 |
| num | 输入 | 存储的`uint64_t`元素个数（重载2）。须大于0。`num>1`时将`vArr[0], vArr[1], ..., vArr[num-1]`分别写入`HBM[addr], HBM[addr+8], ..., HBM[addr+(num-1)*8]`。 |

### 重载3/4参数（Variable地址，间接寻址）

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| addrVar | 输入 | 地址Variable（`Variable`）。运行期该Variable中存储的值被用作HBM目标地址，须已赋有效地址值。 |
| v | 输入 | 源Variable（重载3，num=1）。 |
| vArr | 输入 | 源Variable数组首元素（重载4，num>1）。须通过`ccu::Array<Variable>`申请以保证物理连续。 |
| num | 输入 | 存储的`uint64_t`元素个数（重载4）。须大于0。语义同重载2，地址来自`addrVar`的运行期值。 |

## 返回值

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PARA` | 参数错误：`num`为0；或`num>1`时`vArr`元素物理不连续。 |
| `CCU_E_PTR` | 当前不存在处于注册中的kernel（接口在kernel注册阶段之外被调用）。 |
| `CCU_E_NOT_FOUND` | 传入的`v`/`vArr`句柄（或`vArr[1..num-1]`的相邻句柄）未在当前kernel 注册；或`num`大于`vArr`实际长度时访问到不存在的相邻句柄。 |

## 约束说明

- `addr`须为CCU可访问的物理地址或经token化的VA，直接传入未经token化的进程虚拟地址将触发驱动错误。
- `num`须大于0，传入0返回`CCU_E_PARA`。
- `num>1`时（重载2/4），`vArr`须指向物理连续的Variable数组，必须通过`ccu::Array<Variable>`申请。单独声明多个`Variable`对象不保证物理连续，违反时在`Store(...)`调用处即返回`CCU_E_PARA`。
- `num`必须 ≤ `vArr` 实际长度。本接口不校验`num <= vArr.size()`，`num`越界会访问到不属于`vArr`的相邻句柄或返回`CCU_E_NOT_FOUND`，请自行保证`num`不超过申请长度。
- `addrVar`（重载3/4）在调用本接口前须已被赋予有效地址值（通过`LoadArg`、立即数赋值或算术运算）。
- 立即数地址（重载1/2）在注册阶段确定，运行期不可变；需运行期动态地址时使用重载3/4。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景1：将1个Variable存储到立即数地址（重载1）
CcuResult MyKernel(CcuKernelArg arg) {
    Variable result;
    // ... 对result进行计算 ...
    Store(0x20000000ULL, result);
    return CCU_SUCCESS;
}

// 场景2：将4个Variable批量存储到立即数地址（重载2，num>1）
CcuResult MyKernel2(CcuKernelArg arg) {
    Array<Variable> vArr(4);    // 4个物理连续Variable
    // ... 对vArr进行计算 ...
    Store(0x80000000ULL, vArr, 4);
    return CCU_SUCCESS;
}

// 场景3：将Variable存储到由Variable指定的地址（重载3）
CcuResult MyKernel3(CcuKernelArg arg) {
    Variable dstAddr, result;
    LoadArg(dstAddr, 0);    // host传入目标地址
    // ... 对result进行计算 ...
    Store(dstAddr, result); // 运行期间接寻址写入
    return CCU_SUCCESS;
}
```
