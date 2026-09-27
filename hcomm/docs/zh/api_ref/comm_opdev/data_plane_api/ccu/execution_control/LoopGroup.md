# LoopGroup

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

CCU kernel内的硬件LoopGroup类，将多个`ccu::Loop`对象组织为一组，共享同一份LoopEngine资源池，避免多个独立`Loop`各自独占LoopEngine导致资源耗尽。

构造`ccu::LoopGroup`时，自动将传入的`loops`列表中的每个`Loop`加入该Group。

## 类定义

```cpp
namespace AscendC {
namespace ccu {

class LoopGroup {
public:
    // 构造方式1：config-based（Group参数在注册期已知）
    LoopGroup(const CcuLoopGroupConfig &loopGroupCfg, uint32_t maxLoopNum,
              const std::vector<Loop> &loops);

    // 构造方式2：var-based（Group参数在运行期由Variable决定）
    LoopGroup(Variable &parallelCfg, Variable &offsetCfg, uint32_t maxLoopNum,
              const std::vector<Loop> &loops);
};

} // namespace ccu
} // namespace AscendC
```

## 参数说明

### 构造方式1：config-based

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| loopGroupCfg | 输入 | LoopGroup配置，类型为`CcuLoopGroupConfig`，字段含义见下表。 |
| maxLoopNum | 输入 | 本Group最多容纳的Loop数量，框架据此预留LoopEngine资源池容量。 |
| loops | 输入 | 要加入本Group的`ccu::Loop`对象列表。列表中每个Loop会在`LoopGroup`构造函数内自动被注册到Group。 |

### 构造方式2：var-based

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| parallelCfg | 输入 | 并行配置Variable，运行期决定并行参数。该参数包含64bit，其中[47:41]表示loopgroup内包含的Loop指令个数，其中[54:48]表示Loopgroup中包含的Loop指令需要完成Loop自动展开的Loop偏移，其中[61:55]表示Loop需要展开的次数。举例：X[47:41]=4表示程序中包含4个Loop指令，Xn[54:48]=1表示从编号为1的Loop开始展开，Xn[61:55]=3表示loop1，loop2，loop3分别复制展开3次，loop0不进行复制，经过展开后总Loop个数为4 + (4-1) * 3 = 13 |
| offsetCfg | 输入 | 偏移配置Variable，运行期决定偏移参数。 该参数包含64bit，其中[9:0]表示Loop进行展开后使用的Event资源偏移量，[20:10]表示Loop进行展开后使用的CcuBuffer资源偏移量，[52:21]表示Loop进行展开后各个数据传输类指令使用的Address累加的偏移量。|
| maxLoopNum | 输入 | 同构造方式1。 |
| loops | 输入 | 同构造方式1。 |

### CcuLoopGroupConfig

config-based构造使用的参数结构，字段如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `cloneNum` | `uint32_t` | 并行克隆数量，指定Group内并发执行的实例数。 |
| `cloneLoopOffset` | `uint32_t` | 克隆实例间Loop的偏移量。 |
| `addrOffset` | `uint32_t` | 克隆实例间`Address`的偏移字节数。 |
| `ccuBufferOffset` | `uint32_t` | 克隆实例间`CcuBuffer`切片的偏移片数。 |
| `eventOffset` | `uint32_t` | 克隆实例间`Event`槽位的偏移位数。 |

## 异常

`ccu::LoopGroup`构造失败时抛出异常（携带[CcuResult](../../../datatype_definition/CcuResult.md)错误码）。常见原因：

| 原因 | 错误码 |
| --- | --- |
| `maxLoopNum`为0，或var-based构造的运行期配置Variable为空 | `CCU_E_PARA` |
| 实际加入的Loop数超过`maxLoopNum`（LoopEngine池容量不足） | `CCU_E_PARA` |
| 物理资源（XN/GSA/CKE/MS等）不足 | `CCU_E_UNAVAIL` 等 |

## 约束说明

- `ccu::LoopGroup`构造时会立即遍历`loops`列表，将每个Loop注册到Group，注册后不可再修改Group成员。
- `loops`列表中的`ccu::Loop`对象必须已在`ccu::LoopGroup`构造前完成构造（即body已记录）。
- 同一个`ccu::Loop`对象不应被加入多个`ccu::LoopGroup`。
- Group内各Loop的body约束与独立`ccu::Loop`相同（参见[Loop](Loop.md)的约束说明）。
- `maxLoopNum`必须大于0，为0时构造直接失败（`CCU_E_PARA`）。
- `maxLoopNum`应 ≥ `loops`列表实际大小（即真正会加入Group的Loop数，含展开复用）；偏小会导致后续加入Loop时因LoopEngine池容量不足而失败（`CCU_E_PARA`）。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景：两个Loop共享LoopEngine资源池
CcuResult MyKernel(CcuKernelArg arg) {
    Variable r1, r2, numA, numB;
    numA = 10; numB = 20;

    Func body1([&] { r1 = numA + numB; });
    Func body2([&] { r2 = numA + numA; });

    CcuLoopConfig cfg1;
    cfg1.addrOffset = 0;
    cfg1.iterNum = 2;
    Loop l1(cfg1, body1);

    CcuLoopConfig cfg2;
    cfg2.addrOffset = 0;
    cfg2.iterNum = 3;
    Loop l2(cfg2, body2);

    // 将l1, l2 组织成LoopGroup，共享LoopEngine 资源池
    CcuLoopGroupConfig grpCfg;
    grpCfg.cloneNum = 0;
    grpCfg.cloneLoopOffset = 0;
    grpCfg.addrOffset = 0;
    grpCfg.ccuBufferOffset = 0;
    grpCfg.eventOffset = 0;
    LoopGroup g(grpCfg, /*maxLoopNum=*/2, {l1, l2});

    return CCU_SUCCESS;
}
```

基于variable直接构建loop/loopgroup，要保证variable的值符合指令要求。

```cpp
CcuResult CcuLoopAddDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    auto* args = static_cast<CcuLoopAddKernelArg*>(arg);

    Variable numA{}, numB{}, r1{}, r2{};
    numA = args->numA;
    numB = args->numB;

    // ========== LoopGroup variable-based ==========
    Variable varLoopParam1{}, varLoopParam2{}, varParallel{}, varOffset{};

    // loopParam: ctxId[52:45] | gsaOffset[44:13] | iterNum[12:0]
    varLoopParam1 = 0x0000000002000003ULL; // gsaOffset=4096, iterNum=3
    varLoopParam2 = 0x0000000002000004ULL; // gsaOffset=4096, iterNum=4

    // repeatNum=2, repeatLoopIndex=1, totalLoopNum=2
    varParallel = 0x0101040000000000ULL;

    // gsaOffset=4096, msOffset=1, ckeOffset=1
    varOffset = 0x0000000200000401ULL;

    Func body1([&]() {
        r1 = numA + numB;
    });
    Func body2([&]() {
        r2 = numA + numB;
    });

    Loop loop1(varLoopParam1, body1);
    Loop loop2(varLoopParam2, body2);

    LoopGroup group(varParallel, varOffset, /*maxLoopNum=*/4, {loop1, loop2});

    return CcuResult::CCU_SUCCESS;
}
```

