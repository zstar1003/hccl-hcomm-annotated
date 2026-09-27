# Loop

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

CCU kernel内的硬件Loop类，将循环body交由CCU LoopEngine硬件单元自动迭代。构造`ccu::Loop`对象时立即记录body（构造时body lambda执行一次），运行期由硬件自动推进迭代，无需在body内手写地址自增或计数器更新逻辑。

与软件循环（[CCU_WHILE](CCU_WHILE.md)）相比，硬件Loop每轮迭代不重新执行body指令，而是由LoopEngine硬件单元按offset自动递进地址/buffer/event——适用于大量同结构迭代、body内只有本地搬运类操作的场景，指令开销极小。

> [!CAUTION]注意
> `ccu::Loop` 单独构造只完成body录制，并不会真正下发硬件循环指令。必须把`Loop`加入[`ccu::LoopGroup`](LoopGroup.md)，由 `LoopGroup` 在注册阶段将`iterNum / addrOffset`等参数写入并合成硬件循环指令，硬件才真正按配置迭代。直接`Loop l(cfg, body);`然后不加入任何LoopGroup的写法，body只会被执行一次（注册期录制时），运行期不产生循环效果。

## 类定义

```cpp
namespace AscendC {
namespace ccu {

class Loop {
public:
    // 构造方式1：config-based（迭代参数在注册期已知）
    Loop(const CcuLoopConfig &loopCfg, const Func &func);

    // 构造方式2：var-based（迭代参数在运行期由Variable 决定）
    Loop(Variable &loopCfg, const Func &func);
};

} // namespace ccu
} // namespace AscendC
```

## 参数说明

### 构造方式1：config-based

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| loopCfg | 输入 | Loop配置，类型为`CcuLoopConfig`，字段含义见下表。迭代次数和地址偏移在kernel注册期确定。 |
| func | 输入 | Loop的body，类型为`ccu::Func`，要求为无入参的`Func`（lambda无参数）。构造时body lambda立即执行一次。 |

### 构造方式2：var-based

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| loopCfg | 输入 | Loop配置，类型为`ccu::Variable`，迭代次数等参数在运行期由Variable值决定。该参数包含64bit，其中[12:0]表示该Loop的循环次数，[44:13]表示Loop各次循环中数据传输类指令使用到的地址偏移量，数据地址=Address寄存器的值+[44:13]*循环次数，[52:45]表示该Loop使用的EngineID，用户需填0，运行时框架来填该值。 |
| func | 输入 | 同构造方式1。 |

### CcuLoopConfig

config-based构造使用的参数结构，字段如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `addrOffset` | `uint64_t` | 每轮迭代，body内`Address`自动偏移的字节数。 |
| `iterNum` | `uint64_t` | 迭代总次数。 |

## 异常

`ccu::Loop`构造失败时抛出异常（携带[CcuResult](../../../datatype_definition/CcuResult.md)错误码）。常见原因：

| 原因 | 错误码 |
| --- | --- |
| `func`的lambda有入参（Loop要求无参lambda） | `CCU_E_PARA` |
| 硬件资源不足等 | `CCU_E_UNAVAIL`等 |

## 约束说明

- `ccu::Loop`构造时body立即记录，构造完成后不可再向该Loop追加内容。
- body的`Func`lambda必须无入参，有入参时构造抛出异常。
- Loop body内有以下禁忌：
  - 会被拒绝（返回`CCU_E_NOT_SUPPORT`）：[EventRecord](../synchronization/EventRecord.md)的两个重载——`EventRecord(Event)`与`EventRecord(const char*)`（即[LocalNotifyRecord](../synchronization/LocalNotifyRecord.md)），二者在body内都不可用。
  - 会抛异常（错误码`CCU_E_INTERNAL`）：在body内调用[CallFunc](CallFunc.md)（`ccu::CallFunc<F>`）。
  - 不允许在body内嵌套[CCU_IF](CCU_IF.md)、[CCU_WHILE](CCU_WHILE.md)、[CCU_DO](CCU_DO.md)等软件控制流宏。
  - 建议避免在body内调用[NotifyRecord](../synchronization/NotifyRecord.md)、[WriteVariableWithNotify](../synchronization/WriteVariableWithNotify.md)：这两个接口在body内不会报错，但Loop body会被并行展开，远端notify语义在并行环境中不唯一，应谨慎使用。
- `ccu::Loop`必须加入`ccu::LoopGroup`才能下发硬件循环指令（详见[功能说明](#功能说明)），不要单独使用。
- `ccu::Loop`对象不可在`ccu::LoopGroup`中被重复添加。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 场景：将4个连续4KB数据块依次从HBM搬到MS Buffer
// 硬件Loop自动按addrOffset递进地址，无需手写地址自增
CcuResult MyKernel(CcuKernelArg arg) {
    Variable r1, numA, numB;
    numA = 10;
    numB = 20;

    // 定义body：无入参lambda
    Func body([&] {
        r1 = numA + numB;   // 每轮执行加法
    });

    // config-based：固定迭代4次，地址每轮偏移4096字节
    CcuLoopConfig cfg;
    cfg.addrOffset = 4096;
    cfg.iterNum = 4;
    Loop l(cfg, body);   // 构造时body立即记录（但仅录制IR，未下发硬件循环指令）

    // 关键：必须加入LoopGroup 才会真正下发循环指令
    CcuLoopGroupConfig grpCfg{};   // 单Loop 场景全字段置0 即可
    LoopGroup g(grpCfg, /*maxLoopNum=*/1, {l});

    return CCU_SUCCESS;
}
```
