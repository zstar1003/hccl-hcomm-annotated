# HcommCcuKernelRegister

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

注册一个CCU Kernel函数。本接口在Host侧执行一次用户提供的Kernel函数，该函数体内所有`Ccu*`接口不执行实际硬件操作，仅记录该Kernel的完整操作序列。注册成功后返回Kernel句柄，供后续[HcommCcuKernelLaunch](HcommCcuKernelLaunch.md)使用。

注册失败时框架自动回滚当前Kernel，已注册的其他Kernel不受影响。Kernel函数体内的`CCU_IF`分支若未显式配对`CCU_ELSE`，会在本次`HcommCcuKernelRegister`调用结束时自动关闭。

## 函数原型

```c
CcuResult HcommCcuKernelRegister(CcuInsHandle insHandle, uint32_t dieId,
    const char *kernelFuncName, const void *kernelFunc,
    const void **kernelArgs, uint32_t argNum, CcuKernelHandle *kernelHandle)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| insHandle | 输入 | CCU实例句柄，从Hccl通信域中获取。有效句柄从1开始；传入0或未注册的句柄将返回`CCU_E_PTR`。 |
| dieId | 输入 | 指定Kernel运行的die，需要与channel参数信息中使用的die一致。 |
| kernelFuncName | 输入 | Kernel名称字符串，用于性能分析与调试标识。允许为空指针或空字符串，此时使用默认名称`"CCU_KERNEL"`；长度超过128字符时会被截断为前128字符。该字符串可与Kernel函数的符号名不一致。 |
| kernelFunc | 输入 | 指向Kernel函数的指针，不能为空指针。 |
| kernelArgs | 输入 | Kernel函数入参指针数组。当`argNum`不为`0`时不能为空指针，且实际使用的首元素`kernelArgs[0]`也不能为空；当`argNum`为`0`时该参数被忽略，可传`nullptr`。注册阶段`kernelArgs[0]`原样传入Kernel函数体，Kernel函数体内通过该参数读取的标量值会固化为立即数；运行期需要变化的标量须通过`taskArgs`与`CcuLoadArg`传入。 |
| argNum | 输入 | Kernel函数入参个数。当前仅支持`0`或`1`：`0`表示无入参Kernel，`1`表示单入参Kernel。传入大于`1`的值返回`CCU_E_PARA`。 |
| kernelHandle | 输出 | 出参，注册成功时回填Kernel句柄。不能为空指针。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PTR` | `insHandle`为0或无效，或`kernelFunc`为空指针，或`argNum`不为`0`时`kernelArgs`（含`kernelArgs[0]`）为空指针。 |
| `CCU_E_PARA` | `argNum`大于`1`（当前仅支持`0`或`1`）。 |
| `CCU_E_UNAVAIL` | 硬件资源不足，本Kernel申请的资源超出CCU实例的可用配额。 |
| `CCU_E_INTERNAL` | 内部错误，或Kernel函数抛出未捕获异常；或时序错误：未先调用[HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md)开始一轮注册就调用本接口。 |

## 约束说明

- 必须在[HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md)之后、[HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md)之前调用；若未先调用[HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md)，本接口返回`CCU_E_INTERNAL`。
- 同一实例的同一轮注册内可多次调用本接口，每次调用注册一个独立的Kernel。
- `argNum`当前仅支持`0`或`1`；为`1`时仅`kernelArgs[0]`生效，其余元素被忽略。
- `kernelArgs[0]`指向的标量在注册阶段即被读取，并固化为立即数。若某个标量的值需在每次启动时动态指定，须通过`taskArgs`数组与Kernel内的`CcuLoadArg`接口传入，而不是通过`kernelArgs`。
- 本接口只能在Host侧调用，不能嵌套调用（即Kernel函数体内不能再次调用本接口）。

## 调用示例

```c
// 用户自定义Kernel 入参结构体
typedef struct {
    uint32_t loopCount;
} MyKernelArg;

// 用户自定义Kernel 函数，函数体内调用Ccu* 系列接口
CcuResult MyKernel(CcuKernelArg arg)
{
    // CcuKernelArg即void*（见 ccu_types.h），用户可自定义入参结构体并通过kernelArgs传入
    MyKernelArg *myArg = (MyKernelArg *)arg;
    // 在此调用ccu数据面编程接口，例如LoadArg 等
    // ...
    return CCU_SUCCESS;
}

// insHandle 从通信域中获取
CcuInsHandle insHandle = 0;
uint32_t dieId = 0;                     // 指定Kernel运行的die
MyKernelArg arg = { .loopCount = 10 };
const void *kernelArgs[] = { &arg };    // 入参指针数组
uint32_t argNum = 1;                    // 当前仅支持0 或1
CcuKernelHandle kernelHandle = 0;

CcuResult ret = HcommCcuKernelRegister(
    insHandle,
    dieId,                          // 指定Kernel运行的die
    "MyKernel",                     // Kernel 名称，用于调试，可为空
    (const void *)MyKernel,         // Kernel 函数指针
    kernelArgs,                     // Kernel 入参指针数组
    argNum,                         // 入参个数
    &kernelHandle);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelRegister failed, ret = %d\n", ret);
    return ret;
}
```
