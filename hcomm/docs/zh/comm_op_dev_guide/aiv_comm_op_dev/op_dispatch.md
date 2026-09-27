# 算子下发

开发者完成通信算子任务编排后，需要将Kernel函数下发给具体的通信引擎执行。

定义AIV Kernel函数，格式如下，其中`<aiv_kernel_func_name>`为AIV Kernel函数名字。

```c
extern "C" __global__ __aicore__ void <aiv_kernel_func_name>(void *param)
{
    // Kernel实现
}
```

## 代码示例

以自定义AllGather算子为例，使用AIV通信引擎时，Host侧下发AIV Kernel函数的代码片段如下：

```c
// 获取控核数量
aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &numBlocksLimit);

// 从kernel二进制文件中获取句柄，全局可只加载一次
char realPath[] = "hccl_aiv_all_gather_op.o";
aclrtBinHandle binHandle;
aclrtBinaryLoadOptions loadOptions = {0};
aclrtBinaryLoadOption option;
loadOptions.numOpt = 1;
loadOptions.options = &option;
option.type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD;
option.value.cpuKernelMode = 1;
aclrtBinaryLoadFromFile(realPath, &loadOptions, &binHandle);

// 获取对应kernel的函数句柄
char funcName[] = "aiv_all_gather_bfloat16_t";
aclrtFuncHandle funcHandle;
aclrtBinaryGetFunction(binHandle, funcName, &funcHandle);

// 准备kernel launch的参数
aclrtLaunchKernelCfg cfg;
aclrtLaunchKernelAttr attr[3];
attr[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
attr[0].value.schemMode = 1;
attr[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
attr[1].value.timeoutUs.timeoutLow = 1800 * 1000000;
attr[1].value.timeoutUs.timeoutHigh = 0;
attr[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
attr[2].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
cfg.numAttrs = 3;
cfg.attrs = attr;

// 下发aiv kernel
aclrtLaunchKernelWithHostArgs(funcHandle, numBlocksLimit, stream, &cfg, args, argsSize, nullptr, 0);
```

在Device侧需要定义AIV Kernel函数入口，该函数需编译链接为一个独立的二进制.o文件：

```c
#define EXPORT_AIV_META_INFO(kernel_name) \
static const struct FunLevelKType kernel_name##_kernel_type_section __attribute__ \
((used, section (".ascend.meta." #kernel_name))) \
= {{F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}}

extern "C" __global__ __aicore__ void aiv_all_gather_bfloat16_t(GM_ADDR buffIn, ...) {
    return AivAllGatherMesh<bfloat16_t>();
}
EXPORT_AIV_META_INFO(aiv_all_gather_bfloat16_t)
```
