# HcclKernelFuncInfo

## 功能说明

用于描述AICPU核函数信息，包含动态库名称、核函数名称、参数地址及参数大小。该结构体用于指定需要启动的核函数及其参数。

## 定义原型

```c
typedef struct {
    char kernelSoName[HCCL_KERNEL_SO_NAME_MAX_LEN];
    char kernelFuncName[HCCL_KERNEL_FUNC_NAME_MAX_LEN];
    void *args;
    uint32_t argSize;
} HcclKernelFuncInfo;
```

## 参数说明

- **kernelSoName**：动态库名称，用于指定包含核函数的动态库文件名，最大长度为256字节（HCCL_KERNEL_SO_NAME_MAX_LEN）。
- **kernelFuncName**：核函数名称，用于指定动态库中需要调用的核函数名，最大长度为256字节（HCCL_KERNEL_FUNC_NAME_MAX_LEN）。
- **args**：核函数参数地址，指向参数数据的指针。当argSize大于0时，该参数不能为空指针。
- **argSize**：核函数参数大小，单位为字节。

## 相关常量

```c
const uint32_t HCCL_KERNEL_SO_NAME_MAX_LEN = 256;    // 动态库名称最大长度
const uint32_t HCCL_KERNEL_FUNC_NAME_MAX_LEN = 256;  // 函数名称最大长度
```
