# CcuKernelHandle

## 功能说明

CCU Kernel句柄，由[HcommCcuKernelRegister](../control_plane_api/ccu_kernel_launch_execution/HcommCcuKernelRegister.md)注册后返回，用于标识一个已注册的Kernel。可通过[HcommCcuKernelLaunch](../control_plane_api/ccu_kernel_launch_execution/HcommCcuKernelLaunch.md)多次启动同一句柄。

## 定义原型

```c
typedef uint64_t CcuKernelHandle;
```
