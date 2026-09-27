# CcuKernelArg

## 功能说明

CCU kernel函数的参数类型，用于在CCU kernel注册时传递用户自定义参数。该类型为void*指针，调用方需自行管理参数内存的生命周期。

## 定义原型

```c
typedef void *CcuKernelArg;
```
