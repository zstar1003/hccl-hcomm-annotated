# CcuInsHandle

## 功能说明

CCU实例句柄，从HCCL通信域中获取，用于标识一个CCU实例。后续的Kernel注册、翻译、启动和实例销毁操作均通过此句柄进行。

## 定义原型

```c
typedef uint64_t CcuInsHandle;
```
