# HcclCommStateCallback

## 功能说明

用于定义通信域不同阶段要调用的回调函数类型。

## 定义原型

```c
typedef HcclResult (*HcclCommStateCallback)(HcclComm comm, HcclCommStatePhase state, void *args)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | HCCL通信域。 |
| state | 输入 | 通信域的不同阶段，HcclCommStatePhase类型的定义可参见[HcclCommStatePhase](./HcclCommStatePhase.md)。 |
| args | 输入 | 回调函数传入的用户上下文指针。 |
