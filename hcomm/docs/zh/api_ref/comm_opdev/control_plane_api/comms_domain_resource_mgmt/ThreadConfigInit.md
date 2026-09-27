# ThreadConfigInit

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

初始化线程配置结构体。

## 函数原型

```c
HcommResult ThreadConfigInit(ThreadConfig *config, uint32_t num)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| config | 输出 | 线程配置结构体数组，数组长度为num，函数会初始化该结构体。<br>ThreadConfig类型的定义可参见[ThreadConfig](../../datatype_definition/ThreadConfig.md)。 |
| num | 输入 | 线程配置数量。 |

## 返回值

HcommResult：接口返回值说明如下：

| 返回值 | 描述 |
| --- | --- |
| 0 | 成功初始化线程配置结构体。 |
| 2 | 传入的config为空指针，对齐HCCL_E_PTR。 |
| 4 | 内部错误，对齐HCCL_E_INTERNAL。 |

## 约束说明

- ThreadConfig结构体必须调用该接口进行初始化，否则[HcclThreadAcquireWithConfig](HcclThreadAcquireWithConfig.md)接口将返回参数错误。

- 初始化后，需根据业务需求设置notifyNumPerThread字段，指定每个通信线程的同步资源数量。

## 调用示例

以初始化5个线程配置结构体为例：

```c
uint32_t threadNum = 5;
ThreadConfig configs[5];
ThreadConfigInit(configs, threadNum);
for (uint32_t i = 0; i < threadNum; i++) {
    configs[i].notifyNumPerThread = 2;
}
```
