# HcommBatchModeEnd

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

该接口用于提交并触发批量模式下缓存的所有操作的执行。所有在HcommBatchModeStart和HcommBatchModeEnd之间的数据面接口调用操作将在此时统一执行。

## 函数原型

```c
int32_t HcommBatchModeEnd(const char *batchTag)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| batchTag | 输入 | 批量任务标识符，需要与HcommBatchModeStart中传入的batchTag保持一致。 |

## 返回值

int32_t：接口成功返回0，其他失败。

## 约束说明

HcommBatchModeStart和HcommBatchModeEnd必须成对调用，且需要在同一线程中执行。

<!-- npu="950" id6 -->
仅Ascend 950PR/Ascend 950DT支持批量模式和立即执行模式（不调用批量接口），其他产品必须使用批量模式。
<!-- end id6 -->

## 调用示例

```c
char *tag = "";
// 启动批量模式（临时批量任务）
HcommBatchModeStart(tag);

// 在批量模式中调用数据面接口不会立即执行
// ...

// 结束批量模式并触发执行
HcommBatchModeEnd(tag);
```
