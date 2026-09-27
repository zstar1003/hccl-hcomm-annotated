# HcommAicpuTsTaskCacheLookup

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

查找AICPU Task Cache中是否已缓存指定tag对应的算子展开任务。该接口是AICPU Task Cache使用的入口接口，根据查找结果决定后续走cache hit流程（调用HcommAicpuTsTaskCacheExecute）还是cache miss流程（调用HcommAicpuTsTaskCacheStart和HcommAicpuTsTaskCacheEnd）。

## 函数原型

```c
HcommResult HcommAicpuTsTaskCacheLookup(const char *tag, bool *isHit)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tag | 输入 | 缓存标识符，用于唯一标识一组算子展开任务。 |
| isHit | 输出 | 是否cache hit。true表示已缓存（cache hit），false表示未缓存（cache miss）。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

1. 仅支持AI CPU模式下，在Device侧调用该接口。
2. 相同tag的算子展开操作须串行执行，不可并发。原因在于展开过程采用"先占位、后回填"的缓存写入机制，并发场景下可能导致后续线程命中尚未完备的缓存条目，引发异常。例如，当tag中包含通信域标识（commId）时，相同tag的算子属于同一个通信域，必定按序展开。
3. 该接口需与HcommAicpuTsTaskCacheStart/HcommAicpuTsTaskCacheEnd（cache miss场景）或HcommAicpuTsTaskCacheExecute（cache hit场景）配套使用，且后续接口传入的tag需与该接口一致。

## 调用示例

```c
const char *tag = "op_tag_example";
bool isHit = false;
void *addrs[] = {inputAddr, outputAddr};
uint64_t sizes[] = {inputSize, outputSize};
uint64_t count = 2;

// 查找cache
if (HcommAicpuTsTaskCacheLookup(tag, &isHit) != HCCL_SUCCESS) {
    return 1;
}

if (isHit) {
    // cache hit：刷新并下发
    HcommAicpuTsTaskCacheExecute(tag, addrs, sizes, count);
} else {
    // cache miss：开始缓存
    HcommAicpuTsTaskCacheStart(tag, addrs, sizes, count);
    // 执行算子展开...
    HcommAicpuTsTaskCacheEnd(tag);
}
```
