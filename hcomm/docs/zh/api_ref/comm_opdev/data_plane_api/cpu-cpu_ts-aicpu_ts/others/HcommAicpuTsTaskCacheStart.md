# HcommAicpuTsTaskCacheStart

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

在cache miss场景下，算子展开前调用该接口，通知AICPU Task Cache开始缓存task。该接口会将传入的内存基址和大小信息保存到cache entry中，后续算子展开过程中下发的SQE/WQE将被捕获并缓存。

## 函数原型

```c
HcommResult HcommAicpuTsTaskCacheStart(const char *tag, void **addrs, uint64_t *sizes, uint64_t count)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tag | 输入 | 缓存标识符，需与HcommAicpuTsTaskCacheLookup传入的tag一致。 |
| addrs | 输入 | 内存基址信息数组，指向用户输入/输出内存的基址。 |
| sizes | 输入 | 内存大小信息数组，与addrs一一对应。 |
| count | 输入 | 内存信息数组长度，即内存段的数量。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

1. 仅支持AI CPU模式下，在Device侧调用该接口。
2. 仅在cache miss场景下调用（即HcommAicpuTsTaskCacheLookup返回isHit为false时），否则返回错误。
3. tag需与HcommAicpuTsTaskCacheLookup传入的tag一致，否则返回错误。
4. 该接口需与HcommAicpuTsTaskCacheEnd配套使用，在两者之间执行算子展开操作。
5. 如果AICPU Task Cache容量已满，不会添加新的cache entry，此时算子展开仍可正常执行但不会被缓存。

## 调用示例

```c
const char *tag = "op_tag_example";
void *addrs[] = {inputAddr, outputAddr};
uint64_t sizes[] = {inputSize, outputSize};
uint64_t count = 2;

// cache miss下，开始缓存
if (HcommAicpuTsTaskCacheStart(tag, addrs, sizes, count) != HCCL_SUCCESS) {
    return 1;
}

// 执行算子展开（下发SQE/WQE等）
// ...

// 结束缓存
HcommAicpuTsTaskCacheEnd(tag);
```
