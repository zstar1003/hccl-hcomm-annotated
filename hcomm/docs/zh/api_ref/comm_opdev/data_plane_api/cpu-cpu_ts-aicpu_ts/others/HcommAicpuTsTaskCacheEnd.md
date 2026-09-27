# HcommAicpuTsTaskCacheEnd

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

在cache miss场景下，算子展开完成后调用该接口，通知AICPU Task Cache停止缓存task。该接口会提交cache entry，更新内部地址刷新信息和token信息，并统计cache空间消耗。

## 函数原型

```c
HcommResult HcommAicpuTsTaskCacheEnd(const char *tag)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tag | 输入 | 缓存标识符，需与HcommAicpuTsTaskCacheStart传入的tag一致。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

1. 仅支持AI CPU模式下，在Device侧调用该接口。
2. 仅在cache miss场景下调用（即HcommAicpuTsTaskCacheLookup返回isHit为false时），否则返回错误。
3. tag需与HcommAicpuTsTaskCacheStart传入的tag一致，否则返回错误。
4. 该接口需与HcommAicpuTsTaskCacheStart配套使用，调用该接口后cache上下文会被重置。

## 调用示例

```c
const char *tag = "op_tag_example";
void *addrs[] = {inputAddr, outputAddr};
uint64_t sizes[] = {inputSize, outputSize};
uint64_t count = 2;

// cache miss下，开始缓存
HcommAicpuTsTaskCacheStart(tag, addrs, sizes, count);

// 执行算子展开（下发SQE/WQE等）
// ...

// 结束缓存
if (HcommAicpuTsTaskCacheEnd(tag) != HCCL_SUCCESS) {
    return 1;
}
```
