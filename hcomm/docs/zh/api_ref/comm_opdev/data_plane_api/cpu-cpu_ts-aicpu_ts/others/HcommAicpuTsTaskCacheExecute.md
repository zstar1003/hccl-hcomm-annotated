# HcommAicpuTsTaskCacheExecute

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

在cache hit场景下，根据新传入的内存地址信息刷新已缓存的task并下发执行。该接口会根据新的内存基址刷新SQE/WQE中的地址字段及相关token信息，然后按下发顺序统一提交执行。

## 函数原型

```c
HcommResult HcommAicpuTsTaskCacheExecute(const char *tag, void **addrs, uint64_t *sizes, uint64_t count)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tag | 输入 | 缓存标识符，需与HcommAicpuTsTaskCacheLookup传入的tag一致。 |
| addrs | 输入 | 新的内存基址信息数组，用于刷新cache entry中的地址字段。 |
| sizes | 输入 | 新的内存大小信息数组，需与cache miss时保存的内存大小一致。 |
| count | 输入 | 内存信息数组长度，需与cache miss时保存的数量一致。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

1. 仅支持AI CPU模式下，在Device侧调用该接口。
2. 仅在cache hit场景下调用（即HcommAicpuTsTaskCacheLookup返回isHit为true时），否则返回错误。
3. tag需与HcommAicpuTsTaskCacheLookup传入的tag一致，否则返回错误。
4. count和sizes需与cache miss时通过HcommAicpuTsTaskCacheStart传入的保持一致，否则返回错误。
5. 调用该接口后cache上下文会被重置。

## 调用示例

```c
const char *tag = "op_tag_example";
bool isHit = false;
void *addrs[] = {inputAddr, outputAddr};
uint64_t sizes[] = {inputSize, outputSize};
uint64_t count = 2;

// 查找cache
HcommAicpuTsTaskCacheLookup(tag, &isHit);

if (isHit) {
    // cache hit：刷新地址并下发
    if (HcommAicpuTsTaskCacheExecute(tag, addrs, sizes, count) != HCCL_SUCCESS) {
        return 1;
    }
} else {
    // cache miss：开始缓存
    HcommAicpuTsTaskCacheStart(tag, addrs, sizes, count);
    // 执行算子展开...
    HcommAicpuTsTaskCacheEnd(tag);
}
```
