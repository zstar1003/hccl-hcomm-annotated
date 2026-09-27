# HcommAicpuTsTaskCacheClear

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

清理AICPU Task Cache中指定tag对应的cache entry。释放该cache entry占用的缓存空间，用于在通信域销毁或故障快速恢复等场景下清理不再需要的缓存。

## 函数原型

```c
HcommResult HcommAicpuTsTaskCacheClear(const char *tag)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tag | 输入 | 缓存标识符，指定需要清理的cache entry对应的tag。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

1. 仅支持AI CPU模式下，在Device侧调用该接口。
2. 若指定tag对应的cache entry不存在，接口仍返回成功。

## 调用示例

```c
const char *tag = "op_tag_example";

// 清理指定tag的缓存
if (HcommAicpuTsTaskCacheClear(tag) != HCCL_SUCCESS) {
    return 1;
}
```
