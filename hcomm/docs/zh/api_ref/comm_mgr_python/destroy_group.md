# destroy_group

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
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：支持
<!-- end id5 -->

## 功能说明

销毁集合通信用户自定义group。

## 函数原型

```python
def destroy_group(group)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| group | 输入 | String类型，最大长度为128字节，含结束符。<br>group名称，集合通信group的标识。 |

## 返回值

无。

## 约束说明

- 必须在集合通信初始化完成之后调用。
- 调用该接口的rank必须在当前接口入参group定义的范围内，不在此范围内的rank调用该接口会失败。
- 相同名称的group，[destroy_group](destroy_group.md)和[create_group](create_group.md)配套使用，必须在create_group完成之后调用。
- 如果用户传入的group恰好是hccl_world_group（默认group），销毁group失败。

## 调用示例

```python
from hccl.manage.api import create_group
from hccl.manage.api import destroy_group
create_group("myGroup", 4, [0, 1, 2, 3])
destroy_group("myGroup")
```
