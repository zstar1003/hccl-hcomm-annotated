# get_group_rank_from_world_rank

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

从world rank id，获取该进程在group中的group rank id。

## 函数原型

```python
def get_group_rank_from_world_rank(world_rank_id, group)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| world_rank_id | 输入 | int类型。<br>进程在"hccl_world_group"中的rank id。 |
| group | 输入 | String类型，最大长度为128字节，含结束符。<br>group名称，可以为用户自定义group或者"hccl_world_group"。 |

## 返回值

int类型，正常返回进程在group中的rank id。

## 约束说明

- 必须在集合通信初始化完成之后调用。
- 调用该接口的rank必须在当前接口入参group定义的范围内，不在此范围内的rank调用该接口会失败。
- [create_group](create_group.md)完成之后，调用此API转换world rank id到group rank id。
- 本接口与[get_world_rank_from_group_rank](get_world_rank_from_group_rank.md)互为逆操作，注意参数顺序差异：本接口参数顺序为(world_rank_id, group)，其逆操作参数顺序为(group, group_rank_id)。

## 调用示例

```python
from hccl.manage.api import create_group
from hccl.manage.api import get_group_rank_from_world_rank
create_group("myGroup", 4, [0, 1, 2, 3])
groupRankId = get_group_rank_from_world_rank(1, "myGroup")
```
