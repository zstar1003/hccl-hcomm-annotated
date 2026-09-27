# set_split_strategy_by_size

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

基于梯度数据量百分比，在集合通信group内设置反向梯度切分策略，实现allreduce的融合，用于进行集合通信的性能调优。

## 函数原型

```python
def set_split_strategy_by_size(dataSizeList, group="hccl_world_group")
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dataSizeList | 输入 | list类型。<br>梯度参数数据量百分比列表。<br>  - 梯度数据量百分比列表需为非负，且梯度数据量序列总百分比之和必须为100。<br>  - 梯度的切分最多支持8段。<br>  - 比如模型总共有150MB梯度数据量，需要切分90MB，30MB，30MB三段，则可以设置dataSizeList =[60,20,20]。 |
| group | 输入 | String类型，最大长度为128字节，含结束符。<br>group名称，可以为"hccl_world_group"或自定义group，默认为"hccl_world_group"。 |

## 返回值

无。

## 约束说明

- 必须在集合通信初始化完成之后调用。
- 调用该接口的rank必须在当前接口入参group定义的范围内，不在此范围内的rank调用该接口会失败。
- 在同时基于梯度数据量百分比及梯度的索引id设置反向梯度切分策略时，以基于梯度数据量百分比设置结果优先。
- 若用户不调用梯度切分接口设置切分策略，则会按默认反向梯度切分策略切分。

  默认切分策略：ResNet50的最优切分位置，即按梯度数据量切分为2段，第一段数据量为96.54%，第二段数据量为3.46%。

## 调用示例

```python
from hccl.split.api import *
set_split_strategy_by_size([60, 20, 20], "group")
```
