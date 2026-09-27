# set_split_strategy_by_idx

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

基于梯度的索引id，在集合通信group内设置反向梯度切分策略，实现allreduce的融合，用于进行集合通信的性能调优。

## 函数原型

```python
def set_split_strategy_by_idx(idxList, group="hccl_world_group")
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| idxList | 输入 | list类型。<br>梯度的索引id列表。<br>  - 梯度的索引id列表需为非负，升序序列。<br>  - 梯度的索引id必须基于模型的总梯度参数个数去设置。索引id从0开始，最大值可通过以下方法获得：不调用梯度切分接口设置梯度切分策略进行训练，此时脚本会使用[set_split_strategy_by_size](set_split_strategy_by_size.md)中的默认梯度切分方式进行训练。训练结束后，在INFO级别的Host训练日志中搜索"segment result"关键字，可以得到梯度切分的分段的情况如: segment index list: [0,107] [108,159]。此分段序列中最大的数字（例如159）即总梯度参数索引最大的值。 说明： 完整的训练过程可能出现日志覆盖情况，此时用户可以修改“/var/log/npu/conf/slog/slog.conf”中的配置项LogAgentMaxFileNum，提高Host侧保存日志文件的数量。或可以只进行一次迭代训练。<br>  - 梯度的切分最多支持8段。<br>  - 比如模型总共有160个参数会产生梯度，需要切分[0,20]、[21,100]和[101,159]三段，则可以设置为idxList=[20,100,159]。 |
| group | 输入 | String类型，最大长度为128字节，含结束符。<br>group名称，可以为"hccl_world_group"或自定义group，默认为"hccl_world_group"。 |

## 返回值

无。

## 约束说明

- 必须在集合通信初始化完成之后调用。
- 调用该接口的rank必须在当前接口入参group定义的范围内，不在此范围内的rank调用该接口会失败。
- 若用户不调用梯度切分接口设置切分策略，则会按默认反向梯度切分策略切分。

  默认切分策略：按梯度数据量切分为2段，第一段数据量为96.54%，第二段数据量为3.46%（部分情况可能出现为一段情况）。

## 调用示例

```python
from hccl.split.api import *
set_split_strategy_by_idx([20, 100, 159], "group")
```
