# HcclSymWinGetRemoteAddr

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
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：不支持
<!-- end id5 -->

## 功能说明

根据对称内存窗口资源句柄和偏移，获取指定rank ID的远端对称内存窗口中该偏移对应的地址指针。

<!-- npu="950" id6 -->
针对Ascend 950PR/Ascend 950DT，本接口支持URMA场景。
<!-- end id6 -->

## 函数原型

```c
HcclResult HcclSymWinGetRemoteAddr(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| winHandle | 输入 | 对称内存窗口资源句柄。<br>HcclCommSymWindow类型的定义可参见[HcclCommSymWindow](./data_type_definition/HcclCommSymWindow.md)。 |
| offset | 输入 | 使用[HcclCommSymWinGet](HcclCommSymWinGet.md)获取到的偏移量。 |
| peerRank | 输入 | rank ID，取值范围：[0, rankSize)。 |
| ptr | 输出 | 指向“远端对称内存窗口中对应地址”的指针。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

<!-- npu="950" id7 -->
- 针对Ascend 950PR/Ascend 950DT，仅支持URMA场景。该场景下：
  - 调用该接口前需确保对称内存窗口已完成注册，且相关URMA通信通道已完成建链和远端内存信息更新。

    使用集合通信接口时，相关通道创建和远端内存信息更新由集合通信内部完成；使用独立通信通道资源接口时，需在通道创建成功后再调用该接口。
  - 传入的winHandle必须为已注册的有效对称内存窗口句柄。若通过[HcclCommSymWinGet](HcclCommSymWinGet.md)获取窗口句柄时未命中，返回的winHandle为空，不能继续传入本接口；若对应peerRank的远端内存信息尚未完成更新，本接口将返回错误。
  - 若窗口非URMA模式，本接口将返回错误并将*ptr置为nullptr，此时应改用[HcclSymWinGetPeerPointer](HcclSymWinGetPeerPointer.md)获取地址。
<!-- end id7 -->
- 该接口仅支持通信算子展开模式为AI CPU的场景。
- 该接口仅支持在Device侧调用。

## 调用示例

HOST侧注册完对称内存window以后，将window作为AI CPU kernel的参数传递到AI CPU kernel，该函数需要编译到Device AI CPU侧执行，以下是伪代码描述：

```c
AicpuKernelFunc(param)：
// 从param中获取对称window
HcclCommSymWindow temp_win = param.win;
void *remote_ptr;
uint32_t peerRankId = 1;
// 使用win + offset + peerRank获取到peerRank对应的远端地址
HcclSymWinGetRemoteAddr(temp_win, 0, peerRankId, &remote_ptr);
// 获取出来的地址可以使用数据面local copy直接读写，thread和size需调用者准备好
HcommLocalCopyOnThread(thread, remote_ptr, src_ptr, size);
```
