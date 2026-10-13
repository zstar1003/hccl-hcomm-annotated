# HCCL 常见通信算子一图流

[查看图片](images/collective-operators-overview.png)

图中以 **12 组语义**归纳当前仓库接口，包含集合通信、点对点通信和屏障同步。不是声称 HCCL 总共只有 12 个算子：当前[集合通信接口目录](../hccl/docs/zh/api_ref/comm_op_interface/collective_comm.md)列出 11 项，加上 Send、Recv、BatchSendRecv 与 Barrier，共涉及 15 个接口。AlltoAllV/VC 合为一组；Send/Recv/BatchSendRecv 合为一组。

## 图例与范围

- R0、R1、R2 是 3 个 Rank；有 root 的示例取 R0。
- 橙、蓝、绿标识输入来源，紫色表示归约结果。Scatter 的输入和输出块全部来自 R0，保持橙色。
- 归约示例使用 SUM。实际归约类型和数据类型支持情况依接口、型号及版本而异。
- “—”表示本示例该位置不要求有效输入或不定义归约输出，不代表内存被清零、释放或其他 Rank 不参与。
- 输入表到输出表之间的大箭头表示整个通信操作的语义，不是逐 Rank 独立复制，也不表示特定 Ring/Mesh/NHR 算法。
- V 类示例采用连续存放；实际接口可通过位移配置不同的缓冲区布局。归约散射中，同一目的分片在各源 Rank 的长度必须一致。
- 点对点收发需要匹配对端及数据参数，不要求通信域全员执行同一集合操作；Batch 是批量配对，不等于 AlltoAll。
- Barrier 图表示通信流上的同步屏障，不等于 Host API 返回时设备工作已经全部完成，也不会产生用户数据聚合结果。
- 本图针对本仓公开 API 清单，没有加入清单中未列出的 Gather；它不是所有通信库的通用完整目录。

## 逐项核对

| 图号 | 接口/组 | 核对结果 |
|---|---|---|
| 01 | Broadcast | R0 的 A 复制到 R0/R1/R2，root 也保有结果 |
| 02 | Scatter | root 的 A/B/C 等长分发，各 Rank 只取对应块，数据颜色不变 |
| 03 | Reduce | 1+2+3=6，仅 R0 的归约输出有效 |
| 04 | AllReduce | 同样归约得到 6，三个 Rank 均得到结果 |
| 05 | AllGather | A、B、C 按 Rank 顺序拼接，三个输出均为 ABC，不执行加法 |
| 06 | AllGatherV | 输入长度 1/2/1，各输出均为 A/B1/B2/C |
| 07 | ReduceScatter | 逐位置归约为 111/222/333，再等长分给 R0/R1/R2 |
| 08 | ReduceScatterV | 各源均输入 4 个元素；归约后按 1/2/1 分发为 [111]、[222,333]、[444] |
| 09 | AlltoAll | 字母表示源 Rank，数字表示目的 Rank；输出依次为 A0/B0/C0、A1/B1/C1、A2/B2/C2 |
| 10 | AlltoAllV/VC | 图中矩阵是发送元素个数，行是源、列是目的；C[0][1]=2，零表示该方向本例发送零元素 |
| 11 | Send/Recv/Batch | 三条箭头分别表示配对收发示例；批量接口可组合这些操作 |
| 12 | Barrier | 先到的 Rank 等待；全部 Rank 执行到屏障后，各流后续任务才可继续 |

AlltoAllV 由各 Rank 提供其本地发送/接收计数与位移；AlltoAllVC 使用全局发送计数矩阵。两者都支持变长交换，VC 不表示“再做一次归约”。

数值检查已通过脚本计算：SUM、等长/变长归约分片、AllGatherV 拼接顺序、AlltoAll 行列转换。最终图片另经逐格目视检查，并修正了 Scatter 输出配色及 Barrier 等待线。检查针对图示语义，不替代 Ascend 环境中的 API 运行测试。

## 原始依据

- 集合通信接口文档：[Broadcast](../hccl/docs/zh/api_ref/comm_op_interface/HcclBroadcast.md)、[Scatter](../hccl/docs/zh/api_ref/comm_op_interface/HcclScatter.md)、[Reduce](../hccl/docs/zh/api_ref/comm_op_interface/HcclReduce.md)、[AllReduce](../hccl/docs/zh/api_ref/comm_op_interface/HcclAllReduce.md)、[AllGather](../hccl/docs/zh/api_ref/comm_op_interface/HcclAllGather.md)、[AllGatherV](../hccl/docs/zh/api_ref/comm_op_interface/HcclAllGatherV.md)、[ReduceScatter](../hccl/docs/zh/api_ref/comm_op_interface/HcclReduceScatter.md)、[ReduceScatterV](../hccl/docs/zh/api_ref/comm_op_interface/HcclReduceScatterV.md)、[AlltoAll](../hccl/docs/zh/api_ref/comm_op_interface/HcclAlltoAll.md)、[AlltoAllV](../hccl/docs/zh/api_ref/comm_op_interface/HcclAlltoAllV.md)、[AlltoAllVC](../hccl/docs/zh/api_ref/comm_op_interface/HcclAlltoAllVC.md)。
- [Send、Recv、BatchSendRecv 的公开声明与参数说明](../hccl/include/hccl.h)。
- [HcclBarrier 文档](../hcomm/docs/zh/api_ref/comm_mgr_c/HcclBarrier.md)。
- [官方 AlltoAllVC 说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0059.html)：矩阵表示各源到各目的的数据量。
- [官方 Barrier 说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/hcclapiref/hcclcpp_07_0028.html)：通信域内各 Rank 的流同步。产品支持以当前对应版本文档为准。

制图使用内置 imagegen。核心提示词：16:9、4列×3行、白底蓝框、统一3-Rank输入/输出表、固定SUM数值及数据块来源颜色；点对点与Barrier独立呈现，不将12组语义称为12个API。局部打磨保持其余表格不变，只修正Scatter配色与Barrier等待线。
