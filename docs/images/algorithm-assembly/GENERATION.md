# 算法组装配图：制图依据与复核记录

配套文档：[集合通信算法如何组装](../../ALGORITHM_ASSEMBLY.zh-CN.md)。核对日期：2026-10-15；源码固定为 `9aae2b3abdbaf03f12e47fe8617d9391ce067184`。

六张图由内置 imagegen 根据已核实的源码关系生成，并在生成后逐图检查数据标签、阶段顺序、箭头语义和源码命名。它们是非官方教学示意，不是硬件 trace；图中的相对面积、时间格宽度、颜色均不用于推算实测性能。SUM 示例中的 Σ 表示不同 Rank 在同一张量位置的相加。

## 图片与依据

| 文中顺序 | 图片 | 需要保持的事实 | 对应正文依据 |
|---|---|---|---|
| 图 1 | [组装链](01-assembly-chain.png) | 预先注册组合；规则与成本选择是替代入口；CalcRes 产生资源请求；资源取得后再执行；提交不等于完成 | S01—S09 |
| 图 2 | [AllReduce OneShot / TwoShot](02-oneshot-twoshot.png) | OneShot 每 Rank 得到完整归约结果；TwoShot 的 RS 中间态各持一片，AG 后各持完整结果 | S10—S14 |
| 图 3 | [Reduce / Broadcast TwoShot](05-reduce-broadcast-shots.png) | Reduce 为 RS → Gather 到 root；Broadcast 无归约，root 预发自身片段，非 root 再交换其余片段 | S28—S34 |
| 图 4 | [AllGather / ReduceScatter](06-allgather-reducescatter-stages.png) | AG 拼接而不归约；RS 只输出各自片段；轮数、阶段和全域算子不能混为一谈 | S27、S35—S37 |
| 图 5 | [Concur / Parallel](03-concur-parallel.png) | Concur 示例的两份数据各走一条完整路线；Parallel 的每份数据都经过两层，但顺序相反 | S07、S16、S17 |
| 图 6 | [流水线](04-pipeline.png) | 左侧是通用片段依赖示例；右侧是当前 AICPU OmniPipe 先组织 RS 区域、再组织 AG 区域的代码结构 | S19—S21 |

文件名前缀按生成先后保留，文中图号按阅读顺序排列。源码编号的完整链接见主文档末尾。

## 提示词与迭代

图片统一采用白底、深蓝文字、浅蓝分区和少量彩色数据块，保持技术图的紧凑密度。图片中不加入硬件照片，不把每条指令都画成独立节点，不使用虚构注册名。

以下文件保存实际使用的初始提示词：

- [组装链](prompts/01-assembly-prompt.txt)
- [AllReduce 的阶段对照](prompts/02-shots-prompt.txt)
- [Parallel 与 Concur](prompts/03-parallel-concur-prompt.txt)
- [流水线原理与实现边界](prompts/04-pipeline-prompt.txt)
- [Reduce 与 Broadcast](prompts/05-other-shots-prompt.txt)
- [AllGather 与 ReduceScatter](prompts/06-allgather-reducescatter.prompt.txt)

生成后进行了以下修正，仅将核对后的图片放入正文：

1. **组装链**：删除生成器自行补入的注册项，换为真实组合的简写；预注册组成项之间使用加号；将 CalcRes 到 HCOMM 的箭头标签改为“资源请求”。最后一次标签修正的[提示词](prompts/01-assembly-final-edit.prompt.txt)一并保存。图中数据小块表示模板处理的数据区间，并非主张数据切片只能发生在 KernelRun 之后。
2. **AllReduce 对照**：将容易与 AlltoAll 算子混淆的文字改成“整份数据互换；各 Rank 计算完整归约结果”，清理多余细小线条；检查四行输入、RS 四个中间片段及完整输出。
3. **Reduce / Broadcast**：核对 root 角色和输出归属。直接 Reduce 小图画的是远端输入汇入 root 的过程，root 框内 A0—D0 仍表示其本地输入；后续归约由文字说明。Broadcast 两阶段图逐格确认 R1 为 A/B、R2 为 A/C、R3 为 A/D，最终四个 Rank 都为 A/B/C/D。
4. **AllGather / ReduceScatter**：将底部“分层 AG”的最终标签明确为“一个全域 AG 算子”，避免把两个层次阶段说成同一个阶段；其余两行分别表示多个对端和多轮通信。[局部修正提示词](prompts/06-allgather-reducescatter-edit.prompt.txt)。
5. **Parallel / Concur**：逐格核对 A 路 Mesh RS → NHR RS → NHR AG → Mesh AG，以及 B 路相反顺序；不把两份数据默认解释为等量。
6. **流水线**：核对同一片段始终先上层后下层，只有不同片段可重叠；右侧保留外层 loop 与 RS → AG 边界，不从 PipeLine 名字推断无条件重叠。

文字文档已按源码独立复核，新增跨算子部分另做一次复核。检查范围为静态实现与图示语义，不包含上板测量。
