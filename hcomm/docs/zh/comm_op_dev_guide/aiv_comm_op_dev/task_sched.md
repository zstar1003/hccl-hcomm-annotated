# 任务编排

## 编排步骤

参与集合通信的各个rank协调有序地进行同步与数据搬运，完成一个集合通信操作的过程，称为任务编排。任务编排的主要目的是将不同通信线程上的任务并行执行，实现资源的最大化利用，从而提升整体性能。

针对AIV通信算子，主要基于Ascend C编程语言提供的数据拷贝接口和同步接口来完成算法编排，用户也可以针对通信场景的数据搬运和同步操作继续封装高阶接口以便于多次调用。

在Ascend C编程框架中，AI Core能够访问的外部存储和内部存储分别称为Global Memory和Local Memory，GlobalTensor数据结构用来存放Global Memory的全局数据，LocalTensor数据结构用于存放AI Core中Local Memory的数据。无论是软同步设置、标记位检查还是数据拷贝，底层均使用Ascend C的DataCopy相关接口，该接口支持在GlobalTensor和LocalTensor之间进行双向的数据拷贝，但不同场景的数据流向不一样：

- 软同步Record，标记值在LocalTensor中设置后，通过DataCopy拷贝到GlobalTensor。
- 软同步Wait，标记值在GlobalTensor中设置，拷贝标记值到LocalTensor后，进行标量的比较，如果为预期值，则往下执行，否则重新执行拷贝与比较操作。
- 数据搬运，源地址和目的地址均在GlobalTensor中，需要经过Vector Core的内部存储空间中转，先通过DataCopy将源GlobalTensor拷贝到LocalTensor，再拷贝到目的GlobalTensor。

如果是Reduce类算子，还可通过Ascend C的原子操作类API开启LocalTensor到GlobalTensor数据传输的随路Reduce功能，比如当Reduce类型为SUM时可调用SetAtomicAdd接口，为MAX时可调用SetAtomicMax接口。

具体而言，AIV算子的任务编排主要包含以下几个步骤：

1. 获取本端通信内存，在HCCL中称其为HCCL Buffer。

   > [!NOTE]说明
   > HCCL Buffer是HCCL每个通信域所管理的一块Device上的锁页内存（Pinned Memory），由于通信任务是异步执行的，为确保用户输入的数据在通信任务实际执行时依然有效，因此需要先将输入数据拷贝到拥有固定内存地址的HCCL Buffer中。

2. 切分输入数据，计算偏移量。

    HCCL Buffer默认大小是200MB，若输入数据大小超过该值，则需要切分成多个数据块分别进行处理，需要在Host侧进行多次算子Kernel下发，每次任务编排只针对其中1个数据块。

3. 拷贝算子输入数据到HCCL Buffer，常用的Ascend C接口为DataCopy和DataCopyPad等。
4. 前同步，与其他rank同步确认通信准备是否就绪。常用的Ascend C接口为DataCopy、PipeBarrier、SetFlag和WaitFlag等。
5. 数据搬运，将远端数据拷贝至本端的HCCL Buffer中。常用的Ascend C接口为DataCopy和DataCopyPad等。
6. 后同步，与其他rank同步确认数据传输完成。
7. 拷贝HCCL Buffer中的结果数据到算子输出内存。

## 代码示例

以自定义AllGather算子为例，其在Vector Core侧的任务编排代码片段如下：

其中CpGM2GM基于DataCopy和DataCopyPad封装了支持不对齐数据搬运的片上内存间拷贝接口，Record1vN和WaitNv1基于DataCopy、SetFlag和WaitFlag封装了rank间同步的接口。

```c
if (GetBlockIdx() != rank) {
    // rank间同步，每个核等待其他rank数据就绪
    WaitNv1(tag);
    PipeBarrier<PIPE_ALL>();
    // 读取对端rank的数据到本rank的输出内存
    CpGM2GM(outputGM + GetBlockIdx() * count, cclGMOther, count);
    // 与rank号为GetBlockIdx()的对端做后同步
    Record(GetBlockIdx(), tag);
    Wait(GetBlockIdx(), tag);
} else {
    // 1个核将本rank输入内存上的数据拷贝到中转内存，供其他rank读取
    CpGM2GM(cclGMSelf, inputGM, count);
    PipeBarrier<PIPE_ALL>();
    // rank间同步，通知其他rank来读取数据
    Record1vN(tag);
    // 1个核将本rank中转内存上的数据拷贝到用户输出内存
    CpGM2GM(outputGM + count * rank, cclGMSelf, count);
}
```
