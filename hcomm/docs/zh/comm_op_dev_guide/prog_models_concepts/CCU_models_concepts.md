# CCU编程模型与概念

## CCU架构

### 系统架构

CCU（Collective Communication Unit，集合通信加速单元）是昇腾NPU中的专用集合通信协处理器，位于IO Die上。下图展示了CCU在Ascend 950PR/Ascend 950DT中的位置：

![](figures/ccu_in_950.png)

### 基础概念

CCU内部包含多个关键组件，共同完成集合通信任务：

![](figures/ccu_arch.png)

- **片上缓存**：片上数据缓存，以4KB分片为基础操作单元，支持多片的片上归约操作。
- **寄存器**：分为通用寄存器、同步寄存器和地址寄存器。
  - **通用寄存器**：用于存放CCU执行过程中的参数或数据，支持简单赋值和加法运算。
  - **同步寄存器**：用于实现类似信号量的同步操作，支持Wait或Set操作。
  - **地址寄存器**：用于存放通信操作的内存地址；未来会取消，使用通用寄存器替代。

- **并发执行引擎**：提供CCU的并发与循环执行指令的能力。
- **指令空间**：存储CCU执行的指令序列。
- **Channel表**：存放多个Channel表项，每个表项包含用于UB通信的上下文信息。

### CCU典型用法与优势

在大规模分布式训练和推理业务中，集合通信的性能是影响整体系统性能的关键瓶颈之一。传统通信方式中，集合通信依赖AI CPU、AI Core等计算单元通过软件协议栈构造通信任务描述符，驱动硬件执行通信任务。这种执行方式需要占用计算核或计算算力，并且软件协议栈的调度开销较大，导致通信整体延迟较高。

为了解决上述问题，NPU引入了**集合通信加速单元（Collective Communication Unit，CCU）**，作为专用处理器加速集合通信任务。CCU可接收NPU调度器下发的集合通信任务，执行通信算法指令，完成跨NPU的同步、地址交换、数据传输以及归约（Reduce）运算，从而实现完整的集合通信能力。

CCU典型用法是通过使用片上缓存进行数据中转，以此降低内存访问，如图所示：

![](figures/ccu_typical_usage.png)

- **Reduce操作**：从远端读取数据到片上缓存，与本地内存数据进行归约，最终写回本地内存。

    如果不使用片上缓存，在对每个对端的数据进行归约时，都需要读写一次本地数据，再加上每个对端对本地数据的读取，共需要进行2\(n-1\)次读和n-1次写；如果使用片上缓存，那么只需要1次本地数据的读取以及1次归约后的写回，再加上每个对端对本地数据的读取，共需要进行n次读和1次写。

- **Broadcast操作**：类似地利用片上缓存减少内存访问次数

    同理，Broadcast可将n-1次读和n-1次写降低为1次读和n-1次写。

**CCU优势**：

1. 节省访存带宽

    在分布式训练和推理业务中，访存带宽是性能瓶颈。CCU结合集合通信算子的数据移动特征，通过内置独立的片上MS，将AllReduce等通信算子的访存需求降低约一个数量级，释放出更多访存带宽供计算算子使用，从而提升通信与计算的并发性能。

2. 提升精度归约与确定性

    集合通信中的归约运算存在精度损失和顺序不确定性问题。CCU利用独立片上MS和升精度归约单元，确保加法顺序和浮点数截断误差可控，有效提升归约运算的精度和确定性。

3. 低时延通信不占用计算核

    传统通信方式依赖AICPU/AI Vector核，占用计算资源。CCU通过硬件加速通信任务描述符的构造与下发，在降低通信时延的同时，不占用计算算力，为低时延业务提供独立的硬件支持。

## 资源抽象

| 资源类型 | 说明 | 对应资源 |
| --- | --- | --- |
| `ccu::Variable` | 变量 | 通用寄存器 |
| `ccu::Address` | 地址 | 地址寄存器 |
| `ccu::Event` | 事件 | 同步寄存器 |
| `ccu::CcuBuffer` | 片上缓存分片，大小为4KB | CCU片上缓存 |
| `ccu::LocalAddr` | 本地地址，用于通信操作 | 包含地址和token，分别由Address和Variable保存 |
| `ccu::RemoteAddr` | 远端地址，用于通信操作 | 包含地址和token，分别由Address和Variable保存 |

### 资源创建

- 单个资源创建：默认构造函数直接创建。
- 批量连续资源创建（Loop&LoopGroup操作对资源有连续性要求）：
  
  使用ccu::Array创建；注意使用原生数组创建资源（比如Variable vars\[10\]），不保证资源的连续性。

> [!NOTE]说明
>
> - CCU并发操作对资源的连续性有要求。
> - 使用原生数组创建资源（比如Variable vars\[10\]），不保证资源的连续性。

### 使用示例

```c
// 单个资源
ccu::Variable var; 
ccu::CcuBuffer buf;  
// 批量资源分配
ccu::Array<ccu::Variable> vars(10); 
ccu::Array<ccu::Event> events(5); 
```

## 数据搬运能力

如图所示，CCU具备以下数据搬运能力：

![](figures/ccu_datacopy.png)

- 本地内存 \<-\> Variable/Address寄存器
  - 从本地内存加载数据到寄存器。
  - 将寄存器内容保存到本地内存中。

- 本地内存 \<-\> CcuBuf
  - 从本地内存读取数据到CcuBuf。
  - 将CcuBuf数据写到本地内存。

- CcuBuf \<-\> 远端内存
  - 从远端内存读取数据到CcuBuf。
  - 将CcuBuf数据写到远端内存。

- 本地内存 \<-\> 远端内存
  - 读远端内存。
  - 写远端内存。

- 写远端寄存器
  - 写远端同步寄存器。
  - 写本地Variable数据到远端Variable。

## 计算能力

### 通用寄存器/地址寄存器

- 立即数赋值

    ```c
    // 通用寄存器赋值
    Variable x; x = 5;  
    // 地址寄存器赋值
    uint64_t ptr; 
    // ... 
    Address addr; 
    addr = ptr; 
    ```

- Variable赋值

    ```c
    Variable x, y;
    // ... 
    Variable v; 
    v = x;  
    Address addr; 
    addr = y; 
    ```

- 加法

    ```c
    Variable x=5, y=3, z;  
    z = x+y;  
    Address addr1, addr2;
    // ... 
    addr1 += x; 
    addr2 = addr1 + x; 
    ```

## CcuBuffer

支持最多8个CcuBuffer的按序Reduce计算，输入类型与输出类型一致，支持的Reduce操作：ADD、MAX、MIN，支持的数据类型：FP32、FP16、BF16、INT32、INT16、INT8和UINT8。

对于FP16/BF16的ADD操作，为避免精度损失，CCU会升精度为FP32类型进行归约计算，如下图所示：

![](figures/ccubuffer.png)

## 并发模型

- Loop：CCU任务中的基础并发单元。
  - Loop内的操作串行执行，不同Loop并行执行；
  - Loop可以循环执行内部操作，配合CcuBuffer循环搬运数据；
  - Loop参数：循环次数、每次循环操作的内存地址增量偏移；
  - 使用CcuBuffer搬运数据时，因大小有限，单次数据搬运数据量有限，利用多Loop并发可撑满链路带宽。

    ![](figures/ccu_loop.png)

- LoopGroup：将一组Loop复制多份，增加并发Loop个数。
  - Loop不能单独执行，需要放在LoopGroup中；
  - 对于LoopGroup中的Loop，LoopGroup可以指定从哪个Loop开始复制，以及复制的次数；
  - LoopGroup需要指定复制后每组Loop使用的资源id偏移。

    ![](figures/ccu_loopgroup.png)

- 使用示例

  ```c
  ccu::Func func([](){
      Write(chann, remoteMem, localCcuBuf, 4096);
  }); 
  ccu::Loop loop({4096, 10}, func);  // 循环执行10次，每次地址增量偏移为4096B 
  
  // 对LoopGroup中的Loop，从第0个Loop开始到最后一个，都复制5次
  ccu::LoopGroup loopGroup({5, 0, 4096*10, 10, 10}, {loop});
  ```

## 同步机制

CCU使用同步寄存器实现同步操作：

- Wait操作：等待掩码对应的寄存器比特置位才能返回。
- Record操作：置位掩码对应的寄存器比特位。

同步寄存器分为两种使用场景：

- 单个Device的本地操作，抽象为Event。
- 不同Device之间的通信操作，抽象为Notify。

### 本地事件

本地事件有两种使用场景：

- 异步操作可以传入event对象，通过EventWait可以等待异步操作完成。
- 不同CCU任务，使用EventRecord和EventWait，通过event对象实现同步。

使用示例：

```c
ccu::Event evt;  

ccu::EventRecord(evt, 0x01); 

ccu::EventWait(evt, 0x01); 
```

### 远端通知

两个通信实体建立Channel后，1）可以基于Channel中Notify序号写对端同步寄存器；2）可以基于Channel中Notify序号等待来自对端的通知信号。

| 接口 | 说明 |
| --- | --- |
| NotifyRecord(ch, notifyIdx, mask) | 向远端发送通知 |
| NotifyWait(ch, notifyIdx, mask) | 等待远端通知 |

使用示例：

```c
// 发送通知到远端
ccu::NotifyRecord(channelHandle, 0, 0x12);  
// 等待远端通知
ccu::NotifyWait(channelHandle, 0, 0x12);
```

## 流程控制

- 条件分支

  ```c
  CCU_IF(counter != limit) {
       // then-block} 
  CCU_ELSE {
       // else-block
  } 
  ```

- 循环

  ```c
  CCU_WHILE(counter != limit) {
       accumulator = accumulator + step;
       counter = counter + one; 
  } 
  ```

- 函数块

  | 类型/接口 | 说明 |
  | --- | --- |
  | `ccu::Func` | 函数块定义 |
  | `ccu::CallFunc<func>(...)` | 调用函数块 |

- 使用示例

  ```c
  // 定义函数块
  ccu::Func myFunc([](Variable x, Variable y){
     ...
  });  
  // 调用函数块
  Variable arg1, arg2;
  ...
  ccu::CallFunc<myFunc>(arg1, arg2); 
  ```

## 约束与限制

### 参数加载

- CCU任务描述符最多只支持13个64bit的参数。
- 参数加载指令LoadArg必须在程序最开始调用。

### 资源创建

资源创建出来后，不支持动态回收；比如在局部作用域中定义的资源，比如Variable，在局部作用域结束后Variable对象本身会销毁，但是对应的寄存器资源不会回收。

### 数据搬运

- 数据搬运类操作长度不能等于0，否则硬件行为不可预知。
- 本地内存到远端内存搬运，长度要小于256MB。
- 涉及CcuBuffer的数据搬运操作，长度要小于等于4096B。
- 不支持读写其他设备的CcuBuffer。

### 并发操作

- Loop内只能放置数据搬运指令和Wait指令，其他均不支持。
- Loop的循环次数必须大于0。
- 创建Loop和LoopGroup时，如果使用立即数作为参数，创建后的Loop和LoopGroup会消耗Variable资源用于保存配置参数。
- 创建Loop和LoopGroup前，需要提前计算好需要使用的资源并申请，避免越界。

LoopGroup可以将一组Group复制多份并发执行，每个Loop使用的资源不能冲突；因此LoopGroup除了需要指定复制次数，还需要指定复制后Loop使用的寄存器ID增量偏移，如下图所示，以Loop S为例，Loop S是开发者手写的，LoopGroup对它复制N份，那复制后Loop使用的CcuBuffer是在基础CcuBuffer之上做增量偏移。Loop S使用CcuBuffer x，LoopGroup指定CcuBuffer偏移参数为offset，那基于Loop S复制出来的第N个Loop使用的CcuBuffer x+N\*offset。在开发代码时，只能显式感知CcuBuffer x，那为了避免复制后的Loop使用的资源越界，需要规划好资源使用并提前申请。

### 指令部署

CCU位于一个设备的IO Die上。一个设备可能包含多个IO Die，不同IO Die包含不同的网络设备。实际组网中，不同的网络设备可能与不同的其他设备互联。一个CCU不能跨Die使用另一个IO Die的网络设备进行通信。

因此，在开发运行于某个CCU上的程序时，需要保证它使用的网络设备都在一个Die上，CCU翻译器会根据它使用的网络设备推导出翻译后的指令序列应该部署到哪个CCU设备上。
