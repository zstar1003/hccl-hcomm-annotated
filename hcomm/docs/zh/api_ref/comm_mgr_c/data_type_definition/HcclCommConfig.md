# HcclCommConfig

## 功能说明

初始化具有特定配置的通信域时，此数据类型用于定义通信域配置信息，包含缓存区大小、确定性计算开关、通信域名称、通信算子展开模式、通信算法、对称内存预留等多个配置项。

## 定义原型

```c
const uint32_t HCCL_COMM_CONFIG_INFO_BYTES = 24;
const uint32_t COMM_NAME_MAX_LENGTH = 128;
const uint32_t BUFFER_NAME_MAX_LENGTH = 128;
const uint32_t UDI_MAX_LENGTH = 128;
const uint32_t HCCL_COMM_ALGO_MAX_LENGTH = 1600;
const uint32_t HCCL_COMM_RETRY_ENABLE_MAX_LENGTH = 50;
const uint32_t HCCL_COMM_RETRY_PARAMS_MAX_LENGTH = 128;
typedef struct HcclCommConfigDef {
    char reserved[HCCL_COMM_CONFIG_INFO_BYTES];    /* 保留字段，不可修改 */
    uint32_t hcclBufferSize;
    uint32_t hcclDeterministic;
    char hcclCommName[COMM_NAME_MAX_LENGTH];
    char hcclUdi[UDI_MAX_LENGTH];
    uint32_t hcclOpExpansionMode;
    uint32_t hcclRdmaTrafficClass;
    uint32_t hcclRdmaServiceLevel;
    uint32_t hcclWorldRankID;
    uint64_t hcclJobID;
    uint8_t aclGraphZeroCopyEnable;
    int32_t hcclExecTimeOut;
    char hcclAlgo[HCCL_COMM_ALGO_MAX_LENGTH];
    char hcclRetryEnable[HCCL_COMM_RETRY_ENABLE_MAX_LENGTH];
    char hcclRetryParams[HCCL_COMM_RETRY_PARAMS_MAX_LENGTH];
    char hcclBufferName[BUFFER_NAME_MAX_LENGTH];
    uint32_t hcclQos;
    uint64_t hcclSymWinMaxMemSizePerRank;
    uint32_t hcclChannelSqDepth;
} HcclCommConfig;
```

## 参数说明

- **hcclBufferSize**：共享数据的缓存区大小，需要配置为整数，取值需大于等于1，单位为MB。取值范围和针对不同产品类型的使用约束请参见环境变量[HCCL_BUFFSIZE](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_BUFFSIZE.md)。

  需要注意：

  - 该配置项优先级高于HCCL_BUFFSIZE环境变量配置。
  - 该配置项申请的内存为HCCL独占，不可与其他业务内存复用。
  - 基于该配置项创建的通信域，会独占“2\*hcclBufferSize”大小的内存，保证多通信域并发算子互不影响。
  - 针对集合通信算子，当数据量超过该配置值时，可能会出现性能下降的情况，建议hcclBufferSize配置值大于数据量。

- **hcclDeterministic**：确定性计算开关，支持如下型号：

  下面分别列出不同AI处理器支持的取值及含义，未列出的代表不支持配置。

  <!-- npu="950" id1 -->
  - Ascend 950PR/Ascend 950DT：不支持此配置，可通过HCCL_DETERMINISTIC环境变量配置全局确定性计算开关。
  <!-- end id1 -->

  <!-- npu="910b,A3" id2 -->
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品和Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持的取值及含义如下：
    - 0（默认值）：代表关闭确定性计算。
    - 1：开启归约类通信算子的确定性计算，支持通信算子AllReduce、ReduceScatter、Reduce、ReduceScatterV。
    - 2：开启归约类通信算子的严格确定性计算，即保序功能（在确定性的基础上保证所有bit位的归约顺序均一致）。支持通信算子为AllReduce、ReduceScatter、ReduceScatterV，配置为该参数时需满足以下条件：
      - 仅支持多机对称分布场景，不支持非对称分布的场景。
      - 开启保序时，不支持饱和模式，仅支持INF/NaN模式。
      - 相较于确定性计算，开启保序功能后会产生一定的性能下降，建议在推理场景下使用该功能。
  <!-- end id2 -->

    > [!NOTE]说明
    > 在不开启确定性计算的场景下，多次执行的结果可能不同。这个差异的来源，一般是因为在算子实现中存在异步的多线程执行，会导致浮点数累加的顺序变化。当开启确定性计算后，算子在相同的硬件和输入下，多次执行将产生相同的输出。
    > 默认情况下，无需开启确定性计算，但当发现模型执行多次结果不同或者精度调优时，可以开启确定性计算辅助进行调试调优，但开启后，算子执行时间会变慢，导致性能下降。

- **hcclCommName**：通信域名称，最大长度为128。

  指定的通信域名称需确保与其他通信域中的名称不重复；不指定时由HCCL自动生成。

- **hcclUdi**：用户自定义信息，最大长度为128，默认为空。
- **hcclOpExpansionMode**：配置通信算子的展开模式，为通信域粒度的配置。

  下面分别列出不同AI处理器支持的取值及含义，未列出的代表不支持配置。

  <!-- npu="950" id4 -->
  **针对Ascend 950PR/Ascend 950DT，支持的取值及含义如下：**

  - 0：使用默认算子展开模式，针对**Ascend 950PR/Ascend 950DT**，通信算子默认在AI CPU计算单元展开。
  - 2：通信算子在AI CPU计算单元展开，使用STARS调度器调度运行。

    该配置项支持Broadcast、Reduce、AllReduce、Scatter、ReduceScatter、ReduceScatterV、AllGather、AllGatherV、AlltoAll、AlltoAllV、AlltoAllVC、Send、Recv、BatchSendRecv算子。

    图模式（Ascend IR）或者图捕获（aclgraph）场景，当通信算法采用AI CPU模式时，单卡上的并发图数量不能超过6个，否则可能会因AI CPU核被占满而导致通信阻塞。

  - 3：通信算子在Device侧的Vector Core计算单元展开。
    - 该配置仅支持对称组网、推理特性。
    - 该配置下，若数据量不满足在“Vector Core”上的运行要求，部分算子会自动切换到默认模式。
    - 该配置项仅支持Broadcast、Reduce、AllReduce、ReduceScatter、Scatter、AllGather、AlltoAll、AlltoAllV、AlltoAllVC算子，当前仅支持单机场景。
      - 针对Reduce、AllReduce、ReduceScatter算子，数据类型支持int8、int16、int32、int64、float16、float32、bfp16。

    - 该配置项下，支持控核能力，建议业务根据实际使用场景中计算算子与通信算子的并发情况进行Vector Core核数的配置。

  - 4：代表通信算子在Device侧的Vector Core计算单元展开，但不会随着数据量的变化进行模式切换，始终使用Vector Core计算，如果不满足Vector Core的运行条件，会报错退出。
    - 该配置仅支持对称组网、推理特性。
    - 该配置项支持的算子及约束限制参见配置“3”。

  - 5：通信算子在CCU（Collective Communication Unit，集合通信加速单元）展开，使用MS（Memory Slice）模式。Ascend 950PR不支持此配置。

    MS模式为与多个远端通信时，使用CCU片上Memory Slice作为中转，用于节省内存读写带宽，Memory Slice的特点是大小较小，但速度较快。当CCU资源不足时，系统会自动切换为“2：AI CPU模式”。
    - 该配置项仅支持Broadcast、Reduce、AllReduce、ReduceScatter、AllGather、AllGatherV、ReduceScatterV算子，当前仅支持单机场景。
      - 针对Broadcast、AllGather、AllGatherV算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。
      - 针对Reduce、AllReduce、ReduceScatter、ReduceScatterV算子，数据类型支持int16、int32、float16、float32、bfp16。
      
  - 6：通信算子在CCU展开，使用调度模式。

    调度模式指使用CCU作为调度器，向UB引擎调度UB WQE任务。调度模式下不使用CCU的片上MS，直接在两个rank间进行HBM到HBM的数据传输。

    针对单机通信场景的AllReduce、ReduceScatter、Reduce算子，当数据量超过一定值时，为防止性能下降，系统会自动切换为“2：AI_CPU模式”（该阈值并非固定，会根据算子运行模式及网络规模等因素有所调整）。

    此模式下，ReduceScatterV、AllGatherV算子仅支持单Server场景。

    当CCU资源不足时，系统会自动切换为“2：AI CPU模式”。
  <!-- end id4 -->

  <!-- npu="A3" id5 -->
  **针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的取值及含义如下：**

  - 0：使用默认算子展开模式，Atlas A3 训练系列产品/Atlas A3 推理系列产品默认使用Device侧的AI CPU计算单元。
  - 2：通信算子在AI CPU计算单元展开。
  - 3：通信算子在Device侧的Vector Core计算单元展开。
    - 该配置仅支持对称组网、推理特性。
    - 该配置下，若数据量不满足在“Vector Core”上的运行要求，部分算子会自动切换到默认模式。
    - 该配置项仅支持Broadcast、AllReduce、ReduceScatter、AllGather、AlltoAll、AlltoAllV、AlltoAllVC算子。
      - 针对Broadcast算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16、int64、uint64、float64，仅支持超节点内的单机/多机通信，仅支持单算子模式和Ascend IR图模式，不支持跨超节点间通信。
      - 针对AllReduce算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min，仅支持超节点内的单机/多机通信，不支持跨超节点间通信。
      - 针对ReduceScatter算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min，仅支持超节点内的单机/多机通信，不支持跨超节点间通信。
      - 针对AllGather、AlltoAll、AlltoAllV、AlltoAllVC算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16，仅支持超节点内的单机/多机通信，不支持跨超节点间通信。

    - 针对Broadcast、AllReduce、ReduceScatter、AllGather、AlltoAll（单机通信场景）算子，当数据量超过一定值时，为防止性能下降，系统会自动切换为“2：AI CPU模式”（该阈值并非固定，会根据算子运行模式、是否启动确定性计算及网络规模等因素有所调整）；针对AlltoAllV、AlltoAllVC、AlltoAll（多机通信场景）算子，系统不会自动切换为“2：AI CPU”模式，为避免性能劣化，当任意两个rank之间的最大通信数据量不超过1MB时，建议配置为“3：AIV模式”，否则请采用“2：AI CPU模式”。
    - 该配置项下，集合通信支持控核能力，建议业务根据实际使用场景中计算算子与通信算子的并发情况进行Vector Core核数的配置。

      - 针对Broadcast算子，建议至少分配ranksize个vector核。
      - 针对AllGather、非确定性ReduceScatter算子，建议最少分配max\(2, ceil\(ranksize/20\)\)个vector核。
      - 针对AllReduce、确定性ReduceScatter、AlltoAll、AlltoAllV、AlltoAllVC算子，建议最少分配max\(2, ceil\(ranksize/20\)\)个vector核，且核数需为偶数（若计算结果为奇数则向上取整至下一个偶数）。

      若业务编译分配的Vector Core核数无法满足算法编排的要求，HCCL会报错并提示所需要的最低Vector Core核数。

  - 4：代表通信算子在Device侧的Vector Core计算单元展开，但不会随着数据量的变化进行模式切换，始终使用Vector Core计算，如果不满足Vector Core的运行条件，会报错退出。
    - 该配置仅支持对称组网、推理特性。
    - 该配置项支持AllReduce、ReduceScatter、AllGather、AlltoAll、AlltoAllV、AlltoAllVC算子。相关算子支持的数据类型及场景限制参见配置“3”。
    - 该配置项下，集合通信支持控核能力，不同算子的Vector Core核数要求与配置“3”相同。
  <!-- end id5 -->

  <!-- npu="910b" id6 -->
   **针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的取值及含义如下：**

  - 0：使用默认算子展开模式，Atlas A2 训练系列产品/Atlas A2 推理系列产品默认使用Host侧CPU。
  - 1：通信算子在Host侧CPU展开。
  - 2：通信算子在AI CPU计算单元展开。

    该配置项仅支持AllGather、AlltoAll、AlltoAllV、AlltoAllVC算子。

    图模式（Ascend IR）或者图捕获（aclgraph）场景，当通信算法采用AI CPU模式时，单卡上的并发图数量不能超过6个，否则可能会因AI CPU核被占满而导致通信阻塞。

  - 3：通信算子在Device侧的Vector Core计算单元展开。
    - 该配置仅支持对称组网、推理特性。
    - 该配置下，若数据量不满足在“Vector Core”上的运行要求，部分算子会自动切换到默认模式。
    - 该配置项仅支持Broadcast、AllReduce、AlltoAll、AlltoAllV、AlltoAllVC、AllGather、ReduceScatter、AllGatherV、ReduceScatterV算子。
      - 针对Broadcast算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16，仅支持单机场景8卡以内的单算子模式。
      - 针对AllReduce算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min。
      - 针对AlltoAll、AlltoAllV、AlltoAllVC算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16。针对AlltoAllV、AlltoAllVC算子，仅支持单机场景；针对AlltoAll算子的图模式运行方式，仅支持单机场景。
      - 针对AllGather算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16。针对该算子的图模式运行方式，仅支持单机场景。
      - 针对ReduceScatter算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min。针对该算子的图模式运行方式，仅支持单机场景。
      - 针对AllGatherV算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16，仅支持单算子模式。
      - 针对ReduceScatterV算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min。

    - 该配置项下，集合通信支持控核能力，建议业务根据实际使用场景中计算算子与通信算子的并发情况进行Vector Core核数的配置。

      - 针对AllReduce、ReduceScatter、ReduceScatterV算子，建议最少分配24个核。
      - 针对Broadcast、AlltoAll、AlltoAllV、AlltoAllVC、AllGather、AllGatherV算子，建议最少分配16个核。

      若业务编译分配的Vector Core核数无法满足算法编排的要求，HCCL会报错并提示所需要的最低Vector Core核数。

  - 4：代表通信算子在Device侧的Vector Core计算单元展开，但不会随着数据量的变化进行模式切换，始终使用Vector Core计算，如果不满足Vector Core的运行条件，会报错退出。
    - 该配置仅支持对称组网、推理特性。
    - 该配置项仅支持AllReduce、AlltoAll、AlltoAllV、AlltoAllVC、AllGather、ReduceScatter算子。相关算子支持的数据类型及场景限制参见配置“3”。
    - 该配置项下，集合通信支持控核能力，不同算子的Vector Core核数要求与配置“3”相同。
  <!-- end id6 -->

    > [!NOTE]说明
    > - 多通信域并行场景下，不支持多个通信域同时配置为“3”或“4”（AIV Only模式）。
    > <!-- npu="910b" id7 -->
    > - 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，通信算子展开模式设置为“3”或“4”时，同时设置hcclDeterministic配置为“1”（开启确定性计算），在单机的单算子和图模式场景下，当数据量≤8MB时，仅AllReduce和ReduceScatter算子的确定性计算生效，其他场景和算子则以hcclDeterministic配置为准。
    > - 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，若hcclDeterministic配置为“2”（开启保序功能），hcclOpExpansionMode不支持配置为“3”或“4”，以保序功能为准。
    > <!-- end id7 -->
    > <!-- npu="A3" id8 -->
    > - 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，通信算子展开模式设置为“3”或“4”时，若同时设置hcclDeterministic为“1”（开启确定性计算）或“2”（开启保序功能），当数据量＜8MB时，仅AllReduce和ReduceScatter算子的确定性计算生效，其他场景和算子则以hcclDeterministic配置为准。
    > <!-- end id8 -->

- **hcclRdmaTrafficClass**：配置RDMA网卡的traffic class，取值范围为\[0,255\]，需要配置为4的整数倍。

  在RoCE V2协议中，该值对应IP报文头中ToS（Type of Service）域段。共8个bit，其中，bit\[0,1\]固定为0，bit\[2,7\]为DSCP，因此，该值除以4即为DSCP的值。

  **注意事项：**

  - 0xFFFFFFFF被用作优先级判断标识，当配置为0xFFFFFFFF时，此通信域配置无效，会按照优先级取环境变量配置或默认值132。

- **hcclRdmaServiceLevel**：配置RDMA网卡的service level，取值需要和网卡配置的PFC优先级保持一致，若配置不一致可能导致性能劣化。

  需要配置为无符号整数，取值范围\[0,7\]。

  **注意事项：**

  - 0xFFFFFFFF被用作优先级判断标识，当配置为0xFFFFFFFF时，此通信域配置无效，会按照优先级取环境变量配置或默认值4。

- **hcclWorldRankID**：NSLB-DP（Network Scale Load Balance-Data Plane：数据面网络级负载均衡）场景使用字段，代表当前进程在AI框架（如Pytorch）中的全局rank ID。

  **注意事项：**

  - Ascend 950PR/Ascend 950DT不支持此配置。

- **hcclJobID**：NSLB-DP场景使用字段，代表当前分布式业务的唯一标识，由AI框架生成。

  **注意事项：**

  - Ascend 950PR/Ascend 950DT不支持此配置。

- **aclGraphZeroCopyEnable**：该参数仅在图捕获模式（aclgraph）下对Reduce类算子生效，用于控制其是否开启零拷贝功能。
  - 0（默认值）：关闭零拷贝功能。
  - 1：开启零拷贝功能。

  **注意事项：**

  - Ascend 950PR/Ascend 950DT不支持此配置。

- **hcclExecTimeOut**：不同设备进程在分布式训练或推理过程中存在卡间执行任务不一致的场景（如仅特定进程会保存checkpoint数据），通过该参数可控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步。单位为s，取值范围和针对不同产品类型的使用约束请参见环境变量[HCCL_EXEC_TIMEOUT](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_EXEC_TIMEOUT.md)。

  **配置示例：**

    ```text
    # 整数秒配置，配置为1800s
    hcclExecTimeOut = 1800
    # 十毫秒级精度配置，配置为50ms
    hcclExecTimeOut = 0.05
    ```

  **注意事项：**

  - 0xFFFFFFFF被用作优先级判断标识，当配置为0xFFFFFFFF时，此通信域配置无效，会按照优先级取环境变量配置或默认值1836。
  - Ascend 950PR/Ascend 950DT不支持此配置，可通过HCCL_EXEC_TIMEOUT环境变量配置全局超时时间。

- **hcclAlgo**：用于配置集合通信Server间通信算法以及超节点间通信算法，支持全局配置算法类型与按算子配置算法类型两种配置方式。需注意，HCCL提供自适应算法选择功能，默认会根据产品形态、数据量和Server个数选择合适的算法，一般情况下用户无需手工指定。若通过此参数指定了Server间通信算法，则自适应算法选择功能不再生效。

  配置方式的参数信息及不同产品类型支持的算法类型请参见环境变量[HCCL_ALGO](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_ALGO.md)。

- **hcclRetryEnable**：用于配置是否开启HCCL算子的重执行特性。重执行是指当通信算子执行报SDMA或者RDMA CQE类型的错误时，HCCL会尝试重新执行此通信算子。**仅支持在Atlas A3 训练系列产品/Atlas A3 推理系列产品上使用。**

  通过此参数，开发者可以在Server间、超节点间两个物理层级的通信域中配置是否开启重执行特性，每个层级支持配置两种状态：开启或关闭，使用约束请参见环境变量[HCCL_OP_RETRY_ENABLE](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_OP_RETRY_ENABLE.md)，配置方式为：`hcclRetryEnable = "L1:1, L2:0"`，参数取值如下。

  - L1代表通信域的物理范围为Server间通信域，取值为0表示通信域内Server间通信task不开启重执行，取值为1表示通信域内Server间通信task开启重执行，默认值为0。
  - L2代表通信域的物理范围为超节点间通信域，取值为0表示通信域内超节点间通信task不开启重执行，取值为1表示通信域内超节点间通信task开启重执行，默认值为0。

- **hcclRetryParams**：只有当开发者通过参数**hcclRetryEnable**开启了HCCL的算子重执行特性时，可通过本参数配置第一次重执行的等待时间、最大重执行的次数以及两次重执行的间隔时间。**仅支持在Atlas A3 训练系列产品/Atlas A3 推理系列产品上使用。**

  使用约束请参见环境变量[HCCL_OP_RETRY_PARAMS](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_OP_RETRY_PARAMS.md)。配置方式为`hcclRetryParams = "MaxCnt:3, HoldTime:5000, IntervalTime:1000"`，参数取值如下：

  - MaxCnt：最大重传次数，uint32类型，取值范围为\[1,10\]，默认值为1，单位次。
  - HoldTime：从检测到通信算子执行失败到开始第一次重新执行的等待时间，uint32类型，取值范围\[0,60000\]，默认值为5000，单位ms。
  - IntervalTime：同一个通信算子两次重执行的间隔时间，uint32类型，取值范围\[0,60000\]，默认值为1000，单位ms。

- **hcclBufferName**：CCLBuffer名称，多通信域使用同一Buffer名称，共享同一片CCLBuffer，不指定时默认不共享，最大长度为128。需注意，传入同一CCLBuffer名称的通信域，需将算子下发到同一条Stream上。

  **注意事项：**

  - Ascend 950PR/Ascend 950DT不支持此配置。

- **hcclQos**：用于配置超平面QoS的级别，取值范围：\[0, 7\]。

   针对Ascend 950PR/Ascend 950DT，默认值为：4
   针对Atlas A3 训练系列产品/Atlas A3 推理系列产品和Atlas A2 训练系列产品/Atlas A2 推理系列产品，默认值为：6

- **hcclSymWinMaxMemSizePerRank**：Atlas A3 训练系列产品/Atlas A3 推理系列产品的HCCS场景下，为当前通信域中每个rank预留的对称内存大小，单位GB，取值范围：\[1, 当前环境中允许分配的物理内存最大值\]，默认值16。该参数仅在Atlas A3 训练系列产品/Atlas A3 推理系列产品的HCCS场景下生效。Ascend 950PR/Ascend 950DT的URMA场景使用已申请的Device内存注册对称内存窗口，不依赖该参数配置预留的对称内存大小。

- **hcclChannelSqDepth**：用于配置通信发送队列（SQ）深度，目前仅在A5场景下支持，且仅在AIV展开模式和UB_CTP、UB_RTP协议下可配置。配置值会向上调整为最近的2的整数次幂，调整后的有效取值范围为[16, 设备的max_jfs_depth]。其余场景下保持为无效值。


## 配置优先级说明

以上配置为通信域级别的配置，对于部分参数，HCCL提供了全局级别的环境变量配置，优先级如下：

- 通信域级别（HcclCommConfig）高于环境变量。

  若在HcclCommConfig中配置了某参数，则以该配置值为准。

- 环境变量优先级次之。

  若未在HcclCommConfig中配置对应参数，但设置了环境变量，则使用环境变量的值。

- 默认值最后生效。

  若HcclCommConfig和环境变量均未配置，则使用下列表格中列出的默认值。

**表1**  配置优先级说明详表

| 配置项 | 配置优先级 |
| --- | --- |
| hcclBufferSize | 配置项hcclBufferSize（通信域粒度配置）> 环境变量[HCCL_BUFFSIZE](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_BUFFSIZE.md)（全局配置）> 默认值200。 |
| hcclDeterministic | 配置项hcclDeterministic（通信域粒度配置）> 环境变量[HCCL_DETERMINISTIC](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_DETERMINISTIC.md)（全局配置）> 默认确定性开关（同环境变量HCCL_DETERMINISTIC默认值）。 |
| hcclOpExpansionMode | 配置项hcclOpExpansionMode（通信域粒度配置）> 环境变量[HCCL_OP_EXPANSION_MODE](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_OP_EXPANSION_MODE.md)（全局配置）> 默认算子展开模式（同环境变量HCCL_OP_EXPANSION_MODE默认值）。|
| hcclRdmaTrafficClass | 配置项hcclRdmaTrafficClass（通信域粒度配置） > 环境变量[HCCL_RDMA_TC](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_RDMA_TC.md)（全局配置）> 默认值132。 |
| hcclRdmaServiceLevel | 配置项hcclRdmaServiceLevel（通信域粒度配置）> 环境变量[HCCL_RDMA_SL](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_RDMA_SL.md)（全局配置）> 默认值4。 |
| hcclExecTimeOut | 配置项hcclExecTimeOut（通信域粒度配置）> 环境变量[HCCL_EXEC_TIMEOUT](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_EXEC_TIMEOUT.md)（全局配置）> 默认超时时间（同环境变量HCCL_EXEC_TIMEOUT默认值）。 |
| hcclAlgo | 配置项hcclAlgo（通信域粒度配置）> 环境变量[HCCL_ALGO](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_ALGO.md)（全局配置）> 自适应选择算法。 |
| hcclRetryEnable | 配置项hcclRetryEnable（通信域粒度配置）> 环境变量[HCCL_OP_RETRY_ENABLE](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_OP_RETRY_ENABLE.md)（全局配置）> 默认值0。 |
| hcclRetryParams | 配置项hcclRetryParams（通信域粒度配置）> 环境变量[HCCL_OP_RETRY_PARAMS](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/hccl_env/HCCL_OP_RETRY_PARAMS.md)（全局配置）> 默认配置（MaxCnt：1，HoldTime：5000，IntervalTime：1000）。 |
