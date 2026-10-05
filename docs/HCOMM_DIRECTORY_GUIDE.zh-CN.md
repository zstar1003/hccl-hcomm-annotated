# HCOMM目录与文件详细导读

HCOMM回答的是：**Rank怎样组成通信域，通信资源如何创建和复用，以及数据原语如何变成执行引擎上的任务？** 本仓同时包含通信域管理层L2和基础通信层L3。两层依赖自上而下；HCCL的L1算子通过动态符号适配消费这些能力。

本文针对仓内固定快照`87ce550f8f7c584e0ed89b0ec56699553d9c332d`。[HCCL导读](HCCL_DIRECTORY_GUIDE.zh-CN.md)说明算子选择与编排。目录中名为`hcom`的部分主要在历史兼容框架中，不能把HCOM目录直接等同于整个HCOMM项目。

## 完整的逐目录、逐文件页面

[HCOMM完整目录索引](catalog/HCOMM_DIRECTORY_INDEX.zh-CN.html)覆盖**8,220个受Git跟踪的文件以及所有祖先目录**，包括现行源码、legacy、接口、测试、工具、依赖、文档和隐藏工程配置。GitHub浏览HTML显示源文件，可使用Download raw file保存后在浏览器中离线打开。目录树按实际快照生成；此快照基础资源目录实际叫`resources/`，不将架构文档中的目标拼写`resource/`当作实际路径。

展开左侧树并点击任意层级的“查看”，右侧就会列出该目录的文件。可切换直接文件/包含子目录，按类型筛选，或搜索`HcclChannelAcquire`、`EndpointMgr`、`注册`、`Notify`、`ccu`等关键词。每个条目包含职责摘要、类别和代表性内容定位；点击类型/接口/测试名称进入源码对应行。搜索限制在当前目录，“显示全仓”会清空筛选。

核心文件提供人工职责摘要；其余摘要结合目录、文件类型、文档标题及内容中的代表性符号生成。它适合逐文件定位；没有抽取到符号的条目会提示直接核查原文件。目录导航覆盖全仓，不表示全部历史实现、硬件协议和算法已经逐函数检视。

## 第一层：全仓地图

```text
hcomm/
├── include/                 L2/L3公开API及CCU开发接口
├── pkg_inc/                 包间接口与历史兼容头文件
├── src/
│   ├── coll_communicator_mgr/L2：通信域、拓扑和资源管理
│   ├── base_comm/           L3：基础资源和数据原语
│   └── legacy/              Ascend910/Ascend950旧流程兼容
├── test/                    UT/ST、legacy验证、hccl_vm工具
├── examples/                通信域、图模式和自定义通信示例
├── experimental/            网络插件、诊断等试验扩展
├── external_depends/        外部组件依赖接口
├── python/                  Python绑定
├── docs/                    架构、API、资源/算子开发等资料
├── cmake/、scripts/         构建、依赖、安装打包与维护
├── .agents/                 Agent工作技能与检视规则
└── .gitcode/                协作模板和CI工作流
```

| 根文件 | 阅读用途 |
|---|---|
| [README.md](../hcomm/README.md) | 官方快照定位、接口和构建说明。 |
| [AGENTS.md](../hcomm/AGENTS.md) | 分层、控制面/数据面及legacy约束。 |
| [CMakeLists.txt](../hcomm/CMakeLists.txt) | 顶层目标与依赖组织。 |
| [build.sh](../hcomm/build.sh) | Host/Device打包与UT/ST构建入口。 |
| [LICENSE](../hcomm/LICENSE) | 项目许可入口；第三方文件按自身许可。 |

## 第二层：接口按消费者分开读

| 位置 | 面向谁、提供什么 | 关键文件 |
|---|---|---|
| `include/hccl/` | L2接口：通信域、Rank图、通信域内资源查询和申请。部分接口名字以Hccl开头，但实现属于HCOMM。 | [hccl_comm.h](../hcomm/include/hccl/hccl_comm.h)、[hccl_res.h](../hcomm/include/hccl/hccl_res.h)、[hccl_channel.h](../hcomm/include/hccl/hccl_channel.h)、[hccl_rank_graph.h](../hcomm/include/hccl/hccl_rank_graph.h) |
| `include/hcomm_*.h` | L3基础资源与数据原语，供算子/通信库开发。 | [hcomm_primitives.h](../hcomm/include/hcomm_primitives.h)、[hcomm_res.h](../hcomm/include/hcomm_res.h)、[hcomm_res_defs.h](../hcomm/include/hcomm_res_defs.h)、[hcomm_channel.h](../hcomm/include/hcomm_channel.h) |
| `include/ccu/` | CCU资源、变量、地址、事件、控制流与Kernel发射接口。 | [ccu_primitives.hpp](../hcomm/include/ccu/ccu_primitives.hpp)、[ccu_res.h](../hcomm/include/ccu/ccu_res.h)、[ccu_launch.h](../hcomm/include/ccu/ccu_launch.h) |
| `pkg_inc/hccl/`、`pkg_inc/hcomm/` | 跨软件包使用的资源、诊断等扩展接口，稳定性不等同于公开API。 | 按调用者追踪相关头文件。 |
| `pkg_inc/legacy/` | 历史兼容接口与框架头文件。 | 只在兼容路径阅读相应声明。 |

## 第三层：L2通信域管理coll_communicator_mgr

| 子目录 | 主要职责 | 如何继续深入 |
|---|---|---|
| `api_c_adpt/` | C接口边界：输入检查、句柄转换、新通信域与旧流程分流。 | 先从目标API进入对应`.cc`，然后跟踪通信域对象。 |
| `api_c_adpt/resource/` | Channel、Thread和通信域内存适配。 | `channel_c_adpt.cc`、`thread_c_adpt.cc`、`comm_mem_c_adpt.cc`。 |
| `api_c_adpt/dev/` | 设备侧接口适配。 | 设备侧句柄和资源如何表示。 |
| `communicator/` | 通信域对象与生命周期。 | Host/Device对象、创建及销毁调用。 |
| `communicator/device/` | 设备侧通信域和资源支持。 | 设备上下文和流/资源的关联。 |
| `communicator/group_schedule_mgr/` | 分组调度支持。 | 分组任务如何聚合和调度。 |
| `rank_graph/` | Rank表、物理拓扑、Rank图构建与查询。 | `rank_table_info/`→`phy_topo_builder/`→`rank_graph_builder/`，再看`rank_graph/`查询。 |
| `rank_info_detect/` | Rank信息检测与交换。 | 建域时本端信息怎样发送给其他Rank。 |
| `resource_mgr/` | 通信域内资源协调与复用。 | 继续展开本端/远端资源管理。 |
| `config_mgr/` | 通信域配置。 | 创建参数与运行时配置如何变成内部状态。 |
| `dfx/` | 维测、监控、恢复、性能和任务异常。 | `cluster_monitor/`、`ns_recovery/`、`profiling/`、`taskException/`。 |
| `common/`、`common/loggers/` | L2公共类型和日志支持。 | 保持与L3公共逻辑区分。 |
| `team/hccl/`、`team/hcomm/` | 对应Team接口与内部组织。 | 从相关公开Team接口追踪实现。 |

[coll_comm_res_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_res_c_adpt.cc)是从HCCL申请资源时的重要入口；[exchange_info_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/exchange_info_c_adpt.cc)登记和消费算子一致性信息。两者负责控制信息和资源，不是用户张量搬运实现。

## 第四层：资源管理内部，哪些对象持有什么

```text
resource_mgr/
├── local/
│   └── my_rank/
│       ├── my_rank.cc/.h       本Rank资源协调与建链
│       ├── endpoints/          Endpoint缓存与内存注册
│       └── …                   本端其他资源子模块，完整树见索引
└── remote/
    └── rank_pairs/             远端Rank对、Channel管理与AICPU接口
```

| 核心文件 | 功能小段的阅读重点 |
|---|---|
| [my_rank.cc](../hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc) | 规范描述→准备Socket→取得Endpoint并注册/选择内存→创建/复用Channel→按需等待连接→一致性交换→引擎句柄整理。共享队列tag的分流则从`coll_comm_res_c_adpt.cc`继续追踪。 |
| [endpoint_mgr.cc](../hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints/endpoint_mgr.cc) | 根据端点描述和tag复用端点；用内存版本判断是否需要注册；按tag选MemHandle；解注册内存后再释放关联端点。 |
| [channel_manager.cc](../hcomm/src/coll_communicator_mgr/resource_mgr/remote/rank_pairs/channel_manager.cc) | 远端Rank对Channel管理。资源如何被远端描述和缓存，要从调用者继续核查。 |
| [channel_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/resource/channel_c_adpt.cc) | 向上返回Channel通知容量、远端CCL和注册内存信息。 |
| [thread_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc) | 申请Thread及把用户Stream包装为执行上下文，包含引擎/TS枚举转换。 |
| [comm_mem_c_adpt.cc](../hcomm/src/coll_communicator_mgr/api_c_adpt/resource/comm_mem_c_adpt.cc) | 获取通信域内CCL及登记相关内存；返回地址不等同于把内存所有权交给算子。 |

这里的“复用”有不同粒度：通信域资源、Endpoint、Channel槽位、注册内存版本各有键与生命周期。只有先看缓存键和更新条件，才能判断下一次调用是否真的跳过创建/注册。

## 第五层：L3基础通信base_comm

| 子目录 | 负责什么 |
|---|---|
| `primitives/api_c_adpt/` | 数据与资源C接口适配；按CPU/AICPU_TS/CCU等路径转换句柄、检查参数和进入实现。 |
| `primitives/api_c_adpt/ccu/` | CCU原语、资源适配和Kernel Launch。 |
| `primitives/aicpu/` | AICPU任务缓存适配。 |
| `primitives/dfx/` | 基础原语的诊断适配。 |
| `resources/endpoints/` | 各类通信端点的基础实现。 |
| `resources/endpoint_pairs/` | 管理本端/远端端点对及通道创建和更新。 |
| `resources/endpoint_pairs/channels/` | 通道实现、连接/交换状态及共享jetty支持；引擎子目录含`host/`、`aicpu/`、`aiv/`、`ccu/`等。 |
| `resources/reged_mems/` | 注册内存实现与描述。 |
| `resources/comm_engine_res/` | 引擎资源组织。 |
| `resources/southbound_adpt/` | 底层运行时/驱动适配。 |
| `resources/hccp/` | 网络代理、底层网络接口及相关依赖。 |
| `resources/ccu/` | CCU设备、实例、通道、表示、微码与执行资源。 |
| `config_mgr/`、`common/`、`dfx/` | 基础层配置、公共支持和维测。 |

[endpoint_pair.cc](../hcomm/src/base_comm/resources/endpoint_pairs/endpoint_pair.cc)根据Engine与槽位创建/复用Channel。它连接“资源需求”与具体通道对象；[hcomm_channel_c_adpt.cc](../hcomm/src/base_comm/primitives/api_c_adpt/hcomm_channel_c_adpt.cc)则是基础Channel接口适配边界。

[aicpu_ts_primitives_c_adpt.cc](../hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc)包含LocalCopy/LocalReduce、Channel Read/Write、通知等待/记录、任务提交与通信域占用保护。新设备与旧兼容分支不同；`StreamLite`承载下发任务，远端数据接口依赖Channel恢复出的传输对象。`size`按字节传递的接口与归约`count`按元素传递的接口要分别阅读。

CCU内部可按`ccu_representation/`及其中的`reps/{arithmetic,control,data,loop,sync}/`、`reps/translator/`追踪表示与转换，再看`ccu_microcode/`及`ccu_microcode_opt/`的指令组织，结合`ccu_device/`、`ccu_instance/`和对应发射接口理解资源生命周期。这条链与AICPU逐次排入Read/Write的组织方式不同，应按引擎分开读。

## legacy为什么体量很大

`src/legacy/ascend910/`保留历史算法、框架、平台资源和共享头文件；`framework/hcom/`位于这一历史框架路径。`src/legacy/ascend950/`保留Ascend950旧流程，含框架、service和unified_platform等模块。旧算法目录里的executor/template或历史transport不能直接解释为新流程同名接口的实现。

遇到入口中的版本/设备分流时，先确认条件，再沿实际命中的legacy路径阅读。legacy只承担兼容维护；现行新资源/原语主链从`coll_communicator_mgr/`和`base_comm/`开始。完整索引仍逐个记录历史文件，便于核查兼容分支。

## 测试、工具、示例与依赖

`test/ut/`包含设备、DFX、实现及misc等测试；`test/ut/stub/`提供替身。`test/st/algorithm/`组织算法测试，`test/legacy/`对应历史流程验证。`test/hccl_vm/`包含虚拟执行、任务图及检查/可视化工具，工具前端文件也是索引的一部分。读取测试时要区分被测行为与模拟环境。

`examples/01_communicators/`展示每进程/每线程设备及Rank表建域；`examples/02_aclgraphs/`展示图模式相关用法。其余示例按实存树检索。`experimental/base_comm/nic_plugin/`包含网络插件扩展，`experimental/cluster_link_diag/`包含链路诊断试验；`external_depends/`保存依赖接口，`python/`提供Python层支持。随附`.o`等二进制条目说明其文件类型，不推断未解析的内部行为。

`docs/zh/`和`docs/en/`按语言组织架构、API参考、资源管理、通信算子开发、环境配置与诊断指南。接口文件回答“怎么调用”，实现回答“如何分流和管理”，测试回答“在哪些条件下检查哪些结果”，三者结合才形成完整引导。

## 新流程AICPU_TS AllToAll请求如何串起两个仓

1. HCCL入口检查计数/位移，selector选算法，`op_common.cc`汇总资源需求。
2. 经HCCL的dlsym适配进入HCOMM L2资源接口，获得CCL、Thread、Channel与拓扑信息。
3. `MyRank`协调Endpoint、内存注册和连接；L3的`EndpointPair`准备引擎对应Channel。
4. HCCL将执行参数下发到设备，模板按Peer与Channel展开切片与握手。
5. 模板调用HCOMM L3数据原语，在Thread上排入搬运/同步任务，并进行任务提交。
6. 完成依赖通知/等待协议和必要的后复制；控制面建链成功、任务提交成功、用户输出就绪是不同节点。

[阅读路线](READING_GUIDE.zh-CN.md)提供更细的API顺序；[注释覆盖清单](ANNOTATION_COVERAGE.md)说明本次细化17个文件的范围；[验证说明](VERIFICATION.md)解释如何证明没有改动原始代码文本。

## 维护与验证

执行`python3 scripts/build-directory-guides.py`重建两个完整索引，使用`--check`验证生成内容是否与当前受跟踪文件及符号位置一致。数据完全内嵌，搜索不请求网络；只有主动点击源码链接时打开GitHub。新增/删除文件或插入注释改变行号后要重新生成。该检查不等同于UT/ST或真实设备验证。
