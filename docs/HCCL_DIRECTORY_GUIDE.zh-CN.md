# HCCL目录与文件详细导读

HCCL回答的是：**一次集合通信如何选择算法、安排数据切片，并把操作下发到相应执行引擎？** 它位于算子层L1，向上提供AllGather、AllReduce等API，向下经动态符号适配使用HCOMM的通信域、资源和数据原语。

本文针对仓内固定快照`170ddeec539b4d693028ce6e0cf5c58933e4d46d`。实存目录和架构文档中的“目标目录”可能不同，下面按实存目录说明。[HCOMM导读](HCOMM_DIRECTORY_GUIDE.zh-CN.md)解释下层实现。

## 完整的逐目录、逐文件页面

打开[HCCL完整目录索引](catalog/HCCL_DIRECTORY_INDEX.zh-CN.html)，下载后用浏览器打开即可离线检索。GitHub浏览HTML时显示源文件，可使用Download raw file保存到本地。页面覆盖**1,828个受Git跟踪的文件以及所有祖先目录**，包括隐藏的治理目录，不省略测试、示例、文档、图片和构建配置。

使用方式：展开左侧目录树，点击“查看”选择任意层级；勾选“包含子目录”阅读整个模块，取消则只看这一层的直接文件。右侧逐文件显示职责摘要、类别及从内容抽取的代表性类型/接口/测试符号；点击符号跳到源码对应行。搜索可使用`AllGather`、`HcclExecOp`、`资源`、`mesh`或完整路径。搜索只作用于当前选定目录，必要时点击“显示全仓”。

核心文件的职责摘要经过人工梳理；其余条目结合目录职责、文件类型、文档标题及代表性内容符号生成，是文件级导航。没有抽取到内容符号的文件会明确提示打开原文件核查，不能将目录摘要当作所有函数和算法都已经逐一审计。

## 第一层：从工程入口看全仓

```text
hccl/
├── include/        公开算子与MC2头文件
├── src/
│   ├── ops/        按算子组织：入口、选择器、执行器、模板
│   ├── common/     配置、校验、动态加载、图和框架适配
│   └── algo_plugin/算法插件接口和加载
├── experimental/   试验算子、递归执行器和生态插件
├── examples/       基础通信、集合通信与自定义算子示例
├── test/           UT、算法ST及模拟执行支撑
├── docs/           中文/英文接口、构建、架构和开发资料
├── cmake/          构建目标、依赖和设备侧配置
├── scripts/        构建安装及版本维护脚本
├── .agents/        Agent工作技能与检视规则
└── .gitcode/       Issue/PR模板、CI工作流与辅助脚本
```

| 工程文件 | 作用与阅读时机 |
|---|---|
| [README.md](../hccl/README.md) | 官方快照的项目定位、接口能力和构建入口，先读它了解支持范围。 |
| [AGENTS.md](../hccl/AGENTS.md) | 架构依赖、编码、构建与贡献约束；改源码前阅读。 |
| [CMakeLists.txt](../hccl/CMakeLists.txt) | 顶层构建组织。是否真正编入产物，要顺着此文件及下级CMake看，不能仅因目录存在就认定会编译。 |
| [build.sh](../hccl/build.sh) | 构建脚本入口，接收打包、UT/ST、静态库等选项。 |
| [LICENSE](../hccl/LICENSE) | 本项目许可；依赖组件仍按各自许可处理。 |

## 第二层：公开接口与算子实现

[include/hccl.h](../hccl/include/hccl.h)声明L1算子接口。[include/hccl_mc2.h](../hccl/include/hccl_mc2.h)提供MC2自定义通信相关接口和参数。`Hccl*`前缀不能单独说明实现属于哪个仓：通信域与部分资源API的实现位于HCOMM。

`src/ops/`下每个算子目录通常包含入口`.cc/.h`、`selector/`和`algorithm/`，部分算子还有`op_graph/`。例如AllToAll、AllToAllV、AllToAllVC共用`all_to_all_v/`，不能只按目录名称推断只实现一种API。

| 算子目录 | 数据语义 | 优先入口 |
|---|---|---|
| `all_gather/` | 每个Rank收集所有Rank的等长输入。输出总量为本Rank输入的RankSize倍。 | [all_gather.cc](../hccl/src/ops/all_gather/all_gather.cc) |
| `all_gather_v/` | 各Rank输入长度可不同，收集结果需要变长描述。 | `all_gather_v`入口和选择器 |
| `all_reduce/` | 对相同位置的数据归约，各Rank得到完整结果。 | [all_reduce.cc](../hccl/src/ops/all_reduce/all_reduce.cc) |
| `all_to_all_v/` | 按Peer交换切片，等长、变长与计数矩阵入口共享实现组织。 | [all_to_all_v.cc](../hccl/src/ops/all_to_all_v/all_to_all_v.cc) |
| `reduce_scatter/`、`reduce_scatter_v/` | 归约后各Rank只取对应分片，V版本允许变长。 | 对应入口、分片计算及模板 |
| `broadcast/`、`reduce/`、`scatter/` | 分别从Root广播、归约到Root、从Root分发。 | 对应入口及Root处理 |
| `send/`、`recv/`、`batch_send_recv/` | 点对点发送、接收和批量收发。 | 对应Peer参数和配对关系 |
| `barrier/` | Rank间同步屏障。 | 入口及算法模板 |
| `interface_graph_mode/` | 图模式公共接口组织。 | 图资源如何传入算子 |

## 第三层：一个算子目录内部怎样分工

以`src/ops/all_to_all_v/`为例：

```text
all_to_all_v/
├── all_to_all_v.cc/.h           API入口、校验、OpParam组装
├── selector/                   选择算法与匹配拓扑/引擎
├── algorithm/
│   ├── executor/               组合算法层级与模板，计算资源需求
│   └── template/
│       ├── aicpu/              AICPU指令模板
│       ├── aiv/                AIV算法模板
│       └── ccu/                CCU算法模板/上下文
└── op_graph/                   图模式适配
```

入口负责把用户参数变成公共描述：`count`通常是元素数，乘`dataTypeSize`后才得到字节数；AllToAllV的计数和位移描述按Peer组织。选择器决定使用哪个算法，但不在这里完成数据传输。执行器组合模板和资源；模板才根据Peer、Channel和Thread组织搬运与同步。各算子实存子目录可能不同，逐层展开完整索引查看。

[ins_temp_all_to_all_v_mesh_1D.cc](../hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc)是本次细化注释的代表：先计算资源，随后分Peer轮次，再按Channel切分长度/偏移；普通Read路径要准备发送CCL槽位，普通Write路径收到CCL后要复制到用户输出。远端用户内存访问分支使用不同地址来源。顺着功能段读，持续检查“当前处理哪个Peer、哪个槽位、哪条Thread、数据落在哪里”。

## 第四层：公共执行枢纽op_common

| 目录或文件 | 具体职责 | 接下来追踪什么 |
|---|---|---|
| [op_common.cc](../hccl/src/ops/op_common/op_common.cc) | `HcclExecOp`及公共资源/执行流程；协调拓扑、缓存、资源与Engine分支。 | 资源查询是否命中、哪个Engine被选中、Host向Device传什么。 |
| `op_common.h`、`op_common_ops.h` | 公共执行接口和算子共用入口声明。 | 参数结构与函数边界。 |
| `algorithm/executor/` | 公共执行器接口与实现。 | 算法如何组合模板并导出资源需求。 |
| `algorithm/template/aicpu/` | AICPU模板公共支持与设备入口。 | [kernel_launch.cc](../hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc)如何恢复参数、处理缓存并执行算法。 |
| `algorithm/template/wrapper/` | 传输和同步协议包装。 | [alg_data_trans_wrapper.cc](../hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc)中的Read/Write握手和主从同步。 |
| `algorithm/topo_match/` | 匹配算法所需子拓扑。 | 算法Rank和用户Rank如何对应。 |
| `selector/` | 通用选择、成本及引擎相关逻辑。 | 拓扑与配置怎样改变候选算法。 |
| `topo_info/` | Host拓扑生成、物理层级整理和Rank映射。 | [topo_host.cc](../hccl/src/ops/op_common/topo_info/topo_host.cc)与`physical_level_*.cc`。 |
| `ccu_fallback.cc/.h` | CCU回退支持。 | 回退条件和替代执行路径。 |
| `exec_timeout_manager.cc/.h` | 执行超时相关管理。 | 超时配置如何传给等待。 |
| `order_launch.cc` | 发射顺序相关逻辑。 | 多次算子下发的顺序约束。 |
| `omnipipe_*_data_slice_calc.cc/.h` | OmniPipe数据切片计算。 | 全局布局如何转换为每层、每轮切片。 |

资源缓存命中可能跳过申请和构建；CCU快速发射、AIV缓存重放、单Rank处理等分支也可能提前返回。因此公共流程图表示可选路径，不能当作每次调用都执行的固定步骤表。

## 第五层：common与跨仓边界

| 位置 | 做什么 |
|---|---|
| [param_check.cc/.h](../hccl/src/common/param_check.cc) | 公共输入与类型检查，失败阻止继续下发。 |
| [alg_env_config.cc/.h](../hccl/src/common/alg_env_config.cc) | 算法环境配置；环境变量改变选择或执行方式。 |
| `alg_parse.cc/.h`、`alg_type.cc/.h` | 算法配置解析与类型转换。 |
| `adapter_acl.cc/.h`、`compat.cc` | 运行时适配及兼容处理。 |
| `log.cc`、`config_log.cc/.h`、`adapter_error_manager_pub.cc/.h` | 日志和错误信息组织。 |
| [inconsistent_check.cc](../hccl/src/common/inconsistent_check.cc) | 跨Rank算子描述一致性相关检查。 |
| `hcomm_dlsym/` | 按符号表动态加载HCOMM接口。`.map`控制相关导出符号；`.cc/.h`分别实现/声明加载适配。 |
| `hcomm_dlsym/ccu/` | CCU相关动态符号适配。 |
| `framework/tf_plugin/` | TensorFlow集合通信算子和图适配，包含HCCL及Horovod相关插件。 |
| `op_graph/` | 图优化/转换适配。 |
| `tuner/`、`src/algo_plugin/` | 调优与算法插件连接。 |

真实跨仓调用可从[HCOMM动态加载适配](../hccl/src/common/hcomm_dlsym/hcomm_dlsym.cc)与[原语动态适配](../hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc)进入HCOMM实现。阅读时直接跳到下层源码方便理解，但工程依赖仍遵守动态加载边界。

## 测试、示例、试验代码与文档怎样读

`test/ut/`围绕特定算法和扩展做单元验证。`test/st/algorithm/`组织算法验证；其中`utils/src/{sim_world,hccl_proxy,hccl_depends_stub}/`等提供模拟世界、代理和依赖替身。带`stub`的函数是测试支撑，不能据此推断硬件的真实行为。

`examples/01_point_to_point/`演示Send/Recv；`02_collectives/`按算子列出示例；`03_ai_framework/`展示框架集成；`04_custom_ops_p2p/`、`05_custom_ops_allgather/`、`06_custom_ops_reduce_scatter/`展示自定义通信；`07_tuner_plugin/`展示调优插件。自定义示例中的`aicpu/aiv/ccu`要按对应引擎分别阅读。

`experimental/ops/`包含试验算子及公共扩展；`experimental/eco_system/`包含生态插件。是否参与某个产品构建，以构建配置为准。`docs/zh/`和`docs/en/`对应中文/英文资料；接口参考、构建、架构、用户指南和开发指南分别回答不同问题，图片和图源也全部保留在索引中。

## 建议的完整阅读顺序

1. 从`include/hccl.h`选定API，再读对应入口`.cc`；写下元素数、缓冲区布局及合法分支。
2. 阅读对应`selector/`，确认引擎、算法名和拓扑条件。
3. 进入`op_common.cc`，判断资源新建/复用及参数如何下发。
4. 顺着执行器进入模板，跟踪Peer、切片偏移、Channel、Thread与通知槽位。
5. 从wrapper及dlsym跳到HCOMM的资源或原语实现，核查句柄、远端地址和等待语义。
6. 对照测试和示例验证理解；普通返回成功、任务提交成功与设备实际完成要分开解释。

当前细化注释的范围仍是核心17文件，完整目录索引覆盖全仓。[注释覆盖清单](ANNOTATION_COVERAGE.md)记录范围与数量；[源码阅读路线](READING_GUIDE.zh-CN.md)进一步串联主链；[固定PPT源码对照](PPT_FINAL_SOURCE_MAP.zh-CN.md)保留旧提交的行号链接。

## 维护与验证

执行`python3 scripts/build-directory-guides.py`重新生成两个索引，执行同命令加`--check`检查是否与当前受跟踪文件及内容一致。源文件新增/删除或注释移动符号行号后，应重新生成。此检查证明清单可再生，不替代CANN构建、UT/ST或上板执行。
