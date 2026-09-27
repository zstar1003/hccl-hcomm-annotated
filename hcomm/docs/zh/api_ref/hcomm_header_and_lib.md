# HCOMM头文件与库文件说明

HCOMM（Huawei Communication，华为通信基础库）是HCCL的通信基础库，提供通信域与通信资源管理能力，是CANN集合通信栈的底层基座。HCCL通过dlsym动态加载HCOMM接口，HCCL与HCOMM独立编译、独立版本演进。

本章节介绍HCOMM对外接口的头文件与库文件说明。

## 接口分类

HCOMM对外接口按API分层组织，分层关系与架构约束以[架构简介](../architecture/architecture-brief.md)中「对外API分层关系」与「软件架构约束说明」小节为权威来源：

**表1**接口分类

| 接口类别 | 面向 | 描述 |
| --- | --- | --- |
| 通信域管理（L2-comm） | AI框架层 | 通信域初始化与销毁接口。 |
| 资源与拓扑（L2-res） | 算子开发者 | 通道、内存、拓扑、Endpoint等资源与拓扑查询接口。 |
| 基础原语（L3-prim） | 算子/通信库开发者 | 数据搬运与同步原语类型。 |
| 基础资源（L3-res） | 通信库开发者 | Endpoint/Channel/内存注册等资源管理接口。 |
| CCU算子开发 | CCU算子开发者 | CCU（Collective Communication Unit）资源对象封装与Kernel Launch接口。 |

## 调用接口依赖的头文件和库文件说明

安装固件、驱动及CANN软件包后，编译、运行应用程序时才能引用到HCOMM接口的头文件、库文件。

HCOMM对外头文件在`${INSTALL_DIR}/include/`目录下的hccl/、hcomm/、hcomm/ccu/子目录中，库文件在`${INSTALL_DIR}/lib64/`目录下。${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

> [!CAUTION]注意
> include/目录下的头文件为对外稳定头文件；pkg_inc/目录为HCOMM <-> HCCL、GE等包间接口，安装到单独的包间目录，不对外承诺稳定，请勿在对外业务中直接引用。编译接口程序时，请按照include的头文件依赖对应的库文件，如果引用多余的so文件，可能导致版本功能异常或后续版本升级时存在兼容性问题。

您需要根据实际使用的HCOMM接口来include依赖的文件，各头文件的用途如下表所示。

**表2**头文件列表

| 定义接口的头文件 | 用途 | 对应的库文件 |
| --- | --- | --- |
| hccl/hccl_comm.h | 用于定义通信域初始化与销毁等C接口（弱符号）。 | libhcomm.so |
| hccl/hccl_types.h | 用于定义HCCL返回码枚举与基础类型。 | libhcomm.so |
| hccl/hccn_rping.h | 用于定义HCCN RPing（Remote Ping）网络连通性探测C接口，包括HccnRpingCtx、HccnResult、HccnRpingMode等类型与探测接口。 | libhcomm.so |
| hccl/hccl_rank_graph.h | 用于定义通信拓扑、Endpoint属性与异构组网枚举。 | libhcomm.so |
| hccl/hccl_ccu_res.h | 用于定义查询通信域内CCU实例句柄的C接口。 | libhcomm.so |
| hccl/hccl_res.h | 用于定义HCCL通信域资源管理（Thread/EngineCtx/内存注册）接口与常量。 | libhcomm.so |
| hccl/hccl_channel.h | 用于定义HCCL通道描述、通道创建/销毁、Channel配置等通道管理接口与常量。 | libhcomm.so |
| hccl/hccl_sym_win.h | 用于定义对称内存窗口（Symmetric Window）访问接口。 | libhcomm.so |
| hccl/hccl_launch.h | 用于定义P2P算子描述与Launch相关结构。 | libhcomm.so |
| hcomm/hcomm_primitives.h | 用于定义通道/线程句柄、归约算子等基础原语类型，提供数据搬运与同步原语。 | libhcomm.so |
| hcomm/hcomm_res.h | 用于定义Endpoint/内存注册/线程等基础资源管理C接口。 | libhcomm.so |
| hcomm/hcomm_res_defs.h | 用于定义HCOMM ABI版本、句柄与资源描述结构。 | libhcomm.so |
| hcomm/hcomm_channel.h | 用于定义HCOMM通道描述、通道创建/销毁、Channel配置等通道管理C接口与类型。 | libhcomm.so |
| hcomm/ccu/ccu_primitives.hpp | CCU原语聚合头，含类型别名与资源创建入口。 | libhcomm.so |
| hcomm/ccu/ccu_launch.h | 用于定义CCU Kernel注册与Launch的C接口（弱符号）。 | libhcomm.so |
| hcomm/ccu/ccu_res.h | 用于定义CCU资源管理及内存CCU访问token（HcommCcuGetMemToken）等C接口。 | libhcomm.so |
| hcomm/ccu/ccu_types.h | 用于定义CCU返回码、条件类型等基础类型。 | libhcomm.so |
| hcomm/ccu/ccu_control_flow_macro.h | 用于定义CCU while循环等控制流宏。 | libhcomm.so |
| hcomm/ccu/ccu_address.hpp | CCU地址对象封装（Address类）。 | libhcomm.so |
| hcomm/ccu/ccu_local_addr.hpp | CCU本端地址对象封装（LocalAddr类）。 | libhcomm.so |
| hcomm/ccu/ccu_remote_addr.hpp | CCU远端地址对象封装（RemoteAddr类）。 | libhcomm.so |
| hcomm/ccu/ccu_buffer.hpp | CCU Buffer资源对象封装（CcuBuffer类）。 | libhcomm.so |
| hcomm/ccu/ccu_array.hpp | CCU资源连续容器Array模板，用于批量分配。 | libhcomm.so |
| hcomm/ccu/ccu_event.hpp | CCU Event资源对象封装（Event类）。 | libhcomm.so |
| hcomm/ccu/ccu_variable.hpp | CCU Variable资源对象封装（Variable类）。 | libhcomm.so |
| hcomm/ccu/ccu_loop.hpp | CCU循环（Loop）结构对象封装。 | libhcomm.so |
| hcomm/ccu/ccu_func.hpp | 将lambda包装为CCU Func的工具模板。 | libhcomm.so |
| hcomm/ccu/ccu_utils.hpp | CCU异常类与算子工具等内部辅助定义。 | libhcomm.so |

HCOMM以`cann-hcomm_<version>_linux-<arch>.run`安装包形式发布，包含`libhcomm.so`、对外头文件与`cann-hcomm-compat.tar.gz`兼容升级子包。

- 源码编译与安装流程详见[构建指南](../build/build.md)。
- 通信域管理相关接口的原型定义、参数描述与约束说明详见[通信域创建与管理接口（C语言）](./comm_mgr_c/README.md)。
- 通信算子开发相关接口的原型定义、参数描述与约束说明详见[通信算子开发接口](./comm_opdev/README.md)。
