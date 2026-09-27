# HCOMM

简体中文 | [English](./README_en.md)

## 🔥 Latest News

- [2025/11/30] HCOMM项目开源。

## 🚀 概述

HCOMM（Huawei Communication）是HCCL的通信基础库，提供通信域以及通信资源的管理能力。

HCOMM提供了标准化通信编程接口，具备以下关键特性：

- 支持昇腾设备上的多种通信引擎，充分发挥硬件能力。
- 支持多种通信协议，包括PCIe、HCCS、RDMA等。
- 通信平台与通信算子开发解耦，支持通信算子的独立开发、构建与部署。

<img src="./docs/zh/build/figures/architecture.png" alt="hccl-architecture" style="width: 65%;  height:65%;" />

HCOMM通信基础库采用分层解耦的设计思路，将通信能力划分为控制面和数据面两部分。

- 控制面：提供拓扑信息查询与通信资源管理功能。
- 数据面：提供本地操作、算子间同步、通信操作等数据搬运和计算功能。

  控制面提供通信资源，数据面提供操作资源的方法，所提供的通信编程接口可以让通信算子开发人员聚焦于业务创新，而无需关注芯片底层复杂的实现细节。

## 🔍 目录结构说明

本项目关键目录如下所示：

```text
├── src                                  # 源码目录
│   ├── base_comm                        # 基础通信层
│   │   ├── common                       # 基础通信层公共基础功能目录
│   │   ├── primitives                   # 基础通信原语
│   │   └── resources                    # 基础通信资源
│   ├── coll_communicator_mgr            # 集合通信域管理
│   │   ├── api_c_adpt                   # C接口适配
│   │   ├── common                       # 集合通信层公共基础功能目录
│   │   ├── communicator                 # 通信域
│   │   ├── config_mgr                   # 配置管理
│   │   ├── dfx                          # 维测
│   │   ├── rank_graph                   # 拓扑管理
│   │   └── resource_mgr                 # 资源管理
│   └── legacy                           # 历史版本兼容目录
│       ├── ascend910                    # A2&A3兼容代码
│       │   ├── algorithm                # 通信算法源码目录
│       │   ├── common                   # 公共基础功能目录
│       │   ├── framework                # 通信框架源码目录
│       │   ├── hccd                     # 提供进程间点对点通信能力
│       │   ├── platform                 # 通信平台源码目录
│       │   └── pub_inc                  # 平台接口头文件
│       └── ascend950                    # A5旧流程兼容代码
│           ├── common                   # 公共基础组件
│           ├── framework                # 框架核心实现
│           ├── include                  # 公共接口头文件
│           ├── interface                # 接口适配层
│           ├── local_build              # 本地构建工具
│           ├── service                  # 服务层
│           └── unified_platform         # 统一平台层
├── python                               # Python 包
├── include                              # 对外头文件
├── pkg_inc                              # 包间接口头文件
├── experimental                         # 社区贡献的试验性代码目录
├── test                                 # 测试代码目录
│   ├── ut                               # 单元测试代码目录
│   └── st                               # 系统测试代码目录
├── docs                                 # 资料文档目录
├── examples                             # 样例代码目录
└── build.sh                             # 编译构建脚本
```

## 📝 版本配套

本项目源码会跟随CANN软件版本发布，关于CANN软件版本与本项目标签的对应关系请参阅[release仓库](https://gitcode.com/cann/release-management)中的相应版本说明。
请注意，为确保您的源码定制开发顺利进行，请选择配套的CANN版本与GitCode标签源码，使用master分支可能存在版本不匹配的风险。

## ⚡️ 快速开始

若您希望快速构建并体验本项目，请访问如下简易指南。

- [源码构建](./docs/zh/build/build.md)：了解如何编译、安装本项目，并进行基础测试验证。
- [样例执行](./examples/README.md)：参照详细的示例代码与操作步骤指引，快速体验。

## 📖 学习教程

HCCL提供了使用指南、通信算子开发指南、技术文章、培训视频，详细可参见 [HCCL 参考资料](./docs/README.md)。
此外，HCCL还提供了QuickStart指南、常见FAQ等wiki，详细可参见 [WIKI](https://gitcode.com/cann/hcomm/wiki)。

## 📝 相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
