# 通信域管理 - 每个进程管理一个 NPU 设备（基于 rank table 初始化通信域）

## 样例介绍

本样例展示如何使用 `HcclCommInitClusterInfoConfig()` 接口，根据 `rank_table.json` 配置文件初始化通信域，包含以下功能点：

- 通过 MPI 拉起多个进程，在每个进程中读取 `rank_table.json` 文件，通过 `HcclCommInitClusterInfoConfig()` 接口初始化通信域。
- 调用 `HcclAllReduce()` 算子，并打印结果。

## 目录结构

```text
├── main.cc                              # 样例源文件
├── Makefile                             # 编译/构建配置文件
├── rank_table.json                      # 集群信息配置文件
└── one_device_per_process_rank_table    # 编译生成的可执行文件
```

## 环境准备

### 环境要求

本样例支持以下产品，集群拓扑为单机 2 卡：

- Ascend 950PR/Ascend 950DT

本样例支持以下产品，集群拓扑为单机 8 卡：

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品
- Atlas 训练系列产品
- Atlas 推理系列产品

### 软件依赖

本样例运行依赖安装CANN ops算子包，详细安装步骤可参见 [源码构建](../../../docs/zh/build/build.md) 中的 “安装CANN软件包” 章节。

### 安装 MPI

本样例依赖 MPI 软件在每个 Device 上拉起进程，所以执行本样例前需要安装 MPI，详细安装步骤可参见配套版本的 [《HCCL 性能测试工具用户指南》][1] 中的 “MPI安装与配置” 章节。

[1]: https://hiascend.com/document/redirect/CannCommunityToolHcclTest

### 关闭验签

本源码仓编译生成的`cann-hcomm_<version>_linux-<arch>.run`软件包中包含如下tar.gz子包：

  - `cann-hcomm-compat.tar.gz`: HCOMM兼容升级包。
  - `cann-hccd-compat.tar.gz`: DataFlow兼容升级包。
  - `aicpu_hcomm.tar.gz`: AI CPU通信基础包。

上述tar.gz包会在业务启动时加载至Device，加载过程中默认会由驱动进行安全验签，确保包可信。由于开发者通过本源码仓自行编译生成的tar.gz包中并不含签名头，所以需要关闭驱动安全验签的机制。关闭验签方式参考[关闭验签](../../../docs/zh/build/build.md) 中的 “关闭验签” 章节。

### 配置环境变量

```bash
# 设置 CANN 环境变量，以 root 用户默认安装路径为例
source /usr/local/Ascend/cann/set_env.sh
# 设置 MPI 安装目录，请根据实际情况进行设置
export MPI_HOME=/usr/local/mpich
```

## 编译执行样例

在本样例代码目录下执行如下命令：

```bash
make
make test N=${RANK_SIZE}
```

其中，`RANK_SIZE` 为集群设备数量。针对Ascend 950PR/Ascend 950DT， `RANK_SIZE` 为 2，其他系列产品为8。

> 注意：可通过设置 `HCCL_OP_EXPANSION_MODE` 环境变量配置通信算子的展开模式，不同产品型号支持的范围可参考[环境变量列表](https://hiascend.com/document/redirect/CannCommunityEnvRef) 中该环境变量的使用方法。
>
> ```bash
> # 设置通信算子的展开模式为AI CPU通信引擎
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## 结果示例

每个 rank 的数据初始化为 0~7，经过 AllReduce 操作后，每个 rank 的结果是所有 rank 对应位置数据的和（8 个 rank 的数据相加）。

```text
Found 8 NPU device(s) available
rankId: 0, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 1, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 2, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 3, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 4, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 5, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 6, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 7, output: [ 0 8 16 24 32 40 48 56 ]
```
