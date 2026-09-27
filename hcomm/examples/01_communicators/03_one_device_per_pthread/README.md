# 通信域管理 - 每个线程管理一个 NPU 设备

## 样例介绍

本样例展示如何使用 `HcclCommInitRootInfoConfig()` 接口在单进程中初始化通信域，每个线程管理一个 NPU 设备，包含以下功能点：

- 设备检测，通过 `aclrtGetDeviceCount()` 接口查询可用设备数量。
- 将 rank0 作为 root 节点，通过 `HcclGetRootInfo()` 接口生成 root 节点的 rootinfo 标识信息。

  > rootinfo 标识信息主要包含：Device IP、Device ID 等信息，此信息需广播至集群内所有 rank 用来初始化通信域。

- 在每个线程中，基于 rootinfo 标识信息通过 `HcclCommInitRootInfoConfig()` 接口初始化通信域。
- 调用 `HcclAllReduce()` 算子，并打印结果。

## 目录结构

```text
├── main.cc                   # 样例源文件
├── Makefile                  # 编译/构建配置文件
└── one_device_per_pthread    # 编译生成的可执行文件
```

## 环境准备

### 环境要求

本样例支持以下产品：

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品
- Atlas 训练系列产品
- Atlas 推理系列产品

### 软件依赖

本样例运行依赖安装CANN ops算子包，详细安装步骤可参见 [源码构建](../../../docs/zh/build/build.md) 中的 “安装CANN软件包” 章节。

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
```

## 编译执行样例

在本样例代码目录下执行如下命令：

```bash
make
make test
```

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
