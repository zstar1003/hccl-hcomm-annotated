# AI CPU算子

本节以AI CPU的点对点通信算子为例，介绍使用HCCL通信编程接口开发通信算子的整体流程，帮助用户快速了解通信算子的开发步骤。

## 点对点通信算子

本节以点对点通信算子Send/Receive为例进行介绍：

- Send：将本端rank的数据发送到对端rank。
- Receive：接收对端rank发送过来的数据，需要与Send算子配对使用。

## 产品支持情况

本样例支持以下产品，组网为单机N卡（N>=2，且为偶数）：

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 样例介绍

开发者可以点击[样例链接](https://gitcode.com/cann/hccl/tree/master/examples/04_custom_ops_p2p)获取完整样例代码，该样例使用HCCL通信算子开发接口实现了基于AI CPU通信引擎的Send和Receive算子，主要实现流程如下所示：

- **查询通信域的拓扑信息**：调用拓扑信息查询接口HcclGetRankId\(\)和HcclGetRankSize\(\)获取当前线程操作的rank_id和通信域的rank数量。
- **创建Thread资源**：调用资源管理接口HcclThreadAcquire\(\)分配通信线程资源。
- **建立通信通道**：调用HcclChannelAcquire\(\)接口创建rank间的channel链路。
- **获取远端通信内存信息**：调用HcclChannelGetHcclBuffer\(\)接口获取远端的通信内存地址。
- **准备输入数据**：Send算子调用HcommLocalCopyOnThread\(\)接口将输入数据拷贝到通信内存。
- **前同步**：
  - Send算子HcommChannelNotifyRecordOnThread\(\)接口告知远端数据已经准备完成；
  - Receive算子调用HcommChannelNotifyWaitOnThread\(\)接口等待远端数据准备完成。

- **读取远端数据**：Receive算子调用HcommReadOnThread\(\)接口读取远端通信内存中的数据。
- **后同步**：
  - Receive算子调用HcommChannelNotifyRecordOnThread\(\)接口告知远端已经读取完成。
  - Send算子调用HcommChannelNotifyWaitOnThread\(\)接口等待远端读取完成。

**图1**  Send/Receive算子测试样例示意图  
![Send-Receive算子测试样例示意图](./figures/send_receive_sample.png)

同时，该样例中还包含了测试程序，其中创建1个通信域，rank为偶数的通信成员负责发送数据，rank为奇数的通信成员负责接收数据，发送数据为偶数rank的编号。包含以下功能点：

- 设备检测，通过aclrtGetDeviceCount\(\)接口查询可用设备数量。
- 将rank0作为root节点，通过HcclGetRootInfo\(\)接口生成root节点的rootinfo标识信息。
- 在每个线程中，基于rootinfo标识信息通过HcclCommInitRootInfo\(\)接口初始化通信域。
- 调用算子接口HcclSendCustom\(\)和HcclRecvCustom\(\)，验证基础收发功能，并打印接收结果。

## 编译安装

在CANN/hccl代码仓根目录下执行如下命令，编译并安装自定义算子包：

```bash
# 设置CANN软件包环境变量，此处以root用户默认安装路径为例
source /usr/local/Ascend/cann/set_env.sh

# 执行build.sh脚本进行编译，通过custom_ops_path指定自定义算子工程路径
bash build.sh --vendor=cust --ops=p2p --custom_ops_path=./examples/04_custom_ops_p2p

# 自定义算子安装包在代码仓的build_out目录下
./build_out/cann-hccl_custom_p2p_linux-<arch>.run --install
```

自定义算子包安装信息如下：

- 头文件：\$\{ASCEND_HOME_PATH\}/opp/vendors/cust/include/hccl_custom_p2p.h
- 动态库：\$\{ASCEND_HOME_PATH\}/opp/vendors/cust/lib64/libhccl_custom_p2p.so
- AI CPU算子描述文件：\$\{ASCEND_HOME_PATH\}/opp/vendors/cust/aicpu/config/libp2p_aicpu_kernel.json
- AI CPU算子包：\$\{ASCEND_HOME_PATH\}/opp/vendors/cust/aicpu/kernel/aicpu_hccl_custom_p2p.tar.gz

- 安装脚本：\$\{ASCEND_HOME_PATH\}/opp/vendors/cust/scripts/install.sh

## 执行样例

1. 关闭AI CPU算子验签功能。

    AI CPU算子包会在业务启动时加载至Device，加载过程中驱动默认会执行安全验签，以确保包的可信性。但用户自行编译生成的AI CPU算子包不包含签名头，因此需要手工关闭驱动的验签机制，才可以正常加载。

    参考如下命令，使用root用户在物理机上执行， 以device 0为例：

    ```bash
    npu-smi set -t custom-op-secverify-enable -i 0 -d 1    # 开启验签配置
    npu-smi set -t custom-op-secverify-mode -i 0 -d 0      # 关闭自定义验签
    ```

    > [!NOTE]说明
    > - 关闭驱动安全验签机制存在一定的安全风险，需要用户自行确保自定义通信算子的安全可靠，防止恶意攻击行为。
    > - 更多命令可参考Ascend HDK配套版本的《[npu-smi 命令参考](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743?category=reference-guides&subcategory=command-reference)》中“AI CPU算子验签”相关章节。

2. 修改AI CPU白名单。

    若用户新增AI CPU算子包，需同步将该AI CPU算子包配置到AI CPU白名单中。以root用户默认安装路径为例，编辑ascend_package_load.ini文件：

    ```bash
    vim /usr/local/Ascend/cann/conf/ascend_package_load.ini
    ```

    将下列内容追加到ascend_package_load.ini中：

    ```text
    name:aicpu_hccl_custom_p2p.tar.gz
    install_path:2
    optional:true
    package_path:opp/vendors/cust/aicpu/kernel
    load_as_per_soc:false 
    ```

3. 编译并执行测试样例。

    ```bash
    # 进入样例代码目录
    cd examples/04_custom_ops_p2p/testcase
    # 编译
    make
    # 执行测试用例
    make test
    ```

## 结果解析

rank为偶数的节点负责发送数据，内容为其rank编号，rank为奇数的节点负责接收数据，因此打印结果中各个奇数rank接收到的是上一rank的编号。

```text
Found 8 NPU device(s) available
rankId: 1, output: [ 0 0 0 0 0 0 0 0 ]
rankId: 3, output: [ 2 2 2 2 2 2 2 2 ]
rankId: 5, output: [ 4 4 4 4 4 4 4 4 ]
rankId: 7, output: [ 6 6 6 6 6 6 6 6 ]
```

## 关键代码解析

下面以自定义Send/Receive算子为例，讲解其实现细节。

1. 解析通信域的拓扑信息。

    ```c
    uint32_t rank, rankSize;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    ```

2. 创建资源。

    ```c
    CommEngine engine = CommEngine::COMM_ENGINE_AICPU;
    ACLCHECK(aclrtCreateNotify(&(g_notifies[0]), ACL_NOTIFY_DEFAULT));
    ACLCHECK(aclrtCreateNotify(&(g_notifies[1]), ACL_NOTIFY_DEFAULT));
    AlgResourceCtx resCtxHost;
    for (uint32_t idx = 0; idx < AICPU_CONTROL_NOTIFY_NUM; idx++) {
        ACLCHECK(aclrtGetNotifyId(g_notifies[idx], &(resCtxHost.notifyIds[idx])));
    }
    CHK_RET(HcclThreadAcquire(comm, engine, 1, 0, &(resCtxHost.threadHandle)));
    ```

3. 建立通信链路。

    ```c
    HcclChannelDesc channelDesc;
    HcclChannelDescInit(&channelDesc, 1);
    channelDesc.remoteRank = destRank;
    channelDesc.channelProtocol = CommProtocol::COMM_PROTOCOL_HCCS;
    channelDesc.notifyNum = 2;
    CHK_RET(HcclChannelAcquire(comm, engine, &channelDesc, 1, &(resCtxHost.channelHandle)));
    ```

4. 获取远端的通信内存地址。

    ```c
    CHK_RET(HcclChannelGetHcclBuffer(comm, resCtxHost.channelHandle, &(resCtxHost.remoteBuffer.addr), &(resCtxHost.remoteBuffer.size)));
    ```

5. 读取远端数据。

    ```c
    // 单边读
    CHK_RET(HcommReadOnThread(resCtx->threadHandle, resCtx->channelHandle, param.outputPtr, resCtx->remoteBuffer.addr, size));
    ```
