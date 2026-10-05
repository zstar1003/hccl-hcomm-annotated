/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/ins_temp_all_to_all_v_mesh_1D.h"
#include <cstring>

#define NET_NUM 2

namespace ops_hccl {

std::vector<CostModelParam> InsTempAlltoAllVMesh1D::CalcCostCoeff(CalcCostCoeffParam param)
{
    // 端口数由 executor 通过 topomatch v2 动态传入，直接聚合使用
    bool isSingleChannel = (param.algName != nullptr && strstr(param.algName, "SingleChannel") != nullptr);
    // [中文导读] 汇总 executor 传入的端口数；缺少有效端口信息时按单通道或多通道算法使用默认值。
    int portNum = 0;
    for (auto p : param.portNum) {
        portNum += static_cast<int>(p);
    }
    if (portNum <= 0) {
        portNum = isSingleChannel ? 6 : 8;
    }
    int kernelNum = 10;
    // SingleChannel单通道每peer 5个trans + 4个sync = 9；多通道(channelsPerRank=2)每个通道对应一组trans/sync
    int channelsPerRank = isSingleChannel ? 1 : 2;
    // [中文导读] 把传输与同步任务数按每个 Peer 的通道数扩展，作为发射成本的输入。
    int taskNum
        = (CostModelManager::CalcTransTaskNum(param.rankSize) + CostModelManager::CalcSyncTaskNum(param.rankSize) * 2)
          * channelsPerRank;
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    CostModelManager::Global()->CalcMeshParam(param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    // AICPU模板使用hcclBuff做中转，input->hcclBuff(PreCopy)和hcclBuff->output(PostCopy)都需要local copy
    // 但inputBuffer(INPUT) != scratchBuffer(HCCL_BUFFER)，且outputBuffer(OUTPUT) != scratchBuffer(HCCL_BUFFER)
    // [中文导读] 输入与中转区不同时计入本地复制成本，再组合带宽、复制、延迟和发射系数。
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B);
    }
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

InsTempAlltoAllVMesh1D::InsTempAlltoAllVMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempAlltoAllVMesh1D::~InsTempAlltoAllVMesh1D() {}

// [中文导读] Host侧资源计算：选择本层Peer通道，按并发Peer数和每Peer通道数计算从Thread/通知需求。
// [中文导读] 此函数填resourceRequest，不直接调用HCOMM创建物理Channel；申请由公共资源层完成。
HcclResult InsTempAlltoAllVMesh1D::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && topoInfo->topoLevelNums > 1 && !topoInfo->level0PcieMix) {
        CHK_PRT_RET(
            subCommRanks_.size() != NET_NUM,
            HCCL_ERROR(
                "[InsTempAlltoAllVMesh1D][CalcRes] subCommRankNum[%zu] is not [%u]", subCommRanks_.size(), NET_NUM),
            HCCL_E_PARA);
        subCommRanks_ = {subCommRanks_[1]};
        templateRankSize_ = subCommRanks_[1].size();
    }

    // [中文导读] 按拓扑选择普通 Mesh 请求或 Mesh CLOS 多 Jetty 请求，形成当前模板的通道清单。
    std::vector<HcclChannelDesc> level0Channels;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMeshClosMultiJetty(comm, param, topoInfo, subCommRanks_, myChannelDescs));
        for (auto channel : myChannelDescs) {
            // [中文导读] Mesh CLOS 分支只保留 UB_CTP 通道，将其它协议留给各自对应的层次和模板。
            if (channel.channelProtocol == COMM_PROTOCOL_UB_CTP) {
                level0Channels.push_back(channel);
            }
        }
        HCCL_DEBUG("[InsTempAlltoAllVMesh1D::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    // [中文导读] 单通道专用算法保持既定通道数；其它算法从请求清单统计每 Peer 通道数。
    if (std::string(param.algName) != "AicpuAllToAllSoleMeshSingleChannel") {
        channelsPerRank_ = CalcChannelsPerRank(level0Channels);
    }
    HCCL_INFO("[InsTempAlltoAllVMesh1D][CalcRes] channelsPerRank_ is [%u]", channelsPerRank_);
    // [中文导读] 每个并发 Peer 配置一组通道 Thread，因此从 Thread 数按并发 Peer 数乘通道数计算。
    resourceRequest.slaveThreadNum
        = std::min(ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE, templateRankSize_ - 1) * channelsPerRank_;
    for (u32 index = 0; index < resourceRequest.slaveThreadNum; index++) {
        // 从流的notify数量以rank间channel数的最大值为准，用于和主流同步以及同一个rank多条链路间的同步
        resourceRequest.notifyNumPerThread.push_back(channelsPerRank_);
    }
    // [中文导读] 主 Thread 需要与所有从 Thread 汇合，所以主流通知数按从 Thread 总数预留。
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllVMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // 分组fullmesh，每轮最多通信maxConcurrentSize_个
    // [中文导读] 一次只处理有限个远端 Rank，中转空间倍数与这一并发数量一致。
    concurrentSendRecvNum_ = std::min(ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE, templateRankSize_ - 1);
    return concurrentSendRecvNum_;
}

void InsTempAlltoAllVMesh1D::CalcCommRankSetForOneLoop(
    const u32 roundIdx, const u32 remainRankSize, std::vector<u32>& commRanks) const
{
    commRanks.clear();
    // [中文导读] 以左右邻居成对选择 Peer，并按剩余 Rank 数裁剪最后一轮的配对数量。
    u32 pairNumPerRound = (concurrentSendRecvNum_ + 1) / 2;
    u32 pairSize = (remainRankSize < concurrentSendRecvNum_) ? (remainRankSize + 1) / 2 : pairNumPerRound;
    for (u32 i = roundIdx * pairNumPerRound + 1; i < (roundIdx * pairNumPerRound + pairSize + 1); i++) {
        // [中文导读] 通过环形编号计算本轮两侧 Peer，使相对距离相同的 Rank 对按同一规则通信。
        u32 leftRemoteRank = (myRank_ + templateRankSize_ - i) % templateRankSize_;
        u32 rightRemoteRank = (myRank_ + i) % templateRankSize_;
        // [中文导读] 偶数规模的正对 Peer 可能左右编号相同，只加入一次，避免重复通信。
        if (leftRemoteRank == rightRemoteRank) {
            commRanks.push_back(leftRemoteRank);
            break;
        } else {
            commRanks.push_back(leftRemoteRank);
            commRanks.push_back(rightRemoteRank);
        }
    }
    return;
}

u32 InsTempAlltoAllVMesh1D::CalcCommLoops() const
{
    u32 totalCommRankSize = templateRankSize_ - 1; // 除去本rank
    return (totalCommRankSize + concurrentSendRecvNum_ - 1) / concurrentSendRecvNum_;
}

void InsTempAlltoAllVMesh1D::CalcCclBuffIdx(u32 remoteRank, u32& myRankCclBuffIdx, u32& remoteCclBuffIdx) const
{
    u32 pairNum = (concurrentSendRecvNum_ + 1) / 2;
    // 以myRank为基准，计算remoteRank相对于它的gapRight和gapLeft
    // 反过来就是myRank相对于remoteRank的gapLeft和gapRight
    // [中文导读] 从双方的左右相对距离计算可复用 CCL 槽位，保证发送方和接收方对槽位有一致约定。
    u32 gapRight = (templateRankSize_ + remoteRank - myRank_) % templateRankSize_;
    u32 gapLeft = (templateRankSize_ + myRank_ - remoteRank) % templateRankSize_;
    if (gapLeft < gapRight) {
        // remoteRank是myRank左边的rank，myRank是remoteRank右边的rank
        u32 gap = gapLeft;
        myRankCclBuffIdx = pairNum - 1 - ((gap - 1) % pairNum);
        remoteCclBuffIdx = pairNum + ((gap - 1) % pairNum);
    } else if (gapLeft > gapRight) {
        // remoteRank是myRank右边的rank，myRank是remoteRank右边的rank
        u32 gap = gapRight;
        myRankCclBuffIdx = pairNum + ((gap - 1) % pairNum);
        remoteCclBuffIdx = pairNum - 1 - ((gap - 1) % pairNum);
    } else {
        myRankCclBuffIdx = 0;
        remoteCclBuffIdx = 0;
    }
    HCCL_DEBUG(
        "[InsTempAlltoAllVMesh1D][CalcCclBuffIdx] For my rank[%u] and remote rank[%u], "
        "my ccl buff idx is [%u], remote ccl buff idx is [%u].",
        myRank_, remoteRank, myRankCclBuffIdx, remoteCclBuffIdx);
    return;
}

// [中文导读] 设备模板入口，接收已准备好的Thread、Channel和当前数据块描述。
// [中文导读] 检测到PCIe链路时当前模板采用Read模式，否则采用Write；不能把同一模板固定解释为远端写。
// [中文导读] 本Rank编号先映射到子通信域中的算法Rank，再按轮次与Peer展开任务。
HcclResult InsTempAlltoAllVMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAlltoAllVMesh1D][KernelRun] Run Start");
    // [中文导读] 读取模板已持有的 Thread 数、元素类型及直接远端内存访问能力，作为本次展开状态。
    threadNum_ = templateResource.threads.size();
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = HCCL_SIZE_TABLE[dataType_];
    opType_ = param.opType;
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;

    // [中文导读] 检测整个模板通道集合是否含 PCIe；检测结果决定本次采用 Read 还是 Write 协议。
    bool isPcieProtocol = IsPcieProtocol(templateResource.channels); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocol;                                     // 是否使用Read模式
    HCCL_DEBUG("[InsTempAlltoAllVMesh1D][KernelRun] Use Dma Read[%d]", isDmaRead_);

    myAlgRank_ = 0;
    // [中文导读] 将通信域中的本 Rank 编号映射到子通信域排列中的算法编号，定位自发自收的数据段。
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank_ = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][KernelRun] subCommRanks_ or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }
    if (std::string(param.algName) != "AicpuAllToAllSoleMeshSingleChannel") {
        channelsPerRank_ = CalcChannelsPerRank(templateResource.channels); // 每个rank的channel数量的最大值
    }
    // [中文导读] 交给本数据块的 Peer 轮次调度，后面的 Thread/Channel 操作开始构造具体任务。
    CHK_RET(RunALLtoALL(templateResource.channels, templateResource.threads, tempAlgParams, myAlgRank_));

    HCCL_INFO("[InsTempAlltoAllVMesh1D][KernelRun] Run End");
    return HcclResult::HCCL_SUCCESS;
}

// [中文导读] 发给自己的那一份不经过远端Channel，按发送/接收位移在本地Thread上复制到输出。
// [中文导读] 数量为零时不提交复制；DataSlice同时保存字节长度和元素数量，二者不要混淆。
HcclResult InsTempAlltoAllVMesh1D::LocalCopyForMyRank(
    const TemplateDataParams& tempAlgParams, const ThreadHandle& thread, const u32 myAlgRank, const u32 queIdx) const
{
    // [中文导读] 自发部分用本 Rank 的发送位移定位输入，接收部分用接收位移定位输出。
    DataSlice srcSlice = DataSlice(
        tempAlgParams.buffInfo.inputPtr, tempAlgParams.sdispls[myAlgRank] * dataTypeSize_,
        tempAlgParams.sendCounts[myAlgRank] * dataTypeSize_, tempAlgParams.sendCounts[myAlgRank]);
    DataSlice dstSlice = DataSlice(
        tempAlgParams.buffInfo.outputPtr, tempAlgParams.rdispls[myAlgRank] * dataTypeSize_,
        tempAlgParams.recvCounts[myAlgRank] * dataTypeSize_, tempAlgParams.recvCounts[myAlgRank]);

    if (tempAlgParams.sendCounts[myAlgRank] > 0) {
        CHK_RET(static_cast<HcclResult>(LocalCopy(thread, srcSlice, dstSlice)));
        HCCL_DEBUG(
            "[InsTempAlltoAllVMesh1D][RunALLtoALL] do local copy on thread[%u], data size[%llu].", queIdx,
            tempAlgParams.sendCounts[myAlgRank] * dataTypeSize_);
    }
    return HCCL_SUCCESS;
}

// [中文导读] 这是一个模板数据块内的Peer轮次循环，不是整个算子按CCL容量分块的最外层循环。
// [中文导读] 主从前同步放行从Thread，后同步汇合从Thread，保证最终完成通知不越过未完成的分支。
// [中文导读] Read中转路径先准备可供Peer读取的CCL数据，Write路径则在收取后按需复制到用户输出。
HcclResult InsTempAlltoAllVMesh1D::RunALLtoALL(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParams, const u32 myAlgRank)
{
    // 计算通信轮数
    // [中文导读] 排除自身 Rank 后计算远端通信轮数；剩余数量随每轮实际选中的 Peer 数递减。
    u32 commLoops = CalcCommLoops();
    u32 remainRankSize = templateRankSize_ - 1;
    std::vector<u32> commRanks;

    std::vector<ThreadHandle> subThreads;
    if (threadNum_ > 1) {
        // 只做一次全量的前同步
        // [中文导读] 从 Thread 列表中去掉全局主 Thread，先统一放行所有从 Thread 的通信工作。
        subThreads.assign(threads.begin() + 1, threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
    }
    for (u32 roundIdx = 0; roundIdx < commLoops && remainRankSize > 0; roundIdx++) {
        // [中文导读] 按本轮编号生成 Peer 清单，每个轮次复用有限数量的 Thread 组和中转槽位。
        CalcCommRankSetForOneLoop(roundIdx, remainRankSize, commRanks); // 计算本轮通信rank
        if (isDmaRead_ && !enableRemoteMemAccess_) {
            if (roundIdx == 0) {
                // 如果是read模式，第一轮做统一的前拷贝
                // [中文导读] Read 中转路径的首轮先把各 Peer 的输入片放入本端 CCL，供对端拉取。
                CHK_RET(PreCopyByLoop(commRanks, channels, threads, tempAlgParams));
                if (threadNum_ > 1) {
                    // [中文导读] 首轮复制后先汇合从 Thread，再重新放行，建立读通信前的数据就绪依赖。
                    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
                    CHK_RET(PostSyncInterThreads(
                        threads[0], subThreads, notifyIdxSubToMain_)); // 第1轮通信中将前拷贝与本卡数据拷贝错开
                    CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
                }
                CHK_RET(
                    // [中文导读] 首轮同时在全局主 Thread 处理自身数据；远端收发安排在从 Thread 上。
                    LocalCopyForMyRank(tempAlgParams, threads[0], myAlgRank, 0)); // 在第1轮通信中用0号流做本卡数据拷贝
            }
            CHK_RET(RunSendRecvByLoop(commRanks, tempAlgParams, channels, threads, roundIdx, commLoops));
            remainRankSize -= commRanks.size();
        } else {
            if (roundIdx == 0) {
                CHK_RET(
                    LocalCopyForMyRank(tempAlgParams, threads[0], myAlgRank, 0)); // 在第1轮通信中用0号流做本卡数据拷贝
            }
            CHK_RET(RunSendRecvByLoop(commRanks, tempAlgParams, channels, threads, roundIdx, commLoops));
            remainRankSize -= commRanks.size();
            HCCL_DEBUG(
                "[InsTempAlltoAllVMesh1D][RunALLtoALL] round[%u] finish, commRank size is [%zu], "
                "remainRankSize is [%u].",
                roundIdx, commRanks.size(), remainRankSize);
        }
    }
    // [中文导读] 所有 Peer 轮次完成后通过一次全局后同步汇合，防止主 Thread 越过从 Thread 的尾部任务。
    if (threadNum_ > 1) {
        // 只做一次全量的后同步
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
    }
    return HCCL_SUCCESS;
}

// [中文导读] 一轮内遍历Peer，为每个Peer取出Channel列表，再把该Peer的发送/接收数据分别按端口分片。
// [中文导读] 一条API调用不一定对应一个Peer的全部数据；多Channel时每条链承载各自偏移和长度。
HcclResult InsTempAlltoAllVMesh1D::RunSendRecvByLoop(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const u32 roundIdx, const u32 commLoops)
{
    // 遍历本次通信的所有rank
    for (u32 rankIdx = 0; rankIdx < commRanks.size(); rankIdx++) {
        u32 remoteRank = commRanks[rankIdx];
        // 取出本次通信对端的channel
        // [中文导读] 为当前 Peer 查找预备通道；缺失时立即报参数错误，不继续构造无效传输。
        auto it = channels.find(remoteRank);
        if (it == channels.end()) {
            HCCL_ERROR(
                "[InsTempAlltoAllVMesh1D][RunSendRecvByLoop] remoteRank[%u] "
                "does not exist in channels map!",
                remoteRank);
            return HCCL_E_PARA;
        }
        const std::vector<ChannelInfo>& curChannels = it->second;
        // [中文导读] 实际使用的通道数受 Peer 已有通道数量和模板并行上限共同约束。
        u32 curValidChannelsSize = std::min(static_cast<u32>(curChannels.size()), channelsPerRank_);
        // send数据按照channel分片
        // [中文导读] 按通道端口组划分发送元素数、字节长度与片内偏移，避免多通道传输重叠。
        CHK_RET(CalcDataSplitByPortGroupCommon(
            tempAlgParams.sendCounts[remoteRank], dataTypeSize_, curChannels, sendCountsSplit_, sendSizeSplit_,
            sendOffsetSplit_, curValidChannelsSize));
        // recv数据按照channel分片
        // [中文导读] 接收方向独立分片，因为同一 Peer 的发送数量与接收数量可以不同。
        CHK_RET(CalcDataSplitByPortGroupCommon(
            tempAlgParams.recvCounts[remoteRank], dataTypeSize_, curChannels, recvCountsSplit_, recvSizeSplit_,
            recvOffsetSplit_, curValidChannelsSize));
        CHK_RET(RunSendRecvByChannel(
            tempAlgParams, roundIdx, curValidChannelsSize, curChannels, remoteRank, threads, commLoops));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PreSyncInterThreadsPerRank(
    const ThreadHandle& mainThreadCurRank, const std::vector<ThreadHandle>& subThreadsCurRank) const
{
    // [中文导读] Peer 组内部使用通知槽 1 放行其它通道 Thread，槽 0 留给全局主从同步。
    std::vector<u32> notifyIdxMainToSubCurRank;
    for (u32 subThreadIdx = 0; subThreadIdx < subThreadsCurRank.size(); subThreadIdx++) {
        notifyIdxMainToSubCurRank.emplace_back(1); // 第0个用于和全局的主流通信，第1个用于和rank内部的主流通信
    }
    CHK_RET(PreSyncInterThreads(mainThreadCurRank, subThreadsCurRank, notifyIdxMainToSubCurRank));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PostSyncInterThreadsPerRank(
    const ThreadHandle& mainThreadCurRank, const std::vector<ThreadHandle>& subThreadsCurRank) const
{
    // [中文导读] Peer 组主 Thread 用从槽 1 开始的独立索引收齐其它通道的完成通知。
    std::vector<u32> notifyIdxSubToMainCurRank;
    for (u32 subThreadIdx = 0; subThreadIdx < subThreadsCurRank.size(); subThreadIdx++) {
        notifyIdxSubToMainCurRank.emplace_back(
            subThreadIdx + 1); // 第0个用于和全局的主流通信，从第1个开始用于和rank内部的从流通信
    }
    CHK_RET(PostSyncInterThreads(mainThreadCurRank, subThreadsCurRank, notifyIdxSubToMainCurRank));
    return HcclResult::HCCL_SUCCESS;
}

// [中文导读] 选定当前Peer的CCL槽位及Thread组，逐Channel构造切片并提交对应的收发协议。
// [中文导读] 普通Write中转路径接收数据先落本地CCL，收发依赖之后才执行PostCopy到用户输出。
// [中文导读] enableRemoteMemAccess_分支可直接访问已准备的远端用户内存，不执行同样的中转复制。
HcclResult InsTempAlltoAllVMesh1D::RunSendRecvByChannel(
    const TemplateDataParams& tempAlgParams, const u32 roundIdx, const u32 curValidChannelsSize,
    const std::vector<ChannelInfo>& curChannels, const u32 remoteRank, const std::vector<ThreadHandle>& threads,
    const u32 commLoops) const
{
    u32 myRankCclBuffIdx = 0; // myRank与remoteRank交互时myRank提供的cclbuffer index
    u32 remoteCclBuffIdx = 0; // myRank与remoteRank交互时remoteRank提供的cclbuffer index
    // [中文导读] 按 Peer 的相对位置确定双方中转槽；本端槽号进一步映射为从 Thread 组的起点。
    CalcCclBuffIdx(remoteRank, myRankCclBuffIdx, remoteCclBuffIdx);
    u32 queIdx = myRankCclBuffIdx * channelsPerRank_ + 1;
    const ThreadHandle& mainThreadCurRank = threads[queIdx]; // 当前rank分配到的第一条流（rank内主流）
    std::vector<ThreadHandle> subThreadsCurRank;             // 当前rank的rank内从流
    // [中文导读] 多通道 Peer 使用组内主从同步；后续轮次必须先放行组内从 Thread。
    if (curValidChannelsSize > 1) {
        subThreadsCurRank.assign(threads.begin() + queIdx + 1, threads.begin() + queIdx + curValidChannelsSize);
        if (roundIdx != 0) {
            CHK_RET(PreSyncInterThreadsPerRank(mainThreadCurRank, subThreadsCurRank));
        }
    }
    for (u32 channelId = 0; channelId < curValidChannelsSize; channelId++) {
        // [中文导读] Read 中转的后续轮次在复用槽位上补做当前片的前拷贝；直接访问路径跳过它。
        if (!enableRemoteMemAccess_ && roundIdx != 0 && isDmaRead_ && sendSizeSplit_[channelId] > 0) {
            CHK_RET(static_cast<HcclResult>(PreCopy(
                tempAlgParams, threads[queIdx], myRankCclBuffIdx, remoteRank, sendSizeSplit_[channelId],
                sendCountsSplit_[channelId], sendOffsetSplit_[channelId])));
        }
        const ChannelInfo& channelSend = curChannels[channelId]; // 发给哪个rank
        const ChannelInfo& channelRecv = curChannels[channelId]; // 收哪个rank的数据
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        // [中文导读] 先把本地与远端的逻辑布局落成四组切片，再封装为收发包装层所需的数据描述。
        CHK_RET(BuildDataSlices(
            tempAlgParams, remoteRank, channelSend, channelRecv, channelId, remoteCclBuffIdx, txSrcSlices, txDstSlices,
            rxSrcSlices, rxDstSlices));

        DataInfo sendInfo{channelSend, {txSrcSlices, txDstSlices}, dataType_};
        DataInfo recvInfo{channelRecv, {rxSrcSlices, rxDstSlices}, dataType_};
        SendRecvInfo sendRecvInfo{
            {channelSend, channelRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_};
        // [中文导读] 针对当前通道按有效收发长度选择 Read/Write 握手协议，任务落到该通道对应 Thread。
        CHK_RET(RunSendRecv(sendRecvInfo, sendInfo, recvInfo, threads[queIdx], channelId));
        HCCL_INFO(
            "[InsTempAlltoAllVMesh1D][RunSendRecvByLoop] do send recv write on thread[%u], channelId[%u], "
            "send size[%llu], recv size[%llu], remote rank[%u].",
            queIdx, channelId, sendSizeSplit_[channelId], recvSizeSplit_[channelId], remoteRank);
        // [中文导读] Write 中转接收依赖满足后，把该通道收到的 CCL 片复制到用户输出的对应位置。
        if (!enableRemoteMemAccess_ && !isDmaRead_ && recvSizeSplit_[channelId] > 0) {
            CHK_RET(PostCopy(
                tempAlgParams, threads[queIdx], myRankCclBuffIdx, remoteRank, recvSizeSplit_[channelId],
                recvCountsSplit_[channelId], recvOffsetSplit_[channelId]));
        }
        queIdx++;
    }
    // [中文导读] 还有下一轮时先汇合 Peer 组内的多通道，下一轮才可安全复用该组槽位。
    if (curValidChannelsSize > 1 && roundIdx != commLoops - 1) {
        CHK_RET(PostSyncInterThreadsPerRank(mainThreadCurRank, subThreadsCurRank));
    }
    return HcclResult::HCCL_SUCCESS;
}

// [中文导读] 按Read/Write模式以及本Channel的收发长度选择双向、仅发送、仅接收或无数据分支。
// [中文导读] Send/Recv包装还负责协议同步，不能只统计Write调用就判断AllToAll的完整执行覆盖。
HcclResult InsTempAlltoAllVMesh1D::RunSendRecv(
    const SendRecvInfo& sendRecvInfo, const DataInfo& sendInfo, const DataInfo& recvInfo, const ThreadHandle& thread,
    const u32 channelId) const
{
    if (isDmaRead_) {
        if (sendSizeSplit_[channelId] > 0 && recvSizeSplit_[channelId] > 0) {
            CHK_PRT_RET(
                SendRecvRead(sendRecvInfo, thread),
                HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendRecvInfo failed"), HcclResult::HCCL_E_INTERNAL);
        } else { // 其中一个或者两个为0
            if (sendSizeSplit_[channelId] > 0) {
                CHK_PRT_RET(
                    SendRead(sendInfo, thread), HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL sendInfo failed"),
                    HcclResult::HCCL_E_INTERNAL);
            } else if (recvSizeSplit_[channelId] > 0) {
                CHK_PRT_RET(
                    RecvRead(recvInfo, thread), HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL recvInfo failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        }
    } else {
        if (sendSizeSplit_[channelId] > 0 && recvSizeSplit_[channelId] > 0) {
            CHK_PRT_RET(
                SendRecvWrite(sendRecvInfo, thread),
                HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendRecvInfo failed"), HcclResult::HCCL_E_INTERNAL);
        } else { // 其中一个或者两个为0
            if (sendSizeSplit_[channelId] > 0) {
                CHK_PRT_RET(
                    SendWrite(sendInfo, thread), HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL sendInfo failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
            if (recvSizeSplit_[channelId] > 0) {
                CHK_PRT_RET(
                    RecvWrite(recvInfo, thread), HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL recvInfo failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// [中文导读] 普通Write切片指向本端input和远端CCL；普通Read切片指向远端CCL和本端output。
// [中文导读] 位移包含Peer布局、CCL槽位与Channel分片偏移，远端地址必须来自对应Channel的交换结果。
HcclResult InsTempAlltoAllVMesh1D::BuildDataSlices(
    const TemplateDataParams& tempAlgParams, const u32 remoteRank, const ChannelInfo& channelSend,
    const ChannelInfo& channelRecv, const u32 channelId, const u32 remoteCclBuffIdx,
    std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices, std::vector<DataSlice>& rxSrcSlices,
    std::vector<DataSlice>& rxDstSlices) const
{
    // [中文导读] 直接远端内存访问分支使用注册后的用户区描述；否则构造用户区与远端 CCL 之间的片。
    if (enableRemoteMemAccess_) {
        CHK_RET(BuildRemoteMemSlices(
            tempAlgParams, remoteRank, channelSend, channelRecv, channelId, txSrcSlices, txDstSlices, rxSrcSlices,
            rxDstSlices));
    } else {
        void* remoteCclBuffAddr = channelRecv.remoteCclMem.addr;
        // write模式下，本端src数据input buffer slice
        DataSlice txSrcSlice = DataSlice(
            tempAlgParams.buffInfo.inputPtr,
            tempAlgParams.sdispls[remoteRank] * dataTypeSize_ + sendOffsetSplit_[channelId], sendSizeSplit_[channelId],
            sendCountsSplit_[channelId]);
        // write模式下，远端dst数据ccl buffer slice
        DataSlice txDstSlice = DataSlice(
            remoteCclBuffAddr,
            remoteCclBuffIdx * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff
                + sendOffsetSplit_[channelId],
            sendSizeSplit_[channelId], sendCountsSplit_[channelId]);
        // read模式下，远端src数据ccl buffer slice
        DataSlice rxSrcSlice = DataSlice(
            remoteCclBuffAddr,
            remoteCclBuffIdx * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff
                + recvOffsetSplit_[channelId],
            recvSizeSplit_[channelId], recvCountsSplit_[channelId]);
        // read模式下，本端dst数据output buffer slice
        DataSlice rxDstSlice = DataSlice(
            tempAlgParams.buffInfo.outputPtr,
            tempAlgParams.rdispls[remoteRank] * dataTypeSize_ + recvOffsetSplit_[channelId], recvSizeSplit_[channelId],
            recvCountsSplit_[channelId]);

        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);
        rxSrcSlices.push_back(rxSrcSlice);
        rxDstSlices.push_back(rxDstSlice);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::BuildRemoteMemSlices(
    const TemplateDataParams& tempAlgParams, const u32 remoteRank, const ChannelInfo& channelSend,
    const ChannelInfo& channelRecv, const u32 channelId, std::vector<DataSlice>& txSrcSlices,
    std::vector<DataSlice>& txDstSlices, std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices) const
{
    // [中文导读] VC 的直接访问路径使用对端接收位移，只构造本地输入到对端输出的发送切片。
    if (opType_ == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        // alltoallvc对称内存零拷贝：远端用peerRdispls，只需发送切片
        (void)channelRecv;
        void* remoteOutputAddr = channelSend.remoteOutputGraphMode.addr;
        DataSlice txSrcSlice = DataSlice(
            tempAlgParams.buffInfo.inputPtr,
            tempAlgParams.sdispls[remoteRank] * dataTypeSize_ + sendOffsetSplit_[channelId], sendSizeSplit_[channelId],
            sendCountsSplit_[channelId]);
        DataSlice txDstSlice = DataSlice(
            remoteOutputAddr, tempAlgParams.peerRdispls[remoteRank] * dataTypeSize_ + sendOffsetSplit_[channelId],
            sendSizeSplit_[channelId], sendCountsSplit_[channelId]);
        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);
    } else {
        // alltoall对称内存零拷贝：本端用outputSliceStride，远端用outputSliceStride，完整send+recv切片
        // [中文导读] 等长 AllToAll 按固定步长定位本端 Peer 段，并用本端算法 Rank 定位对端相应段。
        u64 sendBaseOff = remoteRank * tempAlgParams.outputSliceStride + tempAlgParams.buffInfo.inBuffBaseOff;
        u64 recvBaseOff = remoteRank * tempAlgParams.outputSliceStride + tempAlgParams.buffInfo.outBuffBaseOff;
        u64 peerRecvOff = myAlgRank_ * tempAlgParams.outputSliceStride + tempAlgParams.buffInfo.outBuffBaseOff;
        u64 peerSendOff = myAlgRank_ * tempAlgParams.outputSliceStride + tempAlgParams.buffInfo.inBuffBaseOff;

        DataSlice txSrcSlice = DataSlice(
            tempAlgParams.buffInfo.inputPtr, sendBaseOff + sendOffsetSplit_[channelId], sendSizeSplit_[channelId],
            sendCountsSplit_[channelId]);
        DataSlice txDstSlice = DataSlice(
            channelSend.remoteOutputGraphMode.addr, peerRecvOff + sendOffsetSplit_[channelId],
            sendSizeSplit_[channelId], sendCountsSplit_[channelId]);
        DataSlice rxSrcSlice = DataSlice(
            channelRecv.remoteInputGraphMode.addr, peerSendOff + recvOffsetSplit_[channelId], recvSizeSplit_[channelId],
            recvCountsSplit_[channelId]);
        DataSlice rxDstSlice = DataSlice(
            tempAlgParams.buffInfo.outputPtr, recvBaseOff + recvOffsetSplit_[channelId], recvSizeSplit_[channelId],
            recvCountsSplit_[channelId]);

        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);
        rxSrcSlices.push_back(rxSrcSlice);
        rxDstSlices.push_back(rxDstSlice);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PreCopyByLoop(
    const std::vector<u32>& commRanks, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const std::vector<ThreadHandle>& threads, const TemplateDataParams& tempAlgParams)
{
    // [中文导读] 首轮遍历 Peer 并按通道分割发送片，将准备工作分散到各 Peer 的从 Thread 组。
    for (u32 rankIdx = 0; rankIdx < commRanks.size(); rankIdx++) {
        u32 remoteRank = commRanks[rankIdx];
        u32 myRankCclBuffIdx = 0; // myRank与remoteRank交互时myRank提供的cclbuffer index
        u32 remoteCclBuffIdx = 0; // myRank与remoteRank交互时remoteRank提供的cclbuffer index
        CalcCclBuffIdx(remoteRank, myRankCclBuffIdx, remoteCclBuffIdx);
        u32 queIdx = myRankCclBuffIdx * channelsPerRank_ + 1;
        if (channels.find(remoteRank) == channels.end()) {
            HCCL_ERROR("[InsTempAlltoAllVMesh1D][PreCopy] remoteRank[%u] does not exist in channels map!", remoteRank);
            return HCCL_E_PARA;
        }
        const std::vector<ChannelInfo>& curChannels = channels.at(remoteRank);
        u32 curValidChannelsSize = std::min(static_cast<u32>(curChannels.size()), channelsPerRank_);
        // send数据按照channel分片
        CHK_RET(CalcDataSplitByPortGroupCommon(
            tempAlgParams.sendCounts[remoteRank], dataTypeSize_, curChannels, sendCountsSplit_, sendSizeSplit_,
            sendOffsetSplit_, curValidChannelsSize));
        for (u32 channelId = 0; channelId < curValidChannelsSize; channelId++) {
            if (sendSizeSplit_[channelId] > 0) {
                CHK_RET(static_cast<HcclResult>(PreCopy(
                    tempAlgParams, threads[queIdx], myRankCclBuffIdx, remoteRank, sendSizeSplit_[channelId],
                    sendCountsSplit_[channelId], sendOffsetSplit_[channelId])));
            }
            queIdx++;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PreCopy(
    const TemplateDataParams& tempAlgParams, const ThreadHandle& thread, const u32 myRankCclBuffIdx,
    const u32 remoteRank, const u64& sendSize, const u64& sendCount, const u64& sendOffset) const
{
    // [中文导读] 前拷贝从用户输入的 Peer 段读取，写入该 Peer 的本端 CCL 槽和通道分片偏移。
    DataSlice localCopySrcSlice = DataSlice(
        tempAlgParams.buffInfo.inputPtr, tempAlgParams.sdispls[remoteRank] * dataTypeSize_ + sendOffset, sendSize,
        sendCount);
    DataSlice localCopyDstSlice = DataSlice(
        tempAlgParams.buffInfo.hcclBuff.addr,
        myRankCclBuffIdx * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff + sendOffset,
        sendSize, sendCount);
    CHK_RET(static_cast<HcclResult>(LocalCopy(thread, localCopySrcSlice, localCopyDstSlice)));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PostCopy(
    const TemplateDataParams& tempAlgParams, const ThreadHandle& thread, const u32 myRankCclBuffIdx,
    const u32 remoteRank, const u64& recvSize, const u64& recvCount, const u64& recvOffset) const
{
    // ccl buffer的数据搬运到usrout
    // 远端的数据发送到本端ccl buffer的slice
    // [中文导读] 后拷贝从本端收到数据的 CCL 槽读取，按该源 Peer 的接收位移写入用户输出。
    DataSlice localCopySrcSlice = DataSlice(
        tempAlgParams.buffInfo.hcclBuff.addr,
        myRankCclBuffIdx * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff + recvOffset,
        recvSize, recvCount);
    // 本端output buffer slice
    DataSlice localCopyDstSlice = DataSlice(
        tempAlgParams.buffInfo.outputPtr, tempAlgParams.rdispls[remoteRank] * dataTypeSize_ + recvOffset, recvSize,
        recvCount);
    CHK_RET(static_cast<HcclResult>(LocalCopy(thread, localCopySrcSlice, localCopyDstSlice)));
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAlltoAllVMesh1D::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    if (threadNum_ <= 1) {
        return;
    }
    // [中文导读] 每个从 Thread 在各自通知空间使用槽 0 接收全局主 Thread 的放行通知。
    u32 slaveThreadNum = threadNum_ - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAlltoAllVMesh1D::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    // [中文导读] 全局主 Thread 用不同索引接收每个从 Thread 的完成通知，便于逐一汇合。
    u32 notifyNum = threadNum_ - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}
} // namespace ops_hccl
