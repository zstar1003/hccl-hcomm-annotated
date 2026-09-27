/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_multi_deter_pipeline.h"
namespace hccl {
MultiDeterPipeline::MultiDeterPipeline(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

MultiDeterPipeline::~MultiDeterPipeline() {}

HcclResult MultiDeterPipeline::RunAsync() { return HCCL_SUCCESS; }

// ReduceScatterDeterPipeline
HcclResult MultiDeterPipeline::Prepare(
    [[maybe_unused]] HcomCollOpInfo* opInfo, [[maybe_unused]] DeviceMem& buffer, [[maybe_unused]] const u64 count,
    [[maybe_unused]] const u64 offset, [[maybe_unused]] const std::vector<Slice>& slices,
    [[maybe_unused]] const SubCommInfo& level0CommInfo, [[maybe_unused]] const SubCommInfo& level1CommInfo,
    [[maybe_unused]] Stream& mainStream, [[maybe_unused]] std::vector<Stream>& subStream,
    [[maybe_unused]] std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
    [[maybe_unused]] std::vector<std::shared_ptr<LocalNotify>>& notifySub)
{
    return HCCL_SUCCESS;
}

// AllReduceDeterPipeline
HcclResult MultiDeterPipeline::Prepare(
    [[maybe_unused]] HcomCollOpInfo* opInfo, [[maybe_unused]] DeviceMem& inBuffer,
    [[maybe_unused]] DeviceMem& outBuffer, [[maybe_unused]] const u64 count,
    [[maybe_unused]] const std::vector<Slice>& slices, [[maybe_unused]] const SubCommInfo& level0CommInfo,
    [[maybe_unused]] const SubCommInfo& level1CommInfo, [[maybe_unused]] Stream& mainStream,
    [[maybe_unused]] std::vector<Stream>& subStream,
    [[maybe_unused]] std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
    [[maybe_unused]] std::vector<std::shared_ptr<LocalNotify>>& notifySub)
{
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::MainWaitSub(u32 begin, u32 end)
{
    for (u32 signalIndex = begin; signalIndex < end; signalIndex++) {
        CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, streamNotifyMain_[signalIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::SubRecordMain(u32 begin, u32 end)
{
    for (u32 streamIndex = begin; streamIndex < end; streamIndex++) {
        CHK_RET(LocalNotify::Post(subStreams_[streamIndex], dispatcher_, streamNotifyMain_[streamIndex], -1));
    }
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::MainRecordSub(u32 begin, u32 end)
{
    for (u32 signalIndex = begin; signalIndex < end; signalIndex++) {
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, streamNotifySub_[signalIndex], -1));
    }
    return HCCL_SUCCESS;
}

// begin max = 7, end max = 11
HcclResult MultiDeterPipeline::SubWaitMain(u32 begin, u32 end)
{
    for (u32 streamIndex = begin; streamIndex < end; streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            subStreams_[streamIndex], dispatcher_, streamNotifySub_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::GetRemoteCclbufferDeviceMem(
    [[maybe_unused]] u32 inputSliceIndex, [[maybe_unused]] LINK link, [[maybe_unused]] u32 outputSliceIndex,
    [[maybe_unused]] DeviceMem& remoteMem)
{
    return HCCL_SUCCESS;
}

HcclResult
MultiDeterPipeline::GetLocalUserInDeviceMem([[maybe_unused]] u32 rankIdInAllRanks, [[maybe_unused]] DeviceMem& locaMem)
{
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::GetLocalUserOutDeviceMem(
    [[maybe_unused]] u32 rankIdInAllRanks, [[maybe_unused]] DeviceMem& localMem)
{
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::GetLocalInCclbufferDeviceMem(
    [[maybe_unused]] u32 rankIdInAllRanks, [[maybe_unused]] DeviceMem& localMem, [[maybe_unused]] bool ifUseLastSize)
{
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::GetLocalOutCclbufferDeviceMem(
    [[maybe_unused]] u32 rankIdInAllRanks, [[maybe_unused]] DeviceMem& localMem, [[maybe_unused]] bool ifUseLastSize)
{
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::RunLocalCopy() { return HCCL_SUCCESS; }

HcclResult MultiDeterPipeline::RunIntraAlltoallPreSync([[maybe_unused]] u32 step) { return HCCL_SUCCESS; }

HcclResult MultiDeterPipeline::RunIntraAlltoall(u32 step)
{
    u32 recvServerId = GetPreServerIdByStep(step);  // 从上一个收 2
    u32 sendServerId = GetNextServerIdByStep(step); // 发给发下一个 1
    // 机内alltoall full mesh收集数据，是为了收集机内第sendServerId整块的内存（包含intraRankId_块）
    // 该索引是为了计算第sendServerId整块内的内存序号块（0~intraRankId_-1）
    std::vector<u32> localUsrInIndex;
    for (u32 i = intraRankId_ + 1; i < intraRankSize_ + intraRankId_; ++i) {
        localUsrInIndex.push_back(i % intraRankSize_);
    }
    for (u32 i = 0; i < intraRankSize_ - 1; ++i) {
        u32 sendIntraRankId = GetNextIntraRankIdByStep(i + 1);
        LINK sendIntraLink = intraLinks_[sendIntraRankId];
        DeviceMem srcMem;
        DeviceMem dstMem;
        // 从usrin收集发给下一个cclbuffer的数据, 收集的所有数据需要发送给机间序号为sendServerId的server
        u32 needSendInputndex = GetRankIdx(sendServerId, localUsrInIndex[i]);
        // 发送数据到cclbuffer 索引为[intraRankId_, localUsrInIndex[i]]
        u32 recvIntraRankIdx = alltoallRecvBlockIdxMap_[intraRankId_][localUsrInIndex[i]];
        u32 recvCclbufferIndex = GetRankIdx(recvServerId, recvIntraRankIdx);
        CHK_RET(GetLocalUserInDeviceMem(needSendInputndex, srcMem));
        CHK_RET(GetRemoteCclbufferDeviceMem(needSendInputndex, sendIntraLink, recvCclbufferIndex, dstMem));
        // 发送给机内rank索引 [serverId_, sendIntraRankId]
        u32 remoteUserRank = GetRankIdx(serverId_, sendIntraRankId);
        // SDMA copy write语义，因为只能写到cclbuffer
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dstMem, srcMem, subStreams_[i], remoteUserRank, sendIntraLink->GetLinkType()));
        CHK_RET(sendIntraLink->TxDataSignal(subStreams_[i]));
        CHK_RET(sendIntraLink->RxDataSignal(subStreams_[i]));
        HCCL_DEBUG("[%s] intra-server SDMA send, intraRank: [%u] -> [%u]", __func__, intraRankId_, sendIntraRankId);
        HCCL_DEBUG(
            "[%s] intra-server SDMA send, mem: inputMem[%u, %u] -> cclbuffer[%u, %u]; cclbufferNo[%u] -> [%u]",
            __func__, sendServerId, localUsrInIndex[i], recvServerId, recvIntraRankIdx, needSendInputndex,
            recvCclbufferIndex);
    }
    HCCL_INFO("[%s] intra-server step[%u] run alltoall success", __func__, step);
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::GroupTasksByStream(
    u32 activeCount, const std::vector<bool>& isReduceBlock, u32 retIndex,
    std::vector<std::vector<std::vector<std::pair<u32, u32>>>>& batchStreamTasks, // 输出：批次→流→任务
    std::vector<bool>& processed, std::vector<u32>& origIdxMap, u32& newActiveCount) const
{
    batchStreamTasks.clear(); // 清空批次任务
    processed.assign(activeCount, false);
    newActiveCount = 0;

    const u32 mergeStep = 2;
    const u32 batchSize = MAX_REDUCE_STREAM_NUM;
    const u32 totalGroups = (activeCount + mergeStep - 1) / mergeStep;
    const u32 batchNum = (totalGroups + batchSize - 1) / batchSize;

    // 逐批生成流任务
    for (u32 batch = 0; batch < batchNum; batch++) {
        // 初始化当前批次的流任务（MAX_REDUCE_STREAM_NUM条流）
        std::vector<std::vector<std::pair<u32, u32>>> streamTasks(MAX_REDUCE_STREAM_NUM);
        u32 startGroup = batch * batchSize;
        u32 endGroup = std::min((batch + 1) * batchSize, totalGroups);

        // 处理当前批次的分组
        for (u32 group = startGroup; group < endGroup; group++) {
            u32 idx0 = group * mergeStep;
            u32 idx1 = idx0 + 1;
            if (idx1 >= activeCount) {
                processed[idx0] = false;
                continue;
            }

            // 选择dst/src（原有优先级逻辑不变）
            u32 dstIdx, srcIdx;
            if (origIdxMap[idx0] == retIndex) { // 当前idx0的原始索引是目标块，强制为dst
                dstIdx = idx0;
                srcIdx = idx1;
            } else if (origIdxMap[idx1] == retIndex) { // 当前idx1的原始索引是目标块，强制为dst
                dstIdx = idx1;
                srcIdx = idx0;
            } else if (isReduceBlock[idx0] && !isReduceBlock[idx1]) {
                dstIdx = idx0;
                srcIdx = idx1;
            } else if (!isReduceBlock[idx0] && isReduceBlock[idx1]) {
                dstIdx = idx1;
                srcIdx = idx0;
            } else {
                dstIdx = std::max(idx0, idx1);
                srcIdx = std::min(idx0, idx1);
            }

            // 分配到当前批次的流任务中
            u32 batchInnerGroupIdx = group - startGroup;
            u32 streamId = batchInnerGroupIdx % MAX_REDUCE_STREAM_NUM;
            streamTasks[streamId].emplace_back(srcIdx, dstIdx);

            processed[srcIdx] = true;
            processed[dstIdx] = false;

            HCCL_DEBUG(
                "[%s] batch[%u] group[%u] merge src[%u] -> dst[%u] on stream[%u]", __func__, batch, group, srcIdx,
                dstIdx, streamId + reduceStreamBegin_);
        }

        // 将当前批次的流任务加入总批次列表
        batchStreamTasks.push_back(streamTasks);
    }

    newActiveCount = std::count(processed.begin(), processed.end(), false);
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::BatchPostNotifyForStreams(
    [[maybe_unused]] const std::vector<std::vector<std::pair<u32, u32>>>& streamTasks,
    [[maybe_unused]] bool isStartPhase, [[maybe_unused]] bool useMainStream)
{
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::ExecuteStreamTasks(
    const std::vector<std::vector<std::pair<u32, u32>>>& streamTasks, const std::vector<DeviceMem>& validMem,
    std::vector<u32>& origIdxMap, bool useMainStream)
{
    for (u32 s = 0; s < MAX_REDUCE_STREAM_NUM; s++) {
        if (streamTasks[s].empty())
            continue;

        u32 streamIdx = reduceStreamBegin_ + s;
        Stream& subStream = subStreams_[streamIdx];
        Stream& stream = useMainStream ? mainStream_ : subStream;

        for (const auto& task : streamTasks[s]) {
            u32 srcIdx = task.first;
            u32 dstIdx = task.second;
            const DeviceMem& dstMem = validMem[dstIdx];
            const DeviceMem& srcMem = validMem[srcIdx];
            u64 count = srcMem.size() / unitSize_;
            CHK_RET(HcclReduceAsync(
                dispatcher_, srcMem.ptr(), count, dataType_, reductionOp_, stream, dstMem.ptr(), INVALID_VALUE_RANKID,
                LinkType::LINK_ONCHIP, INLINE_REDUCE_BIT));
            HCCL_DEBUG(
                "[%s] stream[%u] execute task: merge src[%u] -> dst[%u], origSrc[%u] -> origDst[%u]", __func__,
                useMainStream ? 0 : streamIdx, srcIdx, dstIdx, origIdxMap[srcIdx], origIdxMap[dstIdx]);
        }
    }
    return HCCL_SUCCESS;
}

void MultiDeterPipeline::CompressActiveSet(
    std::vector<DeviceMem>& validMem, std::vector<bool>& isReduceBlock, std::vector<u32>& origIdxMap,
    const std::vector<bool>& processed, u32& trackedTargetIdx, const u32 origRetIndex) const
{
    std::vector<DeviceMem> newValidMem;
    std::vector<bool> newIsReduceBlock;
    std::vector<u32> newOrigIdxMap;
    u32 newTrackedTargetIdx = 0;
    bool foundTarget = false;

    // 保留未处理的块（processed=false，即dst块）
    for (u32 i = 0; i < validMem.size(); i++) {
        if (!processed[i]) { // 仅保留dst块，移除src块（processed=true）
            newValidMem.push_back(validMem[i]);
            newIsReduceBlock.push_back(isReduceBlock[i]);
            newOrigIdxMap.push_back(origIdxMap[i]);

            // 追踪原始目标块（retIndex）的新索引
            if (!foundTarget && origIdxMap[i] == origRetIndex) {
                newTrackedTargetIdx = newValidMem.size() - 1;
                foundTarget = true;
            }
        }
    }

    validMem.swap(newValidMem);
    isReduceBlock.swap(newIsReduceBlock);
    origIdxMap.swap(newOrigIdxMap);
    // 未找到目标块时，默认指向最后一个块
    trackedTargetIdx = foundTarget ? newTrackedTargetIdx : (validMem.empty() ? 0 : validMem.size() - 1);
    HCCL_DEBUG(
        "[%s] compressed: old size[%llu], new size[%llu], trackedTargetIdx[%u]", __func__, processed.size(),
        validMem.size(), trackedTargetIdx);
}

HcclResult MultiDeterPipeline::LocalReduce(
    std::vector<DeviceMem>& reduceMem, std::vector<bool>& isReduceBlock, u32 retIndex, bool useMainStream)
{
    const u32 totalBlockCount = reduceMem.size();
    // 校验1：容器大小匹配 + retIndex越界
    if (reduceMem.size() != isReduceBlock.size() || retIndex >= totalBlockCount) {
        HCCL_ERROR(
            "[%s] Invalid param (size mismatch: %llu vs %llu, retIndex: %u >= %u)", __func__, reduceMem.size(),
            isReduceBlock.size(), retIndex, totalBlockCount);
        return HCCL_E_PARA;
    }
    // 校验2：目标内存块有效
    const DeviceMem& targetCCLBuffer = reduceMem[retIndex];
    if (targetCCLBuffer.ptr() == nullptr || targetCCLBuffer.size() == 0) {
        HCCL_ERROR(
            "[%s] Target CCLBuffer invalid (ptr: %p, size: %llu)", __func__, targetCCLBuffer.ptr(),
            targetCCLBuffer.size());
        return HCCL_E_MEMORY;
    }

    std::vector<DeviceMem> validMem = std::move(reduceMem); // 外层不使用reduceMem
    std::vector<bool> validIsReduceBlock = std::move(isReduceBlock);
    // 1. 动态追踪目标块索引 2. 原索引→当前索引的映射表
    std::vector<u32> origIdxMap(validMem.size());
    for (size_t i = 0; i < origIdxMap.size(); ++i) {
        origIdxMap[i] = i;
    }

    if (validMem.size() == 1) {
        HCCL_ERROR("[%s] validMem size is one, only target block valid", __func__);
        return HCCL_E_PARA;
    }

    u32 trackedTargetIdx = retIndex;
    u32 activeCount = validMem.size();
    const u32 origRetIndex = retIndex;

    while (activeCount > 1) {
        std::vector<std::vector<std::vector<std::pair<u32, u32>>>> batchStreamTasks; // 批次→流→任务
        std::vector<bool> processed(activeCount, false);
        u32 newActiveCount = 0;

        // 2.1 生成按批次组织的流任务
        CHK_RET(GroupTasksByStream(
            activeCount, validIsReduceBlock, origRetIndex, batchStreamTasks, processed, origIdxMap, newActiveCount));

        // 2.2 逐批执行流任务（串行处理每批）
        for (const auto& streamTasks : batchStreamTasks) {
            // a. 执行start phase notify（仅处理有任务的流）
            CHK_RET(BatchPostNotifyForStreams(streamTasks, true, useMainStream));
            // b. 执行当前批次的流任务（src→dst归约）
            CHK_RET(ExecuteStreamTasks(streamTasks, validMem, origIdxMap, useMainStream));
            // c. 执行sync phase notify（等待当前批次完成）
            CHK_RET(BatchPostNotifyForStreams(streamTasks, false, useMainStream));
        }
        // 2.3 压缩活跃块（移除src块，保留dst块）
        CompressActiveSet(validMem, validIsReduceBlock, origIdxMap, processed, trackedTargetIdx, origRetIndex);
        activeCount = validMem.size();
        HCCL_DEBUG("[LocalReduce] round done: activeCount=%u -> %u", newActiveCount, activeCount);
    }
    HCCL_DEBUG(
        "[%s] Local reduce success (merge to retIndex[%u] CCLBuffer, final tracked idx[%u])", __func__, retIndex,
        trackedTargetIdx);
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::RunIntraLocalReduce([[maybe_unused]] u32 step) { return HCCL_SUCCESS; }

HcclResult MultiDeterPipeline::RunInterSend([[maybe_unused]] u32 step) { return HCCL_SUCCESS; }

HcclResult MultiDeterPipeline::RunFinalReduce() { return HCCL_SUCCESS; }

HcclResult MultiDeterPipeline::AlltoallSync([[maybe_unused]] u32 step, [[maybe_unused]] bool isStartPhase)
{
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::LocalReduceSync([[maybe_unused]] u32 step, [[maybe_unused]] bool isStartPhase)
{
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::AlltoallLocalReduceSync(u32 step, bool isStartPhase)
{
    bool alltoallStep = (step < allSteps_);
    bool localReduceStep = (step > 1 && step < allSteps_ + 1);
    if (alltoallStep) {
        CHK_RET(AlltoallSync(step, isStartPhase));
    }
    if (localReduceStep) {
        CHK_RET(LocalReduceSync(step, isStartPhase));
    }
    return HCCL_SUCCESS;
}

HcclResult MultiDeterPipeline::RunAsyncLocalReduceSerial()
{
    HCCL_INFO(
        "[MultiDeterPipeline] run begin: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p]", userRank_, userRankSize_,
        usrInMemPtr_, usrOutMemPtr_);
    CHK_SMART_PTR_NULL(dispatcher_);
    // 以机内8卡为例主流 + 从流 = 1 + 7 + 4 = 12
    allSteps_ = serverSize_;
    // #1 机间发送，#2 local reduce，#3 机内alltoall
    // 以机间的pairwise来分步，#n表示pairwise的第n步
    for (u32 step = 1; step < allSteps_ + 1; step++) {
        HCCL_DEBUG(
            "[%s] userRank[%u], intraRankId[%u], serverId[%u], step[%u/%u] begin", __func__, userRank_, intraRankId_,
            serverId_, step, allSteps_);
        // alltoall 主从流同步+前同步+拉齐
        CHK_RET(AlltoallSync(step, true));
        if (step < allSteps_ - 1) {
            CHK_RET(RunIntraAlltoallPreSync(step));
        }
        if (step == allSteps_ - 1) {
            CHK_RET(RunIntraAlltoallPreSync(0));
        }
        if (step == 1) {
            CHK_RET(RunLocalCopy());
        }
        if (step > 1) {
            CHK_RET(RunInterSend(step - 1));
        }
        // #1 机内RS alltoall + local reduce串行
        if (step < allSteps_) {
            CHK_RET(RunIntraAlltoall(step));
            CHK_RET(AlltoallSync(step, false));
            CHK_RET(LocalReduceSync(step, true));
            CHK_RET(RunIntraLocalReduce(step));
            CHK_RET(LocalReduceSync(step, false));
        } else {
            // #0 机内RS alltoall + local reduce串行，#0不需要向其他机发送数据
            CHK_RET(RunIntraAlltoall(0));
            CHK_RET(AlltoallSync(0, false));
            CHK_RET(LocalReduceSync(0, true));
            CHK_RET(RunIntraLocalReduce(0));
            CHK_RET(LocalReduceSync(0, false));
        }
    }
    // 总local reduce
    CHK_RET(RunFinalReduce());
    HCCL_INFO("[MultiDeterPipeline] MultiDeterPipeline success userRank[%u] ", userRank_);
    return HCCL_SUCCESS;
}

// 每个server内首先要进行alltoall full mesh收集数据，再进行机内local reduce，最后发送给指定server
HcclResult MultiDeterPipeline::RunAsyncReduceScatterPipeline()
{
    constexpr u64 HCCL_MEDIUM_COUNT_2_MB = 2 * 1024 * 1024;
    // 2机或者数据量小于2MB走localreduce串行算法
    if (serverSize_ <= LOCAL_REDUCE_SERIIAL_ALG_SERVER_NUM || GetLocalReduceSerialThresh() < HCCL_MEDIUM_COUNT_2_MB) {
        CHK_RET(RunAsyncLocalReduceSerial());
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[MultiDeterPipeline] [%s] begin, userRank[%u]", __func__, userRank_);
    // 以机内8卡为例主流 + 从流 = 1 + 7 + 4 = 12
    // pairwise总共需要serverSize_步，#1机内alltoall只能自己执行，无法和其他步骤并行，所以总共需要serverSize_ + 1个步骤
    allSteps_ = serverSize_ + 1;
    // #1 机间发送，#2 local reduce，#3 机内alltoall
    // 以机间的pairwise来分步，#n表示pairwise的第n步
    for (u32 step = 1; step < allSteps_ + 1; step++) {
        HCCL_DEBUG(
            "[%s] userRank[%u], intraRankId[%u], serverId[%u], step[%u/%u] begin", __func__, userRank_, intraRankId_,
            serverId_, step, allSteps_);
        CHK_RET(AlltoallLocalReduceSync(step, true));
        if (step == 1) {
            CHK_RET(RunLocalCopy());
        }
        if (step < allSteps_ - 1) {
            CHK_RET(RunIntraAlltoallPreSync(step));
        }
        if (step == allSteps_ - 1) {
            CHK_RET(RunIntraAlltoallPreSync(0));
        }
        if (step > STEP_OFFSET_TWO) {
            CHK_RET(RunInterSend(step - STEP_OFFSET_TWO));
        }
        // allSteps_最小为3
        if (step > 1 && step < allSteps_) {
            CHK_RET(RunIntraLocalReduce(step - 1));
        }
        // 最后一步，需要额外执行 #0 机内RS local reduce，#0不需要向其他机发送数据
        if (step == allSteps_) {
            CHK_RET(RunIntraLocalReduce(0));
        }
        if (step < allSteps_ - 1) {
            CHK_RET(RunIntraAlltoall(step));
        }
        if (step == allSteps_ - 1) {
            // #0 机内RS alltoall
            CHK_RET(RunIntraAlltoall(0));
        }
        CHK_RET(AlltoallLocalReduceSync(step, false));
    }
    // 总local reduce
    CHK_RET(RunFinalReduce());
    HCCL_INFO("[MultiDeterPipeline] [%s] end, userRank[%u]", __func__, userRank_);
    return HCCL_SUCCESS;
}

// 遍历所有发送方rank和发送方块索引，预计算映射关系，目的是构造出下属矩阵
// srcRank\srcBlockIdx	0	1	2
//         0	       MAX	0	0
//         1	        0  MAX	1
//         2            1	1  MAX
// 例1，发送方是rank0，接收端是rank1，那么接收端非自身 rank 列表为[0, 2]，那么rank0的索引为0，所以就发到rank1的第0块内存
// 例2，发送方是rank0，接收端是rank2，那么接收端非自身 rank 列表为[0, 1]，那么rank0的索引为0，所以就发到rank2的第0块内存
// 例3，发送方是rank1，接收端是rank2，那么接收端非自身 rank 列表为[0, 1]，那么rank1的索引为1，所以就发到rank2的第1块内存
void MultiDeterPipeline::InitAlltoallRecvBlockIdxMap()
{
    const u32 rankSize = intraRankSize_;
    alltoallRecvBlockIdxMap_.resize(rankSize, std::vector<u32>(rankSize, UINT32_MAX));
    if (rankSize <= 1) {
        return;
    }
    // 规则一：接收端的块索引 = 发送方 rank 在「接收端非自身 rank 列表」中的索引
    std::vector<std::vector<u32>> dstRankToRankIndex(rankSize, std::vector<u32>(rankSize, UINT32_MAX));
    for (u32 dstRank = 0; dstRank < rankSize; ++dstRank) {
        for (u32 srcRank = 0; srcRank < rankSize; ++srcRank) {
            if (srcRank == dstRank) {
                continue;
            }
            // 若 srcRank < dstRank：索引 = srcRank, 若 srcRank > dstRank：索引 = srcRank - 1
            const u32 idx = (srcRank < dstRank) ? srcRank : (srcRank - 1);
            dstRankToRankIndex[dstRank][srcRank] = idx;
        }
    }
    for (u32 srcRank = 0; srcRank < rankSize; ++srcRank) {
        for (u32 srcBlockIdx = 0; srcBlockIdx < rankSize; ++srcBlockIdx) {
            // 规则二：发送方块索引 = 接收端 rank
            const u32 dstRank = srcBlockIdx;
            // 跳过自身发送的无效场景
            if (srcRank == dstRank) {
                continue;
            }
            // 查找发送方rank在列表中的索引，存入映射表
            const u32 dstBlockIdx = dstRankToRankIndex[dstRank][srcRank];
            alltoallRecvBlockIdxMap_[srcRank][srcBlockIdx] = dstBlockIdx;
            HCCL_DEBUG(
                "[%s] srcRank[%u], srcBlockIdx[%u] -> dstRank[%u], dstBlockIdx[%u]", __func__, srcRank, srcBlockIdx,
                dstRank, dstBlockIdx);
        }
    }
}

HcclResult MultiDeterPipeline::PrepareTopoInfo(const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo)
{
    serverSize_ = level1CommInfo.localRankSize;
    CHK_PRT_RET(
        serverSize_ < MIN_SERVER_NUM,
        HCCL_ERROR("[%s] Unexpected inter rank size[%u], which should >= 2.", __func__, serverSize_), HCCL_E_PARA);

    intraRankSize_ = level0CommInfo.localRankSize;
    CHK_PRT_RET(
        intraRankSize_ < MIN_INTRA_RANK_NUM,
        HCCL_ERROR("[%s] Unexpected intra rank size[%u], which should >= 3.", __func__, intraRankSize_), HCCL_E_PARA);
    intraRankId_ = level0CommInfo.localRank;
    serverId_ = level1CommInfo.localRank;
    userRankSize_ = intraRankSize_ * serverSize_;
    userRank_ = intraRankId_ + serverId_ * intraRankSize_;

    intraLinks_ = level0CommInfo.links;  // 节点内
    serverLinks_ = level1CommInfo.links; // 节点间
    HCCL_INFO(
        "[%s] opInfo: dataType[%u], unitSize[%u], memSliceSize[%u], usrInMem[%p], usrOutMem[%p], reductionOp[%u]",
        __func__, dataType_, unitSize_, memSliceSize_, usrInMemPtr_, usrOutMemPtr_, reductionOp_);
    HCCL_INFO(
        "[%s] topoInfo: userRank[%u], intraRankId[%u], intraRankSize[%u], serverId[%u], interRankSize[%u]", __func__,
        userRank_, intraRankId_, intraRankSize_, serverId_, serverSize_);
    HCCL_INFO(
        "[%s] topoInfo: serverLinksNum[%zu], intraLinksNum[%zu]", __func__, serverLinks_.size(), intraLinks_.size());
    InitAlltoallRecvBlockIdxMap();
    return HCCL_SUCCESS;
}
} // namespace hccl
