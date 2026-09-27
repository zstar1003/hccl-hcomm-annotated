/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_unified_march.h"
#include "alg_template_register.h"

namespace hccl {
static const u32 NEIGHBORS_NUM_TWO = 2; //  2: 邻居数量
static const u32 NEIGHBORS_NUM_ONE = 1; //  1: 邻居数量
static const u32 DIVISOR_NUM_TWO = 2;

AllGatherUnifiedMarch::AllGatherUnifiedMarch(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllGatherUnifiedMarch::~AllGatherUnifiedMarch() {}

std::string AllGatherUnifiedMarch::GetStreamIndexString()
{
    std::string res = "";
    for (u32 streamIndex = 0; streamIndex < subStreams_.size(); streamIndex++) {
        res += std::to_string(streamIndex) + ", ";
    }
    return res;
}

// 主流所有从流
HcclResult AllGatherUnifiedMarch::NotifySubStreamStart(u32 streamSize)
{
    CHK_PRT_RET(
        streamSize > subStreams_.size() || streamSize > meshSignalSubToMain_.size(),
        HCCL_ERROR(
            "[AllGatherUnifiedMarch][NotifySubStreamStart] streamSize[%u] is out of range "
            "subStreams_ size[%zu] or meshSignalSubToMain_ size[%zu]",
            streamSize, subStreams_.size(), meshSignalSubToMain_.size()),
        HCCL_E_PARA);
    for (u32 streamIndex = 0; streamIndex < streamSize; streamIndex++) {
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, meshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(
            subStreams_[streamIndex], dispatcher_, meshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG(
        "[AllGatherUnifiedMarch][NotifySubStreamStart] userRank [%u] main stream notify substream [%s]", intraRank_,
        GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherUnifiedMarch::WaitSubStreamFinish(u32 streamSize)
{
    CHK_PRT_RET(
        streamSize > subStreams_.size() || streamSize > meshSignalMainToSub_.size(),
        HCCL_ERROR(
            "[AllGatherUnifiedMarch][NotifySubStreamStart] streamSize[%u] is out of range "
            "subStreams_ size[%zu] or meshSignalMainToSub_ size[%zu]",
            streamSize, subStreams_.size(), meshSignalMainToSub_.size()),
        HCCL_E_PARA);
    for (u32 streamIndex = 0; streamIndex < streamSize; streamIndex++) {
        CHK_RET(LocalNotify::Post(
            subStreams_[streamIndex], dispatcher_, meshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, meshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG(
        "[AllGatherUnifiedMarch][WaitSubStreamFinish] userRank [%u] main stream wait substream [%s]", intraRank_,
        GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherUnifiedMarch::DoSerialSDMA(
    void* remoteSrcAddr, u64 remoteOffsetByte, void* dstAddr, Stream& temStream, LINK& tmpLink, u64 memSize, u32 step)
{
    (void)step;
    for (u32 sliceIdx = 0; sliceIdx < (multRingsUserMemSlice_[0].size() / intraRankSize_); sliceIdx++) {
        struct hccl::Transport::Buffer remoteBuf;
        remoteBuf.addr = static_cast<u8*>(remoteSrcAddr) + remoteOffsetByte + multRingsUserMemSlice_[0][sliceIdx].offset
                         + baseOffset_;
        remoteBuf.size = memSize;
        struct hccl::Transport::Buffer localBuf;
        localBuf.addr = static_cast<u8*>(dstAddr) + multRingsUserMemSlice_[0][sliceIdx].offset;
        localBuf.size = memSize;
        HCCL_DEBUG(
            "intraRank[%u] slice[%u] offset[%llu] do SDMA read with remoteBuf[addr[%p], size[%llu]] and "
            "localBuf[addr[%p], size[%llu]]",
            intraRank_, sliceIdx, multRingsUserMemSlice_[0][sliceIdx].offset, remoteBuf.addr, remoteBuf.size,
            localBuf.addr, localBuf.size);
        CHK_RET(tmpLink->ReadSync(localBuf, remoteBuf, temStream));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherUnifiedMarch::NotifyNeighborsStart(LINK& prevIntraLink, LINK& nextIntralLink, u32 neighbors)
{
    // 图模式保持使用Post/Wait接口
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        for (u32 neighborRankId = 0; neighborRankId < neighbors; neighborRankId++) {
            HCCL_DEBUG("[AllGatherUnifiedMarch][NotifyNeighborsStart]neighborRankId is %u", neighborRankId);
            // notify是否越界由平台侧保证
            if (neighborRankId == 0) {
                CHK_RET(nextIntralLink->Post(notifyIdx_, subStreams_[neighborRankId])); // AckRecord
                CHK_RET(prevIntraLink->Wait(notifyIdx_, subStreams_[neighborRankId]));  // AckWait
            } else if (neighborRankId == 1) {
                CHK_RET(prevIntraLink->Post(notifyIdx_, subStreams_[neighborRankId]));  // AckRecord
                CHK_RET(nextIntralLink->Wait(notifyIdx_, subStreams_[neighborRankId])); // AckWait
            }
        }
        HCCL_DEBUG(
            "[AllGatherUnifiedMarch][NotifyNeighborsStart] intraRank[%u] switch on [%u] neighbors done", intraRank_,
            neighbors);
        return HCCL_SUCCESS;
    }

    // 一条流负责一个环
    for (u32 neighborRankId = 0; neighborRankId < neighbors; neighborRankId++) {
        // 交替使用Ack和DataSignal两种notify
        const u32 NOTIFY_IDX_TWO = 2;
        if (neighborRankId == 0) {
            if (notifyIdx_ % NOTIFY_IDX_TWO == 0) {
                CHK_RET(nextIntralLink->TxAck(subStreams_[neighborRankId])); // AckRecord
                CHK_RET(prevIntraLink->RxAck(subStreams_[neighborRankId]));  // AckWait
            } else {
                CHK_RET(nextIntralLink->TxDataSignal(subStreams_[neighborRankId])); // DataRecord
                CHK_RET(prevIntraLink->RxDataSignal(subStreams_[neighborRankId]));  // DataWait
            }
        } else if (neighborRankId == 1) {
            if (notifyIdx_ % NOTIFY_IDX_TWO == 0) {
                CHK_RET(prevIntraLink->TxAck(subStreams_[neighborRankId]));  // AckRecord
                CHK_RET(nextIntralLink->RxAck(subStreams_[neighborRankId])); // AckWait
            } else {
                CHK_RET(prevIntraLink->TxDataSignal(subStreams_[neighborRankId]));  // DataRecord
                CHK_RET(nextIntralLink->RxDataSignal(subStreams_[neighborRankId])); // DataWait
            }
        }
    }
    HCCL_DEBUG(
        "[AllGatherUnifiedMarch][NotifyNeighborsStart] intraRank[%u] switch on [%u] neighbors done", intraRank_,
        neighbors);

    return HCCL_SUCCESS;
}

HcclResult AllGatherUnifiedMarch::NotifyNeighborsEnd(LINK& prevIntraLink, LINK& nextIntralLink, u32 neighbors)
{
    HCCL_DEBUG("[AllGatherUnifiedMarch]NotifyNeighborsEnd start.");
    for (u32 neighborRankId = 0; neighborRankId < neighbors; neighborRankId++) {
        if (neighborRankId == 0) {
            CHK_RET(prevIntraLink->TxDataSignal(subStreams_[neighborRankId])); // DataRecord
            CHK_RET(nextIntralLink->RxDataSignal(subStreams_[neighborRankId]));
        } else if (neighborRankId == 1) {
            CHK_RET(nextIntralLink->TxDataSignal(subStreams_[neighborRankId]));
            CHK_RET(prevIntraLink->RxDataSignal(subStreams_[neighborRankId]));
        }
    }
    HCCL_DEBUG(
        "[AllGatherUnifiedMarch][NotifyNeighborsStart] intraRank[%u] notified [%u] neighbors sdma read done",
        intraRank_, neighbors);

    return HCCL_SUCCESS;
}

HcclResult AllGatherUnifiedMarch::RunSingleStep(u32 ringPrevRank, u32 ringNextRank, u32 step, u32 totalStep)
{
    LINK prevIntraLink = links_[ringPrevRank];
    LINK nextIntralLink = links_[ringNextRank];
    CHK_SMART_PTR_NULL(prevIntraLink);
    CHK_SMART_PTR_NULL(nextIntralLink);
    u32 neighbors = (ringPrevRank == ringNextRank) ? NEIGHBORS_NUM_ONE : NEIGHBORS_NUM_TWO;
    CHK_RET(NotifyNeighborsStart(prevIntraLink, nextIntralLink, neighbors));

    // 拉齐 从流record主流、主流record从流 保证从流同时开始做SDMA
    CHK_RET(WaitSubStreamFinish(neighbors));
    CHK_RET(NotifySubStreamStart(neighbors));

    // 从前向rank读取数据
    void* preRemDMAMemPtr = nullptr;
    CHK_RET(prevIntraLink->GetRemoteMem(UserMemType::OUTPUT_MEM, &preRemDMAMemPtr));
    u32 preDataIndex = (intraRank_ + intraRankSize_ - step - 1) % intraRankSize_;
    u64 preOffsetByte = preDataIndex * blockDataByte_;
    void* preDstAddr = static_cast<u8*>(userOutput_.ptr()) + preOffsetByte;
    u64 preRemoteOffsetByte = preOffsetByte;
    CHK_RET(DoSerialSDMA(
        preRemDMAMemPtr, preRemoteOffsetByte, preDstAddr, subStreams_[0], prevIntraLink, blockDataByte_, step));
    HCCL_INFO(
        "[AllGatherUnifiedMarch][RunSingleStep] intraRank [%u] read from ringPrevRank [%u] done", intraRank_,
        ringPrevRank);

    // 从后向rank读取数据
    if (neighbors > NEIGHBORS_NUM_ONE) {
        void* nextRemDMAMemPtr = nullptr;
        CHK_RET(nextIntralLink->GetRemoteMem(UserMemType::OUTPUT_MEM, &nextRemDMAMemPtr));
        u32 nextDataIndex = (intraRank_ + 1 + step) % intraRankSize_;
        u64 nextOffsetByte = nextDataIndex * blockDataByte_;
        void* nextDstAddr = static_cast<u8*>(userOutput_.ptr()) + nextOffsetByte;
        u64 nextRemoteOffsetByte = nextOffsetByte;
        CHK_RET(DoSerialSDMA(
            nextRemDMAMemPtr, nextRemoteOffsetByte, nextDstAddr, subStreams_[1], nextIntralLink, blockDataByte_, step));
        HCCL_INFO(
            "[AllGatherUnifiedMarch][RunSingleStep] intraRank [%u] read from ringNextRank [%u] done", intraRank_,
            ringNextRank);
    }

    if (step == totalStep - 1) {
        CHK_RET(NotifyNeighborsEnd(prevIntraLink, nextIntralLink, neighbors));
    }
    notifyIdx_++;

    return HCCL_SUCCESS;
}

HcclResult AllGatherUnifiedMarch::RunLastStep(u32 ringPrevRank, u32 ringNextRank, u32 totalStep)
{
    LINK prevIntraLink = links_[ringPrevRank];
    CHK_SMART_PTR_NULL(prevIntraLink);
    LINK nextIntralLink = links_[ringNextRank];
    CHK_SMART_PTR_NULL(nextIntralLink);
    CHK_RET(NotifyNeighborsStart(prevIntraLink, nextIntralLink, NEIGHBORS_NUM_TWO));

    // 拉齐 从流record主流、主流record从流 保证从流同时开始做SDMA
    CHK_RET(WaitSubStreamFinish(NEIGHBORS_NUM_TWO));
    CHK_RET(NotifySubStreamStart(NEIGHBORS_NUM_TWO));

    // 从前向rank读取数据
    void* preRemDMAMemPtr = nullptr;
    CHK_RET(prevIntraLink->GetRemoteMem(UserMemType::OUTPUT_MEM, &preRemDMAMemPtr));
    u32 preDataIndex = (intraRank_ + intraRankSize_ - totalStep) % intraRankSize_;
    u64 preOffsetByte = preDataIndex * blockDataByte_;
    void* preDstAddr = static_cast<u8*>(userOutput_.ptr()) + preOffsetByte;
    CHK_RET(DoSerialSDMA(
        preRemDMAMemPtr, preOffsetByte, preDstAddr, subStreams_[0], prevIntraLink, blockDataByte_ / DIVISOR_NUM_TWO));
    HCCL_INFO(
        "[AllGatherUnifiedMarch][RunLastStep] intraRank [%u] read from ringPrevRank [%u] done", intraRank_,
        ringPrevRank);

    // 从后向rank读取数据
    void* nextRemDMAMemPtr = nullptr;
    CHK_RET(nextIntralLink->GetRemoteMem(UserMemType::OUTPUT_MEM, &nextRemDMAMemPtr));
    u32 nextDataIndex = (intraRank_ + totalStep) % intraRankSize_;
    u64 nextOffsetByte = nextDataIndex * blockDataByte_ + blockDataByte_ / DIVISOR_NUM_TWO;
    void* nextDstAddr = static_cast<u8*>(userOutput_.ptr()) + nextOffsetByte;
    CHK_RET(DoSerialSDMA(
        nextRemDMAMemPtr, nextOffsetByte, nextDstAddr, subStreams_[1], nextIntralLink,
        (blockDataByte_ - blockDataByte_ / DIVISOR_NUM_TWO))); // 兼容单块儿allgather数据量不是2的倍数场景
    HCCL_INFO(
        "[AllGatherUnifiedMarch][RunLastStep] intraRank [%u] read from ringNextRank [%u] done", intraRank_,
        ringNextRank);

    // 单算子使用Ack/Datasignal接口，必须保证两者交替使用
    const u32 NOTIFY_IDX_TWO = 2;
    if (notifyIdx_ % NOTIFY_IDX_TWO != 0 && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        notifyIdx_++;
        CHK_RET(WaitSubStreamFinish(NEIGHBORS_NUM_TWO));
        CHK_RET(NotifySubStreamStart(NEIGHBORS_NUM_TWO));
        CHK_RET(NotifyNeighborsStart(prevIntraLink, nextIntralLink, NEIGHBORS_NUM_TWO));
    }
    CHK_RET(NotifyNeighborsEnd(prevIntraLink, nextIntralLink, NEIGHBORS_NUM_TWO));

    return HCCL_SUCCESS;
}

HcclResult AllGatherUnifiedMarch::RunAsync()
{
    HCCL_INFO("[AllGatherUnifiedMarch][RunAsync] starts.");
    HcclOpMetaInfoDef opMeta = HcclOpMetaInfo::GetOneForAllGather();
    CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));

    // 获取link的收、发
    u32 ringPrevRank = (intraRank_ + intraRankSize_ - 1) % intraRankSize_;
    u32 ringNextRank = (intraRank_ + 1) % intraRankSize_;

    u32 neighbors = (ringPrevRank == ringNextRank) ? NEIGHBORS_NUM_ONE : NEIGHBORS_NUM_TWO;
    CHK_RET(NotifySubStreamStart(neighbors));

    u32 totalStep = intraRankSize_ / DIVISOR_NUM_TWO;
    HCCL_INFO(
        "[AllGatherUnifiedMarch][RunAsync] intraRank [%u] ringPrevRank [%u] ringNextRank [%u] totalStep [%u]",
        intraRank_, ringPrevRank, ringNextRank, totalStep);
    if (totalStep == 1) {
        CHK_RET(RunSingleStep(ringPrevRank, ringNextRank, 0, totalStep));
    } else {
        for (u32 step = 0; step < totalStep - 1; step++) {
            CHK_RET(RunSingleStep(ringPrevRank, ringNextRank, step, totalStep));
        }
        CHK_RET(RunLastStep(ringPrevRank, ringNextRank, totalStep));
    }
    CHK_RET(WaitSubStreamFinish(neighbors));

    CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, subStreams_));

    HCCL_INFO("[AllGatherUnifiedMarch][RunAsync] finished.");
    return HCCL_SUCCESS;
}

HcclResult AllGatherUnifiedMarch::Prepare(
    const Stream& mainStream, SubCommInfo& level0CommInfo, DeviceMem& userInput, DeviceMem& userOutput,
    DeviceMem& usrInMem, DeviceMem& usrOutMem, u64 blockDataByte, std::vector<Stream>& subStreams,
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain,
    const std::vector<std::vector<Slice>>& multRingsUserMemSlice, const u64 baseOffset)
{
    mainStream_ = mainStream;
    intraRank_ = level0CommInfo.localRank;
    intraRankSize_ = level0CommInfo.localRankSize;
    CHK_PRT_RET(
        intraRankSize_ == 0 || (intraRankSize_ % DIVISOR_NUM_TWO != 0),
        HCCL_ERROR("[AllGatherUnifiedMarch][Prepare]intraRankSize_ is zero."), HCCL_E_PARA);
    links_ = level0CommInfo.links;

    userInput_ = userInput;
    userOutput_ = userOutput;
    usrInMem_ = usrInMem;
    usrOutMem_ = usrOutMem;
    HCCL_INFO(
        "userInput_[%p] size[%llu], userOutput_[%p] size[%llu], usrInMem_[%p] size[%llu], usrOutMem_[%p] size[%llu]",
        userInput_.ptr(), userInput_.size(), userOutput_.ptr(), userOutput_.size(), usrInMem_.ptr(), usrInMem_.size(),
        usrOutMem_.ptr(), usrOutMem_.size());

    subStreams_ = subStreams;
    meshSignalMainToSub_ = meshSignalMainToSub;
    meshSignalSubToMain_ = meshSignalSubToMain;
    CHK_PRT_RET(
        subStreams_.size() < NEIGHBORS_NUM_TWO || meshSignalMainToSub_.size() < NEIGHBORS_NUM_TWO
            || meshSignalSubToMain_.size() < NEIGHBORS_NUM_TWO,
        HCCL_ERROR(
            "[AllGatherUnifiedMarch] subStreams_ size[%u] or meshSignalMainToSub_ size[%u] or "
            "meshSignalSubToMain_ size[%u] is less than 2",
            subStreams_.size(), meshSignalMainToSub_.size(), meshSignalSubToMain_.size()),
        HCCL_E_PARA);
    blockDataByte_ = blockDataByte;
    multRingsUserMemSlice_ = multRingsUserMemSlice;
    CHK_PRT_RET(
        multRingsUserMemSlice_[0].size() % intraRankSize_ != 0,
        HCCL_ERROR(
            "[AllGatherUnifiedMarch] multRingsUserMemSlice_[0] size[%u] can not be divided by rank size[%u]",
            multRingsUserMemSlice_[0].size(), intraRankSize_),
        HCCL_E_PARA);

    baseOffset_ = baseOffset;

    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_UNIFIED_MARCH, AllGatherUnifiedMarch);
} // namespace hccl
