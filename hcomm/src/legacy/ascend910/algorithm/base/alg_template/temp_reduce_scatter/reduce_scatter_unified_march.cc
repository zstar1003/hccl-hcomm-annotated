/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_unified_march.h"
#include "alg_template_register.h"

namespace hccl {
static const u32 NEIGHBORS_NUM_TWO = 2; //  2: 邻居数量
static const u32 NEIGHBORS_NUM_ONE = 1; //  1: 邻居数量
static const u32 DIVISOR_NUM_TWO = 2;

ReduceScatterUnifiedMarch::ReduceScatterUnifiedMarch(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

ReduceScatterUnifiedMarch::~ReduceScatterUnifiedMarch() {}

HcclResult ReduceScatterUnifiedMarch::Prepare(
    Stream& mainStream, SubCommInfo& level0CommInfo, DeviceMem& userInput, DeviceMem& userOutput, DeviceMem& usrInMem,
    DeviceMem& scratchMem, u64 totalCount, std::vector<Stream>& subStreams,
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain, const HcclDataType dataType,
    const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>>& multRingsUserMemSlice, u64 reduceAttrBitMap)
{
    reduceAttr_ = reduceAttrBitMap;
    mainStream_ = mainStream;
    intraRank_ = level0CommInfo.localRank;
    intraRankSize_ = level0CommInfo.localRankSize;
    CHK_PRT_RET(
        intraRankSize_ == 0 || (intraRankSize_ % DIVISOR_NUM_TWO != 0),
        HCCL_ERROR("[ReduceScatterUnifiedMarch][Prepare]intraRankSize_ is zero or not divisible by 2"), HCCL_E_PARA);
    links_ = level0CommInfo.links;

    userInput_ = userInput;
    userOutput_ = userOutput;
    usrInMem_ = usrInMem;
    scratchMem_ = scratchMem;
    HCCL_INFO(
        "userInput_[%p] size[%llu], userOutput_[%p] size[%llu], usrInMem_[%p] size[%llu], scratchMem_[%p] size[%llu]",
        userInput_.ptr(), userInput_.size(), userOutput_.ptr(), userOutput_.size(), usrInMem_.ptr(), usrInMem_.size(),
        scratchMem_.ptr(), scratchMem_.size());

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

    totalCount_ = totalCount;
    dataType_ = dataType;
    reductionOp_ = reductionOp;
    blockDataByte_ = totalCount_ * SIZE_TABLE[dataType_];
    multRingsUserMemSlice_ = multRingsUserMemSlice;
    CHK_PRT_RET(
        multRingsUserMemSlice_[0].size() % intraRankSize_ != 0,
        HCCL_ERROR(
            "[ReduceScatterUnifiedMarch] multRingsUserMemSlice_[0] size[%u] can not be divided by rank size[%u]",
            multRingsUserMemSlice_[0].size(), intraRankSize_),
        HCCL_E_PARA);

    return HCCL_SUCCESS;
}

std::string ReduceScatterUnifiedMarch::GetStreamIndexString()
{
    std::string res = "";
    for (u32 streamIndex = 0; streamIndex < subStreams_.size(); streamIndex++) {
        res += std::to_string(streamIndex) + ", ";
    }
    return res;
}

// 主流通知所有从流
HcclResult ReduceScatterUnifiedMarch::NotifySubStreamStart(u32 streamSize)
{
    CHK_PRT_RET(
        streamSize > subStreams_.size() || streamSize > meshSignalSubToMain_.size(),
        HCCL_ERROR(
            "[ReduceScatterUnifiedMarch][NotifySubStreamStart] streamSize[%u] is out of range"
            "subStreams_ size[%zu] or meshSignalSubToMain_ size[%zu]",
            streamSize, subStreams_.size(), meshSignalSubToMain_.size()),
        HCCL_E_PARA);
    for (u32 streamIndex = 0; streamIndex < streamSize; streamIndex++) {
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, meshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(
            subStreams_[streamIndex], dispatcher_, meshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG(
        "[ReduceScatterUnifiedMarch][NotifySubStreamStart] intraRank [%u] main stream notify substream [%s]",
        intraRank_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterUnifiedMarch::WaitSubStreamFinish(u32 streamSize)
{
    CHK_PRT_RET(
        streamSize > subStreams_.size() || streamSize > meshSignalMainToSub_.size(),
        HCCL_ERROR(
            "[ReduceScatterUnifiedMarch][WaitSubStreamFinish] streamSize[%u] is out of range"
            "subStreams_ size[%zu] or meshSignalMainToSub_ size[%zu]",
            streamSize, subStreams_.size(), meshSignalMainToSub_.size()),
        HCCL_E_PARA);
    for (u32 streamIndex = 0; streamIndex < streamSize; streamIndex++) {
        CHK_RET(LocalNotify::Post(
            subStreams_[streamIndex], dispatcher_, meshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, meshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG(
        "[ReduceScatterUnifiedMarch][WaitSubStreamFinish] intraRank [%u] main stream wait substream [%s]", intraRank_,
        GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterUnifiedMarch::NotifyNeighborsStart(LINK& prevIntraLink, LINK& nextIntralLink, u32 neighbors)
{
    // 图模式保持使用Post/Wait接口
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        // notify是否越界由平台侧保证
        for (u32 neighborRankId = 0; neighborRankId < neighbors; neighborRankId++) {
            if (neighborRankId == 0) {
                CHK_RET(nextIntralLink->Post(notifyIdx_, subStreams_[neighborRankId])); // AckRecord
                CHK_RET(prevIntraLink->Wait(notifyIdx_, subStreams_[neighborRankId]));  // AckWait
            } else if (neighborRankId == 1) {
                CHK_RET(prevIntraLink->Post(notifyIdx_, subStreams_[neighborRankId]));  // AckRecord
                CHK_RET(nextIntralLink->Wait(notifyIdx_, subStreams_[neighborRankId])); // AckWait
            }
        }
        HCCL_DEBUG(
            "[ReduceScatterUnifiedMarch][NotifyNeighborsStart] intraRank[%u] switch on [%u]neighbors done", intraRank_,
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
        "[ReduceScatterUnifiedMarch][NotifyNeighborsStart] intraRank[%u] switch on [%u]neighbors done", intraRank_,
        neighbors);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterUnifiedMarch::NotifyNeighborsEnd(LINK& prevIntraLink, LINK& nextIntralLink, u32 neighbors)
{
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
        "[ReduceScatterUnifiedMarch][NotifyNeighborsEnd] intraRank[%u] notify [%u]neighbors reduce done", intraRank_,
        neighbors);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterUnifiedMarch::DoSerialReduce(
    void* remDMAMemPtr, void* dstAddr, u64 memSize, u64 dataCount, Stream& tmpStream, LINK& tmpLink,
    u64 remoteOffsetByte)
{
    for (u32 sliceIdx = 0; sliceIdx < (multRingsUserMemSlice_[0].size() / intraRankSize_); sliceIdx++) {
        DeviceMem srcMem = DeviceMem::create(
            static_cast<u8*>(remDMAMemPtr) + remoteOffsetByte + multRingsUserMemSlice_[0][sliceIdx].offset, memSize);
        DeviceMem dstMem
            = DeviceMem::create(static_cast<u8*>(dstAddr) + multRingsUserMemSlice_[0][sliceIdx].offset, memSize);

        if ((INLINE_REDUCE_BITMASK & reduceAttr_) == 1) { // inlineReduce
            struct hccl::Transport::Buffer remoteBuf;
            remoteBuf.addr = srcMem.ptr();
            remoteBuf.size = srcMem.size();
            struct hccl::Transport::Buffer localBuf;
            localBuf.addr = dstMem.ptr();
            localBuf.size = dstMem.size();
            HCCL_DEBUG(
                "intraRank[%u] slice[%u] offset[%llu] do inlinereduce with remoteBuf[addr[%p], size[%llu]] and "
                "localBuf[addr[%p], size[%llu]]",
                intraRank_, sliceIdx, multRingsUserMemSlice_[0][sliceIdx].offset, remoteBuf.addr, remoteBuf.size,
                localBuf.addr, localBuf.size);
            CHK_RET(tmpLink->ReadReduceSync(localBuf, remoteBuf, dataType_, reductionOp_, tmpStream));
        } else { // TBE_reduce
            // left的inputMem拷到本端的scratchMem
            DeviceMem tempMem = scratchMem_.range(remoteOffsetByte, srcMem.size());
            struct hccl::Transport::Buffer remoteBuf;
            remoteBuf.addr = srcMem.ptr();
            remoteBuf.size = srcMem.size();
            struct hccl::Transport::Buffer localBuf;
            localBuf.addr = tempMem.ptr();
            localBuf.size = tempMem.size();
            HCCL_DEBUG(
                "intraRank[%u] slice[%u] offset[%llu] do SDMA read with remoteBuf[addr[%p], size[%llu]] and "
                "localBuf[addr[%p], size[%llu]]",
                intraRank_, sliceIdx, multRingsUserMemSlice_[0][sliceIdx].offset, remoteBuf.addr, remoteBuf.size,
                localBuf.addr, localBuf.size);
            CHK_RET(tmpLink->ReadSync(localBuf, remoteBuf, tmpStream));
            CHK_RET(HcclReduceAsync(
                dispatcher_, tempMem.ptr(), dataCount, dataType_, reductionOp_, tmpStream, dstMem.ptr(),
                INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, reduceAttr_));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterUnifiedMarch::RunSingleSliceRead(u32 ringPrevRank, u32 ringNextRank, u32 step, u32 totalStep)
{
    LINK prevIntraLink = links_[ringPrevRank];
    CHK_SMART_PTR_NULL(prevIntraLink);
    LINK nextIntralLink = links_[ringNextRank];
    CHK_SMART_PTR_NULL(nextIntralLink);
    u32 neighbors = (ringPrevRank == ringNextRank) ? NEIGHBORS_NUM_ONE : NEIGHBORS_NUM_TWO;
    CHK_RET(NotifyNeighborsStart(prevIntraLink, nextIntralLink, neighbors));

    // 拉齐 从流record主流、主流record从流 保证从流同时开始做SDMA
    CHK_RET(WaitSubStreamFinish(neighbors));
    CHK_RET(NotifySubStreamStart(neighbors));

    // 从前向rank读取数据
    void* preRemDMAMemPtr = nullptr;
    CHK_RET(prevIntraLink->GetRemoteMem(UserMemType::INPUT_MEM, &preRemDMAMemPtr));
    u32 preDataIndex = (intraRank_ + intraRankSize_ - step - totalStep) % intraRankSize_;
    u64 preOffsetByte = preDataIndex * blockDataByte_;
    void* preDstAddr = static_cast<u8*>(userInput_.ptr()) + preOffsetByte;

    CHK_RET(DoSerialReduce(
        preRemDMAMemPtr, preDstAddr, blockDataByte_, totalCount_, subStreams_[0], prevIntraLink, preOffsetByte));
    HCCL_INFO(
        "[ReduceScatterUnifiedMarch][RunSingleSliceRead] intraRank [%u] reduce with ringPrevRank [%u] done", intraRank_,
        ringPrevRank);

    // 从后向rank读取数据
    if (neighbors > NEIGHBORS_NUM_ONE) {
        void* nextRemDMAMemPtr = nullptr;
        CHK_RET(nextIntralLink->GetRemoteMem(UserMemType::INPUT_MEM, &nextRemDMAMemPtr));
        u32 nextDataIndex = (intraRank_ + totalStep + step) % intraRankSize_;
        u64 nextOffsetByte = nextDataIndex * blockDataByte_;
        void* nextDstAddr = static_cast<u8*>(userInput_.ptr()) + nextOffsetByte;

        CHK_RET(DoSerialReduce(
            nextRemDMAMemPtr, nextDstAddr, blockDataByte_, totalCount_, subStreams_[1], nextIntralLink,
            nextOffsetByte));
        HCCL_INFO(
            "[ReduceScatterUnifiedMarch][RunSingleSliceRead] intraRank [%u] "
            "reduce with ringNextRank [%u] done",
            intraRank_, ringNextRank);
    }

    /* 2卡 场景，在最后一步的notifyDone */
    if (step == 0) {
        CHK_RET(NotifyNeighborsEnd(prevIntraLink, nextIntralLink, neighbors));
    }
    notifyIdx_++;

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterUnifiedMarch::RunHalfSliceRead(u32 ringPrevRank, u32 ringNextRank, u32 step, u32 totalStep)
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
    CHK_RET(prevIntraLink->GetRemoteMem(UserMemType::INPUT_MEM, &preRemDMAMemPtr));
    u32 temIdx = (step == 0) ? totalStep : 0;
    u32 preDataIndex = (intraRank_ + intraRankSize_ - temIdx) % intraRankSize_;
    // 考虑总数据量不能被整除的情况
    u32 partOneCount
        = (step != totalStep) ? (totalCount_ / DIVISOR_NUM_TWO) : (totalCount_ - totalCount_ / DIVISOR_NUM_TWO);
    u64 partOneSize = partOneCount * SIZE_TABLE[dataType_];
    u64 preOffsetByte = (step != totalStep) ?
                            (preDataIndex * blockDataByte_) :
                            (preDataIndex * blockDataByte_ + totalCount_ / DIVISOR_NUM_TWO * SIZE_TABLE[dataType_]);
    void* preDstAddr = static_cast<u8*>(userInput_.ptr()) + preOffsetByte;

    CHK_RET(DoSerialReduce(
        preRemDMAMemPtr, preDstAddr, partOneSize, partOneCount, subStreams_[0], prevIntraLink, preOffsetByte));
    HCCL_INFO(
        "[ReduceScatterUnifiedMarch][RunHalfSliceRead] intraRank [%u] reduce with ringPrevRank [%u] done", intraRank_,
        ringPrevRank);

    // 从后向rank读取数据
    void* nextRemDMAMemPtr = nullptr;
    CHK_RET(nextIntralLink->GetRemoteMem(UserMemType::INPUT_MEM, &nextRemDMAMemPtr));
    temIdx = (step == 0) ? totalStep : 0;
    u32 nextDataIndex = (intraRank_ + temIdx) % intraRankSize_;
    u32 partTwoCount = totalCount_ - partOneCount;
    u64 partTwoSize = partTwoCount * SIZE_TABLE[dataType_];
    u64 nextOffsetByte
        = (step != totalStep) ? (nextDataIndex * blockDataByte_ + partOneSize) : nextDataIndex * blockDataByte_;
    void* nextDstAddr = static_cast<u8*>(userInput_.ptr()) + nextOffsetByte;

    CHK_RET(DoSerialReduce(
        nextRemDMAMemPtr, nextDstAddr, partTwoSize, partTwoCount, subStreams_[1], nextIntralLink, nextOffsetByte));
    HCCL_INFO(
        "[ReduceScatterUnifiedMarch][RunHalfSliceRead] intraRank [%u] reduce with ringNextRank [%u] done", intraRank_,
        ringNextRank);

    /* 4卡及以上的 场景，在最后一步的notifyDone */
    // 单算子使用Ack/Datasignal接口，必须保证两者交替使用
    if (step == totalStep) {
        const u32 NOTIFY_IDX_TWO = 2;
        if (notifyIdx_ % NOTIFY_IDX_TWO != 0 && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            notifyIdx_++;
            CHK_RET(WaitSubStreamFinish(NEIGHBORS_NUM_TWO));
            CHK_RET(NotifySubStreamStart(NEIGHBORS_NUM_TWO));
            CHK_RET(NotifyNeighborsStart(prevIntraLink, nextIntralLink, NEIGHBORS_NUM_TWO));
        }
        CHK_RET(NotifyNeighborsEnd(prevIntraLink, nextIntralLink, NEIGHBORS_NUM_TWO));
    }
    notifyIdx_++;

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterUnifiedMarch::RunAsync()
{
    HcclOpMetaInfoDef opMeta = HcclOpMetaInfo::GetOneForReduceScatter();
    CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));

    // 获取link的收、发
    u32 ringPrevRank = (intraRank_ + intraRankSize_ - 1) % intraRankSize_;
    u32 ringNextRank = (intraRank_ + 1) % intraRankSize_;

    u32 neighbors = (ringPrevRank == ringNextRank) ? NEIGHBORS_NUM_ONE : NEIGHBORS_NUM_TWO;
    CHK_RET(NotifySubStreamStart(neighbors));

    // 计算所需的总步骤
    u32 totalStep = intraRankSize_ / DIVISOR_NUM_TWO + 1;
    u32 step = 0;
    if (totalStep == DIVISOR_NUM_TWO) {
        CHK_RET(RunSingleSliceRead(ringPrevRank, ringNextRank, step, totalStep));
    } else {
        // 进行第1步收发
        CHK_RET(RunHalfSliceRead(ringPrevRank, ringNextRank, step, totalStep));

        // 进行第k步收发
        step++;
        for (; step < totalStep - DIVISOR_NUM_TWO; step++) {
            CHK_RET(RunSingleSliceRead(ringPrevRank, ringNextRank, step, totalStep));
        }

        // 进行第totalStep - 1步收发
        CHK_RET(RunHalfSliceRead(ringPrevRank, ringNextRank, step, totalStep));

        // 进行第totalStep步收发
        CHK_RET(RunHalfSliceRead(ringPrevRank, ringNextRank, totalStep, totalStep));
    }

    CHK_RET(WaitSubStreamFinish(neighbors));
    CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, subStreams_));

    HCCL_INFO("[ReduceScatterUnifiedMarch][RunAsync] finished.");
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_UNIFIED_MARCH, ReduceScatterUnifiedMarch);
} // namespace hccl
