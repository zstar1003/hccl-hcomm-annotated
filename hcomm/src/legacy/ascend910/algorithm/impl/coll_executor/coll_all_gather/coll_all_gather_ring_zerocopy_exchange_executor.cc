/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_ring_zerocopy_exchange_executor.h"

namespace hccl {
CollAllGatherRingZerocopyExchangeExecutor::CollAllGatherRingZerocopyExchangeExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherRingZerocopyExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true; // 设为true，以禁用RunLoop中的本地拷贝
    desc_.isZeroCopy = true;
}

HcclResult CollAllGatherRingZerocopyExchangeExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    // 调用父类编排函数建链关系计算函数
    CHK_RET(CollAllGatherRingZerocopyExecutor::CalcCommInfo(opTransport));
    // 额外增加数据交换的建链
    CHK_RET(CalcExchangeCommInfo(opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingZerocopyExchangeExecutor::CalExchangeRemoteRank(u32& remoteRankSend, u32& remoteRankRecv)
{
    // AllGather的发端与收端与ReduceScatter相反
    return CalExchangeRemoteRankForReduceScatter(remoteRankRecv, remoteRankSend);
}

HcclResult
CollAllGatherRingZerocopyExchangeExecutor::CalcExchangeCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    std::set<u32> commTargetUserRankSet;
    u32 remoteRankSend = 0;
    u32 remoteRankRecv = 0;

    CHK_RET(CalExchangeRemoteRank(remoteRankSend, remoteRankRecv));
    HCCL_DEBUG("[%s] remoteRankSend:%d, remoteRankRecv:%d", __func__, remoteRankSend, remoteRankRecv);
    commTargetUserRankSet.insert(remoteRankSend);
    commTargetUserRankSet.insert(remoteRankRecv);
    CommParaInfo commParaInfo(
        COMM_COMBINE_ORDER, CommType::COMM_TAG_PARTIAL_MESH_COMBINED, INVALID_VALUE_RANKID, INVALID_VALUE_RANKID, false,
        false, commTargetUserRankSet);

    TransportMemType inputType = TransportMemType::CCL_INPUT;
    TransportMemType outputType = TransportMemType::CCL_OUTPUT;

    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_COMBINE_ORDER], inputType, outputType));
    LevelNSubCommTransport& subCommTransport = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < subCommTransport.size(); subCommIndex++) {
        for (auto& transportRequest : subCommTransport[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma
                = (topoAttr_.superPodNum > 1
                   || (static_cast<bool>(topoMatcher_->GetExternalInputInterHccsDisable()) && topoAttr_.serverNum > 1));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
CollAllGatherRingZerocopyExchangeExecutor::KernelRunInterServerPreProcess(const OpParam& param, const ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[AllGatherRingZerocopyExchangeExecutor] KernelRunInterServerPreProcess");
    // 计算需要交换数据的通信对端
    u32 remoteRankSend = 0;
    u32 remoteRankRecv = 0;
    CHK_RET(CalExchangeRemoteRank(remoteRankSend, remoteRankRecv));

    Stream stream = param.stream;
    u64 inputMemSize = execMem.inputMem.size();
    if (remoteRankSend != topoAttr_.userRank && remoteRankRecv != topoAttr_.userRank) { // 需要交换数据
        // 获取通信对端的link
        LINK sendLink;
        LINK recvLink;
        CHK_RET(GetTransportForExchange(remoteRankSend, sendLink));
        CHK_RET(GetTransportForExchange(remoteRankRecv, recvLink));
        // 当通信对端恰好是同server的邻居时，复用Level0的建链，其注册的内存是UserMem，需要特殊处理
        // 否则，在CommCombineOrder上建链，其注册内存是CCL Buffer
        if (!IsLevel0Neighbor(remoteRankSend, level0RankSize_)) {
            // user in mem -> ccl in mem
            DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr), inputMemSize);
            DeviceMem dstMem = execMem.inputMem.range(0, inputMemSize);
            HCCL_DEBUG(
                "[%s] not neighbor, srcPtr:%p, dstPtr:%p, size:%llu", __func__, srcMem.ptr(), dstMem.ptr(),
                inputMemSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
        }
        // 执行通信
        recvLink->TxAck(stream);
        sendLink->RxAck(stream);
        u32 remoteLevel1Index = remoteRankSend % (level0RankSize_ * level1RankSize_) / level0RankSize_;
        u32 remoteLevel2Index = remoteRankSend / level0RankSize_ / level1RankSize_;
        u64 txDstOffset = (remoteLevel1Index * level2RankSize_ + remoteLevel2Index) * inputMemSize;
        HCCL_DEBUG(
            "[%s] remoteLevel1Index:%d, remoteLevel2Index:%d, txDstOffset:%llu", __func__, remoteLevel1Index,
            remoteLevel2Index, txDstOffset);
        if (IsLevel0Neighbor(remoteRankSend, level0RankSize_)) {
            sendLink->TxAsync(UserMemType::OUTPUT_MEM, txDstOffset, execMem.inputPtr, inputMemSize, stream);
            HCCL_DEBUG("[%s] neighbor, send data to userMem", __func__);
        } else {
            sendLink->TxAsync(UserMemType::OUTPUT_MEM, txDstOffset, execMem.inputMem.ptr(), inputMemSize, stream);
            HCCL_DEBUG("[%s] not neighbor, send data to ccl buffer", __func__);
        }
        u64 rxDstOffset = (level1Rank_ * level2RankSize_ + level2Rank_) * inputMemSize;
        u64 rxSrcOffset = IsLevel0Neighbor(remoteRankRecv, level0RankSize_) ?
                              static_cast<u8*>(execMem.inputPtr) - static_cast<u8*>(param.inputPtr) :
                              0;
        HCCL_DEBUG("[%s] rxDstOffset:%llu, rxSrcOffset:%llu", __func__, rxDstOffset, rxSrcOffset);
        recvLink->RxAsync(
            UserMemType::INPUT_MEM, rxSrcOffset, static_cast<u8*>(execMem.outputMem.ptr()) + rxDstOffset, inputMemSize,
            stream);
        // 交换数据的两端之间Barrier，确认收发完成
        CHK_RET(recvLink->TxAck(stream));
        CHK_RET(sendLink->RxAck(stream));
        CHK_RET(sendLink->TxDataSignal(stream));
        CHK_RET(recvLink->RxDataSignal(stream));
    } else { // 不需要交换数据，将数据从user in拷到ccl out
        u64 dstMemOffset = (level1Rank_ * level2RankSize_ + level2Rank_) * inputMemSize;
        DeviceMem dstMem = execMem.outputMem.range(dstMemOffset, inputMemSize);
        DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr), inputMemSize);
        HCCL_DEBUG(
            "[%s] not exchange, just copy data from CCLOut[%p] to UserInput[%p]", __func__, dstMem.ptr(), srcMem.ptr());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
    }

    return HCCL_SUCCESS;
}

HcclResult
CollAllGatherRingZerocopyExchangeExecutor::KernelRunInterServerPostProcess(const OpParam& param, const ExecMem& execMem)
{
    // 将通信结果从ccl output搬到user output
    if (level1RankSize_ > 1 || level2RankSize_ > 1) {
        u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
        u64 curSize = execMem.inputMem.size();
        Stream stream = param.stream;
        for (u32 i = 0; i < level1RankSize_ * level2RankSize_; i++) {
            DeviceMem dstMem = DeviceMem::create(
                static_cast<u8*>(execMem.outputPtr)
                    + param.DataDes.count * unitSize * (level0Rank_ * level1RankSize_ * level2RankSize_ + i),
                curSize);
            DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.outputMem.ptr()) + i * curSize, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
            HCCL_DEBUG("[%s] memcopy from CCLOut[%p] to UserOut[%p]", __func__, srcMem.ptr(), dstMem.ptr());
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingZerocopyExchangeExecutor::CalcLevel0DataSlices(
    const OpParam& param, const ExecMem& execMem, std::vector<Slice>& dataSegsSlice)
{
    return CalcIntraServerDataSlicesContinuous(
        param, execMem, level0RankSize_, level1RankSize_, level2RankSize_, dataSegsSlice);
}

REGISTER_EXEC(
    "AllGatherRingZerocopyExchangeExecutor", AllGatherRingZerocopyExchange, CollAllGatherRingZerocopyExchangeExecutor);

} // namespace hccl
