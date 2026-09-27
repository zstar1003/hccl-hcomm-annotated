/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_ring_zerocopy_exchange_executor.h"

namespace hccl {

CollReduceScatterRingZerocopyExchangeExecutor::CollReduceScatterRingZerocopyExchangeExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterRingZerocopyExecutor(dispatcher, topoMatcher)
{}

HcclResult CollReduceScatterRingZerocopyExchangeExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    // 调用父类编排函数建链关系计算函数
    CHK_RET(CollReduceScatterRingZerocopyExecutor::CalcCommInfo(opTransport));
    // 额外增加数据交换的建链
    CHK_RET(CalcExchangeCommInfo(opTransport));
    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterRingZerocopyExchangeExecutor::CalcExchangeCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    std::set<u32> commTargetUserRankSet;
    u32 remoteRankSend = 0;
    u32 remoteRankRecv = 0;

    CHK_RET(CalExchangeRemoteRankForReduceScatter(remoteRankSend, remoteRankRecv));
    commTargetUserRankSet.insert(remoteRankSend);
    commTargetUserRankSet.insert(remoteRankRecv);
    CommParaInfo commParaInfo(
        COMM_COMBINE_ORDER, CommType::COMM_TAG_PARTIAL_MESH_COMBINED, INVALID_VALUE_RANKID, INVALID_VALUE_RANKID, false,
        false, commTargetUserRankSet);

    TransportMemType inputType = TransportMemType::CCL_INPUT;
    TransportMemType outputType = TransportMemType::CCL_OUTPUT;

    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_COMBINE_ORDER], inputType, outputType));
    LevelNSubCommTransport& commTransport = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < commTransport.size(); subCommIndex++) {
        for (auto& transportRequest : commTransport[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma
                = (topoAttr_.superPodNum > 1
                   || (static_cast<bool>(topoMatcher_->GetExternalInputInterHccsDisable()) && topoAttr_.serverNum > 1));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingZerocopyExchangeExecutor::KernelRunInterServerPostProcess(
    const OpParam& param, const ExecMem& execMem)
{
    // 计算需要交换数据的通信对端
    u32 remoteRankSend = 0;
    u32 remoteRankRecv = 0;
    CHK_RET(CalExchangeRemoteRankForReduceScatter(remoteRankSend, remoteRankRecv));

    Stream stream = param.stream;
    u64 outputMemSize = execMem.outputMem.size();
    if (remoteRankSend != topoAttr_.userRank && remoteRankRecv != topoAttr_.userRank) { // 需要交换数据
        // 获取通信对端的link
        LINK sendLink;
        LINK recvLink;
        CHK_RET(GetTransportForExchange(remoteRankSend, sendLink));
        CHK_RET(GetTransportForExchange(remoteRankRecv, recvLink));
        CHK_PTR_NULL(sendLink);
        CHK_PTR_NULL(recvLink);
        // 当通信对端恰好是同server的邻居时，复用Level0的建链，其注册的内存是UserMem，需要特殊处理
        // 否则，在CommCombineOrder上建链，其注册内存是CCL Buffer
        if (IsLevel0Neighbor(remoteRankSend, level0RankSize_)) {
            // ccl in -> user in
            u64 memOffset = (level1Rank_ * level2RankSize_ + level2Rank_) * outputMemSize;
            DeviceMem dstMem = DeviceMem::create(static_cast<u8*>(param.inputPtr), execMem.outputMem.size());
            DeviceMem srcMem = execMem.inputMem.range(memOffset, outputMemSize);
            CHK_SMART_PTR_NULL(dstMem);
            HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollReduceScatterRingZerocopyExchangeExecutor][ExchangeData]ReduceScatter double "
                    "ring memcpy Failed, Offset[%llu], Size[%llu]",
                    memOffset, outputMemSize),
                ret);
            // user in send to remote user out
            recvLink->TxAck(stream);
            sendLink->RxAck(stream);
            sendLink->TxAsync(UserMemType::OUTPUT_MEM, 0, param.inputPtr, outputMemSize, stream);
            if (IsLevel0Neighbor(remoteRankRecv, level0RankSize_)) {
                recvLink->RxAsync(UserMemType::INPUT_MEM, 0, execMem.outputPtr, outputMemSize, stream);
            } else {
                u32 remoteLevel1Index = remoteRankRecv % (level0RankSize_ * level1RankSize_) / level0RankSize_;
                u32 remoteLevel2Index = remoteRankRecv / level0RankSize_ / level1RankSize_;
                u64 rxSrcOffset = (remoteLevel1Index * level2RankSize_ + remoteLevel2Index) * outputMemSize;
                recvLink->RxAsync(UserMemType::INPUT_MEM, rxSrcOffset, execMem.outputMem.ptr(), outputMemSize, stream);
            }
        } else {
            recvLink->TxAck(stream);
            sendLink->RxAck(stream);
            u64 txDstOffset = (level1Rank_ * level2RankSize_ + level2Rank_) * outputMemSize;
            sendLink->TxAsync(
                UserMemType::OUTPUT_MEM, 0, static_cast<u8*>(execMem.inputMem.ptr()) + txDstOffset, outputMemSize,
                stream);
            if (IsLevel0Neighbor(remoteRankRecv, level0RankSize_)) {
                recvLink->RxAsync(UserMemType::INPUT_MEM, 0, execMem.outputPtr, outputMemSize, stream);
            } else {
                u32 remoteLevel1Index = remoteRankRecv % (level0RankSize_ * level1RankSize_) / level0RankSize_;
                u32 remoteLevel2Index = remoteRankRecv / level0RankSize_ / level1RankSize_;
                u64 rxSrcOffset = (remoteLevel1Index * level2RankSize_ + remoteLevel2Index) * outputMemSize;
                recvLink->RxAsync(UserMemType::INPUT_MEM, rxSrcOffset, execMem.outputMem.ptr(), outputMemSize, stream);
            }
        }
        // 交换数据的两端之间Barrier，确认收发完成
        CHK_RET(recvLink->TxAck(stream));
        CHK_RET(sendLink->RxAck(stream));
        CHK_RET(sendLink->TxDataSignal(stream));
        CHK_RET(recvLink->RxDataSignal(stream));
    } else { // 不需要交换数据，将数据从ccl in拷到ccl out
        u64 memOffset = (level1Rank_ * level2RankSize_ + level2Rank_) * outputMemSize;
        DeviceMem dstMem = execMem.outputMem;
        DeviceMem srcMem = execMem.inputMem.range(memOffset, outputMemSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
    }

    // 如果不需要交换数据，或者收端不是邻居，那么结果数据还在CCL Out上，需要搬到User Out上去
    if ((remoteRankRecv == topoAttr_.userRank) || !IsLevel0Neighbor(remoteRankRecv, level0RankSize_)) {
        DeviceMem srcMem = execMem.outputMem;
        DeviceMem dstMem = DeviceMem::create(static_cast<u8*>(execMem.outputPtr), execMem.outputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
    }

    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingZerocopyExchangeExecutor::CalcLevel0DataSlices(
    const OpParam& param, const ExecMem& execMem, std::vector<Slice>& dataSegsSlice)
{
    return CalcIntraServerDataSlicesContinuous(
        param, execMem, level0RankSize_, level1RankSize_, level2RankSize_, dataSegsSlice);
}

HcclResult CollReduceScatterRingZerocopyExchangeExecutor::KernelRunInterServerPreProcess(
    const OpParam& param, const ExecMem& execMem)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollReduceScatterRingZerocopyExchangeExecutor][KernelRun] userRank[%u] starts.",
        topoAttr_.userRank);
    u32 unitSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, unitSize));

    DeviceMem dstMem;
    DeviceMem srcMem;
    u64 curSize = execMem.outputMem.size();
    Stream stream = param.stream;
    for (u32 i = 0; i < level1RankSize_ * level2RankSize_; i++) {
        // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为output的size
        dstMem = execMem.inputMem.range(i * curSize, curSize);
        srcMem = DeviceMem::create(
            static_cast<u8*>(execMem.inputPtr)
                + param.DataDes.count * unitSize * level1RankSize_ * level2RankSize_ * level0Rank_
                + param.DataDes.count * unitSize * i,
            curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC(
    "ReduceScatterRingZerocopyExchangeExecutor", ReduceScatterRingZerocopy,
    CollReduceScatterRingZerocopyExchangeExecutor);
} // namespace hccl
