/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_ring_zerocopy_pipeline_executor.h"

namespace hccl {
CollAllGatherRingZerocopyPipelineExecutor::CollAllGatherRingZerocopyPipelineExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true; // 设为true，以禁用RunLoop中的本地拷贝
    desc_.isZeroCopy = true;
    desc_.level1SupportedAlgos
        = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB, AlgTypeLevel1::ALG_LEVEL1_RING};
    desc_.level2SupportedAlgos = {AlgTypeLevel2::ALG_LEVEL2_PIPELINE};
}

HcclResult CollAllGatherRingZerocopyPipelineExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 1
                         + (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? (LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE + 1) :
                                                                              LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[%s] tag[%s] streamNum[%u]", __func__, tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

u64 CollAllGatherRingZerocopyPipelineExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u32 serverNumPerSuperPod = topoAttr_.serverNum / topoAttr_.superPodNum;
    u32 bufferSliceNum = std::max(2U, serverNumPerSuperPod);
    u64 maxCountPerLoop = cclBuffSize / bufferSliceNum / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;
    HCCL_INFO("[%s] tag[%s] maxCountPerLoop[%u]", __func__, tag_.c_str(), maxCountPerLoop);
    return maxCountPerLoop;
}

HcclResult CollAllGatherRingZerocopyPipelineExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    // 额外增加数据交换的建链
    CHK_RET(CalcExchangeCommInfo(opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingZerocopyPipelineExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingZerocopyPipelineExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    LevelNSubCommTransport& commTransportLevel0 = opTransport[COMM_LEVEL0];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        commTransportLevel0[subCommIndex].isZeroCopy = true;
    }
    return HCCL_SUCCESS;
}

// PipeLine模式下使用Ring算法
HcclResult CollAllGatherRingZerocopyPipelineExecutor::CalcLevel2CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL2, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult
CollAllGatherRingZerocopyPipelineExecutor::CalcExchangeCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    std::set<u32> commTargetUserRankSet;
    u32 remoteRankSend = 0;
    u32 remoteRankRecv = 0;
    CHK_RET(CalExchangeRemoteRank(remoteRankSend, remoteRankRecv));
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
            transportRequest.isUsedRdma = topoAttr_.isUsedRdmaMap.at(transportRequest.remoteUserRank);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingZerocopyPipelineExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherRingZerocopyPipelineExecutor][Orchestrate] begins.");

    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    mainStream_ = param.stream;
    subStreams_ = algResResp_->slaveStreams;
    rdmaMainStream_ = subStreams_.back();
    sdmaSubStreams_.assign(subStreams_.begin(), subStreams_.end() - 1);
    notifyMainToRdma_ = algResResp_->notifiesAux.back();  // 主流通知从流使用Aux
    notifyRdmaToMain_ = algResResp_->notifiesMain.back(); // 从流通知主流使用Main
    notifySdmaMain_.assign(algResResp_->notifiesMain.begin(), algResResp_->notifiesMain.end() - 1);
    notifySdmaSub_.assign(algResResp_->notifiesAux.begin(), algResResp_->notifiesAux.end() - 1);

    CHK_RET(GetCommRankInfoNormal(
        level0Rank_, level0RankSize_, level1Rank_, level1RankSize_, level2Rank_, level2RankSize_, false));
    unitSize_ = SIZE_TABLE[param.DataDes.dataType];
    totalSize_ = param.DataDes.count * unitSize_;
    blockSize_ = totalSize_ * level0RankSize_ * level1RankSize_;

    CHK_RET(RunLoop(param));

    HCCL_INFO(
        "tag[%s], Allgather executor orchestrate success, take time [%lld]us.", tag_.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingZerocopyPipelineExecutor::RunLoop(OpParam& param)
{
    u8* curInputPtr = static_cast<u8*>(param.inputPtr);
    u8* curOutputPtr = static_cast<u8*>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(algResResp_->cclInputMem.size(), unitSize_);
    CHK_PRT_RET(
        maxCountPerLoop == 0,
        HCCL_ERROR(
            "[CollAllGatherRingZerocopyPipelineExecutor][RunLoop]tag[%s] userRankSize[%u] maxCountPerLoop[%llu]",
            tag_.c_str(), topoAttr_.userRankSize, maxCountPerLoop),
        HCCL_E_PARA);

    u32 bufferLoopNum = (param.DataDes.count + maxCountPerLoop - 1) / maxCountPerLoop;
    u64 countLeft = param.DataDes.count;
    for (u32 loopIdx = 0; loopIdx < bufferLoopNum; loopIdx++) {
        bool isLastLoop = (loopIdx == bufferLoopNum - 1);
        u64 curCount = countLeft > maxCountPerLoop ? maxCountPerLoop : countLeft;
        countLeft -= curCount;
        u64 curSize = curCount * unitSize_;

        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algResResp_->cclInputMem;
        execMem.outputMem = algResResp_->cclOutputMem;
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;
        CHK_RET(KernelRunWithLoop(param, execMem, isLastLoop));
        CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, subStreams_));

        curInputPtr += curSize;
        curOutputPtr += curSize;
    }

    return HCCL_SUCCESS;
}

HcclResult
CollAllGatherRingZerocopyPipelineExecutor::KernelRunWithLoop(const OpParam& param, ExecMem& execMem, bool isLastLoop)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherRingZerocopyPipelineExecutor][KernelRunWithLoop]KernelRun begins.");
    u64 curSize = execMem.count * unitSize_;

    // Local Copy: UserIn -> Ccl
    DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr), curSize);
    DeviceMem dstMem = execMem.inputMem.range(0, curSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, mainStream_));

    memIdx_ = 0;
    blockIdx_ = level2Rank_;

    for (u32 step = 0; step < level2RankSize_; step++) { // level2 ring算法的步数 + 结尾一步
        // 启动L2 RDMA
        if (step < level2RankSize_ - 1) {
            CHK_RET(NotifyRdmaStreamStart());
        }

        // L1 SDMA
        CHK_RET(KernelRunInterServer(param, execMem));

        // L0 Zcopy 最后一次ccl循环时，需要启动Server内零拷贝步骤
        if (isLastLoop) {
            u64 inputSize = totalSize_ * level1RankSize_;
            u8* inputPtr = static_cast<u8*>(param.outputPtr) + blockSize_ * blockIdx_ + level0Rank_ * inputSize;
            u8* outputPtr = static_cast<u8*>(param.outputPtr) + blockSize_ * blockIdx_;
            ExecMem level0ExecMem;
            level0ExecMem.count = param.DataDes.count * level1RankSize_;
            level0ExecMem.inputMem = DeviceMem::create(inputPtr, inputSize);
            level0ExecMem.outputMem = DeviceMem::create(outputPtr, blockSize_);
            level0ExecMem.inputPtr = inputPtr;
            level0ExecMem.outputPtr = outputPtr;
            CHK_RET(KernelRunIntraServerPost(param, level0ExecMem));
        }

        // L2 RDMA
        if (step < level2RankSize_ - 1) {
            blockIdx_ = (blockIdx_ + level2RankSize_ - 1) % level2RankSize_;
            CHK_RET(KernelRunInterSuperPod(param, execMem));
            memIdx_ = 1 - memIdx_;
            CHK_RET(WaitRdmaStreamFinish());
        }
    }

    return HCCL_SUCCESS;
}

/* 超节点间1步 RMDA通信 */
HcclResult CollAllGatherRingZerocopyPipelineExecutor::KernelRunInterSuperPod(const OpParam& param, ExecMem& execMem)
{
    (void)param;
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    u32 prevLevel2Rank = (level2Rank_ + level2RankSize_ - 1) % level2RankSize_;
    u32 nextLevel2Rank = (level2Rank_ + 1) % level2RankSize_;
    LINK prevLevel2Link = level2CommInfo.links[prevLevel2Rank];
    LINK nextLevel2Link = level2CommInfo.links[nextLevel2Rank];

    CHK_RET(prevLevel2Link->TxAck(rdmaMainStream_));
    CHK_RET(nextLevel2Link->RxAck(rdmaMainStream_));

    u64 curSize = execMem.count * unitSize_;
    CHK_PRT_RET(memIdx_ > 1, HCCL_ERROR("[KernelRunInterSuperPod]memIdx[%u] is not valid", memIdx_), HCCL_E_PARA);
    u64 srcOffset = memIdx_ * curSize; // memIdx=0时发是0，收是1；memIdx=1时发是1，收是0
    u64 dstOffset = (1 - memIdx_) * curSize;
    HCCL_INFO(
        "[KernelRunInterSuperPod] local rank[%u] to[%u] from[%u] srcOffset[%llu] dstOffset[%llu] size[%llu]",
        level2Rank_, nextLevel2Rank, prevLevel2Rank, srcOffset, dstOffset, curSize);

    CHK_RET(nextLevel2Link->TxAsync(
        UserMemType::INPUT_MEM, dstOffset, static_cast<u8*>(execMem.inputMem.ptr()) + srcOffset, curSize,
        rdmaMainStream_));
    CHK_RET(prevLevel2Link->RxAsync(
        UserMemType::INPUT_MEM, srcOffset, static_cast<u8*>(execMem.inputMem.ptr()) + dstOffset, curSize,
        rdmaMainStream_));
    CHK_RET(prevLevel2Link->PostFinAck(rdmaMainStream_));
    CHK_RET(nextLevel2Link->WaitFinAck(rdmaMainStream_));
    return HCCL_SUCCESS;
}

/* Server内的零拷贝通信 */
HcclResult CollAllGatherRingZerocopyPipelineExecutor::KernelRunIntraServerPost(const OpParam& param, ExecMem& execMem)
{
    // 计算slice
    u64 sliceSize = execMem.count * unitSize_;
    std::vector<Slice> dataSegsSlice;
    CalcDataSlices(sliceSize, level0RankSize_, dataSegsSlice);

    // 执行AllGather
    std::vector<std::vector<Slice>> multRingsUserMemSlice = {dataSegsSlice};
    u64 baseOffset = blockIdx_ * blockSize_;
    if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) { // 只有1个ring环，只用到一条流，没有流之间同步
        HCCL_INFO("[%s] single ring AllGather", __func__);
        CHK_RET(MultiRingAllGather(
            tag_, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, multRingsUserMemSlice,
            mainStream_, PROF_STAGE_0, baseOffset, nullptr, multRingsUserMemSlice));
    } else {
        CHK_PRT_RET(
            topoType_ != TopoType::TOPO_TYPE_NP_DOUBLE_RING,
            HCCL_ERROR("[%s] unknown topoType: %u", __func__, topoType_), HCCL_E_NOT_SUPPORT);
        HCCL_INFO("[%s] semi ring AllGather", __func__);
        CHK_RET(SemiRingAllGather(
            tag_, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, multRingsUserMemSlice,
            mainStream_, PROF_STAGE_0, baseOffset, nullptr, multRingsUserMemSlice));
    }

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingZerocopyPipelineExecutor::SemiRingAllGather(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count, const HcclDataType& dataType,
    const std::vector<std::vector<Slice>>& multRingsSliceZero, const Stream& stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo* opInfo, const std::vector<std::vector<Slice>>& multRingsUserMemSlice)
{
    (void)tag;
    (void)multRingsSliceZero;
    (void)opInfo;
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 执行
    std::unique_ptr<AlgTemplateBase> level0Template
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_UNIFIED_MARCH, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_UNIFIED_MARCH in COMM_LEVEL0", __func__);
    CHK_SMART_PTR_NULL(level0Template);

    CHK_RET(level0Template->Prepare(
        stream, level0CommInfo, inputMem, outputMem, inputMem, outputMem, count * SIZE_TABLE[dataType], sdmaSubStreams_,
        notifySdmaMain_, notifySdmaSub_, multRingsUserMemSlice, baseOffset));
    HcclResult ret = level0Template->RegisterProfiler(
        ((COMM_INDEX_0 + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID)
            + (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllGatherRingZerocopyPipelineExecutor][SemiRingAllGather]SemiRing AllGather failed, ret[%d]", ret),
        ret);
    CHK_RET(level0Template->RunAsync());
    return ret;
}

/* 超节点内的节点间通信 */
HcclResult CollAllGatherRingZerocopyPipelineExecutor::KernelRunInterServer(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherRingZerocopyPipelineExecutor][KernelRunInterServer] starts");

    // 前处理
    CHK_RET(KernelRunInterServerPreProcess(param, execMem));

    if (level1RankSize_ > 1) {
        // 计算slice
        u64 sliceSize = execMem.count * unitSize_;
        std::vector<Slice> level1DataSegsSlice;
        CalcDataSlices(sliceSize, level1RankSize_, level1DataSegsSlice);

        std::unique_ptr<AlgTemplateBase> level1AGTemplate;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1AGTemplate
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1AGTemplate
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1AGTemplate
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL1", __func__);
        } else {
            HCCL_ERROR("[KernelRunInterServer] unsupported level1 algtype [%s]", AlgTypeToStr(algType_).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_SMART_PTR_NULL(level1AGTemplate);
        // 执行算法编排
        CHK_RET(level1AGTemplate->Prepare(
            execMem.outputMem, execMem.outputMem, execMem.inputMem, INVALID_U64, param.DataDes.dataType, mainStream_,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice));
        CHK_RET(level1AGTemplate->RegisterProfiler(
            (level1RankSize_ << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1Rank_, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
            mainStream_));
        CHK_RET(CheckCommSize(COMM_LEVEL1, level0Rank_ + 1));
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0Rank_);
        CHK_RET(RunTemplate(level1AGTemplate, level1CommInfo));
    }

    // 后处理
    CHK_RET(KernelRunInterServerPostProcess(param, execMem));

    HCCL_INFO("[CollAllGatherRingZerocopyPipelineExecutor][KernelRunInterServer] run success");
    return HCCL_SUCCESS;
}

/* 数据交换 */
HcclResult
CollAllGatherRingZerocopyPipelineExecutor::KernelRunInterServerPreProcess(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherRingZerocopyPipelineExecutor] KernelRunInterServerPreProcess");
    u64 curSize = execMem.count * unitSize_;

    u32 remoteRankSend = 0;
    u32 remoteRankRecv = 0;
    CHK_RET(CalExchangeRemoteRank(remoteRankSend, remoteRankRecv));

    DeviceMem srcMem = execMem.inputMem.range(memIdx_ * curSize, curSize);
    DeviceMem dstMem = execMem.outputMem.range(level1Rank_ * curSize, curSize);
    if (remoteRankSend != topoAttr_.userRank && remoteRankRecv != topoAttr_.userRank) { // 需要交换数据
        HCCL_DEBUG("[%s] rank [%u] need exchange", __func__, topoAttr_.userRank);
        // 获取通信对端的link
        LINK sendLink;
        LINK recvLink;
        CHK_RET(GetTransportForExchange(remoteRankSend, sendLink));
        CHK_RET(GetTransportForExchange(remoteRankRecv, recvLink));

        bool IsRemoteRankSendNeighbor = IsLevel0Neighbor(remoteRankSend, level0RankSize_);
        bool IsRemoteRankRecvNeighbor = IsLevel0Neighbor(remoteRankRecv, level0RankSize_);
        // 当通信对端恰好是同server邻居时，复用Level0建链，其注册内存是UserMem，需特殊处理：经过UserOut中转CCL的数据
        // 否则，在CommCombineOrder上建链，其注册内存是CCL Buffer
        u64 blockOffset = blockSize_ * blockIdx_;
        u64 tmpOffset = blockOffset + totalSize_ * level0Rank_ * level1RankSize_;
        DeviceMem tmpMem = DeviceMem::create(static_cast<u8*>(execMem.outputPtr) + tmpOffset, curSize);
        if (IsRemoteRankSendNeighbor) {
            HCCL_DEBUG(
                "[%s] neighbor process srcPtr[%p] tmpPtr[%p] size[%llu]", __func__, srcMem.ptr(), tmpMem.ptr(),
                curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, tmpMem, srcMem, mainStream_));
        }

        // 执行通信
        CHK_RET(recvLink->TxAck(mainStream_));
        CHK_RET(sendLink->RxAck(mainStream_));

        u32 remoteLevel1RankSend = remoteRankSend % (level0RankSize_ * level1RankSize_) / level0RankSize_;
        u64 txDstOffset = remoteLevel1RankSend * curSize;
        HCCL_INFO("[%s] remoteRankSend[%u] txDstOffset[%llu]", __func__, remoteRankSend, txDstOffset);
        if (IsRemoteRankSendNeighbor) {
            CHK_RET(sendLink->TxAsync(UserMemType::OUTPUT_MEM, txDstOffset, tmpMem.ptr(), curSize, mainStream_));
        } else {
            CHK_RET(sendLink->TxAsync(UserMemType::OUTPUT_MEM, txDstOffset, srcMem.ptr(), curSize, mainStream_));
        }

        u64 rxSrcOffset = memIdx_ * curSize;
        if (IsRemoteRankRecvNeighbor) {
            u32 remoteLevel0RankRecv = remoteRankRecv % level0RankSize_;
            rxSrcOffset = static_cast<u8*>(execMem.outputPtr) - static_cast<u8*>(param.outputPtr) + blockOffset
                          + totalSize_ * remoteLevel0RankRecv * level1RankSize_;
            CHK_RET(recvLink->RxAsync(UserMemType::OUTPUT_MEM, rxSrcOffset, dstMem.ptr(), curSize, mainStream_));
        } else {
            CHK_RET(recvLink->RxAsync(UserMemType::INPUT_MEM, rxSrcOffset, dstMem.ptr(), curSize, mainStream_));
        }
        HCCL_INFO("[%s] remoteRankRecv[%u] rxSrcOffset[%llu]", __func__, remoteRankRecv, rxSrcOffset);

        // 交换数据的两端之间Barrier，确认收发完成
        CHK_RET(recvLink->TxAck(mainStream_));
        CHK_RET(sendLink->RxAck(mainStream_));
        CHK_RET(sendLink->TxDataSignal(mainStream_));
        CHK_RET(recvLink->RxDataSignal(mainStream_));
    } else { // 不需要交换数据，将数据从ccl in拷到ccl out
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, mainStream_));
    }

    return HCCL_SUCCESS;
}

/* 将通信结果从ccl output搬到user output */
HcclResult
CollAllGatherRingZerocopyPipelineExecutor::KernelRunInterServerPostProcess(const OpParam& param, ExecMem& execMem)
{
    (void)param;
    u64 blockOffset = blockIdx_ * blockSize_;
    u64 curSize = execMem.count * unitSize_;
    for (u32 i = 0; i < level1RankSize_; i++) {
        DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.outputMem.ptr()) + i * curSize, curSize);
        u64 outputOffset = blockOffset + totalSize_ * (level0Rank_ * level1RankSize_ + i);
        DeviceMem dstMem = DeviceMem::create(static_cast<u8*>(execMem.outputPtr) + outputOffset, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, mainStream_));
        HCCL_DEBUG("[%s] memcopy from CCLOut[%p] to UserOut[%p]", __func__, srcMem.ptr(), dstMem.ptr());
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingZerocopyPipelineExecutor::NotifyRdmaStreamStart()
{
    CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, notifyMainToRdma_, INVALID_VALUE_STAGE));
    CHK_RET(LocalNotify::Wait(rdmaMainStream_, dispatcher_, notifyMainToRdma_, INVALID_VALUE_STAGE));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingZerocopyPipelineExecutor::WaitRdmaStreamFinish()
{
    CHK_RET(LocalNotify::Post(rdmaMainStream_, dispatcher_, notifyRdmaToMain_, INVALID_VALUE_STAGE));
    CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, notifyRdmaToMain_, INVALID_VALUE_STAGE));
    return HCCL_SUCCESS;
}

/* 建链时也需要调用，还没有commInfo */
HcclResult CollAllGatherRingZerocopyPipelineExecutor::CalExchangeRemoteRank(u32& remoteRankSend, u32& remoteRankRecv)
{
    u32 userRank = topoAttr_.userRank;
    u32 userRankSize = topoAttr_.userRankSize;
    u32 level2RankSize = topoAttr_.superPodNum;
    CHK_PRT_RET(level2RankSize == 0, HCCL_ERROR("[CalExchangeRemoteRank]level2RankSize is 0"), HCCL_E_PARA);
    u32 level1RankSize = topoAttr_.serverNum / level2RankSize;
    CHK_PRT_RET(level1RankSize == 0, HCCL_ERROR("[CalExchangeRemoteRank]level1RankSize is 0"), HCCL_E_PARA);
    u32 level0RankSize = userRankSize / level1RankSize / level2RankSize;
    u32 level0Rank = userRank % level0RankSize;
    u32 level1Rank = userRank % (level0RankSize * level1RankSize) / level0RankSize;
    u32 level2Rank = userRank / level0RankSize / level1RankSize;

    u32 level2StartRank = level2Rank * level0RankSize * level1RankSize;
    // 计算本超节点内本端将要接收数据的源rank
    remoteRankRecv = level2StartRank + level0Rank * level1RankSize + level1Rank;
    // 计算本超节点内本端将要发送数据的目标rank
    u32 srcLevel0Rank = (userRank - level2StartRank) / level1RankSize;
    u32 srcLevel1Rank = (userRank - level2StartRank) % level1RankSize;
    remoteRankSend = level2StartRank + srcLevel1Rank * level0RankSize + srcLevel0Rank;

    HCCL_INFO(
        "[%s] rank[%u:%u/%u/%u] remoteRankSend[%u], remoteRankRecv[%u]", __func__, topoAttr_.userRank, level2Rank,
        level1Rank, level0Rank, remoteRankSend, remoteRankRecv);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingZerocopyPipelineExecutor::CalcDataSlices(
    u64 sliceSize, u32 rankSize, std::vector<Slice>& dataSegsSlice) const
{
    dataSegsSlice.resize(rankSize);
    for (u32 i = 0; i < rankSize; i++) {
        dataSegsSlice[i].size = sliceSize;
        dataSegsSlice[i].offset = i * sliceSize;
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC(
    "AllGatherRingZerocopyPipelineExecutor", AllGatherRingZerocopyPipeline, CollAllGatherRingZerocopyPipelineExecutor);
} // namespace hccl
