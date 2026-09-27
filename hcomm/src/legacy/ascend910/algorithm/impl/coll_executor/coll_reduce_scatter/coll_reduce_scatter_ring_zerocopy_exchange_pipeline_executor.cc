/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_ring_zerocopy_exchange_pipeline_executor.h"

namespace hccl {

CollReduceScatterRingZerocopyExchangePipelineExecutor::CollReduceScatterRingZerocopyExchangePipelineExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    CCLMemSlice_ = false;
    DMAReduceFlag_ = true;   // 设为true，以禁用RunLoop中的本地拷贝
    desc_.isZeroCopy = true; // 执行RunLoop的KernelRunInterServer分支
    desc_.deterministic = 1;
    desc_.level1SupportedAlgos = {
        AlgTypeLevel1::ALG_LEVEL1_RING,
        AlgTypeLevel1::ALG_LEVEL1_NHR,
        AlgTypeLevel1::ALG_LEVEL1_NB,
    };
    desc_.level2SupportedAlgos = {AlgTypeLevel2::ALG_LEVEL2_PIPELINE};
}

void CollReduceScatterRingZerocopyExchangePipelineExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    root_ = param.root;
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
    opType_ = param.opType;

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * unitSize;
}

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::CalcStreamNum(u32& streamNum)
{
    // level0 需要的stream数，double ring需要2条，single ring需要1条直接用主流
    u32 totalStreamNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE : 0;
    // level1 用NHR等ring算法，需要1条stream。但level0与level1串行，直接用主流
    // level2 用单ring，level2与level0/level1并行，需要1条额外的流
    totalStreamNum += 1;
    streamNum = totalStreamNum;
    HCCL_INFO("[CalcStreamNum] tag[%s] streamNum[%u] topoType_[%d]", tag_.c_str(), streamNum, topoType_);

    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterRingZerocopyExchangePipelineExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO(
        "[CalcCommInfo] tag[%s] algoLevel0[%d] algoLevel1[%d] algoLevel2[%d]", tag_.c_str(), algType_.algoLevel0,
        algType_.algoLevel1, algType_.algoLevel2);

    TransportMemType inputType = TransportMemType::CCL_INPUT;
    TransportMemType outputType = TransportMemType::CCL_OUTPUT;
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcExchangeCommInfo(opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::CalcLevel0CommInfo(
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

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::CalcExchangeCommInfo(
    std::vector<LevelNSubCommTransport>& opTransport)
{
    std::set<u32> commTargetUserRankSet;
    u32 remoteRankSend = 0;
    u32 remoteRankRecv = 0;

    CHK_RET(CalExchangeRemoteRank(remoteRankSend, remoteRankRecv));
    HCCL_INFO(
        "[CalcExchangeCommInfo] tag[%s] userRank[%u] remoteRankSend[%u] remoteRankRecv[%u]", tag_.c_str(),
        topoAttr_.userRank, remoteRankSend, remoteRankRecv);
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

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::CalExchangeRemoteRank(
    u32& remoteRankSend, u32& remoteRankRecv) const
{
    u32 l2Size = topoAttr_.superPodNum;
    CHK_PRT_RET(l2Size == 0, HCCL_ERROR("[CalExchangeRemoteRank] invalid rank size, level2RankSize is 0"), HCCL_E_PARA);
    u32 l1Size = topoAttr_.serverNum / l2Size;
    CHK_PRT_RET(l1Size == 0, HCCL_ERROR("[CalExchangeRemoteRank] invalid rank size, level1RankSize is 0"), HCCL_E_PARA);
    u32 l0Size = topoAttr_.userRankSize / l2Size / l1Size;
    CHK_PRT_RET(l0Size == 0, HCCL_ERROR("[CalExchangeRemoteRank] invalid rank size, level0RankSize is 0"), HCCL_E_PARA);

    // 根据rankId计算出坐标(i, j, k)
    u32 l2Index = topoAttr_.userRank / l1Size / l0Size;
    u32 l1Index = (topoAttr_.userRank % (l1Size * l0Size)) / l0Size;
    u32 l0Index = topoAttr_.userRank % l0Size;

    // 计算本端将要发送数据的目标rank
    remoteRankSend = l2Index * l1Size * l0Size + l0Index * l1Size + l1Index;

    // 计算本端将要接收数据的目标rank
    u32 r = l1Index * l0Size + l0Index; // 超节点内相对rankid
    l0Index = r / l1Size;
    l1Index = r % l1Size;
    remoteRankRecv = l2Index * l1Size * l0Size + l1Index * l0Size + l0Index;
    return HCCL_SUCCESS;
}

u64 CollReduceScatterRingZerocopyExchangePipelineExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    u64 maxCountPerLoop
        = ((inCCLbufferSize_ / topoAttr_.serverNum / HCCL_MIN_SLICE_ALIGN) * HCCL_MIN_SLICE_ALIGN) / unitSize;
    return maxCountPerLoop;
}

HcclResult
CollReduceScatterRingZerocopyExchangePipelineExecutor::KernelRunIntraServerPre(const OpParam& param, ExecMem& execMem)
{
    (void)execMem;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, unitSize_));
    CHK_RET(GetCommRankInfoNormal(
        level0Rank_, level0RankSize_, level1Rank_, level1RankSize_, level2Rank_, level2RankSize_, false));
    CHK_RET(CalExchangeRemoteRank(exchangeRemoteRankSend_, exchangeRemoteRankRecv_));

    HCCL_INFO(
        "[KernelRunIntraServerPre] rank[%u:%u,%u,%u], rankSize[%u, %u, %u] exchange remoteRank[send:%u Recv:%u]",
        topoAttr_.userRank, level2Rank_, level1Rank_, level0Rank_, level2RankSize_, level1RankSize_, level0RankSize_,
        exchangeRemoteRankSend_, exchangeRemoteRankRecv_);
    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterRingZerocopyExchangePipelineExecutor::KernelRunInterServer(const OpParam& param, ExecMem& execMem)
{
    curSize_ = execMem.count * unitSize_;
    HCCL_INFO(
        "[CollReduceScatterRingZerocopyExchangePipelineExecutor] run start, rank[%u:%u,%u,%u], curSize_[%llu]",
        topoAttr_.userRank, level2Rank_, level1Rank_, level0Rank_, curSize_);

    for (u32 step = 0; step < level2RankSize_; step++) {
        if (!intraServerDone_) {
            // 只有第一个loop才需要执行节点内RS
            CHK_RET(RunIntraServer(param, execMem, step));
        }

        // 准备节点间RS的数据，user in搬运到ccl in
        CHK_RET(RunInterServerPreProcess(param, execMem, step));
        // 超节点内、节点间通信执行RS，编排在主流上
        if (level1RankSize_ > 1) {
            // 节点间RS完成后数据在ccl in
            CHK_RET(RunInterServer(param, execMem, step));
        }
        // 数据最终在ccl out
        CHK_RET(RunInterServerPostProcess(param, execMem, step));

        // 从steep 1开始要进行reduce，将本轮超节点间获取的数据与本轮超节点内的数据进行reduce
        if ((step > 0) && (level2RankSize_ > 1)) {
            CHK_RET(RunSuperPodPostSync(param));
            // 超节点间通信 与 超节点内通信 都完成后，本地进行reduce操作
            CHK_RET(RunSuperPodAndInterServerPostProcess(param, execMem, step));
        }

        if (step < (level2RankSize_ - 1)) {
            // 超节点间通信, 编排在最后一个slaveStreams上
            CHK_RET(RunSuperPodPreSync(param));
            CHK_RET(RunSuperPod(param, execMem, step + 1));
        }
    }

    // 将最终数据从ccl out搬到user out
    CHK_RET(RunFinallyProcess(param, execMem));

    intraServerDone_ = true;
    HCCL_INFO(
        "[CollReduceScatterRingZerocopyExchangePipelineExecutor] run success, rank[%u:%u,%u,%u]", topoAttr_.userRank,
        level2Rank_, level1Rank_, level0Rank_);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::RunSuperPodPreSync(const OpParam& param)
{
    Stream stream = param.stream;
    Stream slaveStream = algResResp_->slaveStreams.back();
    // 主流RS完成后，通知超节点间通信开始
    CHK_RET(LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux.back(), INVALID_VALUE_STAGE));
    // 从流等待超节点内RS完成
    CHK_RET(LocalNotify::Wait(slaveStream, dispatcher_, algResResp_->notifiesAux.back(), INVALID_VALUE_STAGE));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::RunSuperPodPostSync(const OpParam& param)
{
    Stream stream = param.stream;
    Stream slaveStream = algResResp_->slaveStreams.back();
    // 从流通知主流，超节点间数据搬运完成
    CHK_RET(LocalNotify::Post(slaveStream, dispatcher_, algResResp_->notifiesMain.back(), INVALID_VALUE_STAGE));
    // 主流等待超节点通信完成
    CHK_RET(LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain.back(), INVALID_VALUE_STAGE));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::RunIntraServer(
    const OpParam& param, const ExecMem& execMem, u32 step)
{
    (void)execMem;
    // 计算slice信息, 将user in分成level2RankSize_块, 每个step处理一块blockIndex, 每个block需要分成level0RankSize_片
    u64 level0Count = param.DataDes.count * level1RankSize_;
    u32 blockIndex = (level2Rank_ + level2RankSize_ - (step + 1)) % level2RankSize_;
    u64 sliceSize = level0Count * unitSize_;
    u64 blockOffset = blockIndex * sliceSize * level0RankSize_;

    HCCL_DEBUG(
        "[RunIntraServer] rank[%u:%u,%u,%u] step[%u] blockIndex[%u], level0Count[%llu]", topoAttr_.userRank,
        level2Rank_, level1Rank_, level0Rank_, step, blockIndex, level0Count);

    std::vector<Slice> dataSegsSlice(level0RankSize_);
    for (u32 i = 0; i < level0RankSize_; i++) {
        dataSegsSlice[i].offset = blockOffset + sliceSize * i; // 相对于param.inputPtr偏移
        dataSegsSlice[i].size = sliceSize;
    }
    std::vector<std::vector<Slice>> multRingsUserMemSlice = {dataSegsSlice};

    // 算法编排
    if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
        CHK_RET(MultiRingReduceScatter(
            param.tag, algResResp_->paramInputMem, algResResp_->paramInputMem, level0Count, param.DataDes.dataType,
            param.reduceType, multRingsUserMemSlice, param.stream, PROF_STAGE_1, 0, nullptr, multRingsUserMemSlice));
    } else {
        CHK_PRT_RET(
            topoType_ != TopoType::TOPO_TYPE_NP_DOUBLE_RING,
            HCCL_ERROR("[RunIntraServer] unknown topoType: %u", topoType_), HCCL_E_NOT_SUPPORT);
        CHK_RET(SemiRingReduceScatter(
            param.tag, algResResp_->paramInputMem, algResResp_->paramInputMem, level0Count, param.DataDes.dataType,
            param.reduceType, multRingsUserMemSlice, param.stream, PROF_STAGE_1, 0, nullptr, multRingsUserMemSlice));
    }

    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::SemiRingReduceScatter(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream,
    s32 profStage, const u64 baseOffset, const HcomCollOpInfo* opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    (void)tag;
    (void)multRingsSliceZero;
    (void)baseOffset;
    (void)opInfo;
    HCCL_DEBUG(
        "[SemiRingReduceScatter] starts, rank[%u:%u,%u,%u]", topoAttr_.userRank, level2Rank_, level1Rank_, level0Rank_);

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 此处计算reduceAttr计算，outputmem使用的是scratchmem
    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);
    // 执行
    std::unique_ptr<AlgTemplateBase> executor = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_UNIFIED_MARCH, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_UNIFIED_MARCH in COMM_LEVEL0", __func__);
    CHK_SMART_PTR_NULL(executor);

    CHK_RET(executor->Prepare(
        stream, level0CommInfo, algResResp_->paramInputMem, algResResp_->paramOutputMem, inputMem, outputMem, count,
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, dataType, reductionOp,
        multRingsUserMemSlice, reduceAttr));

    HcclResult ret = executor->RegisterProfiler(
        ((COMM_INDEX_0 + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID)
            + (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[SemiRingReduceScatter] Double ring ReduceScatter failed,return[%d]", ret),
        ret);

    CHK_RET(executor->RunAsync());

    HCCL_DEBUG(
        "[SemiRingReduceScatter] run success, rank[%u:%u,%u,%u]", topoAttr_.userRank, level2Rank_, level1Rank_,
        level0Rank_);
    return ret;
}

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::RunInterServerPreProcess(
    const OpParam& param, const ExecMem& execMem, u32 step)
{
    // 数据准备，将节点内RS的结果从user in搬到ccl in
    u32 blockIndex = (level2Rank_ + level2RankSize_ - (step + 1)) % level2RankSize_;
    u32 cclSliceIndex = blockIndex * level1RankSize_;
    u32 usrInSliceIndex = blockIndex * level1RankSize_ * level0RankSize_ + level1RankSize_ * level0Rank_;
    Stream stream = param.stream;

    HCCL_DEBUG(
        "[RunInterServerPreProcess] rank[%u:%u,%u,%u] step[%u] blockIndex[%u] sliceIndex[%u, %u]", topoAttr_.userRank,
        level2Rank_, level1Rank_, level0Rank_, step, blockIndex, cclSliceIndex, usrInSliceIndex);
    // 本地 user in -> ccl in
    for (u32 i = 0; i < level1RankSize_; i++) {
        u64 ccInOffset = (cclSliceIndex + i) * curSize_;
        u64 userInOffset = (usrInSliceIndex + i) * param.DataDes.count * unitSize_; // 相对于execMem.inputPtr偏移
        DeviceMem dstMem = execMem.inputMem.range(ccInOffset, curSize_);
        DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr) + userInOffset, curSize_);
        HCCL_DEBUG(
            "[RunInterServerPreProcess] rank[%u:%u,%u,%u] step[%u] userInOffset[%llu] -> ccInOffset[%llu]",
            topoAttr_.userRank, level2Rank_, level1Rank_, level0Rank_, step, userInOffset, ccInOffset);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
    }
    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterRingZerocopyExchangePipelineExecutor::RunInterServer(const OpParam& param, ExecMem& execMem, u32 step)
{
    // 计算slice信息，也就是在ccl in的偏移
    std::vector<Slice> level1DataSegsSlice(level1RankSize_);
    u32 blockIndex = (level2Rank_ + level2RankSize_ - (step + 1)) % level2RankSize_;
    u32 sliceIndex = blockIndex * level1RankSize_;

    HCCL_DEBUG(
        "[RunInterServer] rank[%u:%u,%u,%u] step[%u] blockIndex[%u] sliceStart[%u] sliceCnt[%u]", topoAttr_.userRank,
        level2Rank_, level1Rank_, level0Rank_, step, blockIndex, sliceIndex, level1RankSize_);

    for (u32 i = 0; i < level1RankSize_; i++) {
        level1DataSegsSlice[i].offset = (sliceIndex + i) * curSize_;
        level1DataSegsSlice[i].size = curSize_;
    }

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL1", __func__);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR in COMM_LEVEL1", __func__);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr, false));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_LEVEL1", __func__);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    }
    CHK_SMART_PTR_NULL(level1TempAlg);

    // 执行算法编排, 主流上执行，只会使用ccl in，执行完成后数据在ccl in
    CHK_RET(CheckCommSize(COMM_LEVEL1, level0Rank_ + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0Rank_);
    CHK_RET(level1TempAlg->Prepare(
        execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, param.DataDes.dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, level1DataSegsSlice));
    CHK_RET(level1TempAlg->RegisterProfiler(
        (level1RankSize_ << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1Rank_, PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET,
        param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));

    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::ExchangeData(
    const OpParam& param, const ExecMem& execMem, u32 step, u32 remoteRankSend, u32 remoteRankRecv)
{
    // 获取通信对端的link
    LINK sendLink;
    LINK recvLink;
    CHK_RET(GetTransportForExchange(remoteRankSend, sendLink));
    CHK_RET(GetTransportForExchange(remoteRankRecv, recvLink));
    CHK_PTR_NULL(sendLink);
    CHK_PTR_NULL(recvLink);

    // 当通信对端恰好是同server的邻居时，复用Level0的建链，其注册的内存是UserMem
    // 否则，在CommCombineOrder上建链，其注册内存是ccl buf
    Stream stream = param.stream;
    u32 blockIndex = (level2Rank_ + level2RankSize_ - (step + 1)) % level2RankSize_;
    u32 sliceIndexSnd = blockIndex * level1RankSize_ + level1Rank_; // 要发送的数据块在本地ccl in的位置
    u32 sliceIndexCclOut = blockIndex * level1RankSize_;

    HCCL_DEBUG(
        "[RunInterServerPostProcess] rank[%u:%u,%u,%u] step[%u] send blockIndex[%u] sliceIndex[%u] cclout[%u]",
        topoAttr_.userRank, level2Rank_, level1Rank_, level0Rank_, step, blockIndex, sliceIndexSnd, sliceIndexCclOut);

    bool remoteSndl0Neighbor = IsLevel0Neighbor(remoteRankSend, level0RankSize_);
    bool remoteRcvl0Neighbor = IsLevel0Neighbor(remoteRankRecv, level0RankSize_);
    if (remoteSndl0Neighbor) {
        // 先本地 ccl in -> user in
        u32 usrInSliceIndex
            = blockIndex * level1RankSize_ * level0RankSize_ + level1RankSize_ * level0Rank_ + level1Rank_;
        u64 userInOffset = usrInSliceIndex * param.DataDes.count * unitSize_; // 相对于param.inputPtr偏移
        DeviceMem srcMem = execMem.inputMem.range(sliceIndexSnd * curSize_, curSize_);
        DeviceMem dstMem = DeviceMem::create(static_cast<u8*>(param.inputPtr) + userInOffset, curSize_);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
        HCCL_DEBUG(
            "[RunInterServerPostProcess] rank[%u:%u,%u,%u] step[%u] blockIndex[%u] ci[%u]->ui[%u]", topoAttr_.userRank,
            level2Rank_, level1Rank_, level0Rank_, step, blockIndex, sliceIndexSnd, usrInSliceIndex);

        // user in send to remote user out
        CHK_RET(recvLink->TxAck(stream));
        CHK_RET(sendLink->RxAck(stream));
        CHK_RET(sendLink->TxAsync(
            UserMemType::OUTPUT_MEM, 0, static_cast<u8*>(param.inputPtr) + userInOffset, curSize_, stream));
    } else {
        // ccl in send to remote ccl out
        CHK_RET(recvLink->TxAck(stream));
        CHK_RET(sendLink->RxAck(stream));
        CHK_RET(sendLink->TxAsync(
            UserMemType::OUTPUT_MEM, sliceIndexCclOut * curSize_,
            static_cast<u8*>(execMem.inputMem.ptr()) + sliceIndexSnd * curSize_, curSize_, stream));
    }

    u32 remoteL1Rank = (remoteRankRecv % (level1RankSize_ * level0RankSize_)) / level0RankSize_;
    u32 sliceIndexRcv = blockIndex * level1RankSize_ + remoteL1Rank; // 要接收的数据块在对端ccl in的位置
    if (remoteRcvl0Neighbor) {
        u32 usrInSliceIndexPeer
            = blockIndex * level1RankSize_ * level0RankSize_ + level1Rank_ * level0RankSize_ + level0Rank_;
        u64 userInOffsetPeer = usrInSliceIndexPeer * param.DataDes.count * unitSize_; // 相对于param.inputPtr偏移
        CHK_RET(recvLink->RxAsync(UserMemType::INPUT_MEM, userInOffsetPeer, execMem.outputPtr, curSize_, stream));
    } else {
        CHK_RET(recvLink->RxAsync(
            UserMemType::INPUT_MEM, sliceIndexRcv * curSize_,
            static_cast<u8*>(execMem.outputMem.ptr()) + sliceIndexCclOut * curSize_, curSize_, stream));
        CHK_RET(recvLink->PostFinAck(stream));
    }

    if (!remoteSndl0Neighbor) {
        CHK_RET(sendLink->WaitFinAck(stream));
    }

    // 交换数据的两端之间Barrier，确认收发完成
    CHK_RET(recvLink->TxAck(stream));
    CHK_RET(sendLink->RxAck(stream));
    CHK_RET(sendLink->TxDataSignal(stream));
    CHK_RET(recvLink->RxDataSignal(stream));

    if (remoteRcvl0Neighbor) {
        // 本地 user out -> ccl out
        DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.outputPtr), curSize_);
        DeviceMem dstMem = execMem.outputMem.range(sliceIndexCclOut * curSize_, curSize_);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
        HCCL_DEBUG(
            "[RunInterServerPostProcess] rank[%u:%u,%u,%u] step[%u] blockIndex[%u] uo->co[%u]", topoAttr_.userRank,
            level2Rank_, level1Rank_, level0Rank_, step, blockIndex, sliceIndexCclOut);
    }
    HCCL_DEBUG(
        "[RunInterServerPostProcess] rank[%u:%u,%u,%u] step[%u] recv blockIndex[%u] sliceIndex[%u] cclout[%u]",
        topoAttr_.userRank, level2Rank_, level1Rank_, level0Rank_, step, blockIndex, sliceIndexRcv, sliceIndexCclOut);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::RunInterServerPostProcess(
    const OpParam& param, const ExecMem& execMem, u32 step)
{
    // 超节点内数据交换
    u32 remoteRankSend = exchangeRemoteRankSend_;
    u32 remoteRankRecv = exchangeRemoteRankRecv_;

    HCCL_DEBUG(
        "[RunInterServerPostProcess] rank[%u:%u,%u,%u] step[%u] remoteRankSend[%u] remoteRankRecv[%u]",
        topoAttr_.userRank, level2Rank_, level1Rank_, level0Rank_, step, remoteRankSend, remoteRankRecv);
    if (remoteRankSend == topoAttr_.userRank && remoteRankRecv == topoAttr_.userRank) { // 不需要交换数据
        // 本地 ccl in -> ccl out
        Stream stream = param.stream;
        u32 blockIndex = (level2Rank_ + level2RankSize_ - (step + 1)) % level2RankSize_;
        u32 srcSliceIndex = blockIndex * level1RankSize_ + level1Rank_;
        u32 dstSliceIndex = blockIndex * level1RankSize_;
        DeviceMem srcMem = execMem.inputMem.range(srcSliceIndex * curSize_, curSize_);
        DeviceMem dstMem = execMem.outputMem.range(dstSliceIndex * curSize_, curSize_);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
        HCCL_DEBUG(
            "[RunInterServerPostProcess] rank[%u:%u,%u,%u] step[%u] blockIndex[%u] ci[%u]->co[%u]", topoAttr_.userRank,
            level2Rank_, level1Rank_, level0Rank_, step, blockIndex, srcSliceIndex, dstSliceIndex);
        return HCCL_SUCCESS;
    }

    return ExchangeData(param, execMem, step, remoteRankSend, remoteRankRecv);
}

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::RunSuperPod(
    const OpParam& param, const ExecMem& execMem, u32 step)
{
    (void)param;
    Stream slaveStream = algResResp_->slaveStreams.back();
    // 发送前回RS好的数据
    u32 blockIndexSnd = (level2Rank_ + level2RankSize_ - step) % level2RankSize_;
    u32 sliceIndexSnd = blockIndexSnd * level1RankSize_; // 要发送的数据处于本地的哪个slice

    // 接受上一超节点发来的数据
    u32 blockIndexRcv = (level2Rank_ + level2RankSize_ - (step + 1)) % level2RankSize_;
    u32 sliceIndexRcv = blockIndexRcv * level1RankSize_; // 要接收的数据处于对端的哪个slice

    u32 preRank = (level2Rank_ + level2RankSize_ - 1) % level2RankSize_;
    u32 nextRank = (level2Rank_ + 1) % level2RankSize_;
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    LINK sendLink = level0CommInfo.links[nextRank];
    LINK recvLink = level0CommInfo.links[preRank];
    CHK_PTR_NULL(sendLink);
    CHK_PTR_NULL(recvLink);

    // 将数据发给nextRank前回的ccl in范围
    u32 remoteBlockIndexSndTo = (nextRank + level2RankSize_ - step) % level2RankSize_;
    u32 remoteSliceIndexSndTo = remoteBlockIndexSndTo * level1RankSize_; // 对端在哪个slice收对应的数据
    HCCL_DEBUG(
        "[RunSuperPod] rank[%u:%u,%u,%u] step[%u] send blockIndex[%u] sliceIndex[%u]->[%u]", topoAttr_.userRank,
        level2Rank_, level1Rank_, level0Rank_, step, blockIndexSnd, sliceIndexSnd, remoteSliceIndexSndTo);
    HCCL_DEBUG(
        "[RunSuperPod] rank[%u:%u,%u,%u] step[%u] recv blockIndex[%u] sliceIndex[%u]<-[%u]", topoAttr_.userRank,
        level2Rank_, level1Rank_, level0Rank_, step, blockIndexRcv, sliceIndexSnd, sliceIndexRcv);

    CHK_RET(recvLink->TxAck(slaveStream));
    CHK_RET(sendLink->RxAck(slaveStream));
    // 建链时其注册内存是ccl in与ccl out
    // ccl out send to remote ccl in
    CHK_RET(sendLink->TxAsync(
        UserMemType::INPUT_MEM, remoteSliceIndexSndTo * curSize_,
        static_cast<s8*>(execMem.outputMem.ptr()) + sliceIndexSnd * curSize_, curSize_, slaveStream));
    CHK_RET(recvLink->RxAsync(
        UserMemType::OUTPUT_MEM, sliceIndexRcv * curSize_,
        static_cast<s8*>(execMem.inputMem.ptr()) + sliceIndexSnd * curSize_, curSize_, slaveStream));
    CHK_RET(recvLink->PostFinAck(slaveStream));
    CHK_RET(sendLink->WaitFinAck(slaveStream));

    // 交换数据的两端之间Barrier，确认收发完成
    CHK_RET(recvLink->TxAck(slaveStream));
    CHK_RET(sendLink->RxAck(slaveStream));
    CHK_RET(sendLink->TxDataSignal(slaveStream));
    CHK_RET(recvLink->RxDataSignal(slaveStream));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::RunSuperPodAndInterServerPostProcess(
    const OpParam& param, const ExecMem& execMem, u32 step)
{
    // ccl in -> ccl out执行reduce
    u32 blockIndexPreStep = (level2Rank_ + level2RankSize_ - step) % level2RankSize_;
    u32 blockIndex = (level2Rank_ + level2RankSize_ - (step + 1)) % level2RankSize_;
    u32 sliceIndex = blockIndex * level1RankSize_;
    u64 dstOffset = sliceIndex * curSize_;
    u64 srcOffset = blockIndexPreStep * level1RankSize_ * curSize_;
    HCCL_DEBUG(
        "[RunSuperPodAndInterServerPostProcess] rank[%u:%u,%u,%u] step[%u] reduce blockIndex[%u] sliceIndex[%u]",
        topoAttr_.userRank, level2Rank_, level1Rank_, level0Rank_, step, blockIndex, sliceIndex);

    Stream stream = param.stream;
    CHK_RET(HcclReduceAsync(
        dispatcher_, static_cast<s8*>(execMem.inputMem.ptr()) + srcOffset, execMem.count, param.DataDes.dataType,
        param.reduceType, stream, static_cast<s8*>(execMem.outputMem.ptr()) + dstOffset, topoAttr_.userRank,
        LinkType::LINK_RESERVED, INLINE_REDUCE_BIT));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingZerocopyExchangePipelineExecutor::RunFinallyProcess(
    const OpParam& param, const ExecMem& execMem) const
{
    HCCL_DEBUG("[RunFinallyProcess] rank[%u:%u,%u,%u] ccl out -> user out");
    u32 sliceIndex = level2Rank_ * level1RankSize_;
    u64 offset = sliceIndex * curSize_;
    DeviceMem srcMem = execMem.outputMem.range(offset, curSize_);
    DeviceMem dstMem = DeviceMem::create(static_cast<u8*>(execMem.outputPtr), curSize_);
    Stream stream = param.stream;
    return HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream);
}

REGISTER_EXEC(
    "ReduceScatterRingZerocopyExchangePipelineExecutor", ReduceScatterRingZerocopyExchangePipeline,
    CollReduceScatterRingZerocopyExchangePipelineExecutor);
} // namespace hccl
