/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_ring_zerocopy_executor.h"
#include "alg_template_register.h"

namespace hccl {

constexpr u32 TWO_RING = 2;

CollAllReduceRingZerocopyExecutor::CollAllReduceRingZerocopyExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true; // 设为true，以禁用RunLoop中的本地拷贝
    desc_.isZeroCopy = true;
    desc_.deterministic = 1;
    desc_.level1SupportedAlgos
        = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB, AlgTypeLevel1::ALG_LEVEL1_RING,
           AlgTypeLevel1::ALG_LEVEL1_AHC, AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE};
    desc_.level2SupportedAlgos
        = {AlgTypeLevel2::ALG_LEVEL2_NHR, AlgTypeLevel2::ALG_LEVEL2_NB, AlgTypeLevel2::ALG_LEVEL2_RING,
           AlgTypeLevel2::ALG_LEVEL2_HD};
}

HcclResult CollAllReduceRingZerocopyExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum
        = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
                                                             LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllReduceRingZerocopyExecutor][CalcStreamNum] tag[%s] streamNum_[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingZerocopyExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult
CollAllReduceRingZerocopyExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;
    HCCL_INFO(
        "[CollAllReduceRingZerocopyExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingZerocopyExecutor::CalcLevel0CommInfo(
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

HcclResult CollAllReduceRingZerocopyExecutor::DoubleRingReduceScatter(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream,
    s32 profStage, const u64 baseOffset, const HcomCollOpInfo* opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    (void)tag;
    HCCL_INFO("[CollAllReduceRingZerocopyExecutor][DoubleRingReduceScatter] DoubleRingReduceScatter starts");

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 适配AlignedDoubleRing的入参
    u32 ringSize = multRingsSliceZero[0].size();
    std::vector<std::vector<u32>> rankOrders(TWO_RING, std::vector<u32>(ringSize));
    for (u32 i = 0; i < ringSize; i++) {
        rankOrders[0][i] = i;
        rankOrders[1][i] = (i == 0) ? 0 : (ringSize - i);
    }

    // 执行算法编排
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_DB_RING, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_DB_RING in COMM_LEVEL0", __func__);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(
        inputMem, inputMem, outputMem, count, dataType, stream, multRingsSliceZero, reductionOp, LEVEL0_BRIDGE_RANK_ID,
        baseOffset, false, reduceAttr, opInfo, topoAttr_.userRank, algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux, rankOrders, multRingsUserMemSlice));
    u32 ringIndexOp = COMM_INDEX_0;
    u32 rankSize = level0RingCommInfo.localRankSize;
    CHK_RET(tempAlg->RegisterProfiler(
        ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
            + level0RingCommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream));
    CHK_RET(RunTemplate(tempAlg, level0RingCommInfo));

    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingZerocopyExecutor::KernelRunIntraServerPre(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllReduceRingZerocopyExecutor][Run]The CollAllReduceRingZerocopyExecutor starts");
    bool isAHCAlgo = algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
                     || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
    CHK_RET(GetCommRankInfoNormal(
        level0Rank_, level0RankSize_, level1Rank_, level1RankSize_, level2Rank_, level2RankSize_, isAHCAlgo));

    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    std::vector<Slice> level0Datalices;
    CHK_RET(AlgTemplateBase::PrepareSliceData(param.DataDes.count, perDataSize, level0RankSize_, 0, level0Datalices));

    HcomCollOpInfo reduceScatterOpInfo
        = {"", execMem.inputMem.ptr(), nullptr, execMem.count, param.DataDes.dataType, param.root, param.reduceType, 0};

    if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
        // 构造slice数据
        level0MultiRingDataSlices_ = {level0Datalices};
        // 执行算法编排
        CHK_RET(MultiRingReduceScatter(
            param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.reduceType,
            level0MultiRingDataSlices_, param.stream, PROF_STAGE_0, 0, &reduceScatterOpInfo,
            level0MultiRingDataSlices_));
    } else {
        CHK_PRT_RET(
            topoType_ != TopoType::TOPO_TYPE_NP_DOUBLE_RING,
            HCCL_ERROR("[%s] unknown topoType: %u", __func__, topoType_), HCCL_E_NOT_SUPPORT);
        // 构造slice数据（适配AlignedDoubleRing算法）
        level0MultiRingDataSlices_.resize(TWO_RING);
        level0MultiRingDataSlices_[0].resize(level0RankSize_);
        level0MultiRingDataSlices_[1].resize(level0RankSize_);
        for (u32 i = 0; i < level0Datalices.size(); i++) {
            level0MultiRingDataSlices_[0][i].offset = level0Datalices[i].offset;
            level0MultiRingDataSlices_[0][i].size = level0Datalices[i].size / perDataSize / TWO_RING * perDataSize;
            u32 j = (i == 0) ? 0 : (level0RankSize_ - i);
            level0MultiRingDataSlices_[1][j].offset
                = level0MultiRingDataSlices_[0][i].offset + level0MultiRingDataSlices_[0][i].size;
            level0MultiRingDataSlices_[1][j].size = level0Datalices[i].size - level0MultiRingDataSlices_[0][i].size;
        }
        // 执行算法编排
        CHK_RET(DoubleRingReduceScatter(
            param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.reduceType,
            level0MultiRingDataSlices_, param.stream, PROF_STAGE_0, 0, &reduceScatterOpInfo,
            level0MultiRingDataSlices_));
    }

    HCCL_INFO("AllReduce double ring stage0 run success");
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingZerocopyExecutor::KernelRunInterServer(const OpParam& param, ExecMem& execMem)
{
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    Stream stream = param.stream;

    // copy data from user_in -> ccl_in
    std::vector<Slice> level0Datalices;
    CHK_RET(AlgTemplateBase::PrepareSliceData(param.DataDes.count, perDataSize, level0RankSize_, 0, level0Datalices));
    u64 level1DataSize = execMem.count * perDataSize;
    DeviceMem dstMem = execMem.inputMem.range(0, level1DataSize);
    DeviceMem srcMem
        = DeviceMem::create(static_cast<u8*>(execMem.inputPtr) + level0Datalices[level0Rank_].offset, level1DataSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
    bool isSelectAHC
        = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
           || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    if (topoAttr_.superPodNum <= 1 || isSelectAHC) {
        CHK_RET(KernelRunInterServerAllReduceSingleSuperpod(param, execMem, level1DataSize));
    } else {
        CHK_RET(KernelRunInterServerAllReduceMultiSuperpod(param, execMem, level1DataSize));
    }

    // copy results from ccl_out -> user_out
    srcMem = execMem.outputMem.range(0, level1DataSize);
    dstMem
        = DeviceMem::create(static_cast<u8*>(execMem.outputPtr) + level0Datalices[level0Rank_].offset, level1DataSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));

    HCCL_INFO("AllReduce double ring stage2 run success");
    return HCCL_SUCCESS;
}

// 单超节点场景，Level1直接执行AllReduce编排
HcclResult CollAllReduceRingZerocopyExecutor::KernelRunInterServerAllReduceSingleSuperpod(
    const OpParam& param, const ExecMem& execMem, const u64 level1DataSize)
{
    // 获取Level1通信域
    bool isSelectAHC
        = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
           || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    CommPlane commPlaneLevel1 = isSelectAHC ? COMM_LEVEL1_AHC : COMM_LEVEL1;
    CHK_RET(CheckCommSize(commPlaneLevel1, level0Rank_ + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(commPlaneLevel1, level0Rank_);

    // 获取Template
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    u64 level1DataCount = level1DataSize / perDataSize;
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_RING in COMM_LEVEL1", __func__);
        CHK_SMART_PTR_NULL(level1TempAlg);
    } else if (
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
        || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        // 获取通信域分组信息
        std::vector<std::vector<std::vector<u32>>> globalSubGroups;
        std::map<AHCConcOpType, TemplateType> ahcAlgOption;
        CHK_RET(topoMatcher_->GetGlobalSubGroups(commPlaneLevel1, globalSubGroups));
        topoMatcher_->GetAHCAlgOption(ahcAlgOption);
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_AHC, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_AHC in COMM_LEVEL1", __func__);
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_REDUCE_AHC_BROKE, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_AHC_BROKE in COMM_LEVEL1", __func__);
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(level1DataCount, globalSubGroups, ahcAlgOption));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NB, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NB in COMM_LEVEL1", __func__);
        CHK_SMART_PTR_NULL(level1TempAlg);
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        HCCL_DEBUG(
            "AllReduce ring: level1DataSize[%llu] deviceNumPerAggregation[%u] commLevel0Size[%u]", level1DataSize,
            topoAttr_.deviceNumPerAggregation, level0RankSize_);
        if (level1DataSize / topoAttr_.deviceNumPerAggregation <= NHR_ALLREDUCE_SMALL_SIZE) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_REDUCE_NHR_ONESHOT, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NHR_ONESHOT in COMM_LEVEL1", __func__);
        } else {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NHR in COMM_LEVEL1", __func__);
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
    } else {
        HCCL_ERROR("AllReduce ring: unsupported level1 algtype [%s]", AlgTypeToStr(algType_).c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    // 执行算法编排
    DeviceMem allreduceInput = execMem.inputMem.range(0, level1DataSize);
    DeviceMem allreduceOutput = execMem.outputMem.range(0, level1DataSize);
    u64 reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, param.DataDes.dataType, param.reduceType);
    CHK_RET(level1TempAlg->Prepare(reduceAttr));
    CHK_RET(level1TempAlg->Prepare(
        allreduceInput, allreduceOutput, allreduceOutput, level1DataCount, param.DataDes.dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), 0));
    CHK_RET(level1TempAlg->RegisterProfiler(
        (level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_1,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));

    return HCCL_SUCCESS;
}

// 单超节点场景，Level1先ReduceScatter，Level2再AllReduce，最后Level1再做AllGather
HcclResult CollAllReduceRingZerocopyExecutor::KernelRunInterServerAllReduceMultiSuperpod(
    const OpParam& param, const ExecMem& execMem, const u64 level1DataSize)
{
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    CHK_RET(CheckCommSize(COMM_LEVEL1, level0Rank_ + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0Rank_);

    // 根据数据量计算level1的数据切分
    u64 level1DataCount = level1DataSize / perDataSize;
    std::vector<Slice> level1DataSlices;
    CHK_RET(AlgTemplateBase::PrepareSliceData(level1DataCount, perDataSize, level1RankSize_, 0, level1DataSlices));
    u64 level2DataCount = level1DataSlices[level1Rank_].size / perDataSize;
    DeviceMem level1InputMem = execMem.inputMem.range(0, level1DataSize);
    DeviceMem level1OutputMem = execMem.outputMem.range(0, level1DataSize);
    DeviceMem level2InputMem
        = level1InputMem.range(level1DataSlices[level1Rank_].offset, level1DataSlices[level1Rank_].size);
    DeviceMem level2OutputMem
        = level1OutputMem.range(level1DataSlices[level1Rank_].offset, level1DataSlices[level1Rank_].size);

    // Step1：超节点内、节点间做ReduceScatter
    if (level1RankSize_ > 1) {
        // 获取Template
        u64 reduceAttr = GetReduceAttr(level1InputMem, level1OutputMem, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<AlgTemplateBase> level1RSTempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1RSTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            CHK_SMART_PTR_NULL(level1RSTempAlg);
            CHK_RET(level1RSTempAlg->Prepare(reduceAttr));
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1RSTempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            CHK_SMART_PTR_NULL(level1RSTempAlg);
            CHK_RET(level1RSTempAlg->Prepare(reduceAttr));
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1RSTempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
            CHK_SMART_PTR_NULL(level1RSTempAlg);
            CHK_RET(level1RSTempAlg->Prepare(reduceAttr, false));
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR in COMM_LEVEL1", __func__);
        } else {
            HCCL_ERROR("AllReduce ring: unsupported level1 algtype [%s]", AlgTypeToStr(algType_).c_str());
            return HCCL_E_NOT_SUPPORT;
        }

        // 执行算法编排
        CHK_RET(level1RSTempAlg->Prepare(
            level1InputMem, level1InputMem, level1OutputMem, level1DataCount, param.DataDes.dataType, param.stream,
            param.reduceType, LEVEL0_BRIDGE_RANK_ID, level1DataSlices, 0));
        CHK_RET(level1RSTempAlg->RegisterProfiler(
            (level1RankSize_ << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1Rank_, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
            param.stream));
        CHK_RET(RunTemplate(level1RSTempAlg, level1CommInfo));
        HCCL_INFO("AllReduce double ring [superpod] level1 ReduceScatter run success");
    }

    // Step2：超节点间做allreduce
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    // ==> 获取Template
    std::unique_ptr<AlgTemplateBase> level2ARTempAlg;
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        level2ARTempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NB, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NB in COMM_LEVEL2", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        level2ARTempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NHR in COMM_LEVEL2", __func__);
        if (algoAttr_.isSupportAtomicWrite) {
            CHK_SMART_PTR_NULL(level2ARTempAlg);
            level2ARTempAlg->CloseBarrier();
        }
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_RING) {
        level2ARTempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_RING in COMM_LEVEL2", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
        level2ARTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING in COMM_LEVEL2", __func__);
    } else {
        HCCL_ERROR("AllReduce ring: unsupported level2 algtype [%s]", AlgTypeToStr(algType_).c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_SMART_PTR_NULL(level2ARTempAlg);
    // ==> 执行算法编排
    u64 reduceAttr = GetReduceAttr(level2InputMem, level2OutputMem, param.DataDes.dataType, param.reduceType);
    CHK_RET(level2ARTempAlg->Prepare(reduceAttr));
    CHK_RET(level2ARTempAlg->Prepare(
        level2InputMem, level2OutputMem, level2OutputMem, level2DataCount, param.DataDes.dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), level1DataSlices[level1Rank_].offset));
    CHK_RET(level2ARTempAlg->RegisterProfiler(
        (level2RankSize_ << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2Rank_, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
        param.stream));
    CHK_RET(RunTemplate(level2ARTempAlg, level2CommInfo));
    HCCL_INFO("AllReduce double ring [superpod] level2 AllReduce run success");

    // Step3：超节点内、节点间做allgather
    if (level1RankSize_ > 1) {
        // 获取Template
        std::unique_ptr<AlgTemplateBase> level1AGTempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1AGTempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1AGTempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1AGTempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL1", __func__);
        } else {
            HCCL_ERROR("AllReduce ring: algType_[%u] is not supported", algType_.algoLevel1);
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_SMART_PTR_NULL(level1AGTempAlg);
        // 执行算法编排
        CHK_RET(level1AGTempAlg->Prepare(
            level1OutputMem, level1OutputMem, level1OutputMem, level1DataCount, param.DataDes.dataType, param.stream,
            HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, level1DataSlices, 0));
        CHK_RET(level1AGTempAlg->RegisterProfiler(
            (level1RankSize_ << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1Rank_, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
            param.stream));
        CHK_RET(RunTemplate(level1AGTempAlg, level1CommInfo));
        HCCL_INFO("AllReduce double ring [superpod] level1 AllGather run success");
    }

    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingZerocopyExecutor::DoubleRingAllGather(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream, s32 profStage, const u64 baseOffset,
    HcomCollOpInfo* opInfo, const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    (void)tag;
    HCCL_INFO("[CollAllReduceRingZerocopyExecutor][DoubleRingAllGather] DoubleRingAllGather starts");

    // 拿到ring环映射关系
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 适配AlignedDoubleRing的入参
    u32 ringSize = multRingsSliceZero[0].size();
    std::vector<std::vector<u32>> rankOrders(TWO_RING, std::vector<u32>(ringSize));
    for (u32 i = 0; i < ringSize; i++) {
        rankOrders[1][i] = (i == 0) ? 0 : (ringSize - i);
        rankOrders[0][i] = i;
    }

    // 执行算法编排
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALIGNED_ALL_GATHER_DOUBLE_RING in COMM_LEVEL0", __func__);
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALIGNED_ALL_GATHER_DOUBLE_RING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(
        opInfo, topoAttr_.userRank, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        rankOrders, multRingsUserMemSlice));
    CHK_RET(tempAlg->Prepare(
        outputMem, outputMem, inputMem, count, dataType, stream, multRingsSliceZero, HCCL_REDUCE_RESERVED,
        LEVEL0_BRIDGE_RANK_ID, baseOffset));
    CHK_RET(tempAlg->RegisterProfiler(
        ((COMM_INDEX_0 + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (level0RankSize_ << PROF_RANKSIZE_OFFSET_OF_PLANEID)
            + level0Rank_,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream));
    CHK_RET(RunTemplate(tempAlg, level0RingCommInfo));

    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingZerocopyExecutor::KernelRunIntraServerPost(const OpParam& param, ExecMem& execMem)
{
    HcomCollOpInfo allgatherOpInfo = {
        "", nullptr, execMem.outputMem.ptr(), execMem.count, param.DataDes.dataType, param.root, param.reduceType, 0};

    if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
        CHK_RET(MultiRingAllGather(
            param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
            level0MultiRingDataSlices_, param.stream, PROF_STAGE_2, 0, &allgatherOpInfo, level0MultiRingDataSlices_));
    } else {
        CHK_PRT_RET(
            topoType_ != TopoType::TOPO_TYPE_NP_DOUBLE_RING,
            HCCL_ERROR("[%s] unknown topoType: %u", __func__, topoType_), HCCL_E_NOT_SUPPORT);
        CHK_RET(DoubleRingAllGather(
            param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
            level0MultiRingDataSlices_, param.stream, PROF_STAGE_2, 0, &allgatherOpInfo, level0MultiRingDataSlices_));
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceRingZerocopyExecutor", AllReduceRingZerocopy, CollAllReduceRingZerocopyExecutor);

} // namespace hccl
