/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_ring_zerocopy_executor.h"
#include "alg_template_register.h"

namespace hccl {

constexpr u32 TWO_RING = 2;

CollBroadCastRingZerocopyExecutor::CollBroadCastRingZerocopyExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true; // 设为true，以禁用RunLoop中的本地拷贝
    desc_.isZeroCopy = true;
    desc_.level1SupportedAlgos = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB};
    desc_.level2SupportedAlgos
        = {AlgTypeLevel2::ALG_LEVEL2_NHR, AlgTypeLevel2::ALG_LEVEL2_NB, AlgTypeLevel2::ALG_LEVEL2_HD};
}

HcclResult CollBroadCastRingZerocopyExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    u32 ringFactor = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? (LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE) :
                                                                         (LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum = ringFactor * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    } else {
        totalStreamNum = ringFactor;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollBroadCastRingZerocopyExecutor][CalcStreamNum] tag[%s] streamNum_[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingZerocopyExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingZerocopyExecutor::CalcLevel0CommInfo(
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

HcclResult CollBroadCastRingZerocopyExecutor::DoubleRingScatter(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::vector<Slice>> multRingsSliceZero, u32 root, Stream stream, HcomCollOpInfo* opInfo,
    const u64 baseOffset)
{
    (void)tag;
    HCCL_INFO("[BroadCastOperator][CollBroadCastRingZerocopyExecutor] DoubleRingScatter starts");
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 适配AlignedDoubleRing的入参
    u32 ringSize = multRingsSliceZero[0].size();
    std::vector<std::vector<u32>> rankOrders(TWO_RING, std::vector<u32>(ringSize));
    for (u32 i = 0; i < ringSize; i++) {
        rankOrders[0][i] = i;
        rankOrders[1][i] = (i == 0) ? 0 : (ringSize - i);
    }
    u32 level0RankSubRing = (level0Rank_ == 0) ? 0 : (level0RankSize_ - level0Rank_);

    // 执行算法编排
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_SCATTER_DOUBLE_RING_DIRECT, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_SCATTER_DOUBLE_RING_DIRECT in COMM_LEVEL0", __func__);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(
        opInfo, topoAttr_.userRank, level0RankSubRing, algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux, rankOrders, multRingsSliceZero, multRingsSliceZero));
    u32 rootRank = 0;
    HcclResult ret = GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, root, rootRank);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadCastRingZerocopyExecutor][DoubleRingScatter]invalid root [%u] to get userrank", root),
        ret);
    CHK_RET(tempAlg->Prepare(
        inputMem, inputMem, outputMem, count, dataType, stream, HCCL_REDUCE_RESERVED, rootRank, std::vector<Slice>(0),
        baseOffset));
    CHK_RET(tempAlg->RegisterProfiler(
        (level0RankSize_ << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0Rank_, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
        stream));
    CHK_RET(RunTemplate(tempAlg, level0RingCommInfo));

    HCCL_INFO("[CollBroadCastRingZerocopyExecutor] double ring scatter run success");
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingZerocopyExecutor::KernelRunIntraServerPre(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG,
        "[BroadCastOperator][CollBroadCastRingZerocopyExecutor] The CollBroadCastRingZerocopyExecutor starts");
    CHK_RET(GetCommRankInfoNormal(
        level0Rank_, level0RankSize_, level1Rank_, level1RankSize_, level2Rank_, level2RankSize_));

    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    std::vector<Slice> level0Datalices;
    CHK_RET(AlgTemplateBase::PrepareSliceData(param.DataDes.count, perDataSize, level0RankSize_, 0, level0Datalices));

    HcomCollOpInfo scatterOpInfo
        = {"",         execMem.inputPtr, nullptr, level0Datalices[0].size / perDataSize, param.DataDes.dataType,
           param.root, param.reduceType, 0};

    if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
        // 构造slice数据
        level0MultiRingDataSlices_ = {level0Datalices};
        // 执行算法编排
        CHK_RET(MultiRingScatter(
            param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
            level0MultiRingDataSlices_, param.root, param.stream, nullptr));
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
        CHK_RET(DoubleRingScatter(
            param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
            level0MultiRingDataSlices_, param.root, param.stream, &scatterOpInfo));
    }

    HCCL_INFO("[CollBroadCastRingZerocopyExecutor][KernelRun] level0-scatter run success");

    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingZerocopyExecutor::KernelRunInterServer(const OpParam& param, ExecMem& execMem)
{
    HCCL_INFO("[BroadCastOperator][CollBroadCastRingZerocopyExecutor] KernelRunInterServer starts");

    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    Stream stream = param.stream;

    // copy data from user_in -> ccl_in
    std::vector<Slice> level0Datalices;
    CHK_RET(AlgTemplateBase::PrepareSliceData(param.DataDes.count, perDataSize, level0RankSize_, 0, level0Datalices));
    u64 level1DataSize = execMem.count * perDataSize;
    HCCL_DEBUG("[BroadCastOperator][CollBroadCastRingZerocopyExecutor]level1DataSize is %llu", level1DataSize);
    DeviceMem dstMem = execMem.inputMem.range(0, level1DataSize);
    DeviceMem srcMem
        = DeviceMem::create(static_cast<u8*>(execMem.inputPtr) + level0Datalices[level0Rank_].offset, level1DataSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));

    if (topoAttr_.superPodNum <= 1) {
        CHK_RET(KernelRunInterServerBroadcastSingleSuperpod(param, execMem, level1DataSize));
    } else {
        CHK_RET(KernelRunInterServerBroadcastMultiSuperpod(param, execMem, level1DataSize));
    }

    // copy results from ccl_out -> user_out
    srcMem = execMem.outputMem.range(0, level1DataSize);
    dstMem
        = DeviceMem::create(static_cast<u8*>(execMem.outputPtr) + level0Datalices[level0Rank_].offset, level1DataSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));

    return HCCL_SUCCESS;
}

// 单超节点场景，Level1直接执行Broadcast编排
HcclResult CollBroadCastRingZerocopyExecutor::KernelRunInterServerBroadcastSingleSuperpod(
    const OpParam& param, ExecMem& execMem, const u64 level1DataSize)
{
    HCCL_INFO("Broadcast double ring No level2");

    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    // 获取Level1通信域
    CHK_RET(CheckCommSize(COMM_LEVEL1, level0Rank_ + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0Rank_);

    // 获取Template
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        HCCL_DEBUG(
            "broadcast ring: level1DataSize[%llu] deviceNumPerAggregation[%u] commLevel0Size[%u]", level1DataSize,
            topoAttr_.deviceNumPerAggregation, level0RankSize_);
        if (level1DataSize / topoAttr_.deviceNumPerAggregation <= NHR_BCAST_SMALL_SIZE) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_NHR_ONESHOT, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NHR_ONESHOT in COMM_LEVEL1", __func__);
        } else {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NHR in COMM_LEVEL1", __func__);
        }
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        if (ShouldUseBinaryBroadcastOfNB(
                level1DataSize / topoAttr_.deviceNumPerAggregation, level1RankSize_, topoAttr_.userRankSize,
                topoAttr_.deviceNumPerAggregation)) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_NB_BINARY, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NB_BINARY in COMM_LEVEL1", __func__);
        } else {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NB in COMM_LEVEL1", __func__);
        }
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_BROADCAST_RECURSIVE_HD, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_RECURSIVE_HD in COMM_LEVEL1", __func__);
    } else {
        HCCL_ERROR("broadcast ring: unsupported level1 algtype [%s]", AlgTypeToStr(algType_).c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_SMART_PTR_NULL(level1TempAlg);

    // 获取level1层级的root
    u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
    CHK_PRT_RET(
        subUserrankRoot == INVALID_VALUE_RANKID,
        HCCL_ERROR(
            "[CollBroadCastRingZerocopyExecutor][KernelRun]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
            subUserrankRoot, topoAttr_.userRank, param.root),
        HCCL_E_INTERNAL);
    u32 planeRoot = 0;
    CHK_RET(GetRankByUserRank(COMM_LEVEL1, level0Rank_, subUserrankRoot, planeRoot));

    // 按需调用prepare函数，并执行算法编排
    u64 level1DataCount = level1DataSize / perDataSize;
    CHK_RET(level1TempAlg->Prepare(
        execMem.inputMem, execMem.inputMem, execMem.outputMem, level1DataCount, param.DataDes.dataType, param.stream,
        HCCL_REDUCE_RESERVED, planeRoot, std::vector<Slice>(0), 0));
    CHK_RET(level1TempAlg->RegisterProfiler(
        (level1RankSize_ << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1Rank_, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
        param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));

    HCCL_INFO("Broadcast double ring stage1 run success");
    return HCCL_SUCCESS;
}

// 单超节点场景，Level1先Scatter，Level2再Broadcast，最后Level1再做AllGather
HcclResult CollBroadCastRingZerocopyExecutor::KernelRunInterServerBroadcastMultiSuperpod(
    const OpParam& param, ExecMem& execMem, const u64 level1DataSize)
{
    HCCL_INFO("Broadcast double ring with Level2");

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

    // Step1：超节点内、节点间做Scatter
    if (level1RankSize_ > 1) {
        // 获取Template
        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_SCATTER_NHR in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_SCATTER_NB in COMM_LEVEL1", __func__);
        } else {
            HCCL_ERROR("broadcast ring: unsupported level1 algtype [%s]", AlgTypeToStr(algType_).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_SMART_PTR_NULL(level1TempAlg);

        // 获取level1层级的root
        u32 subPodRoot = topoMatcher_->GetSubRootWithSuperPod(topoAttr_.userRank, param.root);
        u32 subServerRootUsrRank = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, subPodRoot);
        u32 level1RootRank = INVALID_VALUE_RANKID;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, level0Rank_, subServerRootUsrRank, level1RootRank));
        CHK_PRT_RET(
            level1RootRank == INVALID_VALUE_RANKID,
            HCCL_ERROR("[CollBroadCastRingZerocopyExecutor][KernelRun] get rootRank IDX in level1 failed"),
            HCCL_E_PARA);

        // 执行算法编排
        CHK_RET(level1TempAlg->Prepare(
            level1InputMem, level1InputMem, level1InputMem, level1DataCount, param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, level1RootRank, level1DataSlices, 0));
        CHK_RET(level1TempAlg->RegisterProfiler(
            (level1RankSize_ << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1Rank_, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
            param.stream));
        CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));

        HCCL_INFO("Broadcast double ring [superpod] level1 run success");
    }

    // Step2：超节点间做broadcast
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    // ==> 获取Template
    std::unique_ptr<AlgTemplateBase> level2TempAlg;
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        level2TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NB, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NB in COMM_LEVEL2", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        level2TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NHR in COMM_LEVEL2", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
        level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_BROADCAST_RECURSIVE_HD, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_RECURSIVE_HD in COMM_LEVEL2", __func__);
    } else {
        HCCL_ERROR("broadcast ring: unsupported level2 algtype [%s]", AlgTypeToStr(algType_).c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_SMART_PTR_NULL(level2TempAlg);
    // ==> 获取level2层级的root
    u32 subUserrankRootSupperPod = topoMatcher_->GetSubRootUserRankWithSuperPod(topoAttr_.userRank, param.root);
    CHK_PRT_RET(
        subUserrankRootSupperPod == INVALID_VALUE_RANKID,
        HCCL_ERROR(
            "[CollBroadCastRingZerocopyExecutor][KernelRun]subUserrankRootSupperPod[%u] is invalid,userRank[%u],"
            "root[%u]",
            subUserrankRootSupperPod, topoAttr_.userRank, param.root),
        HCCL_E_INTERNAL);
    u32 planeRootSupperPod = 0;
    CHK_RET(GetRankByUserRank(COMM_LEVEL2, COMM_INDEX_0, subUserrankRootSupperPod, planeRootSupperPod));
    HCCL_DEBUG(
        "level2 get root info as: subUserrankRootSupperPod[%u], planeRootSupperPod[%u]", subUserrankRootSupperPod,
        planeRootSupperPod);
    // ==> 执行算法编排
    CHK_RET(level2TempAlg->Prepare(
        execMem.inputMem, execMem.inputMem, execMem.outputMem, level2DataCount, param.DataDes.dataType, param.stream,
        HcclReduceOp::HCCL_REDUCE_RESERVED, planeRootSupperPod, std::vector<Slice>(0),
        level1DataSlices[level1Rank_].offset));
    HCCL_DEBUG(
        "[superpod]Broadcast level2-broadcast : level1DataSlices[localRank].offset[%llu] "
        "level1DataSlices[localRank].size[%llu]",
        level1DataSlices[level1Rank_].offset, level1DataSlices[level1Rank_].size);
    CHK_RET(level2TempAlg->RegisterProfiler(
        (level2RankSize_ << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2Rank_, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
        param.stream));
    CHK_RET(RunTemplate(level2TempAlg, level2CommInfo));
    HCCL_INFO("[CollBroadCastRingZerocopyExecutor][superpod]Broadcast level2-broadcast run success");

    // Step3：超节点内、节点间做allgather
    if (level1RankSize_ > 1) {
        // 获取Template
        std::unique_ptr<AlgTemplateBase> level1AGTempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1AGTempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1AGTempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL1", __func__);
        } else {
            HCCL_ERROR("broadcast ring: unsupported level1 algtype [%s]", AlgTypeToStr(algType_).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_SMART_PTR_NULL(level1AGTempAlg);
        // 执行算法编排
        CHK_RET(level1AGTempAlg->Prepare(
            level1InputMem, level1OutputMem, level1OutputMem, level1DataCount, param.DataDes.dataType, param.stream,
            HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, level1DataSlices, 0));
        CHK_RET(level1AGTempAlg->RegisterProfiler(
            (level1RankSize_ << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1Rank_, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
            param.stream));
        CHK_RET(RunTemplate(level1AGTempAlg, level1CommInfo));
        HCCL_INFO("[CollBroadCastRingZerocopyExecutor]broadcast [superpod] level1 allgather run success");
    }
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingZerocopyExecutor::DoubleRingAllGather(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream, s32 profStage, const u64 baseOffset,
    HcomCollOpInfo* opInfo, const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    (void)tag;
    HCCL_INFO("[CollBroadCastRingZerocopyExecutor][DoubleRingAllGather] DoubleRingAllGather starts");

    // 拿到ring环映射关系
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 构造rankOrders
    u32 ringSize = multRingsSliceZero[0].size();
    std::vector<std::vector<u32>> rankOrders(TWO_RING, std::vector<u32>(ringSize));
    for (u32 i = 0; i < ringSize; i++) {
        rankOrders[0][i] = i;
        rankOrders[1][i] = (i == 0) ? 0 : (ringSize - i);
    }

    // 执行算法编排
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALIGNED_ALL_GATHER_DOUBLE_RING, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALIGNED_ALL_GATHER_DOUBLE_RING in COMM_LEVEL0", __func__);
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

HcclResult CollBroadCastRingZerocopyExecutor::KernelRunIntraServerPost(const OpParam& param, ExecMem& execMem)
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

REGISTER_EXEC("BroadCastRingZerocopyExecutor", BroadCastRingZerocopy, CollBroadCastRingZerocopyExecutor);

} // namespace hccl
