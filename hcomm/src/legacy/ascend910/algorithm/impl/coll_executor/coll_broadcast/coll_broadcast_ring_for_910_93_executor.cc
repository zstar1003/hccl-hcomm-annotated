/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_ring_for_910_93_executor.h"
#include "alg_template_register.h"

namespace hccl {

CollBroadCastRingFor91093::CollBroadCastRingFor91093(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DMAReduceFlag_ = true;
    } else {
        DMAReduceFlag_ = false;
    }
    desc_.level2SupportedAlgos
        = {AlgTypeLevel2::ALG_LEVEL2_NHR, AlgTypeLevel2::ALG_LEVEL2_NB, AlgTypeLevel2::ALG_LEVEL2_HD};
    desc_.level1SupportedAlgos = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB};
}

HcclResult CollBroadCastRingFor91093::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    u32 ringFactor = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? (LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE) :
                                                                         (LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
    HCCL_DEBUG("[CollBroadCastRingFor91093][CalcStreamNum]ringFactor is [%u]", ringFactor);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum = ringFactor * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    } else {
        totalStreamNum = ringFactor;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollBroadCastRingFor91093][CalcStreamNum] tag[%s] streamNum_[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingFor91093::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingFor91093::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingFor91093::CalcLevel2CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_DEBUG("[CollBroadCastRingFor91093][CalcLevel2CommInfo]cal for level2CommInfo");
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX, root_);
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s]Calc NHRCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[%s]Calc NBCommInfo", __func__);
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_HALVING_DOUBLING;
        HCCL_INFO("[%s]Calc HDCommInfo", __func__);
    }

    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingFor91093::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[BroadCastOperator][CollBroadCastRingFor91093] The CollBroadCastRingFor91093 starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(
        perDataSize == 0,
        HCCL_ERROR(
            "[CollBroadCastRingFor91093][KernelRun]errNo[0x%016llx] datatype[%d] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), param.DataDes.dataType),
        HCCL_E_PARA);
    std::vector<Slice> dataSegsSlice;             // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> mulRingSlice; // 数据基于该rank上环0的偏移

    u32 ringNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? (LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE) :
                                                                      (LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 按ranksize得到内存切分slice数
    u32 sliceNum = level0CommInfo.localRankSize;
    // 将根节点数据切分成sliceNum份
    CHK_RET(AlgTemplateBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));
    HCCL_DEBUG("[CollBroadCastRingFor91093][KernelRun]ringNum[%u] sliceNum[%u]", ringNum, sliceNum);

    /* 节点内 scatter */
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        // 将每slice再切分成2份，按各ring的dev顺序排列
        mulRingSlice = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
        CHK_PRT_RET(
            mulRingSlice.size() != ringNum,
            HCCL_ERROR(
                "[CollBroadCastRingFor91093][KernelRun] ringNum[%u] !=mulRingSlice size[%zu]", ringNum,
                mulRingSlice.size()),
            HCCL_E_INTERNAL);
    } else {
        mulRingSlice.push_back(dataSegsSlice); // 应该offset全为0，而大小和dataSegsSlice中一样,里面的offset不使用
    }

    HcomCollOpInfo* scatterOpInfoPtr = nullptr;
    HcomCollOpInfo scatterOpInfo
        = {"", execMem.inputPtr, nullptr, param.DataDes.count, param.DataDes.dataType, param.root, param.reduceType, 0};

    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        scatterOpInfoPtr = &scatterOpInfo;
        CHK_RET(ActiveSlaveStreams(param.stream));
        CHK_RET(DoubleRingScatter(
            param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, mulRingSlice,
            param.root, param.stream, scatterOpInfoPtr));
    } else {
        if (DMAReduceFlag_) {
            scatterOpInfoPtr = &scatterOpInfo;
        }
        CHK_RET(MultiRingScatter(
            param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, mulRingSlice,
            param.root, param.stream, scatterOpInfoPtr));
    }
    HCCL_INFO("[CollBroadCastRingFor91093][KernelRun] level0-scatter run success");

    u64 level1DataSize = 0;
    u32 commIndex = 0;
    u32 segmentIdx = 0;
    CHK_RET(PrepareLevel1CommInfo(segmentIdx, commIndex, level1DataSize, level0CommInfo, mulRingSlice, param.tag));
    u64 level1DataCount = level1DataSize / perDataSize;
    HCCL_DEBUG(
        "[CollBroadCastRingFor91093][KernelRun]usrRank[%u] level1 use level1DataCount[%llu]", topoAttr_.userRank,
        level1DataCount);

    // level 1 通信域获取
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    HCCL_DEBUG(
        "[CollBroadCastRingFor91093][KernelRun]commIdx:%u TagCommInfo[%s].commLevel1.size():%u", commIndex,
        param.tag.c_str(), level1CommInfo.localRankSize);
    if (topoAttr_.superPodNum <= 1) {
        HCCL_INFO("Broadcast double ring No level2.");
        /* step2: server间 broadcast */
        bool isUsedRegister = false;
        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            HCCL_DEBUG(
                "broadcast ring: curSize[%llu] deviceNumPerAggregation[%u] commLevel0Size[%u]", curSize,
                topoAttr_.deviceNumPerAggregation, level0CommInfo.localRankSize);
            if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_BCAST_SMALL_SIZE) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_BROADCAST_NHR_ONESHOT, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NHR_ONESHOT in COMM_LEVEL1", __func__);
            } else {
                level1TempAlg
                    = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NHR in COMM_LEVEL1", __func__);
            }
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            isUsedRegister = true;
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR_V1, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NHR_V1 in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            const u32 level1RankSize = level1CommInfo.localRankSize;
            if (ShouldUseBinaryBroadcastOfNB(
                    curSize / topoAttr_.deviceNumPerAggregation, level1RankSize, topoAttr_.userRankSize,
                    topoAttr_.deviceNumPerAggregation)) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_BROADCAST_NB_BINARY, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NB_BINARY in COMM_LEVEL1", __func__);
            } else {
                level1TempAlg
                    = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NB, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NB in COMM_LEVEL1", __func__);
            }
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_RECURSIVE_HD, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_RECURSIVE_HD in COMM_LEVEL1", __func__);
        }
        CHK_SMART_PTR_NULL(level1TempAlg);

        u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
        CHK_PRT_RET(
            subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR(
                "[CollBroadCastRingFor91093][KernelRun]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
                subUserrankRoot, topoAttr_.userRank, param.root),
            HCCL_E_INTERNAL);
        u32 planeRoot = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));
        u32 ranksize = level1CommInfo.localRankSize;
        // 节点间的hd 使用环0来记录
        if (isUsedRegister) {
            PrepareData prepareData;
            prepareData.inputMem = execMem.inputMem;
            prepareData.outputMem = execMem.inputMem;
            prepareData.scratchMem = execMem.outputMem;
            prepareData.count = level1DataCount;
            prepareData.dataType = param.DataDes.dataType;
            prepareData.stream = param.stream;
            prepareData.reductionOp = HCCL_REDUCE_RESERVED;
            prepareData.root = planeRoot;
            prepareData.baseOffset = dataSegsSlice[segmentIdx].offset;
            CHK_RET(level1TempAlg->Prepare(prepareData));
        } else {
            CHK_RET(level1TempAlg->Prepare(
                execMem.inputMem, execMem.inputMem, execMem.outputMem, level1DataCount, param.DataDes.dataType,
                param.stream, HCCL_REDUCE_RESERVED, planeRoot, std::vector<Slice>(0),
                dataSegsSlice[segmentIdx].offset));
        }

        CHK_RET(level1TempAlg->RegisterProfiler(
            (ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_1,
            HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));

        HCCL_INFO("Broadcast double ring stage1 run success");
    } else {
        HCCL_INFO("Broadcast double ring with Level2.");
        /* step2: 节点间 scatter */
        // 按level1RankSize得到内存切分slice数
        u32 level1RankSize = level1CommInfo.localRankSize;
        u64 level1Offset = dataSegsSlice[segmentIdx].offset;
        CHK_RET(AlgTemplateBase::PrepareSliceData(level1DataCount, perDataSize, level1RankSize, 0, dataSegsSlice));

        DeviceMem level1InputMem = execMem.inputMem.range(level1Offset, level1DataSize);
        CHK_SMART_PTR_NULL(level1InputMem);
        DeviceMem level1OutputMem = execMem.outputMem.range(level1Offset, level1DataSize);
        CHK_SMART_PTR_NULL(level1OutputMem);

        if (level1RankSize > 1) {
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
                HCCL_ERROR("broadcast level1 only supports NB/NHR algo. not support algType_[%u]", algType_.algoLevel1);
                return HCCL_E_NOT_SUPPORT;
            }
            CHK_SMART_PTR_NULL(level1TempAlg);

            /* 获取每个超节点内的subroot */
            u32 subPodRoot = topoMatcher_->GetSubRootWithSuperPod(topoAttr_.userRank, param.root);
            /* 获取超节点内的节点在每个level1通信域的对应卡subServerRootUsrRank */
            u32 subServerRootUsrRank = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, subPodRoot);
            u32 level1RootRank = INVALID_VALUE_RANKID;
            /* 用此卡subServerRootUsrRank 获取每个level1通信域的相对root idx */
            CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subServerRootUsrRank, level1RootRank));
            CHK_PRT_RET(
                level1RootRank == INVALID_VALUE_RANKID,
                HCCL_ERROR("[CollBroadCastRingFor91093][KernelRun] get rootRank IDX in level1 failed."), HCCL_E_PARA);

            CHK_RET(level1TempAlg->Prepare(
                level1InputMem, level1InputMem, level1InputMem, level1DataCount, param.DataDes.dataType, param.stream,
                HCCL_REDUCE_RESERVED, level1RootRank, dataSegsSlice, level1Offset));
            CHK_RET(level1TempAlg->RegisterProfiler(
                (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_1,
                HCCL_EXEC_STEP_NOT_SET, param.stream));

            CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
            HCCL_INFO("Broadcast double ring [superpod] level1 run success");
        }

        /* step3: 超节点间 broadcast */
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        u32 level2RankSize = level2CommInfo.localRankSize;
        u32 localRank = level1CommInfo.localRank;
        u32 subUserrankRootSupperPod = topoMatcher_->GetSubRootUserRankWithSuperPod(topoAttr_.userRank, param.root);
        CHK_PRT_RET(
            subUserrankRootSupperPod == INVALID_VALUE_RANKID,
            HCCL_ERROR(
                "[CollBroadCastRingFor91093][KernelRun]subUserrankRootSupperPod[%u] is invalid,userRank[%u],"
                "root[%u]",
                subUserrankRootSupperPod, topoAttr_.userRank, param.root),
            HCCL_E_INTERNAL);
        u32 planeRootSupperPod = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL2, COMM_INDEX_0, subUserrankRootSupperPod, planeRootSupperPod));
        HCCL_DEBUG(
            "level2 get root info as: subUserrankRootSupperPod[%u], planeRootSupperPod[%u]", subUserrankRootSupperPod,
            planeRootSupperPod);

        std::unique_ptr<AlgTemplateBase> level2TempAlg;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level2TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NB in COMM_LEVEL2", __func__);
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level2TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NHR in COMM_LEVEL2", __func__);
        } else {
            level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_RECURSIVE_HD, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_RECURSIVE_HD in COMM_LEVEL2", __func__);
        }
        CHK_SMART_PTR_NULL(level2TempAlg);
        u64 bcastCount = dataSegsSlice[localRank].size / perDataSize;

        CHK_RET(level2TempAlg->Prepare(
            execMem.inputMem, execMem.inputMem, execMem.outputMem, bcastCount, param.DataDes.dataType, param.stream,
            HcclReduceOp::HCCL_REDUCE_RESERVED, planeRootSupperPod, std::vector<Slice>(0),
            dataSegsSlice[localRank].offset + level1Offset));
        HCCL_DEBUG(
            "[superpod]Broadcast level2-broadcast : dataSegsSlice[localRank].offset[%llu] "
            "dataSegsSlice[localRank].size[%llu] level1Offset[%llu]",
            dataSegsSlice[localRank].offset, dataSegsSlice[localRank].size, level1Offset);

        CHK_RET(level2TempAlg->RegisterProfiler(
            (level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank, PROF_STAGE_1,
            HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2TempAlg, level2CommInfo));
        HCCL_INFO("[CollBroadCastRingFor91093][superpod]Broadcast level2-broadcast run success");

        /* step4: 节点间 allgather */
        if (level1RankSize > 1) {
            std::unique_ptr<AlgTemplateBase> level1AGTempAlg;
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
                level1AGTempAlg
                    = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL1", __func__);
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
                level1AGTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL1", __func__);
            } else {
                HCCL_ERROR("AllGather ring: algType_[%u] is not supported.", algType_.algoLevel1);
                return HCCL_E_NOT_SUPPORT;
            }

            CHK_SMART_PTR_NULL(level1AGTempAlg);
            CHK_RET(level1AGTempAlg->Prepare(
                level1InputMem, level1OutputMem, level1OutputMem, bcastCount, param.DataDes.dataType, param.stream,
                HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, level1Offset));
            CHK_RET(level1AGTempAlg->RegisterProfiler(
                (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_1,
                HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(level1AGTempAlg, level1CommInfo));
            HCCL_INFO("[CollBroadCastRingFor91093]broadcast [superpod] level1 allgather run success");
        }
    }

    /* step 3 or 5: 节点内 allgatherring */
    HcomCollOpInfo allgatherOpInfo = {
        "", nullptr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, param.root, param.reduceType, 0};
    HcomCollOpInfo* allgatherOpInfoPtr = (DMAReduceFlag_) ? (&allgatherOpInfo) : (nullptr);

    CHK_RET(MultiRingAllGather(
        param.tag, execMem.inputMem, execMem.outputMem, level1DataCount, param.DataDes.dataType, mulRingSlice,
        param.stream, PROF_STAGE_2, 0, allgatherOpInfoPtr));
    HCCL_INFO("Broadcast double ring stage2 run success");

    return HCCL_SUCCESS;
}
HcclResult CollBroadCastRingFor91093::DoubleRingScatter(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::vector<Slice>> multRingsSliceZero, u32 root, Stream stream, const HcomCollOpInfo* opInfo,
    const u64 baseOffset)
{
    HCCL_INFO("[BroadCastOperator][CollBroadCastRingFor91093] DoubleRingScatter starts.");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    // 拿到ring环映射关系
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    SubCommInfo level0CommInfo1 = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_1);

    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder
        = GetRingsOrderByTopoType(level0CommInfo.localRankSize, topoType_, nicList);

    std::vector<std::vector<u32>> doubleRingsOrders;
    std::vector<std::vector<Slice>> doubleRingUserMemInputSlices;
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(
            singleRingSliceZero.empty(),
            HCCL_ERROR("[CollBroadCastRingFor91093][DoubleRingScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        CHK_RET(
            CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemInputSlices));
        doubleRingUserMemInputSlices.push_back(userMemInputSlices);
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));
        doubleRingsOrders.push_back(rankOrder);
    }
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_SCATTER_DOUBLE_RING_DIRECT, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_SCATTER_DOUBLE_RING_DIRECT in COMM_LEVEL0", __func__);

    CHK_RET(tempAlg->Prepare(
        const_cast<HcomCollOpInfo*>(opInfo), topoAttr_.userRank, level0CommInfo1.localRank, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, doubleRingsOrders, multRingsSliceZero,
        doubleRingUserMemInputSlices));

    u32 rootRank = 0;
    ret = GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, root, rootRank);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadCastRingFor91093][DoubleRingScatter]invalid root [%u] to get userrank", root), ret);
    ret = tempAlg->Prepare(
        inputMem, inputMem, outputMem, count, dataType, stream, HCCL_REDUCE_RESERVED, rootRank, std::vector<Slice>(0),
        baseOffset);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadCastRingFor91093][DoubleRingScatter]scatter(ring) prepare failed, return[%d]", ret), ret);

    u32 rankSize = level0CommInfo.localRankSize;
    ret = tempAlg->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
        stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollBroadCastRingFor91093][DoubleRingScatter]scatter(ring) register profiler failed,return[%d]", ret),
        ret);

    ret = RunTemplate(tempAlg, level0CommInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadCastRingFor91093][DoubleRingScatter]scatter(ring) run failed, return[%d]", ret), ret);

    HCCL_INFO("[CollBroadCastRingFor91093] double ring scatter run success");
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingFor91093::Getlevel1CommRank(SubCommInfo& level1CommInfo)
{
    if (CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1) != HCCL_SUCCESS) {
        return HCCL_E_UNAVAIL;
    }
    level1CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    return HCCL_SUCCESS;
}

HcclResult CollBroadCastRingFor91093::SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NB, dispatcher_);
            HCCL_INFO("[superpod]Broadcast level2-broadcast: using nonuniform-bruck algo inter-superPod.");
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR, dispatcher_);
            HCCL_INFO("[superpod]Broadcast level2-broadcast: using nonuniform-hierarchical-ring algo inter-superPod.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_RECURSIVE_HD, dispatcher_);
            HCCL_INFO("[superpod]Broadcast level2-broadcast: using Recursive halving-doubling algo inter-superPod.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}

REGISTER_EXEC("BroadCastRingFor91093Executor", BroadCastRingFor91093, CollBroadCastRingFor91093);

} // namespace hccl
