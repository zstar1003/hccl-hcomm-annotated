/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_ring_for_910_93_executor.h"

namespace hccl {
CollAllGatherRingFor91093Executor::CollAllGatherRingFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    desc_.level1SupportedAlgos
        = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB, AlgTypeLevel1::ALG_LEVEL1_RING,
           AlgTypeLevel1::ALG_LEVEL1_AHC, AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE};
    desc_.level2SupportedAlgos
        = {AlgTypeLevel2::ALG_LEVEL2_NHR, AlgTypeLevel2::ALG_LEVEL2_NB, AlgTypeLevel2::ALG_LEVEL2_RING};
}

HcclResult CollAllGatherRingFor91093Executor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum
        = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
                                                             LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }

    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllGatherRingFor91093Executor][CalcStreamNum] tag[%s] streamNum_[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
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
CollAllGatherRingFor91093Executor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollAllGatherRingFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcLevel2CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
        || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        HCCL_INFO("[CollAllGatherRingFor91093Executor][CalcLevel2CommInfo] select AHC bypass level2 comm calculate");
        return HCCL_SUCCESS;
    }

    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    HCCL_DEBUG("[CollAllGatherRingFor91093Executor][CalcLevel2CommInfo]Level2CommInfo start set");
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s]Calc NHRCommInfo.", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[%s]Calc NBCommInfo.", __func__);
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s]Calc RingCommInfo.", __func__);
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollAllGatherRingFor91093Executor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}

HcclResult CollAllGatherRingFor91093Executor::RunIntraSeverAllGather(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count, const HcclDataType& dataType,
    const std::vector<std::vector<Slice>>& multRingsSliceZero, const Stream& stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo* opInfo, const std::vector<std::vector<Slice>>& multRingsUserMemSlice)
{
    CHK_RET(MultiRingAllGather(
        tag, inputMem, outputMem, count, dataType, multRingsSliceZero, stream, profStage, baseOffset, opInfo,
        multRingsUserMemSlice, logicalLevel0plane_));
    return HCCL_SUCCESS;
}

u64 CollAllGatherRingFor91093Executor::CalcDstMemOffset(
    [[maybe_unused]] const OpParam& param, [[maybe_unused]] u32 perDataSize, u64 inputMemSize) const
{
    return topoAttr_.userRank * inputMemSize;
}

HcomCollOpInfo CollAllGatherRingFor91093Executor::GetHcomCollOpInfo(const OpParam& param, const ExecMem& execMem) const
{
    HcomCollOpInfo opInfo
        = {"", execMem.inputPtr,     execMem.outputPtr,        param.DataDes.count, param.DataDes.dataType,
           0,  HCCL_REDUCE_RESERVED, param.DataDes.strideCount};
    if (!DMAReduceFlag_ && (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING)) {
        opInfo.inputAddr = execMem.inputMem.ptr();
        opInfo.outputAddr = execMem.outputMem.ptr();
    }
    return opInfo;
}

HcclResult CollAllGatherRingFor91093Executor::PrepareSlicesL0(
    std::vector<std::vector<Slice>>& multRingsSlice, const OpParam& param, const SubCommInfo& level2CommInfo,
    const SubCommInfo& level1CommInfo, const SubCommInfo& level0CommInfo, [[maybe_unused]] u32 perDataSize,
    u64 inputMemSize)
{
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;

    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize, dataSegsSlice));

    // 多环数据切分
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    bool ARSFlag = topoMatcher_->GetARSFlag();
    bool ARSDoubleRing = (ARSFlag && (level0RankSize > FACTOR_TWO) && topoAttr_.isARSDoubleRing);

    if (ARSDoubleRing) {
        std::vector<u32> mockNicList;
        mockNicList.reserve(level0RankSize);
        for (u32 rankIndex = 0; rankIndex < level0RankSize; rankIndex++) {
            mockNicList.push_back(rankIndex);
        }
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, mockNicList);
    } else if (
        topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING
        && !IsSupportUnifiedMarch(param, topoType_, topoAttr_.serverNum, topoAttr_.superPodNum)) {
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice);
    }
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level2DataSlice;
        CHK_RET(CalculateLevel2AllgatherSlice(
            inputMemSize, level0RankSize, level1RankSize, level2RankSize, multRingsSliceZero, level2DataSlice,
            ringIndex));
        multRingsSlice.push_back(level2DataSlice);
    }

    return HCCL_SUCCESS;
}

std::vector<Slice> CollAllGatherRingFor91093Executor::PrepareSlicesL1(
    [[maybe_unused]] const OpParam& param, const SubCommInfo& level2CommInfo, const SubCommInfo& level1CommInfo,
    const SubCommInfo& level0CommInfo, [[maybe_unused]] u32 perDataSize, u64 inputMemSize) const
{
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level0ServerIndex = level0CommInfo.localRank;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    std::vector<Slice> level1DataSegsSlice;
    for (u32 j = 0; j < level1RankSize; j++) {
        for (u32 i = 0; i < level2RankSize; i++) {
            Slice level1Slice;
            level1Slice.size = inputMemSize;
            level1Slice.offset
                = inputMemSize * (i * level1RankSize * level0RankSize + j * level0RankSize + level0ServerIndex);

            HCCL_DEBUG(
                "[CollAllGatherRingFor91093Executor][PrepareSlicesL1] rank[%u], level1index[%u], level2index[%u], "
                "slices.offset=%llu, slices.size=%llu",
                level0CommInfo.localRank, j, i, level1Slice.offset, level1Slice.size);

            level1DataSegsSlice.push_back(level1Slice);
        }
    }
    return level1DataSegsSlice;
}

std::vector<Slice> CollAllGatherRingFor91093Executor::PrepareSlicesL2(
    [[maybe_unused]] const OpParam& param, const SubCommInfo& level2CommInfo, const SubCommInfo& level1CommInfo,
    const SubCommInfo& level0CommInfo, [[maybe_unused]] u32 perDataSize, u64 inputMemSize) const
{
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level0ServerIndex = level0CommInfo.localRank;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level1ServerIndex = level1CommInfo.localRank;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    std::vector<Slice> level2DataSegsSlice;
    for (u32 i = 0; i < level2RankSize; i++) {
        Slice sliceTemp;
        sliceTemp.size = inputMemSize;
        sliceTemp.offset
            = inputMemSize
              * (i * level1RankSize * level0RankSize + level1ServerIndex * level0RankSize + level0ServerIndex);
        level2DataSegsSlice.push_back(sliceTemp);
    }
    return level2DataSegsSlice;
}

HcclResult CollAllGatherRingFor91093Executor::PrepareUserMemSlices(
    std::vector<std::vector<Slice>>& userMemSlices, const std::vector<std::vector<Slice>>& multRingsSlice,
    const OpParam& param, [[maybe_unused]] const SubCommInfo& level2CommInfo,
    [[maybe_unused]] const SubCommInfo& level1CommInfo, [[maybe_unused]] const SubCommInfo& level0CommInfo,
    u32 perDataSize, u64 inputMemSize)
{
    CHK_PRT_RET(
        0 < param.DataDes.strideCount && param.DataDes.strideCount < param.DataDes.count,
        HCCL_ERROR(
            "[CollAllGatherRingFor91093Executor][KernelRun]strideCount[%llu] is smaller than opCount[%llu]",
            param.DataDes.strideCount, param.DataDes.count),
        HCCL_E_PARA);
    HCCL_DEBUG(
        "[CollAllGatherRingFor91093Executor][KernelRun]strideCount[%llu], opCount[%llu]", param.DataDes.strideCount,
        param.DataDes.count);

    if (!DMAReduceFlag_) {
        userMemSlices = multRingsSlice;
        // 图模式，根据strideCount更新slice的offset
        if (param.DataDes.strideCount != 0) {
            CHK_RET(UpdateOffsetBasedOnStrideCount(param, userMemSlices));
        }
    } else {
        for (u32 ringIndex = 0; ringIndex < multRingsSlice.size(); ringIndex++) {
            std::vector<Slice> userMemSlice;
            for (const auto& cclSlice : multRingsSlice[ringIndex]) {
                Slice tmpSlice;
                u64 count = (param.DataDes.strideCount == 0) ? param.DataDes.count : param.DataDes.strideCount;
                tmpSlice.size = cclSlice.size;
                tmpSlice.offset
                    = (cclSlice.offset / inputMemSize) * count * perDataSize + multRingsSlice[ringIndex][0].offset;
                userMemSlice.push_back(tmpSlice);
                HCCL_DEBUG(
                    "rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]", topoAttr_.userRank, ringIndex,
                    tmpSlice.offset, tmpSlice.size);
            }
            userMemSlices.push_back(userMemSlice);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::GetLevelCommInfo()
{
    logicalLevel0plane_ = COMM_LEVEL0;
    CHK_RET(CheckCommSize(logicalLevel0plane_, COMM_INDEX_0 + 1));
    logicalLevel0CommInfo_ = GetSubCommInfo(logicalLevel0plane_, COMM_INDEX_0);
    u32 commIndex = logicalLevel0CommInfo_.localRank;
    bool isSelectAHC
        = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
           || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    logicalLevel1plane_ = isSelectAHC ? COMM_LEVEL1_AHC : COMM_LEVEL1;
    CHK_RET(CheckCommSize(logicalLevel1plane_, commIndex + 1));
    logicalLevel1CommInfo_ = GetSubCommInfo(logicalLevel1plane_, commIndex);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[%s] The AllGatherRingExecutor starts, topoType_[%u], agv[%u]", __func__, topoType_, isAllGatherV_);
    CHK_RET(GetLevelCommInfo()); // 设置逻辑通信域
    CHK_RET(ActiveSlaveStreams(param.stream));
    const HcclDataType dataType = param.GetDataType();
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
    CHK_PRT_RET(
        perDataSize == 0,
        HCCL_ERROR(
            "[CollAllGatherRingFor91093Executor][KernelRun]errNo[0x%016llx] datatype[%s] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), GetDataTypeEnumStr(dataType).c_str()),
        HCCL_E_PARA);

    bool isSelectAHC
        = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
           || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);

    u32 level1RankSize = logicalLevel1CommInfo_.localRankSize;

    SubCommInfo level2CommInfo;
    if (isSelectAHC) {
        level2CommInfo = logicalLevel1CommInfo_;
        level2CommInfo.localRankSize = 1; // AHC bypass level2
    } else {
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    }
    const u32 level2RankSize = level2CommInfo.localRankSize;

    //  第一步，将数据从input内存拷贝到output内存的对应位置
    u64 inputMemSize = execMem.inputMem.size();
    u64 dstMemOffset = CalcDstMemOffset(param, perDataSize, inputMemSize);
    DeviceMem dstMem = execMem.outputMem.range(dstMemOffset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);

    HcomCollOpInfo opInfo = GetHcomCollOpInfo(param, execMem);
    HcomCollOpInfo* opInfoPtr
        = (DMAReduceFlag_ || (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING)) ? &opInfo : nullptr;

    // 图模式opinfo不为空，但需要将数据从ccl input拷贝到ccl output上
    HcclResult ret = HCCL_SUCCESS;
    if (!DMAReduceFlag_) {
        ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollAllGatherRingFor91093Executor][KernelRun]AllGather double "
                "ring memcpy Failed, Offset[%llu], Size[%llu]",
                dstMemOffset, inputMemSize),
            ret);
    } else {
        // 先做server间算法，带有消减拷贝场景数据需要从user input取，拷贝到ccl output上
        if (level1RankSize > 1 || level2RankSize > 1) {
            DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr), inputMemSize);
            ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream));
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollAllGatherRingFor91093Executor][KernelRun]AllGather double "
                    "ring user memcpy Failed, Offset[%llu], Size[%llu]",
                    dstMemOffset, inputMemSize),
                ret);
        }
    }
    if (level2RankSize > 1) {
        std::unique_ptr<AlgTemplateBase> level2AGExecutor;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level2AGExecutor
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL2", __func__);
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level2AGExecutor
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL2", __func__);
        } else {
            level2AGExecutor
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL2", __func__);
        }
        CHK_SMART_PTR_NULL(level2AGExecutor);

        std::vector<Slice> level2DataSegsSlice = PrepareSlicesL2(
            param, level2CommInfo, logicalLevel1CommInfo_, logicalLevel0CommInfo_, perDataSize, inputMemSize);
        CHK_RET(level2AGExecutor->Prepare(
            execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count, dataType, param.stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level2DataSegsSlice, 0));

        CHK_RET(level2AGExecutor->RegisterProfiler(
            (level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank, PROF_STAGE_0,
            HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level2AGExecutor, level2CommInfo));
        HCCL_INFO(
            "AllGather ring [superpod] level2 AllGather run success, topoType_[%u], agv[%u]", topoType_, isAllGatherV_);
    }
    if (level1RankSize > 1) {
        // 计算slice, 不同超节点相同slice
        std::vector<Slice> level1DataSegsSlice = PrepareSlicesL1(
            param, level2CommInfo, logicalLevel1CommInfo_, logicalLevel0CommInfo_, perDataSize, inputMemSize);

        std::unique_ptr<AlgTemplateBase> level1AGExecutor;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1AGExecutor
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1AGExecutor
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1AGExecutor
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL1", __func__);
        } else if (isSelectAHC) {
            // 获取通信域分组信息
            std::vector<std::vector<std::vector<u32>>> globalSubGroups;
            std::map<AHCConcOpType, TemplateType> ahcAlgOption;
            CHK_RET(topoMatcher_->GetGlobalSubGroups(logicalLevel1plane_, globalSubGroups));
            topoMatcher_->GetAHCAlgOption(ahcAlgOption);
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
                level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_AHC, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_AHC in COMM_LEVEL1", __func__);
            } else {
                level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_AHC_BROKE, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_AHC_BROKE in COMM_LEVEL1", __func__);
            }
            CHK_SMART_PTR_NULL(level1AGExecutor);
            CHK_RET(level1AGExecutor->Prepare(execMem.count, globalSubGroups, ahcAlgOption));
        } else {
            HCCL_ERROR("AllGather ring: unsupported algtype [%s].", AlgTypeToStr(algType_).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_SMART_PTR_NULL(level1AGExecutor);
        CHK_RET(level1AGExecutor->Prepare(
            execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count, dataType, param.stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, 0));

        CHK_RET(level1AGExecutor->RegisterProfiler(
            (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank, PROF_STAGE_1,
            HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level1AGExecutor, logicalLevel1CommInfo_));
        HCCL_INFO(
            "AllGather ring [superpod] level1 AllGather run success, topoType_[%u], agv[%u]", topoType_, isAllGatherV_);
    }
    // 节点内做AllGather ring
    std::vector<std::vector<Slice>> multRingsSlice;
    CHK_RET(PrepareSlicesL0(
        multRingsSlice, param, level2CommInfo, logicalLevel1CommInfo_, logicalLevel0CommInfo_, perDataSize,
        inputMemSize));

    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    CHK_RET(PrepareUserMemSlices(
        multRingsUserMemSlice, multRingsSlice, param, level2CommInfo, logicalLevel1CommInfo_, logicalLevel0CommInfo_,
        perDataSize, inputMemSize));

    if (DMAReduceFlag_ && (level1RankSize > 1 || level2RankSize > 1)) {
        // allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
        opInfo.inputAddr = nullptr;
    }
    CHK_RET(RunIntraSeverAllGather(
        param.tag, execMem.inputMem, execMem.outputMem, execMem.count, dataType, multRingsSlice, param.stream,
        PROF_STAGE_2, 0, opInfoPtr, multRingsUserMemSlice));
    HCCL_INFO("AllGather ring run success. topoType_[%u], agv[%u]", topoType_, isAllGatherV_);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
{
    HCCL_INFO("[CollAllGatherRingFor91093Executor][Getlevel1CommRank] Entry Getlevel1CommRank.");
    bool isSelectAHC
        = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
           || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    if (isSelectAHC) {
        SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
        u32 level0ServerIndex = level0CommInfo.localRank;

        CommPlane commPlaneLevel1 = COMM_LEVEL1;
        CHK_RET(CheckCommSize(commPlaneLevel1, level0ServerIndex + 1));
        level1CommInfo = GetSubCommInfo(commPlaneLevel1, level0ServerIndex);
        u32 level1RankSize = level1CommInfo.localRankSize;
        HCCL_INFO("Getlevel1CommRank. level1RankSize[%u]", level1RankSize);
        return HCCL_SUCCESS;
    }
    if (CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1) != HCCL_SUCCESS) {
        HCCL_INFO("[AllGatherRingFor91093] Getlevel1CommRank size not match.");
        return HCCL_E_UNAVAIL;
    }
    level1CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    return HCCL_SUCCESS;
}

HcclResult
CollAllGatherRingFor91093Executor::SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize)
{
    HCCL_INFO("[AllGatherRingFor91093] Entry SelectTempAlg, level1RankSize = [%u].", level1RankSize);
    bool isSelectAHC
        = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
           || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    if (isSelectAHC) {
        CommPlane commPlaneLevel1 = COMM_LEVEL1;
        // 获取通信域分组信息
        std::vector<std::vector<std::vector<u32>>> globalSubGroups;
        std::map<AHCConcOpType, TemplateType> ahcAlgOption;
        CHK_RET(topoMatcher_->GetGlobalSubGroups(commPlaneLevel1, globalSubGroups));
        topoMatcher_->GetAHCAlgOption(ahcAlgOption);
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_AHC, dispatcher_);
            HCCL_INFO("allgather comm: using ahc algo inter-server.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_AHC_BROKE, dispatcher_);
            HCCL_INFO("allgather comm: using ahc-broke algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(NSLBDP_MIN_COUNT, globalSubGroups, ahcAlgOption));
        return HCCL_SUCCESS;
    }
    if (level1RankSize > 1) {
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_INFO("AllGather ring: using nonuniform-bruck algo inter-superPod.");
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_INFO("AllGather ring: using nonuniform-hierarchical-ring algo inter-superPod.");
        } else {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_INFO("AllGather ring: using ring algo inter-superPod.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}

REGISTER_EXEC("AllGatherRingFor91093Executor", AllGatherRingFor91093, CollAllGatherRingFor91093Executor);

} // namespace hccl
