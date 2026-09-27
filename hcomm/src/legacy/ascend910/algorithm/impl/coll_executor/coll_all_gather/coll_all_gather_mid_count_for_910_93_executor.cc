/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_mid_count_for_910_93_executor.h"

namespace hccl {
CollAllGatherMidCountFor91093Executor::CollAllGatherMidCountFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    desc_.level1SupportedAlgos = {
        AlgTypeLevel1::ALG_LEVEL1_NHR,
    };
    desc_.level2SupportedAlgos = {
        AlgTypeLevel2::ALG_LEVEL2_NHR,
    };
}

HcclResult CollAllGatherMidCountFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMidCountFor91093Executor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        HCCL_ERROR("AllGatherMidCountFor91093Executor do not support offload mode");
        return HCCL_E_UNAVAIL;
    }
    HCCL_INFO(
        "[CollAllGatherMidCountFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMidCountFor91093Executor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaCombineL1(COMM_COMBINE_L1, CommType::COMM_TAG_WHOLE_NHR);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaCombineL1, opTransport[COMM_COMBINE_L1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMidCountFor91093Executor::CalcLevel2CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollAllGatherMidCountFor91093Executor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop
        = cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN_910_93 / unitSize * HCCL_MIN_SLICE_ALIGN_910_93;
    if (cclBuffSize != 0 && maxCountPerLoop == 0) {
        maxCountPerLoop = cclBuffSize / topoAttr_.userRankSize / unitSize;
        HCCL_INFO(
            "[CollAllGatherMidCountFor91093Executor][CalcLoopMaxCount]"
            "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize",
            maxCountPerLoop);
    }
    if (maxCountPerLoop == 0) {
        HCCL_ERROR("[CollAllGatherMidCountFor91093Executor][CalcLoopMaxCount] cclbuffer size is too small");
    }
    return maxCountPerLoop;
}

u64 CollAllGatherMidCountFor91093Executor::CalcDstMemOffset(const OpParam& param, u64 inputMemSize) const
{
    (void)param;
    return topoAttr_.userRank * inputMemSize;
}

HcclResult CollAllGatherMidCountFor91093Executor::PrepareL2DataSlices(
    const OpParam& param, const SubCommInfo& level1CommInfo, const SubCommInfo& level2CommInfo, u64 inputMemSize,
    std::vector<Slice>& dataSlices) const
{
    (void)param;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level1RankIndex = level1CommInfo.localRank;
    const u32 level2RankSize = level2CommInfo.localRankSize;

    std::vector<Slice> level2DataSlices;
    for (u32 i = 0; i < level2RankSize; i++) {
        Slice sliceTemp;
        sliceTemp.size = inputMemSize;
        sliceTemp.offset = inputMemSize * (i * level1RankSize + level1RankIndex);
        level2DataSlices.push_back(sliceTemp);
    }

    dataSlices = std::move(level2DataSlices);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMidCountFor91093Executor::RunLevel2ByNHR(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level1CommInfo, SubCommInfo& level2CommInfo) const
{
    const u32 level2RankSize = level2CommInfo.localRankSize;
    const u32 multiSuperPodMode = 1;
    if (level2RankSize > multiSuperPodMode) {
        u32 unitSize = 0;
        const HcclDataType dataType = param.GetDataType();
        CHK_RET(SalGetDataTypeSize(dataType, unitSize));
        u64 inputMemSize = execMem.count * unitSize;

        std::unique_ptr<AlgTemplateBase> level2AGExecutor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        CHK_SMART_PTR_NULL(level2AGExecutor);

        std::vector<Slice> dataSlices;
        CHK_RET(PrepareL2DataSlices(param, level1CommInfo, level2CommInfo, inputMemSize, dataSlices));
        CHK_RET(level2AGExecutor->Prepare(
            execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count, param.GetDataType(), param.stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, dataSlices, 0));

        CHK_RET(level2AGExecutor->RegisterProfiler(
            (level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank, PROF_STAGE_0,
            HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level2AGExecutor, level2CommInfo));
    }
    HCCL_INFO("MidCountAllGather run success in level2");
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMidCountFor91093Executor::PrepareL1DataSlices(
    const OpParam& param, const SubCommInfo& level1CommInfo, const SubCommInfo& level2CommInfo, u64 inputMemSize,
    u32 moduleId, std::vector<Slice>& dataSlices) const
{
    (void)level2CommInfo;
    u32 unitSize = 0;
    CHK_RET(SalGetDataTypeSize(param.GetDataType(), unitSize));
    const u32 level1RankSize = level1CommInfo.localRankSize;
    std::vector<Slice> level1DataSlices;
    for (u32 i = 0; i < level1RankSize; i++) {
        Slice sliceTemp;
        sliceTemp.size = inputMemSize;
        sliceTemp.offset = inputMemSize * (level1RankSize * moduleId + i);
        level1DataSlices.push_back(sliceTemp);
    }
    dataSlices = std::move(level1DataSlices);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMidCountFor91093Executor::RunLevel1ByNHR(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level1CommInfo, SubCommInfo& level2CommInfo) const
{
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    const u32 multiRankMode = 1;
    if (level1RankSize <= multiRankMode) {
        return HCCL_SUCCESS;
    }

    u32 unitSize = 0;
    const HcclDataType dataType = param.GetDataType();
    CHK_RET(SalGetDataTypeSize(dataType, unitSize));
    u64 inputMemSize = execMem.count * unitSize;

    for (u32 moduleId = 0; moduleId < level2RankSize; moduleId++) {
        std::vector<Slice> dataSlices;
        CHK_RET(PrepareL1DataSlices(param, level1CommInfo, level2CommInfo, inputMemSize, moduleId, dataSlices));

        // 计算slice, 不同超节点相同slice
        std::unique_ptr<AlgTemplateBase> level1AGExecutor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        CHK_SMART_PTR_NULL(level1AGExecutor);

        CHK_RET(level1AGExecutor->Prepare(
            execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, dataSlices, 0));

        CHK_RET(level1AGExecutor->RegisterProfiler(
            (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank, PROF_STAGE_1,
            HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level1AGExecutor, level1CommInfo));
    }
    HCCL_INFO("MidCountAllGather run success in level1");
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMidCountFor91093Executor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] The MidCountFor91093Executor starts", __func__);

    SubCommInfo level1CommInfo;
    SubCommInfo level2CommInfo;
    CHK_RET(CheckCommSize(COMM_COMBINE_L1, COMM_INDEX_0 + 1));
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    level1CommInfo = GetSubCommInfo(COMM_COMBINE_L1, COMM_INDEX_0);
    level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    u32 unitSize = 0;
    const HcclDataType dataType = param.GetDataType();
    CHK_RET(SalGetDataTypeSize(dataType, unitSize));

    u64 inputMemSize = execMem.count * unitSize;
    const u32 SINGLE_RANK_FLAG = 1;

    if (topoAttr_.userRankSize == SINGLE_RANK_FLAG) {
        DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr), inputMemSize);
        DeviceMem dstMem = DeviceMem::create(static_cast<u8*>(execMem.outputPtr), inputMemSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        return HCCL_SUCCESS;
    }

    // 先做server间算法，带有消减拷贝场景数据需要从user input取，拷贝到ccl output上
    if (DMAReduceFlag_) {
        u64 dstMemOffset = CalcDstMemOffset(param, inputMemSize);
        DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr), inputMemSize);
        DeviceMem dstMem = execMem.outputMem.range(dstMemOffset, inputMemSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    CHK_RET(RunLevel2ByNHR(param, execMem, level1CommInfo, level2CommInfo));
    CHK_RET(RunLevel1ByNHR(param, execMem, level1CommInfo, level2CommInfo));

    if (DMAReduceFlag_) {
        for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
            DeviceMem srcMem
                = DeviceMem::create(static_cast<u8*>(execMem.outputMem.ptr()) + inputMemSize * i, inputMemSize);
            DeviceMem dstMem = DeviceMem::create(
                static_cast<u8*>(execMem.outputPtr) + param.DataDes.count * unitSize * i, inputMemSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        }
    }
    HCCL_INFO("MidCountAllGather run success.");
    return HCCL_SUCCESS;
}
REGISTER_EXEC("AllGatherMidCountFor91093Executor", AllGatherMidCountFor91093, CollAllGatherMidCountFor91093Executor);
} // namespace hccl
