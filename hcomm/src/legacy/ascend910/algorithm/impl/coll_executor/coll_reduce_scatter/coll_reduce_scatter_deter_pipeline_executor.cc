/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_deter_pipeline_executor.h"

namespace hccl {

CollReduceScatterDeterPipelineExecutor::CollReduceScatterDeterPipelineExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    scratchMemFlag_ = (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    DMAReduceFlag_ = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
}

void CollReduceScatterDeterPipelineExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    curOffset_ = 0;
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
}

HcclResult CollReduceScatterDeterPipelineExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    scratchMemSize = scratchMemFlag_ ? totalSize_ + topoAttr_.userRankSize * HCCL_MIN_SLICE_ALIGN_910B : 0U;
    HCCL_INFO(
        "[CollReduceScatterDeterPipelineExecutor][CalcScratchMemSize]tag[%s] scratchMemSize[%llu]", tag_.c_str(),
        scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterPipelineExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = topoAttr_.deviceNumPerAggregation + 3U; // (deviceNum - 1)机内 + 4Reduce + 1机间 - 1主流
    HCCL_INFO("[CollReduceScatterDeterPipelineExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterPipelineExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterPipelineExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaInfo.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterPipelineExecutor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL1, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterPipelineExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::SCRATCH;
    }
    HCCL_INFO(
        "[CollReduceScatterDeterPipelineExecutor][CalcTransportMemType] tag[%s] inputType[%d], "
        "outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

u64 CollReduceScatterDeterPipelineExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，这里每一块的地址不要求128byte对齐
    u64 maxCountPerLoop = ((inCCLbufferSize_ / topoAttr_.userRankSize) - HCCL_MIN_SLICE_ALIGN_910B) / unitSize;
    HCCL_INFO("[CollReduceScatterDeterPipelineExecutor][CalcLoopMaxCount] maxCountPerLoop[%llu]", maxCountPerLoop);
    return maxCountPerLoop;
}

HcclResult CollReduceScatterDeterPipelineExecutor::RunLoop(OpParam& param, AlgResourceResponse& algRes)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollReduceScatterDeterPipelineExecutor][RunLoop] tag[%s], userRank[%u] begins.", tag_.c_str(),
        topoAttr_.userRank);

    CHK_PRT_RET(
        (param.reduceType == HCCL_REDUCE_PROD) || (param.DataDes.dataType == HCCL_DATA_TYPE_INT64),
        HCCL_ERROR(
            "[CollReduceScatterDeterPipelineExecutor] unsupported reduceType[%u] or unsupported dataType[%u]",
            param.reduceType, param.DataDes.dataType),
        HCCL_E_INTERNAL);

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    u8* curInputPtr = static_cast<u8*>(param.inputPtr);
    u8* curOutputPtr = static_cast<u8*>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(unitSize);
    HCCL_INFO(
        "[CollReduceScatterDeterPipelineExecutor][RunLoop]tag[%s], userRankSize is [%u], maxCountPerLoop "
        "is [%llu].",
        tag_.c_str(), topoAttr_.userRankSize, maxCountPerLoop);

    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();

    for (u64 countLeft = param.DataDes.count, curCount = 0, curSize = 0; countLeft > 0; countLeft -= curCount) {
        curInputPtr += curSize;
        curOutputPtr += curSize;

        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        curSize = curCount * unitSize;

        HCCL_CONFIG_DEBUG(
            HCCL_ALG,
            "[CollReduceScatterDeterPipelineExecutor][RunLoop]tag[%s], curOffset[%llu],"
            "curInputPtr[%p], curOutputPtr[%p], curCount[%llu], dataType[%d].",
            tag_.c_str(), curOffset_, curInputPtr, curOutputPtr, curCount, param.DataDes.dataType);

        constexpr s64 HCCL_MEDIUM_COUNT_2_MB = 2 * 1024 * 1024;
        bool smallData = curSize < HCCL_MEDIUM_COUNT_2_MB ? 1 : 0;
        bool hugeData = IsHugeData(curSize);
        auto meta = HcclOpMetaInfo::GetOneForReduceScatter(
            autoSelectedAlgTypeLevel1, param.DataDes.dataType, ReduceType::INLINE_REDUCE, hugeData, smallData,
            CopyPattern::ZCOPY, false, deterministic, false);
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), meta.isEnableCache, meta.GetCacheKey(), true));
        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.cclOutputMem;
        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;

        CHK_RET(KernelRun(param, execMem));

        CHK_RET(LaunchTaskExtend(
            dispatcher_, const_cast<Stream&>(param.stream),
            const_cast<std::vector<Stream>&>(algResResp_->slaveStreams)));

        curOffset_ += curSize;
    }
    HCCL_INFO(
        "[CollReduceScatterDeterPipelineExecutor][RunLoop] tag[%s], userRank[%u] run loop success.", tag_.c_str(),
        topoAttr_.userRank);

    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterPipelineExecutor::PrepareDataSlice(
    const OpParam& param, const ExecMem& execMem, const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo,
    std::vector<Slice>& bufferSlices) const
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    bufferSlices.resize(topoAttr_.userRankSize);
    u64 totalOutputSize = param.DataDes.count * unitSize;
    u32 rankIdLevel0 = level0CommInfo.localRank;
    u32 rankSizeLevel0 = level0CommInfo.localRankSize;
    u32 rankIdLevel1 = level1CommInfo.localRank;
    u32 rankSizeLevel1 = level1CommInfo.localRankSize;

    for (u32 i = 0; i < rankSizeLevel1; i++) {
        u32 inputBlockIndex = (rankIdLevel1 + i) % rankSizeLevel1;
        u32 outputBlockIndex = (rankIdLevel1 + rankSizeLevel1 - i) % rankSizeLevel1;
        u32 inputSliceIndex = inputBlockIndex * rankSizeLevel0 + rankIdLevel0;
        u64 inputSliceOffset = totalOutputSize * inputSliceIndex + curOffset_;
        for (u32 j = 0; j < rankSizeLevel0; j++) {
            u32 outputSliceIndex = outputBlockIndex * rankSizeLevel0 + j;
            bufferSlices[outputSliceIndex].size = execMem.count * unitSize;
            u64 outputSliceOffset
                = (bufferSlices[outputSliceIndex].size + HCCL_MIN_SLICE_ALIGN_910B) * outputSliceIndex;
            u64 outputInSliceOffset = (HCCL_MIN_SLICE_ALIGN_910B + (inputSliceOffset % HCCL_MIN_SLICE_ALIGN_910B)
                                       - (outputSliceOffset % HCCL_MIN_SLICE_ALIGN_910B))
                                      % HCCL_MIN_SLICE_ALIGN_910B;
            bufferSlices[outputSliceIndex].offset = outputSliceOffset + outputInSliceOffset;
            HCCL_DEBUG(
                "[CollReduceScatterDeterPipelineExecutor][PrepareDataSlice]tag[%s], buffer slice i[%u], "
                "size[%llu], offset[%llu], outputInSliceOffset[%llu], inputSliceIndex[%u], inputSliceOffset[%llu], "
                "curOffset[%llu]",
                tag_.c_str(), outputSliceIndex, bufferSlices[outputSliceIndex].size,
                bufferSlices[outputSliceIndex].offset, outputInSliceOffset, inputSliceIndex, inputSliceOffset,
                curOffset_);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterPipelineExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollReduceScatterDeterPipelineExecutor][KernelRun] tag[%s], userRank[%u] starts.", tag_.c_str(),
        topoAttr_.userRank);

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    std::vector<Slice> bufferSlices; // 数据分成ranksize份，每份的起始偏移和大小
    CHK_RET(PrepareDataSlice(param, execMem, level0CommInfo, level1CommInfo, bufferSlices));

    CHK_RET(ActiveSlaveStreams(param.stream));

    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_MULTI_DETERMINISTIC_PIPELINE, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);

    HcomCollOpInfo opInfo = {"",
                             execMem.inputPtr,
                             execMem.outputPtr,
                             param.DataDes.count,
                             param.DataDes.dataType,
                             param.root,
                             param.reduceType,
                             0};

    CHK_RET(tempAlg->Prepare(
        &opInfo, execMem.scratchMem, execMem.count, curOffset_, bufferSlices, level0CommInfo, level1CommInfo,
        const_cast<Stream&>(param.stream), algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux));
    CHK_RET(tempAlg->RunAsync());

    HCCL_INFO(
        "[CollReduceScatterDeterPipelineExecutor][KernelRun] tag[%s], userRank[%u] run success.", tag_.c_str(),
        topoAttr_.userRank);
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterDeterPipelineExecutor", ReduceScatterDeterPipeline, CollReduceScatterDeterPipelineExecutor);

} // namespace hccl
