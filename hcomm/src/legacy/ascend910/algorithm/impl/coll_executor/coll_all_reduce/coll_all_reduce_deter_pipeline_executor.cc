/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_deter_pipeline_executor.h"

namespace hccl {

CollAllReduceDeterPipelineExecutor::CollAllReduceDeterPipelineExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
}

void CollAllReduceDeterPipelineExecutor::ParseParam(const OpParam& param) { tag_ = param.tag; }

HcclResult CollAllReduceDeterPipelineExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = topoAttr_.deviceNumPerAggregation + 3U; // (deviceNum - 1)机内 + 4Reduce + 1机间 - 1主流
    HCCL_INFO("[CollAllReduceDeterPipelineExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceDeterPipelineExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceDeterPipelineExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaInfo.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceDeterPipelineExecutor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL1, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceDeterPipelineExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollAllReduceDeterPipelineExecutor][CalcTransportMemType] tag[%s] inputType[%d], "
        "outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

u64 CollAllReduceDeterPipelineExecutor::CalcCountPerSlice(const u64& totalCount, const u32& unitSize)
{
    u64 sizePerBlock = (totalCount + topoAttr_.userRankSize - 1) / topoAttr_.userRankSize * unitSize;
    sizePerBlock = AlgTemplateBase::RoundUpWithDivisor(sizePerBlock, HCCL_MIN_SLICE_ALIGN_910B);
    if (sizePerBlock * (topoAttr_.userRankSize - 1) < totalCount * unitSize) {
        return sizePerBlock;
    }
    sizePerBlock = (totalCount + topoAttr_.userRankSize - 1) / topoAttr_.userRankSize * unitSize;
    sizePerBlock = AlgTemplateBase::RoundUpWithDivisor(sizePerBlock, HCCL_MIN_SLICE_ALIGN);
    return sizePerBlock;
}

HcclResult
CollAllReduceDeterPipelineExecutor::RunLoopInner(OpParam& param, const ReduceType& reduceType, ExecMem& execMem)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 curSize = execMem.count * unitSize; // 单位：字节
    HCCL_DEBUG(
        "[CollAllReduceDeterPipelineExecutor][RunLoopInner]inputMem[%p][%llu], outputMem[%p][%llu], "
        "inputPtr[%p], outputPtr[%p], curCount[%llu], curSize[%llu]",
        execMem.inputMem.ptr(), execMem.inputMem.size(), execMem.outputMem.ptr(), execMem.outputMem.size(),
        execMem.inputPtr, execMem.outputPtr, execMem.count, curSize);
    CHK_PRT_RET(
        (execMem.count == 0), HCCL_ERROR("[CollAllReduceDeterPipelineExecutor][RunLoop]In OP_BASE curCount is zero."),
        HCCL_E_PARA);

    /* 设置子图复用标志 */
    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    bool hugeData = IsHugeData(curSize);                                   // override
    bool smallData = IsSmallData(param.DataDes.count * unitSize, curSize); // override
    constexpr s64 HCCL_MEDIUM_COUNT_2_MB = 2 * 1024 * 1024;
    u64 sliceNum = (curSize / topoAttr_.userRankSize) < HCCL_MEDIUM_COUNT_2_MB ? 1 : 0;
    bool dataSplit = false;
    u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();
    auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(
        autoSelectedAlgTypeLevel1, param.DataDes.dataType, reduceType, smallData, 1, hugeData, CopyPattern::ZCOPY,
        sliceNum, false, true, dataSplit, deterministic);
    CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey(), false));

    execMem.inputMem = DeviceMem::create(execMem.inputMem.ptr(), curSize);
    execMem.outputMem = DeviceMem::create(execMem.outputMem.ptr(), curSize);

    // 执行
    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllReduceDeterPipelineExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], "
            "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d], reduce op type[%d]",
            HCCL_ERROR_CODE(ret), param.tag.c_str(), execMem.inputMem.ptr(), execMem.outputMem.ptr(), execMem.count,
            param.DataDes.dataType, param.reduceType),
        ret);

    CHK_RET(LaunchTaskExtend(
        dispatcher_, const_cast<Stream&>(param.stream), const_cast<std::vector<Stream>&>(algResResp_->slaveStreams)));
    return ret;
}

HcclResult CollAllReduceDeterPipelineExecutor::PrepareDataSlice(
    const ExecMem& execMem, const u32& unitSize, std::vector<Slice>& bufferSlices)
{
    bufferSlices.resize(topoAttr_.userRankSize);
    u64 totalSize = execMem.count * unitSize;
    u64 sliceSize = CalcCountPerSlice(execMem.count, unitSize);
    for (u32 sliceIndex = 0; sliceIndex < topoAttr_.userRankSize; sliceIndex++) {
        bufferSlices[sliceIndex].size = totalSize > sliceSize ? sliceSize : totalSize;
        bufferSlices[sliceIndex].offset = sliceIndex * sliceSize;
        totalSize -= bufferSlices[sliceIndex].size;
        HCCL_DEBUG(
            "[CollAllReduceDeterPipelineExecutor][PrepareDataSlice]tag[%s], buffer slice i[%u], "
            "size[%llu], offset[%llu], left size[%llu]",
            tag_.c_str(), bufferSlices[sliceIndex].size, bufferSlices[sliceIndex].offset, totalSize);
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceDeterPipelineExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollAllReduceDeterPipelineExecutor][KernelRun] tag[%s], userRank[%u] starts.", tag_.c_str(),
        topoAttr_.userRank);

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    std::vector<Slice> bufferSlices; // 数据分成ranksize份，每份的起始偏移和大小
    CHK_RET(PrepareDataSlice(execMem, unitSize, bufferSlices));

    CHK_RET(ActiveSlaveStreams(param.stream));

    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_REDUCE_MULTI_DETERMINISTIC_PIPELINE, dispatcher_);
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
        &opInfo, execMem.inputMem, execMem.outputMem, execMem.count, bufferSlices, level0CommInfo, level1CommInfo,
        const_cast<Stream&>(param.stream), algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux));
    CHK_RET(tempAlg->RunAsync());

    HCCL_INFO(
        "[CollAllReduceDeterPipelineExecutor][KernelRun] tag[%s], userRank[%u] run success.", tag_.c_str(),
        topoAttr_.userRank);
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceDeterPipelineExecutor", AllReduceDeterPipeline, CollAllReduceDeterPipelineExecutor);

} // namespace hccl
