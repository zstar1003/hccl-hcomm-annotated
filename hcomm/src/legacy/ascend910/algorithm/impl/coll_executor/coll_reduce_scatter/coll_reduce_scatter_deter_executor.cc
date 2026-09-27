/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_deter_executor.h"

namespace hccl {

CollReduceScatterDeterExecutor::CollReduceScatterDeterExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    desc_.deterministic = 1;
    DMAReduceFlag_ = true;
    CCLMemSlice_ = false;
}

void CollReduceScatterDeterExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    // 是否需要scratch memory 选中确定性计算Executor，其他条件必定满足，只需区分是否为图模式
    scratchMemFlag_ = (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    // 记录图模式总数据量
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollReduceScatterDeterExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    if (scratchMemFlag_) { // 确定性计算只有图模式需要scratch memory
        scratchMemSize = totalSize_;
    } else {
        scratchMemSize = 0U;
    }
    HCCL_INFO(
        "[CollReduceScatterDeterExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%llu]", tag_.c_str(),
        scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        totalStreamNum = topoAttr_.deviceNumPerAggregation - 1U;
    } else {
        totalStreamNum = topoAttr_.deviceNumPerAggregation;
    }
    streamNum = totalStreamNum - 1U;
    const u32 subStreamNum = 3;
    if (topoAttr_.serverNum != 1) {
        streamNum = subStreamNum;
    }
    HCCL_INFO("[CollReduceScatterDeterExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterDeterExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::CCL_OUTPUT;
        }
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::PARAM_OUTPUT;
        }
    }
    HCCL_INFO(
        "[CollReduceScatterDeterExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    if (topoAttr_.serverNum == 1) {
        CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
        commParaLevel0.meshSinglePlane = true;
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    } else {
        CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
        CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara, opTransport[COMM_COMBINE_ORDER], inputType, outputType));
    }
    return HCCL_SUCCESS;
}

u64 CollReduceScatterDeterExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    u64 maxCountPerLoop;
    bool isLocalReduce91073 = ((((topoAttr_.userRankSize & (topoAttr_.userRankSize - 1)) != 0) || aicpuUnfoldMode_
                                || (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB))
                               && (topoAttr_.deviceType == DevType::DEV_TYPE_910_93))
                              && (topoAttr_.serverNum == 1);

    bool isLocalReduce910B = ((totalSize_ > HCCL_SMALL_COUNT_32_KB)
                              || (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB)
                              || ((topoAttr_.deviceNumPerAggregation != DEVICE_EIGHT)
                                  && (topoAttr_.deviceNumPerAggregation != DEVICE_FOUR)))
                             && (topoAttr_.deviceType == DevType::DEV_TYPE_910B);
    if (isLocalReduce91073 || isLocalReduce910B) {
        maxCountPerLoop = (inCCLbufferSize_ - HCCL_MIN_SLICE_ALIGN_910B * topoAttr_.deviceNumPerAggregation) / unitSize
                          / (topoAttr_.deviceNumPerAggregation - 1);
        maxCountPerLoop = maxCountPerLoop / HCCL_MIN_SLICE_ALIGN_910B;
        maxCountPerLoop = maxCountPerLoop * HCCL_MIN_SLICE_ALIGN_910B;
    } else {
        const u32 base = 2;
        maxCountPerLoop = inCCLbufferSize_ * base / topoAttr_.userRankSize;
    }
    return maxCountPerLoop;
}

bool CollReduceScatterDeterExecutor::IsHugeData(const u64 curSize, [[maybe_unused]] OpParam* param)
{
    // 只有server内通信，多QP哈希散列下不刷新子图
    bool hugeData = curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

bool CollReduceScatterDeterExecutor::IsSmallData(const u64 totalSize, [[maybe_unused]] const u64 curSize)
{
    bool smallData = false;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        smallData = true;
    } else {
        smallData = totalSize <= HCCL_SMALL_COUNT_32_KB;
    }
    return smallData;
}

HcclResult CollReduceScatterDeterExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] userRank[%u] starts.", __func__, topoAttr_.userRank);
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::unique_ptr<AlgTemplateBase> level0TempAlg;
    CommPlane commPlane = COMM_LEVEL0;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93 && topoAttr_.serverNum != 1) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CHK_RET(CheckCommSize(commPlane, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(commPlane, COMM_INDEX_0);

    CHK_RET(ActiveSlaveStreams(param.stream));

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    HcomCollOpInfo opInfo = {"",
                             execMem.inputPtr,
                             execMem.outputPtr,
                             param.DataDes.count,
                             param.DataDes.dataType,
                             param.root,
                             param.reduceType,
                             0};

    bool isLocalReduce91073 = ((((topoAttr_.userRankSize & (topoAttr_.userRankSize - 1)) != 0) || aicpuUnfoldMode_
                                || (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB))
                               && (topoAttr_.deviceType == DevType::DEV_TYPE_910_93))
                              && (topoAttr_.serverNum == 1);

    bool isLocalReduce910B = ((param.DataDes.count * unitSize > HCCL_SMALL_COUNT_32_KB)
                              || (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB)
                              || ((topoAttr_.deviceNumPerAggregation != DEVICE_EIGHT)
                                  && (topoAttr_.deviceNumPerAggregation != DEVICE_FOUR)))
                             && (topoAttr_.deviceType == DevType::DEV_TYPE_910B);

    if (isLocalReduce91073 || isLocalReduce910B) {
        level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_LOCAL_REDUCE, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_LOCAL_REDUCE in COMM_COMBINE_ORDER", __func__);
    } else {
        level0TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_HDSTAGE, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_HDSTAGE in COMM_COMBINE_ORDER", __func__);
    }

    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(
        execMem.inputMem, execMem.scratchMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0, reduceAttr, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank, &opInfo));

    CHK_RET(level0TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    HCCL_INFO("ReduceScatter mesh deter run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterDeterExecutor", ReduceScatterDeter, CollReduceScatterDeterExecutor);
} // namespace hccl
