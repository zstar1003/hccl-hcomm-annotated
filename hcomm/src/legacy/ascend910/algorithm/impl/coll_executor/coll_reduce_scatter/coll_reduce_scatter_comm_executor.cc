/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_comm_executor.h"
#include "alg_template_register.h"

namespace hccl {

CollReduceScatterCommExecutor::CollReduceScatterCommExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    desc_.deterministic = 1;
    DMAReduceFlag_ = false;
}

void CollReduceScatterCommExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        scratchMemFlag_ = false;
    } else {
        scratchMemFlag_ = true;
    }

    // 记录图模式总数据量
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollReduceScatterCommExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    if (scratchMemFlag_) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            scratchMemSize = inCCLbufferSize_;
        } else {
            scratchMemSize = totalSize_;
        }
    } else {
        scratchMemSize = 0U;
    }

    HCCL_INFO(
        "[CollReduceScatterCommExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%llu]", tag_.c_str(),
        scratchMemSize);
    return HCCL_SUCCESS;
}

bool CollReduceScatterCommExecutor::IsSmallData(
    [[maybe_unused]] const u64 totalSize, [[maybe_unused]] const u64 curSize)
{
    return topoAttr_.deviceType == DevType::DEV_TYPE_910_93;
}

HcclResult CollReduceScatterCommExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcCombinedCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterCommExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
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
        "[CollReduceScatterCommExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterCommExecutor::CalcCombinedCommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommPlane commPlane = COMM_COMBINE_ORDER;

    CommParaInfo commParaInfo(commPlane, CommType::COMM_TAG_MAX);

    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1;
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
        commParaInfo.commType = CommType::COMM_TAG_HALVING_DOUBLING;
    } else {
        commParaInfo.commType = CommType::COMM_TAG_RING_INNER;
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[commPlane], inputType, outputType));

    return HCCL_SUCCESS;
}

u64 CollReduceScatterCommExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop
        = inCCLbufferSize_ / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}

bool CollReduceScatterCommExecutor::IsHugeData(const u64 curSize, [[maybe_unused]] OpParam* param)
{
    bool hugeData = (curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE)
                    || (curSize > SDMA_SEND_MAX_SIZE);
    return hugeData;
}

HcclResult CollReduceScatterCommExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] userRank[%u] starts.", __func__, topoAttr_.userRank);
    CommPlane commPlane = COMM_COMBINE_ORDER;

    CHK_RET(CheckCommSize(commPlane, COMM_INDEX_0 + 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(commPlane, COMM_INDEX_0);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    // 构造ring algorithm对应的reduce-scatter实例
    std::unique_ptr<AlgTemplateBase> tempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR in COMM_COMBINE_ORDER", __func__);
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(reduceAttr, false));
        CHK_RET(tempAlg->Prepare(
            execMem.inputMem, execMem.outputMem, execMem.scratchMem, execMem.count, param.DataDes.dataType,
            param.stream, param.reduceType));
        if (topoAttr_.deviceType != DevType::DEV_TYPE_910_93 || algoAttr_.isSupportAtomicWrite) {
            tempAlg->CloseBarrier();
        }
        CHK_RET(RunTemplate(tempAlg, combinedCommInfo));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        tempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR_V1, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR_V1 in COMM_COMBINE_ORDER", __func__);
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(reduceAttr));
        CHK_RET(tempAlg->Prepare(
            execMem.inputMem, execMem.outputMem, execMem.scratchMem, execMem.count, param.DataDes.dataType,
            param.stream, param.reduceType));
        CHK_RET(RunTemplate(tempAlg, combinedCommInfo));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_COMBINE_ORDER", __func__);
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(reduceAttr));
        CHK_RET(tempAlg->Prepare(
            execMem.inputMem, execMem.outputMem, execMem.scratchMem, execMem.count, param.DataDes.dataType,
            param.stream, param.reduceType));
        CHK_RET(RunTemplate(tempAlg, combinedCommInfo));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_RECURSIVE_HD, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RECURSIVE_HD in COMM_COMBINE_ORDER", __func__);
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(reduceAttr));
        DeviceMem scratchMem = execMem.scratchMem.range(0, execMem.inputMem.size());
        u64 inputDataCount = execMem.inputMem.size() / SIZE_TABLE[param.DataDes.dataType];
        CHK_RET(tempAlg->Prepare(
            execMem.inputMem, execMem.inputMem, scratchMem, inputDataCount, param.DataDes.dataType, param.stream,
            param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        CHK_RET(RunTemplate(tempAlg, combinedCommInfo));
        u64 dataSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        DeviceMem srcMem = execMem.inputMem.range(dataSize * topoAttr_.userRank, dataSize);
        DeviceMem dstMem = execMem.outputMem.range(0, dataSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    } else {
        tempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_COMBINE_ORDER", __func__);
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(reduceAttr));
        CHK_RET(tempAlg->Prepare(
            execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, param.DataDes.dataType, param.stream,
            param.reduceType));
        CHK_RET(RunTemplate(tempAlg, combinedCommInfo));
        // 将cclInBuffer中与userRank_对应的部分拷贝至cclOutBuffer
        u64 dataSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        DeviceMem srcMem = execMem.inputMem.range(dataSize * topoAttr_.userRank, dataSize);
        DeviceMem dstMem = execMem.outputMem.range(0, dataSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterComm", ReduceScatterComm, CollReduceScatterCommExecutor);
} // namespace hccl
