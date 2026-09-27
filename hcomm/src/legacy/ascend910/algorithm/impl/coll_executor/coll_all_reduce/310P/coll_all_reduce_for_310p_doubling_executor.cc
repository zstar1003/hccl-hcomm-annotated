/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_for_310p_doubling_executor.h"

namespace hccl {
CollAllReduceFor310PDoublingExecutor::CollAllReduceFor310PDoublingExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollAllReduceFor310PDoublingExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;

    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceFor310PDoublingExecutor::CalcTransportMemType(
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
        "[CollAllReduceFor310PDoublingExecutor][CalcTransportMemType]"
        "tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceFor310PDoublingExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollAllReduceFor310PDoublingExecutor][CalcLevel0CommInfo]tag[%s] start", tag_.c_str());

    if (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_HD) {
        CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_HALVING_DOUBLING);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    } else if (
        algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING
        && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING) {
        CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    } else {
        HCCL_ERROR("unsupported algType %d plus %d", algType_.algoLevel0, algType_.algoLevel1);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[CollAllReduceFor310PDoublingExecutor][CalcLevel0CommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceFor310PDoublingExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    bool isInlineReduce = IsSupportSDMAReduce(
        execMem.inputMem.ptr(), execMem.outputMem.ptr(), param.DataDes.dataType, param.reduceType);
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollAllReduceFor310PDoublingExecutor][KernelRun] userRank[%u] starts with isInlineReduce[%d]",
        topoAttr_.userRank, isInlineReduce);
    u64 reduceAttr = 0;
    if (isInlineReduce) {
        SalSetBitOne(reduceAttr, ATTR_POS_INLINE_REDUCE);
    }

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    std::unique_ptr<AlgTemplateBase> tempAlg;
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_DOUBLING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr));

    CHK_RET(tempAlg->Prepare(
        execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), 0));

    CHK_RET(tempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_0,
        HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(tempAlg, level0CommInfo));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceDoubling", AllReduceFor310PDoubling, CollAllReduceFor310PDoublingExecutor);
} // namespace hccl
