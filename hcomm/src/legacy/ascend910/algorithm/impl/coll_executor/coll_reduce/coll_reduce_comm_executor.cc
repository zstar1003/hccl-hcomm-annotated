/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_comm_executor.h"

namespace hccl {

CollReduceCommExecutor::CollReduceCommExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceExecutor(dispatcher, topoMatcher)
{
    desc_.deterministic = 1;
}

HcclResult CollReduceCommExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CalcTransportMemType(inputType, outputType);
    CHK_RET(CalcCombinedCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceCommExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollReduceCommExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(), inputType,
        outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceCommExecutor::CalcCombinedCommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommPlane commPlane = COMM_COMBINE;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CommParaInfo commParaInfo(commPlane, CommType::COMM_TAG_MAX);
    commParaInfo.commType = CommType::COMM_TAG_RING_INNER;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[commPlane], inputType, outputType));

    return HCCL_SUCCESS;
}

HcclResult CollReduceCommExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] ReduceCommExecutor starts.", __func__);
    CommPlane commPlane = COMM_COMBINE;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CHK_RET(CheckCommSize(commPlane, COMM_INDEX_0 + 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(commPlane, COMM_INDEX_0);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    std::unique_ptr<AlgTemplateBase> tempAlg;
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCE_RING, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCE_RING in COMM_COMBINE/COMM_COMBINE_ORDER", __func__);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr));

    // 获取root
    u32 root = 0;
    CHK_RET(GetRankByUserRank(commPlane, COMM_INDEX_0, param.root, root));

    u32 rankSize = combinedCommInfo.localRankSize;
    CHK_RET(tempAlg->Prepare(
        execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.stream,
        param.reduceType, root, std::vector<Slice>(0), 0));

    CHK_RET(tempAlg->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + combinedCommInfo.localRank, PROF_STAGE_0,
        HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(tempAlg, combinedCommInfo));
    HCCL_INFO("[CollReduceCommExecutor] ReduceCommExecutor run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceComm", ReduceComm, CollReduceCommExecutor);

} // namespace hccl
