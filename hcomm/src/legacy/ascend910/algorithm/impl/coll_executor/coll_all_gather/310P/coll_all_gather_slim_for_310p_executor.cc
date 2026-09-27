/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_slim_for_310p_executor.h"

namespace hccl {
CollAllGatherSlimFor310PExecutor::CollAllGatherSlimFor310PExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollAllGatherSlimFor310PExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult
CollAllGatherSlimFor310PExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollAllGatherSlimFor310PExecutor][CalcTransportMemType]"
        "tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherSlimFor310PExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    LevelNSubCommTransport& commTransportLevel0 = opTransport[COMM_LEVEL0];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto& transportRequest : commTransportLevel0[subCommIndex].transportRequests) {
            // transportRequest.notifyNum = topoAttr_.nicList.size(); //卡间notify num数量
            transportRequest.notifyNum = HCCL_NIC_MAX_NUM; // 卡间notify num数量;
            HCCL_INFO(
                "[CollAllGatherSlimFor310PExecutor][CalcLevel0CommInfo] notifyNum[%u]", transportRequest.notifyNum);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherSlimFor310PExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherSlimFor310PExecutor] AllGather starts.");
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_SLIM_RING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);

    CHK_RET(tempAlg->Prepare(
        execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.stream,
        param.reduceType));

    u32 rankSize = level0CommInfo.localRankSize;
    CHK_RET(tempAlg->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
        param.stream));

    CHK_RET(RunTemplate(tempAlg, level0CommInfo));
    HCCL_INFO("AllGather for 310P run success");

    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherSlimRingFor310PExecutor", AllGatherSlimFor310P, CollAllGatherSlimFor310PExecutor);

} // namespace hccl
