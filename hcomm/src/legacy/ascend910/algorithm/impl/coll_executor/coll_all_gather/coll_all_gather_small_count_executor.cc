/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_small_count_executor.h"

namespace hccl {
CollAllGatherSmallCountExecutor::CollAllGatherSmallCountExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
}

HcclResult CollAllGatherSmallCountExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcCombinedCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherSmallCountExecutor::CalcStreamNum(u32& streamNum)
{
    constexpr u64 streamForSmallCount = 3;
    streamNum = streamForSmallCount;
    HCCL_INFO("[CollAllGatherSmallCountExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

bool CollAllGatherSmallCountExecutor::IsSmallData([[maybe_unused]] const u64 size) { return true; }

u64 CollAllGatherSmallCountExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = cclBuffSize / (unitSize * topoAttr_.userRankSize);
    if (topoAttr_.userRankSize % HCCL_DEVICE_NUM_FOUR == 0) {
        maxCountPerLoop = maxCountPerLoop * HCCL_DEVICE_NUM_FOUR;
    } else if (topoAttr_.userRankSize % HCCL_DEVICE_NUM_TWO == 0) {
        maxCountPerLoop = maxCountPerLoop * HCCL_DEVICE_NUM_TWO;
    }
    HCCL_INFO(
        "[CollAllGatherSmallCountExecutor][CalcLoopMaxCount]"
        "maxCountPerLoop[%llu]",
        maxCountPerLoop);
    return maxCountPerLoop;
}

HcclResult
CollAllGatherSmallCountExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollAllGatherSmallCountExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherSmallCountExecutor::CalcCombinedCommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommPlane commPlane = COMM_COMBINE;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CommParaInfo commParaInfo(commPlane, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[commPlane], inputType, outputType));

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherSmallCountExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] starts.", __func__);
    CommPlane commPlane = COMM_COMBINE_ORDER;

    CHK_RET(CheckCommSize(commPlane, COMM_INDEX_0 + 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(commPlane, COMM_INDEX_0);

    // 构造ring algorithm对应的all_gather实例
    std::unique_ptr<AlgTemplateBase> algTemplate
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_HD_STAGE, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_HD_STAGE in COMM_COMBINE_ORDER", __func__);
    CHK_RET(ActiveSlaveStreams(param.stream));
    HcomCollOpInfo opInfo = {"",
                             execMem.inputPtr,
                             execMem.outputPtr,
                             param.DataDes.count,
                             param.DataDes.dataType,
                             param.root,
                             param.reduceType,
                             0};
    CHK_SMART_PTR_NULL(algTemplate);

    PrepareData prepareData;
    prepareData.userRank = topoAttr_.userRank;
    prepareData.opInfo = &opInfo;
    prepareData.inputMem = execMem.inputMem;
    prepareData.outputMem = execMem.outputMem;
    prepareData.scratchMem = execMem.outputMem;
    prepareData.count = execMem.count;
    prepareData.dataType = param.DataDes.dataType;
    prepareData.stream = param.stream;
    prepareData.subStreamsPtr = &algResResp_->slaveStreams;
    prepareData.signalPtr = &algResResp_->notifiesMain;
    prepareData.signalAuxPtr = &algResResp_->notifiesAux;

    CHK_RET(algTemplate->Prepare(prepareData));

    CHK_RET(RunTemplate(algTemplate, combinedCommInfo));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherSmallCount", AllGatherSmallCount, CollAllGatherSmallCountExecutor);

} // namespace hccl
