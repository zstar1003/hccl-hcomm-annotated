/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_v_for_310p_executor.h"

namespace hccl {

CollRunAlltoAllVFor310PExecutor::CollRunAlltoAllVFor310PExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{}

HcclResult CollRunAlltoAllVFor310PExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    tag_ = param.tag;
    algResResp_ = &algRes;
    AlltoAllVParam_ = param;
    ExecMem execMem;
    execMem.count = 0;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.inputMem = algRes.cclInputMem;
    execMem.outputMem = algRes.cclOutputMem;
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollRunAlltoAllVFor310PExecutor][Orchestrate]errNo[0x%016llx]executor run failed", HCCL_ERROR_CODE(ret)),
        ret);

    HCCL_INFO(
        "tag[%s], AlltoAllVFor310P orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclOpMetaInfo CollRunAlltoAllVFor310PExecutor::GetOpMeta(HcclCMDType opType, const u64 size)
{
    (void)opType;
    HcclOpMetaInfoDef opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::ZCOPY, size, true);
    return opMeta;
}

HcclResult CollRunAlltoAllVFor310PExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0U;
    if (topoAttr_.userRank % MINORS_NUM_TWO == 0) {
        streamNum = MINORS_NUM_TWO;
    } else {
        streamNum = 1;
    }

    HCCL_INFO("[CollRunAlltoAllVFor310PExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVFor310PExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult
CollRunAlltoAllVFor310PExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;

    HCCL_INFO(
        "[CollRunAlltoAllVFor310PExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVFor310PExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;

    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVFor310PExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollRunAlltoAllVFor310PExecutor][KernelRun] AllToAll for 310p start.");

    // 获取通信域
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    // 执行
    std::unique_ptr<AlgTemplateBase> executor
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_2_ALL_V_FOR310P, dispatcher_);
    CHK_SMART_PTR_NULL(executor);
    CHK_RET(executor->Prepare(
        algResResp_->paramInputMem, algResResp_->paramOutputMem, execMem.inputMem, execMem.outputMem,
        algResResp_->notifiesMain, algResResp_->notifiesAux, const_cast<Stream&>(param.stream),
        algResResp_->slaveStreams, outerCommInfo.links, topoAttr_.userRank, topoAttr_.userRankSize,
        allMeshAggregationSendRecvInfo_));

    u32 rankSize = outerCommInfo.localRankSize;
    CHK_RET(executor->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
        param.stream));

    CHK_RET(executor->RunAsync());

    HCCL_INFO("[CollRunAlltoAllVFor310PExecutor] executor run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("RunAlltoAllVFor310PExecutor", AlltoAllVFor310P, CollRunAlltoAllVFor310PExecutor);
} // namespace hccl
