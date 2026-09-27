/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_comm_executor.h"

namespace hccl {

CollBroadcastCommExecutor::CollBroadcastCommExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{}

HcclResult CollBroadcastCommExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcCombinedCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastCommExecutor::CalcCombinedCommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommPlane commPlane = COMM_COMBINE;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CommParaInfo commParaInfo(commPlane, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1;
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
    } else {
        commParaInfo.commType = CommType::COMM_TAG_RING_INNER;
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[commPlane], inputType, outputType));

    return HCCL_SUCCESS;
}

HcclResult CollBroadcastCommExecutor::CalcStreamNum(u32& streamNum)
{
    // 只传递从流数量
    streamNum = 0;
    HCCL_INFO("[CollBroadcastCommExecutor][CalcStreamNum]tag[%s] streamNum_ is [%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

void SetPrepareData(PrepareData& prepareData, const OpParam& param, const ExecMem& execMem, const u32& rootRank)
{
    prepareData.inputMem = execMem.inputMem;
    prepareData.outputMem = execMem.outputMem;
    prepareData.scratchMem = execMem.outputMem;
    prepareData.count = execMem.count;
    prepareData.dataType = param.DataDes.dataType;
    prepareData.stream = param.stream;
    prepareData.reductionOp = HCCL_REDUCE_RESERVED;
    prepareData.root = rootRank;
    prepareData.baseOffset = 0;
}

HcclResult CollBroadcastCommExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollBroadcastCommExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    CommPlane commPlane = COMM_COMBINE;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CHK_RET(CheckCommSize(commPlane, COMM_INDEX_0 + 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(commPlane, COMM_INDEX_0);

    bool isUsedRegister = false;
    std::unique_ptr<AlgTemplateBase> tempAlg;
    u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        if (curSize <= NHR_BCAST_SMALL_SIZE) {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_NHR_ONESHOT, dispatcher_);
            HCCL_CONFIG_INFO(
                HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NHR_ONESHOT in COMM_COMBINE/COMM_COMBINE_ORDER", __func__);
        } else {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NHR in COMM_COMBINE/COMM_COMBINE_ORDER", __func__);
        }
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        isUsedRegister = true;
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR_V1, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NHR_V1 in COMM_COMBINE/COMM_COMBINE_ORDER", __func__);
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        if (ShouldUseBinaryBroadcastOfNB(
                curSize, combinedCommInfo.localRankSize, topoAttr_.userRankSize, topoAttr_.deviceNumPerAggregation)) {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_NB_BINARY, dispatcher_);
            HCCL_CONFIG_INFO(
                HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NB_BINARY in COMM_COMBINE/COMM_COMBINE_ORDER", __func__);
        } else {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NB in COMM_COMBINE/COMM_COMBINE_ORDER", __func__);
        }
    } else {
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_RING in COMM_COMBINE/COMM_COMBINE_ORDER", __func__);
    }
    CHK_SMART_PTR_NULL(tempAlg);

    // 获取root
    u32 rootRank = 0;
    CHK_RET(GetRankByUserRank(commPlane, COMM_INDEX_0, param.root, rootRank));

    if (isUsedRegister) {
        PrepareData prepareData;
        SetPrepareData(prepareData, param, execMem, rootRank);
        CHK_RET(tempAlg->Prepare(prepareData));
    } else {
        CHK_RET(tempAlg->Prepare(
            execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, rootRank));
    }

    CHK_RET(RunTemplate(tempAlg, combinedCommInfo));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("BroadCastComm", BroadcastComm, CollBroadcastCommExecutor);

} // namespace hccl
