/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_hccs_sio_executor.h"

namespace hccl {
CollAllGatherHccsSioExecutor::CollAllGatherHccsSioExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    desc_.isZeroCopy = false;
    DMAReduceFlag_ = true;
}

HcclResult CollAllGatherHccsSioExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation + 1U;
    // 本地拷贝两条流并发，跨片两条流并发
    static u32 MULTIPLIER = 2;
    streamNum = MULTIPLIER * (totalStreamNum - 1);
    HCCL_INFO("[CollAllGatherHccsSioExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);

    return HCCL_SUCCESS;
}

u64 CollAllGatherHccsSioExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = cclBuffSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;
    HCCL_INFO(
        "[CollAllGatherHccsSioExecutor][CalcLoopMaxCount]"
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.",
        maxCountPerLoop);
    return maxCountPerLoop;
}

HcclResult CollAllGatherHccsSioExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherHccsSioExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_HCCS_PLUS_SIO); // 新建通信平面
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));

    LevelNSubCommTransport& commTransportLevel0 = opTransport[COMM_LEVEL0];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto& transportRequest :
             commTransportLevel0[subCommIndex].transportRequests) { // 两个transport分别设置ishccs为false、true
            // 根据子通信索引设置isHccs的值
            transportRequest.linkType = (subCommIndex == 0) ? TransportLinkType::SIO : TransportLinkType::HCCS;
            HCCL_INFO(
                "[CollAllGatherHccsSioExecutor][CalcLevel0CommInfo] set extra notifyNum[%u]",
                transportRequest.notifyNum);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
CollAllGatherHccsSioExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }

    HCCL_INFO(
        "[CollAllGatherHccsSioExecutor][CalcTransportMemType]"
        "tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherHccsSioExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] The AllGatherHccsSioExecutor starts.", __func__);
    HcomCollOpInfo opInfo = {"",
                             execMem.inputPtr,
                             execMem.outputPtr,
                             param.DataDes.count,
                             param.DataDes.dataType,
                             param.root,
                             param.reduceType,
                             0};

    // step 1 先获取 comm inner \ comm outer 的value
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfoHccs = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_1 + 1));
    SubCommInfo outerCommInfoSio = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_1);

    // 执行
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_HCCS_SIO, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_HCCS_SIO in COMM_LEVEL0", __func__);
    CHK_SMART_PTR_NULL(tempAlg);

    CHK_RET(tempAlg->Prepare(
        outerCommInfoHccs, outerCommInfoSio, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
        param.stream, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, &opInfo));

    CHK_RET(tempAlg->RegisterProfiler(
        (outerCommInfoHccs.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfoHccs.localRank,
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(tempAlg, outerCommInfoHccs));

    HCCL_INFO("[CollAllGatherHccsSioExecutor][KernelRun] AllGatherHccsSioExecutor ends.");

    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherHccsSioExecutor", AllGatherHccsSio, CollAllGatherHccsSioExecutor);

} // namespace hccl
