/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_executor_base.h"
#include "hccl_aiv.h"

namespace hccl {

CollExecutorBase::CollExecutorBase(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : dispatcher_(dispatcher),
      topoMatcher_(topoMatcher)
{}

HcclResult CollExecutorBase::SetAlgType(const AlgType algType)
{
    const std::vector<AlgTypeLevel0>& l0Algo = desc_.level0SupportedAlgos;
    const std::vector<AlgTypeLevel1>& l1Algo = desc_.level1SupportedAlgos;
    const std::vector<AlgTypeLevel2>& l2Algo = desc_.level2SupportedAlgos;

    algType_ = algType;
    if (!l0Algo.empty() && std::find(l0Algo.begin(), l0Algo.end(), algType_.algoLevel0) == l0Algo.end()) {
        HCCL_WARNING("[%s] not support level0 algo[%d], reset to algo[%d]", __func__, algType_.algoLevel0, l0Algo[0]);
        algType_.algoLevel0 = l0Algo[0];
    }
    if (!l1Algo.empty() && std::find(l1Algo.begin(), l1Algo.end(), algType_.algoLevel1) == l1Algo.end()) {
        HCCL_WARNING("[%s] not support level1 algo[%d], reset to algo[%d]", __func__, algType_.algoLevel1, l1Algo[0]);
        algType_.algoLevel1 = l1Algo[0];
    }
    if (!l2Algo.empty() && std::find(l2Algo.begin(), l2Algo.end(), algType_.algoLevel2) == l2Algo.end()) {
        HCCL_WARNING("[%s] not support level2 algo[%d], reset to algo[%d]", __func__, algType_.algoLevel2, l2Algo[0]);
        algType_.algoLevel2 = l2Algo[0];
    }
    // 记录刷新后的算法类型到executor的描述中
    desc_.algType = algType_;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetCCLInBuffer(u64 cclbufferSize)
{
    inCCLbufferSize_ = cclbufferSize;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetIsSupportSDMAReduce(bool isSupportSDMAReduce)
{
    isSupportSDMAReduce_ = isSupportSDMAReduce;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::RunTemplate(const std::unique_ptr<AlgTemplateBase>& tempAlg, const SubCommInfo& commInfo)
{
    HcclResult ret = tempAlg->RunAsync(commInfo.localRank, commInfo.localRankSize, commInfo.links);
    CHK_PRT_RET(
        ret == HCCL_E_AGAIN,
        HCCL_WARNING("[CollExecutorBase][RunTemplate]"
                     "group has been destroyed. Break!"),
        ret);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollExecutorBase][RunTemplate]run tempAlg rank[%u] rank size[%u] failed", commInfo.localRank,
            commInfo.localRankSize),
        ret);
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::GetAivExecParam(
    [[maybe_unused]] const OpParam& param, [[maybe_unused]] AlgResourceResponse& algRes,
    [[maybe_unused]] AivSuperKernelArgs& args)
{
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::PrepareCommInfoToDevice(AlgResourceResponse& algResource)
{
    (void)algResource;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetRmaInfo(void* rmaInfo)
{
    CHK_PTR_NULL(rmaInfo);
    rmaInfo_ = rmaInfo;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::CalcIncreLinkRequest(
    [[maybe_unused]] const OpParam& param, std::set<u32>& ranksLinked,
    [[maybe_unused]] AlgResourceRequest& resourceRequest, bool& needIncreLink)
{
    (void)ranksLinked;
    (void)needIncreLink;
    return HCCL_SUCCESS;
}

HcclResult
CollExecutorBase::CreatePairWiseList([[maybe_unused]] HcclSendRecvItem* sendRecvInfo, [[maybe_unused]] u32 itemNum)
{
    return HCCL_SUCCESS;
}

HcclResult
CollExecutorBase::GetPairWiseList([[maybe_unused]] std::vector<std::vector<HcclSendRecvItem*>>& sendRecvPairList)
{
    return HCCL_SUCCESS;
}

HcclResult
CollExecutorBase::GetPairWiseList([[maybe_unused]] HcclSendRecvItem* sendRecvInfo, [[maybe_unused]] u32 itemNum)
{
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetAlgOpContext(AlgOpContext algOpContext)
{
    algOpContext_ = algOpContext;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetAivClearEnable(bool aivClearEnable)
{
    aivClearEnable_ = aivClearEnable;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::CalNumBlocks(
    u32& numBlocks, u32 rankSize, [[maybe_unused]] u64 dataSize, [[maybe_unused]] HcclCMDType cmdType)
{
    numBlocks = rankSize;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::GetNumBlocks(u32& numBlocks) const
{
    numBlocks = numBlocks_;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetNumBlocks(const u32& numBlocks)
{
    numBlocks_ = numBlocks;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::GetCache(HcclCacheInfo& cacheInfo)
{
    cacheInfo = cacheInfo_;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::SetOpCounter(const OpCounterInfo& opCounter)
{
    opCounter_ = opCounter;
    return HCCL_SUCCESS;
}
HcclResult CollExecutorBase::GetAdjInfo([[maybe_unused]] AlgResourceResponse& algRes, AdjInfo& adjInfo)
{
    (void)adjInfo;
    return HCCL_SUCCESS;
}

HcclResult CollExecutorBase::MarkNeedAlltoallvCache()
{
    HCCL_ERROR("[CollExecutorBase][MarkNeedAlltoallvCache] not supported for current executor!");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CollExecutorBase::GetHcclOffsetDstRanksMap(
    std::unordered_map<uint64_t, std::vector<uint32_t>>& hcclOffsetDstRanksMap) const
{
    UNUSED_PARAM(hcclOffsetDstRanksMap);
    HCCL_ERROR("[CollExecutorBase][GetHcclOffsetDstRanksMap] not supported for current executor!");
    return HCCL_E_NOT_SUPPORT;
}

void CollExecutorBase::SetWorkflowMode(HcclWorkflowMode workflowMode)
{
    UNUSED_PARAM(workflowMode);
    HCCL_DEBUG("[CollExecutorBase][SetWorkflowMode] base class default, no effect");
}

HcclWorkflowMode CollExecutorBase::GetExecutorWorkflowMode() const
{
    HCCL_ERROR("[CollExecutorBase][GetExecutorWorkflowMode] base class should not be called");
    return HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
}

} // namespace hccl
