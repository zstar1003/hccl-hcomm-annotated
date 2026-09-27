/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COLL_EXECUTOR_BASE_H
#define HCCL_COLL_EXECUTOR_BASE_H

#include "hccl_impl.h"
#include "topo_matcher.h"
#include "coll_alg_param.h"
#include "executor_impl.h"
#include "coll_alg_utils.h"
#include "hccl_aiv.h"

namespace hccl {

class CollExecutorBase {
public:
    CollExecutorBase(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    virtual ~CollExecutorBase() = default;

    // 每次构造完必须调用 SetAlgType
    HcclResult SetAlgType(const AlgType algType);

    HcclResult SetCCLInBuffer(u64 cclbufferSize);
    HcclResult SetIsSupportSDMAReduce(bool isSupportSDMAReduce);
    HcclResult SetAlgOpContext(AlgOpContext algOpContext);
    HcclResult SetAivClearEnable(bool aivClearEnable);

    virtual HcclResult CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest) = 0;
    virtual HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) = 0;
    virtual HcclResult GetAivExecParam(const OpParam& param, AlgResourceResponse& algRes, AivSuperKernelArgs& args);
    virtual HcclResult GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo);
    // AIV拷贝通信域信息到device上
    virtual HcclResult PrepareCommInfoToDevice(AlgResourceResponse& algResource);
    // AIV支持Roce直通Rma信息拷贝
    HcclResult SetRmaInfo(void* rmaInfo);

    // batchsendrecv需要增量建链
    virtual HcclResult CalcIncreLinkRequest(
        const OpParam& param, std::set<u32>& ranksLinked, AlgResourceRequest& resourceRequest, bool& needIncreLink);

    virtual void SetWorkflowMode(HcclWorkflowMode workflowMode);
    virtual HcclWorkflowMode GetExecutorWorkflowMode() const;

    static HcclResult RunTemplate(const std::unique_ptr<AlgTemplateBase>& tempAlg, const SubCommInfo& commInfo);

    // batchsendrecv retry使用
    virtual HcclResult CreatePairWiseList(HcclSendRecvItem* sendRecvInfo, u32 itemNum);
    virtual HcclResult GetPairWiseList(std::vector<std::vector<HcclSendRecvItem*>>& sendRecvPairList);
    virtual HcclResult GetPairWiseList(HcclSendRecvItem* sendRecvInfo, u32 itemNum);
    virtual HcclResult
    CalNumBlocks(u32& numBlocks, u32 rankSize, u64 dataSize = 0, HcclCMDType cmdType = HcclCMDType::HCCL_CMD_INVALID);
    HcclResult GetNumBlocks(u32& numBlocks) const;
    HcclResult SetNumBlocks(const u32& numBlocks);
    HcclResult GetCache(HcclCacheInfo& cacheInfo);
    HcclResult SetOpCounter(const OpCounterInfo& opCounter);

    inline AlgDesc GetAlgDesc() { return desc_; }

    // 用于alltoallv算子的aicpu展开cache
    virtual HcclResult MarkNeedAlltoallvCache();
    virtual HcclResult
    GetHcclOffsetDstRanksMap(std::unordered_map<uint64_t, std::vector<uint32_t>>& hcclOffsetDstRanksMap) const;

protected:
    const HcclDispatcher dispatcher_;
    u64 inCCLbufferSize_{0}; // CCLIN大小，用于计算scratch
    AlgType algType_;        // 算法类型
    std::unique_ptr<TopoMatcher>& topoMatcher_;
    bool isSupportSDMAReduce_ = false;
    AlgOpContext algOpContext_;
    bool aivClearEnable_ = false;
    u32 numBlocks_ = MAX_NUM_BLOCKS;
    OpCounterInfo opCounter_;
    AlgDesc desc_;
    void* rmaInfo_ = nullptr;
    HcclCacheInfo cacheInfo_;
};
} // namespace hccl
#endif
