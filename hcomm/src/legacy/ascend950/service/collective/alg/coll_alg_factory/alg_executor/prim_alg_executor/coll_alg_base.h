/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_BASE
#define HCCLV2_COLL_ALG_BASE

#include <unordered_set>

#include "executor_utils.h"
#include "alg_template_base_v2.h"
#include "hccl_params_pub.h"

namespace Hccl {
class CollAlgBase {
public:
    CollAlgBase();
    virtual ~CollAlgBase();

    using PrimitiveIterator = BaseConstIterator<vector, std::unique_ptr<Primitive>>;
    using SlaveIterator = BaseConstIterator<vector, std::shared_ptr<PrimQueue>>;

    void SetMyRank(RankId myRank);
    void SetRankSize(u32 rankSize);
    void SetDevType(DevType devType);

    // data Align
    void SetAllignSize(u64 allignSize);
    void EnableDataAllign(bool enableAllign);

    // detour
    void EnableDetour(bool enableDetour);

    // dma mode
    void SetDmaMode(const DmaMode dmaMode);

    virtual std::string Describe() const = 0;

    // host
    virtual HcclResult
    GenPrimQues(const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, PrimQuePtr primQue)
        = 0;
    virtual HcclResult CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq) = 0;
    virtual HcclResult CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq) = 0;

    // device
    virtual HcclResult GenPrimQuesAIC(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        PrimQuePtr primQue)
        = 0;

protected:
    // check if enable counterNotify
    bool IsEnableCounterNotify() const;

    // init and check params
    HcclResult Init(const CollAlgOperator& op, const CollAlgParams& params, PrimQuePtr primQue);

    // load params
    HcclResult InitParams(const CollAlgOperator& op, const CollAlgParams& params);
    HcclResult GenPrimQueMap(PrimQuePtr primQue);

    // queue prepare
    HcclResult InitQueue(const u32& requiredQueNum, std::vector<PrimQuePtr>& requiredQue);

    // link prepare
    HcclResult SetLinkPrty(const std::vector<BasePortType>& linkPriority);
    LinkReq GetSeqLinksUnion(const LinkReq& linkReq0, const LinkReq& linkReq1) const;
    HcclResult
    AllocTempResLinks(const ResLinks& execResLinks, const LinkReq& tempLinkReq, ResLinks& tempResLinks) const;

    // CollAlg base params
    RankId myRank_ = INVALID_RANKID;
    u32 rankSize_ = 0;
    DevType devType_ = DevType::DEV_TYPE_NOSOC;

    // CollAlgOperator
    OpType opType_;
    ReduceOp redOp_;
    DataType dataType_;
    DataType outputDataType_;
    u64 dataCount_ = 0;
    u32 root_ = INVALID_U32;

    // CollAlgParams
    OpMode opMode_;
    u64 maxTmpMemSize_ = 0;

    u64 dataSize_ = 0;

    // data alignment
    u64 allignSize_ = 0;
    bool enableAllign_ = false;

    // detour requirements
    bool enableDetour_ = false;

    // dma mode
    DmaMode dmaMode_ = DmaMode::DEFAULT;

    // queue management
    std::map<u32, PrimQuePtr> queId2PrimQue_;

    // link priority
    std::vector<BasePortType> linkPriority_ = DEFAULT_LINK_PRIORITY;
};

} // namespace Hccl

#endif // !HCCLV2_COLL_ALG_BASE
