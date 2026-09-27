/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_COLL_ALG_BASE
#define HCCLV2_INS_COLL_ALG_BASE

#include <unordered_set>
#include <string>
#include <vector>
#include <map>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "types/types.h"
#include "op_type.h"
#include "op_mode.h"
#include "dev_mode.h"
#include "data_type.h"
#include "reduce_op.h"
#include "topo/port.h"
#include "dev_type.h"
#include "rank_gph.h"
#include "virtual_topo.h"
#include "coll_operator.h"
#include "coll_alg_params.h"
#include "connected_link_mgr.h"
#include "template_utils.h"
#include "executor_utils.h"
#include "ins_alg_template_base.h"
#ifndef CCL_KERNEL_AICPU
#include "ccu_alg_template_base.h"
#endif
#include "dma_mode.h"
#include "hccl_params_pub.h"
#include "alg_data_trans_wrapper.h"
#include "rmt_data_buffer_mgr.h"

namespace Hccl {

class InsCollAlgBase {
public:
    InsCollAlgBase();
    virtual ~InsCollAlgBase();

    void SetMyRank(RankId myRank);
    void SetRankSize(u32 rankSize);
    void SetDevType(DevType devType);
    void SetSendRecvRemoteRank(RankId sendRecvRemoteRank);
    virtual void SetOp(const CollAlgOperator& op);

    // data Align
    void SetAllignSize(u64 allignSize);
    void EnableDataAllign(bool enableAllign);

    // detour
    void EnableDetour(bool enableDetour);

    // dma mode
    void SetDmaMode(const DmaMode dmaMode);

    // rmaDataBufferMgr
    virtual void SetRmaDataBufferMgr(const RmtDataBufferMgr* rmaDataBufferMgr);

    virtual std::string Describe() const = 0;

    // host
    virtual HcclResult
    Orchestrate(const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
        = 0;
    virtual HcclResult CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq) = 0;
    virtual HcclResult CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq) = 0;
    virtual HcclResult CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit);

    // device
    virtual HcclResult Orchestrate(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        InsQuePtr insQue)
        = 0;

    // load params
    virtual HcclResult InitParams(const CollAlgOperator& op, const CollAlgParams& params);

protected:
    // check if enable counterNotify
    bool IsEnableCounterNotify() const;

    // init and check params
    HcclResult Init(const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue);

    HcclResult GenInsQueMap(InsQuePtr insQue);

    // queue prepare
    HcclResult InitQueue(const u32& requiredQueNum, std::vector<InsQuePtr>& requiredQue);

    // link prepare
    HcclResult SetLinkPrty(const std::vector<BasePortType>& linkPriority);

    HcclResult CalcParallelNotifyReq(
        const u32 primQueueNum, const u32 IntraqueNum, std::vector<std::tuple<QId, QId, u32>>& queueNotifys) const;
    HcclResult CalcLocalRankSize(
        const RankId myRank, const std::vector<std::vector<RankId>>& virtRanks, u32& rankSizeLevel0,
        u32& rankSizeLevel1) const;
    LinkReq GetSeqLinksUnion(const LinkReq& linkReq0, const LinkReq& linkReq1) const;

    CollAlgOperator op_;
    // CollAlg base params
    RankId myRank_ = INVALID_RANKID;
    u32 rankSize_ = 0;
    DevType devType_ = DevType::DEV_TYPE_NOSOC;
    RankId sendRecvRemoteRank_ = INVALID_RANKID;

    // CollAlgOperator
    OpType opType_;
    // opInfo
    ReduceOp redOp_;
    u32 root_ = INVALID_U32;
    // dataInfo
    DataType dataType_;
    DataType outputDataType_;
    u64 dataCount_ = 0;

    // CollAlgParams
    OpMode opMode_;
    u64 maxTmpMemSize_ = 0;

    // dataSize
    u64 dataSize_ = 0;
    u64 dataTypeSize_ = 0;

    // data alignment
    bool enableAllign_ = false;
    u64 allignSize_ = 0;

    // detour requirements
    bool enableDetour_ = false;

    // dma mode
    DmaMode dmaMode_ = DmaMode::DEFAULT;

    // queue management
    std::map<u32, InsQuePtr> queId2InsQue_;

    // link priority
    std::vector<BasePortType> linkPriority_ = DEFAULT_LINK_PRIORITY;

    // 管理远端地址
    RmtDataBufferMgr* rmaDataBufferMgr_{nullptr};
};

} // namespace Hccl

#endif // !HCCLV2_INS_COLL_ALG_BASE
