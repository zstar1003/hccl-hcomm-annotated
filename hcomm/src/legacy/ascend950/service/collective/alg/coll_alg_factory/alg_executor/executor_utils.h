/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_EXECUTOR_UTILS
#define HCCLV2_EXECUTOR_UTILS

#include "data_type.h"
#include "log.h"
#include "virtual_topo.h"
#include "template_utils.h"

namespace Hccl {

const std::vector<BasePortType> DEFAULT_LINK_PRIORITY
    = {BasePortType(PortDeploymentType::P2P, ConnectProtoType::HCCS),
       BasePortType(PortDeploymentType::P2P, ConnectProtoType::UB),
       BasePortType(PortDeploymentType::P2P, ConnectProtoType::PCIE),
       BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB),
       BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA),
       BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::TCP),
       BasePortType(PortDeploymentType::HOST_NET, ConnectProtoType::RDMA),
       BasePortType(PortDeploymentType::HOST_NET, ConnectProtoType::TCP)};

bool IsEnableCounterNotifyByDevType(const RankId myRank, const DevType devType);

struct TemplateDataParams {
    BuffInfo buffInfo;
    u64 sliceSize{0};
    u64 inputSliceStride{0};
    u64 outputSliceStride{0};
    u64 repeatNum{0};
    u64 inputRepeatStride{0};
    u64 outputRepeatStride{0};
    u64 tailSize{0};
};

HcclResult InitOpInfo(const CollAlgOperator& op, OpType& opType, ReduceOp& redOp, u32& root);
HcclResult InitDataInfo(const CollAlgOperator& op, DataType& dataType, DataType& outputDataType, u64& dataCount);

// link prepare
const std::vector<NetInstance::Path>
GetPathsFromRankGraph(const RankGraph* rankGraph, const RankId srcRank, const RankId dstRank);
HcclResult AddToResLinks(const RankId vNeighborRank, const LinkData& linkData, ResLinks& resLinks);
HcclResult PrepResLinks(
    const RankId myRank, const RankGraph* rankGraph, const std::vector<BasePortType>& linkPriority,
    const LinkReq& linkReq,
    ResLinks& resLinks); // host
HcclResult PrepResLinks(
    const RankId myRank, const LinkReq& linkReq, ConnectedLinkMgr* linkMgr,
    ResLinks& resLinks); // aicpu
HcclResult CalcResLinks(
    const RankId myRank, const RankGraph* rankGraph, const std::vector<BasePortType>& linkPriority,
    const LinkReq& linkReq, std::vector<LinkData>& links);
HcclResult CalcLinkInfo(
    const RankId myRank, const RankGraph* rankGraph, const LinkReq& linkReq,
    std::vector<std::pair<u32, RankId>>& algTempLinksInfo);
HcclResult SetPathNumMapByRankGraphMultiLevel(
    const RankGraph* rankGraph, std::vector<std::vector<RankId>>& virtRanks_, RankId myRank_,
    std::vector<map<u32, u32>>& rank2PathNumMap);

HcclResult SetPathNumMapByRankGraphMultiLevel(
    const RankGraph* rankGraph, std::vector<RankId>& virtRanks_, RankId myRank_, std::map<u32, u32>& rank2PathNumMap);

HcclResult SetPathNumMapByLinkMgrMultiLevel(
    ConnectedLinkMgr* linkMgr, std::vector<std::vector<RankId>>& virtRanks_, RankId myRank_,
    std::vector<map<u32, u32>>& rank2PathNumMap);

HcclResult SetPathNumMapByLinkMgrMultiLevel(
    ConnectedLinkMgr* linkMgr, std::vector<RankId>& virtRanks_, RankId myRank_, map<u32, u32>& rank2PathNumMap);

} // namespace Hccl

#endif // !HCCLV2_EXECUTOR_UTILS
