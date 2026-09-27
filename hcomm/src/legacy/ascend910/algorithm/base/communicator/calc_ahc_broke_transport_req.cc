/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "calc_ahc_broke_transport_req.h"

namespace hccl {
CalcAHCBrokeTransportReq::CalcAHCBrokeTransportReq(
    std::vector<std::vector<u32>>& subCommPlaneVector, std::vector<bool>& isBridgeVector, u32 userRank,
    std::vector<std::vector<std::vector<u32>>>& globalSubGroups, std::map<AHCConcOpType, TemplateType>& ahcAlgOption,
    std::unordered_map<u32, bool>& isUsedRdmaMap)
    : CalcAHCTransportReqBase(
          subCommPlaneVector, isBridgeVector, userRank, globalSubGroups, ahcAlgOption, isUsedRdmaMap)
{}

CalcAHCBrokeTransportReq::~CalcAHCBrokeTransportReq() {}

HcclResult CalcAHCBrokeTransportReq::DisposeSubGroups(u32 rank)
{
    std::vector<std::vector<u32>> level1SubGroups;
    CommAHCBaseInfo::DisposeSubGroups(rank, globalSubGroups_, subGroups_, level1SubGroups);
    return HCCL_SUCCESS;
}

HcclResult CalcAHCBrokeTransportReq::CalcDstRanks(u32 rank, std::set<u32>& dstRanks, u32 ringIndex)
{
    (void)ringIndex;

    // 获取 rank 对应分组信息
    DisposeSubGroups(rank);

    // commAHCBaseInfo_ 初始化
    CHK_RET(CommAHCInfoInit(subGroups_));

    // 计算 AHC建链的目的dstRanks
    commAHCBaseInfo_->CalcDstRanks(rank, dstRanks);

    return HCCL_SUCCESS;
}

HcclResult CalcAHCBrokeTransportReq::CommAHCInfoInit(std::vector<std::vector<u32>>& subGroups)
{
    commAHCBaseInfo_.reset(new (std::nothrow) CommBrokeAlignInfo(subGroups));
    CHK_SMART_PTR_NULL(commAHCBaseInfo_);
    CHK_RET(commAHCBaseInfo_->Init(opType_, ahcAlgOption_));
    return HCCL_SUCCESS;
}
} // namespace hccl
