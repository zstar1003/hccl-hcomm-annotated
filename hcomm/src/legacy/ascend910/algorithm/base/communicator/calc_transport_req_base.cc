/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "calc_transport_req_base.h"

namespace hccl {
CalcTransportReqBase::CalcTransportReqBase(
    const std::vector<std::vector<u32>>& subCommPlaneVector, const std::vector<bool>& isBridgeVector, u32 userRank)
    : subCommPlaneVector_(subCommPlaneVector),
      isBridgeVector_(isBridgeVector),
      userRank_(userRank)
{}

CalcTransportReqBase::~CalcTransportReqBase() {}

HcclResult CalcTransportReqBase::CalcTransportRequest(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] TransportMemType inputMemType,
    [[maybe_unused]] TransportMemType outputMemType, [[maybe_unused]] const CommParaInfo& commParaInfo,
    [[maybe_unused]] std::vector<SingleSubCommTransport>& commTransport, [[maybe_unused]] u32 subUserRankRoot)
{
    return HCCL_SUCCESS;
}

u32 CalcTransportReqBase::GetSubCollectiveRank(const std::vector<u32>& vecPara) const
{
    // 在vecPara数据中，查询本user rank，查询到的vec下标就是rank值
    u32 tmpRank = INVALID_VALUE_RANKID;

    for (u32 rankIndex = 0; rankIndex < vecPara.size(); rankIndex++) {
        if (userRank_ == vecPara[rankIndex]) {
            tmpRank = rankIndex;
            break;
        }
    }

    return tmpRank;
}

HcclResult CalcTransportReqBase::GetRankByUserRank(const std::vector<u32>& vecPara, const u32 userRank, u32& rank) const
{
    // 在vecPara数据中，查询指定userRank，查询到的vec下标就是rank值
    rank = INVALID_VALUE_RANKID;

    for (u32 rankIndex = 0; rankIndex < vecPara.size(); rankIndex++) {
        if (userRank == vecPara[rankIndex]) {
            rank = rankIndex;
            break;
        }
    }
    HCCL_INFO("[Get][RankByUserRank]userRank[%u] --> rank[%u]", userRank, rank);
    return HCCL_SUCCESS;
}

} // namespace hccl
