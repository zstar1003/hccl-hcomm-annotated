/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "calc_nb_transport_req.h"
#include "calc_ahc_template_register.h"

namespace hccl {
CalcNBTransportReq::CalcNBTransportReq(
    std::vector<std::vector<u32>>& subCommPlaneVector, std::vector<bool>& isBridgeVector, u32 userRank)
    : CalcTransportReqBase(subCommPlaneVector, isBridgeVector, userRank)
{}

CalcNBTransportReq::~CalcNBTransportReq() {}

HcclResult CalcNBTransportReq::CalcTransportRequest(
    const std::string& tag, TransportMemType inputMemType, TransportMemType outputMemType,
    const CommParaInfo& commParaInfo, std::vector<SingleSubCommTransport>& commTransport, u32 subUserRankRoot)
{
    (void)subUserRankRoot;
    u32 ringSize = subCommPlaneVector_.size();
    commTransport.resize(ringSize);

    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        if (commParaInfo.commPlane == COMM_LEVEL1 && !isBridgeVector_[ringIndex]) {
            continue; // 跳出本次循环
        }

        u32 rank = GetSubCollectiveRank(subCommPlaneVector_[ringIndex]);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        u32 rankSize = subCommPlaneVector_[ringIndex].size();
        SingleSubCommTransport& subCommTransport = commTransport[ringIndex];
        subCommTransport.transportRequests.resize(rankSize);
        // 只有一张卡时不需要建链
        if (rankSize == HCCL_RANK_SIZE_EQ_ONE) {
            HCCL_INFO("comm base needn't to create links, rankSize_[%u].", rankSize);
            return HCCL_SUCCESS;
        }

        for (u32 delta = 1; delta < rankSize; delta <<= 1) {
            const u32 targetRankPos = static_cast<u32>(rank + delta) % rankSize;
            TransportRequest& tmpTransport = subCommTransport.transportRequests[targetRankPos];
            tmpTransport.isValid = true;
            tmpTransport.localUserRank = userRank_;
            tmpTransport.remoteUserRank = subCommPlaneVector_[ringIndex][targetRankPos];
            tmpTransport.inputMemType = inputMemType;
            tmpTransport.outputMemType = outputMemType;
            HCCL_INFO(
                "[CalcNBTransportReq][CalcTransportRequest] param_.tag[%s], ringIndex[%u], localRank[%u], "
                "remoteRank[%u], inputMemType[%d], outputMemType[%d]",
                tag.c_str(), ringIndex, userRank_, tmpTransport.remoteUserRank, inputMemType, outputMemType);

            const u32 targetRankNeg = static_cast<u32>(rank + rankSize - delta) % rankSize;
            TransportRequest& tmpTransportNeg = subCommTransport.transportRequests[targetRankNeg];
            tmpTransportNeg.isValid = true;
            tmpTransportNeg.localUserRank = userRank_;
            tmpTransportNeg.remoteUserRank = subCommPlaneVector_[ringIndex][targetRankNeg];
            tmpTransportNeg.inputMemType = inputMemType;
            tmpTransportNeg.outputMemType = outputMemType;
            HCCL_INFO(
                "[CalcNBTransportReq][CalcTransportRequest] param_.tag[%s], ringIndex[%u], localRank[%u], "
                "remoteRank[%u], inputMemType[%d], outputMemType[%d]",
                tag.c_str(), ringIndex, userRank_, tmpTransportNeg.remoteUserRank, inputMemType, outputMemType);
        }
        subCommTransport.enableUseOneDoorbell = true;
    }
    return HCCL_SUCCESS;
}

HcclResult CalcNBTransportReq::CalcDstRanks(const u32 rank, const std::vector<u32> commGroups, std::set<u32>& dstRanks)
{
    CHK_PRT_RET(
        rank >= commGroups.size(),
        HCCL_ERROR(
            "[CalcNBTransportReq][CalcDstRanks] rank [%u] exceed commGroups Size [%u] error", rank, commGroups.size()),
        HCCL_E_INTERNAL);

    for (auto i = 0; static_cast<u32>(1 << i) < commGroups.size(); ++i) {
        // 正方向第2^i个节点的rank号
        const u32 targetRankPos = static_cast<u32>(rank + (1 << i)) % commGroups.size();
        dstRanks.insert(commGroups[targetRankPos]);

        // 反方向第2^i个节点的rank号
        const u32 targetRankNeg = static_cast<u32>(rank + commGroups.size() - (1 << i)) % commGroups.size();

        HCCL_DEBUG(
            "[CalcNBTransportReq][CalcDstRanks] local rank[%u], remote rank[%u]", commGroups[rank],
            commGroups[targetRankNeg]);

        dstRanks.insert(commGroups[targetRankNeg]);
    }

    return HCCL_SUCCESS;
}

REGISTER_AHC_COMM_CALC_FUNC(AHCTemplateType::AHC_TEMPLATE_NB, CalcNBTransportReq, CalcNBTransportReq::CalcDstRanks);
} // namespace hccl
