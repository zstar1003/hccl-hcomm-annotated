/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "calc_nhr_v1_transport_req.h"
#include "nonuniform_hierarchical_ring_v1_base_pub.h"

namespace hccl {
CalcNHRV1TransportReq::CalcNHRV1TransportReq(
    std::vector<std::vector<u32>>& subCommPlaneVector, std::vector<bool>& isBridgeVector, u32 userRank)
    : CalcTransportReqBase(subCommPlaneVector, isBridgeVector, userRank)
{}

CalcNHRV1TransportReq::~CalcNHRV1TransportReq() {}

HcclResult CalcNHRV1TransportReq::CalcTransportRequest(
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

        RingInfo info = NHRV1Base::GetRingInfo(subCommPlaneVector_[ringIndex].size());

        u32 hIndex = info.GetHIndex(rank);
        u32 vIndex = info.GetVIndex(rank);
        u32 hSize = info.GetHSizeByVIndex(vIndex);
        u32 vSize = info.GetVSizeByHIndex(hIndex);

        // 收集邻居
        std::set<u32> links;
        // -- 水平方向：左邻居
        links.insert(info.GetRank(vIndex, (hIndex + hSize - 1) % hSize));
        // -- 水平方向：右邻居
        links.insert(info.GetRank(vIndex, (hIndex + 1) % hSize));
        // -- 垂直方向：上邻居
        links.insert(info.GetRank((vIndex + vSize - 1) % vSize, hIndex));
        // -- 垂直方向：下邻居
        links.insert(info.GetRank((vIndex + 1) % vSize, hIndex));

        u32 rankOffset = info.GetRankOffset();
        if (rankOffset != rankSize) {
            // -- 水平方向：建立(x,0)和(x, sqrt-1)之间的链接
            if (hIndex == info.GetSqrtRankSize() - 1) {
                links.insert(info.GetRank(vIndex, 0));
            } else if (hIndex == 0) {
                links.insert(info.GetRank(vIndex, info.GetSqrtRankSize() - 1));
            }
            // -- 垂直方向：建立(0,y)和(y,sqrt)之间、(sqrt-1,y)和(y,sqrt)之间的链接
            if (hIndex < info.GetSqrtRankSize()) {
                if (info.GetHSizeByVIndex(hIndex) > info.GetSqrtRankSize()
                    && (vIndex == info.GetVSizeByHIndex(hIndex) - 1 || vIndex == 0)) {
                    links.insert(info.GetRank(hIndex, info.GetSqrtRankSize()));
                }
            } else {
                links.insert(info.GetRank(0, vIndex));
                links.insert(info.GetRank(info.GetVSizeByHIndex(vIndex) - 1, vIndex));
            }
        }
        for (u32 dstRank : links) {
            if (dstRank != rank) {
                TransportRequest& tmpTransport = subCommTransport.transportRequests[dstRank];
                tmpTransport.isValid = true;
                tmpTransport.localUserRank = userRank_;
                tmpTransport.remoteUserRank = subCommPlaneVector_[ringIndex][dstRank];
                tmpTransport.inputMemType = inputMemType;
                tmpTransport.outputMemType = outputMemType;
                HCCL_INFO(
                    "[CommFactory][CalcNHRV1CommInfo] param_.tag[%s] ringIndex[%u], localRank[%u], "
                    "remoteRank[%u], inputMemType[%d], outputMemType[%d]",
                    tag.c_str(), ringIndex, userRank_, tmpTransport.remoteUserRank, inputMemType, outputMemType);
            }
        }
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
