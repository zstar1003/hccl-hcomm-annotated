/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "calc_p2p_transport_req.h"

namespace hccl {
CalcP2PTransportReq::CalcP2PTransportReq(
    std::vector<std::vector<u32>>& subCommPlaneVector, std::vector<bool>& isBridgeVector, u32 userRank)
    : CalcTransportReqBase(subCommPlaneVector, isBridgeVector, userRank)
{}

CalcP2PTransportReq::~CalcP2PTransportReq() {}

HcclResult CalcP2PTransportReq::CalcTransportRequest(
    const std::string& tag, TransportMemType inputMemType, TransportMemType outputMemType,
    const CommParaInfo& commParaInfo, std::vector<SingleSubCommTransport>& commTransport,
    [[maybe_unused]] u32 subUserRankRoot)
{
    u32 planeSize = subCommPlaneVector_.size();
    commTransport.resize(planeSize);

    for (u32 planeIndex = 0; planeIndex < planeSize; planeIndex++) {
        u32 rankSize = subCommPlaneVector_[planeIndex].size();
        SingleSubCommTransport& subCommTransport = commTransport[planeIndex];
        subCommTransport.transportRequests.resize(rankSize);

        if (userRank_ == commParaInfo.peerUserRank) {
            HCCL_ERROR("[CalcP2PCommInfo]p2p dstRank_[%u] is not support to create link with itself", userRank_);
            return HCCL_E_PARA;
        }
        TransportRequest& tmpTransport = subCommTransport.transportRequests[0];

        tmpTransport.isValid = true;
        tmpTransport.localUserRank = userRank_;
        tmpTransport.remoteUserRank = commParaInfo.peerUserRank;
        tmpTransport.inputMemType = inputMemType;
        tmpTransport.outputMemType = outputMemType;
        HCCL_INFO(
            "[CommFactory][CalcP2PCommInfo] param_.tag[%s] planeIndex[%u], localRank[%u],"
            "remoteRank[%u], inputMemType[%d], outputMemType[%d]",
            tag.c_str(), planeIndex, userRank_, tmpTransport.remoteUserRank, inputMemType, outputMemType);
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
