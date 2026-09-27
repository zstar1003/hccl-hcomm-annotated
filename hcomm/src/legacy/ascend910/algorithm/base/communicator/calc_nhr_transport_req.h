/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CALC_NHR_TRANSPORT_REQ_H
#define CALC_NHR_TRANSPORT_REQ_H

#include "calc_transport_req_base.h"

namespace hccl {
class CalcNHRTransportReq : public CalcTransportReqBase {
public:
    explicit CalcNHRTransportReq(
        std::vector<std::vector<u32>>& subCommPlaneVector, std::vector<bool>& isBridgeVector, u32 userRank);

    ~CalcNHRTransportReq() override;

    HcclResult CalcTransportRequest(
        const std::string& tag, TransportMemType inputMemType, TransportMemType outputMemType,
        const CommParaInfo& commParaInfo, std::vector<SingleSubCommTransport>& commTransport,
        u32 subUserRankRoot = INVALID_VALUE_RANKID) override;

    static HcclResult CalcDstRanks(const u32 rank, const std::vector<u32> commGroups, std::set<u32>& dstRanks);
};
} // namespace hccl

#endif /* CALC_NHR_TRANSPORT_REQ_H */
