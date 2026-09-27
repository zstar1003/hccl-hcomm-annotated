/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CALC_AHC_BROKE_TRANSPORT_REQ_H
#define CALC_AHC_BROKE_TRANSPORT_REQ_H

#include "calc_ahc_transport_req_base.h"

namespace hccl {
class CalcAHCBrokeTransportReq : public CalcAHCTransportReqBase {
public:
    explicit CalcAHCBrokeTransportReq(
        std::vector<std::vector<u32>>& subCommPlaneVector, std::vector<bool>& isBridgeVector, u32 userRank,
        std::vector<std::vector<std::vector<u32>>>& globalSubGroups,
        std::map<AHCConcOpType, TemplateType>& ahcAlgOption, std::unordered_map<u32, bool>& isUsedRdmaMap);

    ~CalcAHCBrokeTransportReq() override;

    HcclResult CalcDstRanks(u32 rank, std::set<u32>& dstRanks, u32 ringIndex) override;

private:
    HcclResult DisposeSubGroups(u32 rank) override;
    HcclResult CommAHCInfoInit(std::vector<std::vector<u32>>& subGroups) override;
};
} // namespace hccl
#endif /* CALC_AHC_BROKE_TRANSPORT_REQ_H */
