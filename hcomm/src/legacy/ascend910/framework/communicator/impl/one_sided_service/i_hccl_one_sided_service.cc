/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "i_hccl_one_sided_service.h"

namespace hccl {
using namespace std;

IHcclOneSidedService::IHcclOneSidedService(
    unique_ptr<HcclSocketManager>& socketManager, unique_ptr<NotifyPool>& notifyPool)
    : socketManager_(socketManager),
      notifyPool_(notifyPool),
      trafficClass_(HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET),
      serviceLevel_(HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET)
{}

HcclResult IHcclOneSidedService::Config(
    const HcclDispatcher& dispatcher, const HcclRankLinkInfo& localRankInfo, const RankTable_t* rankTable,
    std::string identifier, bool isStandardCard, const std::unordered_set<u32>& enableP2PRankIds)
{
    CHK_PTR_NULL(dispatcher);
    CHK_PTR_NULL(rankTable);

    dispatcher_ = dispatcher;
    localRankInfo_ = localRankInfo;
    localRankVnicInfo_ = localRankInfo;
    rankTable_ = rankTable;
    identifier_ = identifier;
    isStandardCard_ = isStandardCard;
    enableP2PRankIds_ = enableP2PRankIds;

    return HCCL_SUCCESS;
}

HcclResult IHcclOneSidedService::SetNetDevCtx(const HcclNetDevCtx& netDevCtx, bool useRdma)
{
    if (useRdma) {
        netDevRdmaCtx_ = netDevCtx;
        CHK_PTR_NULL(netDevRdmaCtx_);
    } else {
        netDevIpcCtx_ = netDevCtx;
        CHK_PTR_NULL(netDevIpcCtx_);
    }
    return HCCL_SUCCESS;
}

HcclResult IHcclOneSidedService::GetNetDevCtx(HcclNetDevCtx& netDevCtx, bool useRdma)
{
    if (useRdma) {
        netDevCtx = netDevRdmaCtx_;
    } else {
        netDevCtx = netDevIpcCtx_;
    }
    return HCCL_SUCCESS;
}

HcclResult IHcclOneSidedService::DeInit() { return HCCL_SUCCESS; }

void IHcclOneSidedService::SetTCAndSL(u32 trafficClass, u32 serviceLevel)
{
    trafficClass_ = trafficClass;
    serviceLevel_ = serviceLevel;
}

} // namespace hccl
