/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ORION_ADPT_UTILS_H
#define ORION_ADPT_UTILS_H

#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "hccl_net_dev_defs.h"
#include "hccp_ctx.h"

// Orion
#include "ip_address.h"
#include "virtual_topo.h"

namespace hcomm {

constexpr u32 UB_SQ_DEPTH_MIN = 16U;

HcclResult CommAddrToIpAddress(const CommAddr& commAddr, Hccl::IpAddress& ipAddr);
HcclResult IpAddressToCommAddr(const Hccl::IpAddress& ipAddr, CommAddr& commAddr);
HcclResult CommProtocolToLinkProtocol(CommProtocol commProtocol, Hccl::LinkProtocol& linkProtocol);
HcclResult CommAddrTypeToHcclAddressType(CommAddrType commAddrType, HcclAddressType& hcclAddressType);
Hccl::LinkData BuildDefaultLinkData();
HcclResult EndpointDescPairToLinkData(
    const EndpointDesc& locEp, const EndpointDesc& rmtEp, Hccl::LinkData& linkData, u32 reuseIdx = 0);
HcclResult EndpointDescPairToLinkDataWithRankIds(
    const uint32_t myRank, const uint32_t rmtRank, const EndpointDesc& locEp, const EndpointDesc& rmtEp,
    Hccl::LinkData& linkData, uint32_t devicePhyId, uint32_t remoteDevicePhyId, u32 reuseIdx = 0);

struct UbConnBuildContext {
    Hccl::LinkProtocol protocol{};
    Hccl::IpAddress locAddr{};
    Hccl::IpAddress rmtAddr{};
    s32 deviceLogicId{0};
    u8 qosPre{0};
    u32 sqDepth{0xFFFFFFFFU};
};

HcclResult PrepareUbConnBuildContext(
    const EndpointDesc& locEp, const EndpointDesc& rmtEp, const HcommChannelDesc& channelDesc, UbConnBuildContext& ctx);

HcclResult CheckUbSqDepth(const UbConnBuildContext& ctx, const DevBaseAttr& devBaseAttr);

} // namespace hcomm

#endif // ORION_ADPT_UTILS_H
