/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "endpoint.h"
#include "channel.h"
#include "orion_adpt_utils.h"
#include "host_peer_ra_init.h"
#include <mutex>
#include <unordered_set>
#include "hccp.h"
#include "log.h"
#include "config_plf_log_v2.h"

namespace hcomm_experimental {

using Hccl::PLF_CHANNEL;

Endpoint::Endpoint(const EndpointDesc& endpointDesc) : endpointDesc_(endpointDesc) {}

HcclResult Endpoint::CreateEndpoint(const EndpointDesc&, std::unique_ptr<Endpoint>&) { return HCCL_E_NOT_SUPPORT; }

HcclResult Endpoint::CreateEndpointBase(const EndpointDesc&, std::unique_ptr<Endpoint>&) { return HCCL_E_NOT_SUPPORT; }

HcclResult Endpoint::GetAsyncEventsContext(uint32_t, struct AsyncEvent[], uint32_t&) { return HCCL_E_NOT_SUPPORT; }

HcclResult Channel::CreateChannel(EndpointHandle, CommEngine, HcommChannelDesc, std::shared_ptr<Channel>&)
{
    return HCCL_E_NOT_SUPPORT;
}

ChannelStatus Channel::TransportStatusToChannelStatus(
    Hccl::TransportStatus ts, const EndpointDesc& localEp, const HcommChannelDesc& channelDesc)
{
    if (Hccl::GetPlfDebugConfigValue() & PLF_CHANNEL) {
        Hccl::IpAddress localAddr{};
        std::string localEid = "invalid";
        if (hcomm::CommAddrToIpAddress(localEp.commAddr, localAddr) == HCCL_SUCCESS) {
            localEid = localAddr.Describe();
        }

        Hccl::IpAddress remoteAddr{};
        std::string remoteEid = "invalid";
        if (hcomm::CommAddrToIpAddress(channelDesc.remoteEndpoint.commAddr, remoteAddr) == HCCL_SUCCESS) {
            remoteEid = remoteAddr.Describe();
        }

        const char* socketTag = channelDesc.channelName != nullptr ? channelDesc.channelName : "anonymous";

        PLF_CONFIG_INFO(
            PLF_CHANNEL, "status[%d], localEid[%s], remoteEid[%s], socketTag[%s].", static_cast<int>(ts),
            localEid.c_str(), remoteEid.c_str(), socketTag);
    }
    switch (ts) {
        case Hccl::TransportStatus::INIT:
            return ChannelStatus::INIT;
        case Hccl::TransportStatus::SOCKET_OK:
            return ChannelStatus::SOCKET_OK;
        case Hccl::TransportStatus::SOCKET_TIMEOUT:
            return ChannelStatus::SOCKET_TIMEOUT;
        case Hccl::TransportStatus::READY:
            return ChannelStatus::READY;
        default:
            HCCL_ERROR("[Channel][%s] Invalid TransportStatus[%d]", __func__, ts);
            return ChannelStatus::FAILED;
    }
}

HcclResult Channel::GetUserRemoteMem(CommMem**, char***, uint32_t*) { return HCCL_SUCCESS; }

HcclResult Channel::UpdateMemInfo(HcommMemHandle*, uint32_t)
{
    HCCL_WARNING("[UpdateMemInfo] not support.");
    return HCCL_SUCCESS;
}

HcommChannelKind Channel::GetChannelKind() const { return channelKind_; }

HcclResult Channel::Serialize(std::shared_ptr<hccl::DeviceMem>& out)
{
    out.reset();
    return HCCL_E_NOT_SUPPORT;
}

void Channel::AddPtrArrayDevMem(std::shared_ptr<hccl::DeviceMem> ptrArrayMem)
{
    if (ptrArrayMem == nullptr || !(*ptrArrayMem)) {
        HCCL_WARNING("[Channel][%s] invalid ptrArrayMem.", __func__);
        return;
    }
    ptrArrayDevMems_.push_back(std::move(ptrArrayMem));
}

void Channel::ReleasePtrArrayDevMems() { ptrArrayDevMems_.clear(); }

HcclResult InitHostPeerRaOnce(uint32_t hostResourceId, const char* logPrefix)
{
    static std::mutex peerRaMutex;
    static std::unordered_set<uint32_t> initializedHostResources;

    std::lock_guard<std::mutex> lock(peerRaMutex);
    if (initializedHostResources.count(hostResourceId) != 0) {
        return HCCL_SUCCESS;
    }

    Hccl::HRaInitConfig cfg;
    cfg.phyId = hostResourceId;
    cfg.mode = Hccl::HrtNetworkMode::PEER;
    EXCEPTION_CATCH(Hccl::HrtRaInit(cfg), return HCCL_E_INTERNAL);
    initializedHostResources.insert(hostResourceId);
    HCCL_INFO(
        "[%s][%s] host peer RA init success, hostResourceId[%u].",
        logPrefix == nullptr ? "ExperimentalEndpoint" : logPrefix, __func__, hostResourceId);
    return HCCL_SUCCESS;
}
} // namespace hcomm_experimental
