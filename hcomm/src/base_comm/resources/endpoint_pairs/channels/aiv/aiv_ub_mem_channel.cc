/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_ub_mem_channel.h"
#include "endpoint.h"

namespace hcomm {

AivUbMemChannel::AivUbMemChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc)
{}

HcclResult AivUbMemChannel::ParseInputParam()
{
    Endpoint* localEpPtr = static_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    localEp_ = localEpPtr->GetEndpointDesc();
    socket_ = static_cast<Hccl::Socket*>(channelDesc_.socket);
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::BuildTransport()
{
    EXCEPTION_CATCH(transport_ = std::make_unique<AivUbMemTransport>(socket_, channelDesc_), return HCCL_E_PTR);
    CHK_PTR_NULL(transport_);
    CHK_RET(transport_->Init());
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::Init()
{
    // TODO: 处理抛异常
    CHK_RET(ParseInputParam());
    CHK_RET(BuildTransport());
    return HCCL_SUCCESS;
}

ChannelStatus AivUbMemChannel::GetStatus()
{
    return Channel::TransportStatusToChannelStatus(transport_->GetStatus(), localEp_, GetChannelDesc());
}

HcclResult AivUbMemChannel::GetNotifyNum([[maybe_unused]] uint32_t* notifyNum) const
{
    HCCL_INFO("AivUbMemChannel GetNotifyNum is not supported.");
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    return transport_->GetRemoteMems(memNum, remoteMem, memInfos);
}

HcclResult AivUbMemChannel::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    return transport_->UpdateMemInfo(memHandles, memHandleNum);
}

HcclResult AivUbMemChannel::Clean()
{
    // 该模式当前不支持N秒快恢
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::Resume()
{
    // 该模式当前不支持N秒快恢
    return HCCL_SUCCESS;
}

HcclResult AivUbMemChannel::NotifyRecord([[maybe_unused]] const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AivUbMemChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AivUbMemChannel::NotifyWait([[maybe_unused]] const uint32_t localNotifyIdx, [[maybe_unused]] const uint32_t timeout)
{
    HCCL_INFO("[AivUbMemChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AivUbMemChannel::WriteWithNotify(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] const uint64_t len,
    [[maybe_unused]] uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AivUbMemChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AivUbMemChannel::Write([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AivUbMemChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AivUbMemChannel::Read([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AivUbMemChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AivUbMemChannel::ChannelFence()
{
    HCCL_INFO("[AivUbMemChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}
} // namespace hcomm
