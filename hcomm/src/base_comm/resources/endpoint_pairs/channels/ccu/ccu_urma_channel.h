/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_URMA_CHANNEL_H
#define CCU_URMA_CHANNEL_H

#include <memory>
#include <vector>
#include <atomic>
#include <mutex>

#include "../channel.h"

#include "urma_endpoint.h"
#include "ccu_transport_.h"

namespace hcomm {

class CcuUrmaChannel : public Channel {
public:
    // 当前仅支持交换hccl buffer
    CcuUrmaChannel(const EndpointHandle locEndpointHandle, const HcommChannelDesc& channelDesc);
    ~CcuUrmaChannel() override;

    HcclResult Init() override;
    ChannelStatus GetStatus() override;

    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum) override;
    HcclResult CcuGetRmtMemToken(uint64_t srcVa, uint64_t& tokenInfo);

    HcclResult Clean() override;
    HcclResult Resume() override;

    HcclResult ChannelFence() override;
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override;
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override;
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override;
    HcclResult Write(void* dst, const void* src, uint64_t len) override;
    HcclResult Read(void* dst, const void* src, uint64_t len) override;

public:
    uint32_t GetDieId() const;
    uint32_t GetChannelId() const;
    HcclResult GetRmtSignalAddrByIndex(uint32_t index, uint64_t& rmtCkeAddr) const;
    HcclResult GetRmtCcuBufferTokenInfo(uint32_t& rmtTokenId, uint32_t& rmtTokenValue) const;
    HcclResult GetRmtVarAddrByIndex(uint32_t index, uint64_t& rmtXnAddr) const;
    HcclResult GetLocCkeByIndex(const uint32_t index, uint32_t& locCkeId) const;
    HcclResult GetLocXnByIndex(const uint32_t index, uint32_t& locXnId) const;

    HcclResult GetRmtCkeByIndex(const uint32_t index, uint32_t& rmtCkeId) const;
    HcclResult GetRmtXnByIndex(const uint32_t index, uint32_t& rmtXnId) const;

    HcclResult GetRmtBuffer(uint64_t& addr, uint32_t& size, uint32_t& tokenId, uint32_t& tokenValue) const;

    EndpointHandle GetlocEndPointHandle() { return locEndpointHandle_; }
    const HcommChannelDesc& GetChannelDesc() const override { return channelDesc_; }

private:
    ChannelStatus TryPrepareAndConstruct();

    std::atomic<bool> isFirstPrintChannelInfo_{true}; // 是否第一次打印通道建链信息，避免重复打印日志刷屏
    std::unique_ptr<CcuTransport> impl_{nullptr};
    ChannelStatus channelStatus_{ChannelStatus::INIT};
    // 保护 GetStatus 懒建链与状态读写：并发轮询同一 channel 时避免 impl_/channelStatus_ 竞态
    std::mutex statusMtx_{};
    EndpointHandle locEndpointHandle_{nullptr};
    HcommChannelDesc channelDesc_{};
    UrmaEndpoint* ccuEndpoint_{nullptr};
    Hccl::Socket* socket_{nullptr};
    std::vector<HcommMemHandle> memHandles_{};
};

} // namespace hcomm
#endif // CCU_URMA_CHANNEL_H
