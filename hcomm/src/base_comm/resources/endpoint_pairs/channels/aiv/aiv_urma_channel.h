/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_URMA_CHANNEL_H
#define AIV_URMA_CHANNEL_H

#include <cstdint>
#include <memory>
#include <vector>
#include "../channel.h"

// Orion
#include "aiv_urma_transport.h"
#include "buffer_key.h"
#include "resource/notify/ub_local_notify.h"
#include "resource/socket/socket.h"
#include "hcomm_adapter_hccp.h"

namespace hcomm {

struct UbConnBuildContext;

class AivUrmaChannel : public Channel {
public:
    AivUrmaChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc);
    ~AivUrmaChannel() override;

    HcclResult Init() override;
    ChannelStatus GetStatus() override;

    HcclResult Clean() override;
    HcclResult Resume() override;
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum) override;
    const HcommChannelDesc& GetChannelDesc() const override { return channelDesc_; }
    HcclResult BuildChannelEntityToDevice(void** devChannelPtr);
    HcclResult PreAllocChannelEntityToDevice(void** devChannelPtr);
    HcclResult FillChannelEntityToDevice();

    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override;
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override;
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override;
    HcclResult Write(void* dst, const void* src, uint64_t len) override;
    HcclResult Read(void* dst, const void* src, uint64_t len) override;
    HcclResult ChannelFence() override;

private:
    HcclResult ParseInputParam();
    HcclResult StartListen();
    HcclResult BuildSocket();
    HcclResult BuildAttr();
    HcclResult BuildConnection();
    HcclResult BuildAivUrmaTransport();
    void PutSocketIfNeeded();
    void ReleaseDeviceChannelEntity();

    HcclResult CreateUbConnectionByProtocol(
        const UbConnBuildContext& ctx, std::unique_ptr<Hccl::DevUbConnection>& ubConn,
        Hccl::DevUbConnection::JettyMode jettyMode = Hccl::DevUbConnection::JettyMode::SELF_CREATE);
    HcclResult AcquireSharedJettyInBuildConnection(const UbConnBuildContext& ctx, Hccl::DevUbConnection* connection);
    static std::unique_ptr<Hccl::DevUbConnection> CreateSharedJettyConnection(
        Hccl::RdmaHandle rdmaHandle, const Hccl::IpAddress& locAddr, const Hccl::IpAddress& rmtAddr, u8 qosPre,
        Hccl::LinkProtocol protocol, u8 taTimeOut, u32 sqDepth);

    // --------------------- 转换参数 ---------------------
    EndpointDesc localEp_{};
    EndpointDesc remoteEp_{};
    uint32_t notifyNum_{0};

    // --------------------- 入参 ---------------------
    EndpointHandle endpointHandle_;
    CommEngine engine_{COMM_ENGINE_RESERVED};
    HcommChannelDesc channelDesc_;

    // --------------------- 具体成员 ---------------------
    Hccl::Socket* socket_{nullptr};
    std::unique_ptr<Hccl::AivUrmaTransport> transport_{nullptr};
    Hccl::BaseMemTransport::Attribution attr_{};
    Hccl::BaseMemTransport::CommonLocRes commonRes_{};
    std::vector<std::unique_ptr<Hccl::DevUbConnection>> connections_{};
    RdmaHandle rdmaHandle_{nullptr};
    void* devChannelEntity_{nullptr};
    void* devChannelEntitySlab_{nullptr};
    size_t devChannelEntitySlabSize_{0};
    std::vector<hccl::DeviceMem> deviceMemories_{};
    DevBaseAttr devBaseAttr_{};
    std::unique_ptr<Hccl::SocketConfig> socketConfigHolder_{nullptr};
    const Hccl::SocketConfig* socketConfig_{nullptr};
    uint32_t devicePhyId_{};

    // 共享 jetty 模式下从 JettyContext::Ctx 取得的 PI/CI device 内存指针，
    // BuildChannelEntityToDevice 时绑给 transport，使同 endpoint 下多 channel 共用同一 PI/CI。
    void* sharedSqPiPtr_{nullptr};
    void* sharedSqCiPtr_{nullptr};
    void* sharedCqPiPtr_{nullptr};
    void* sharedCqCiPtr_{nullptr};
};

} // namespace hcomm

#endif // AIV_URMA_CHANNEL_H
