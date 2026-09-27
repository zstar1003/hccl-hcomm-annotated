/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_ROCE_CHANNEL_H
#define AICPU_TS_ROCE_CHANNEL_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "../channel.h"
#include "aicpu_ts_channel_helper.h"
#include "hccl_dispatcher_ctx.h"
#include "hccl_socket.h"
#include "transport_pub.h"

struct HcommRoceChannelRes;

namespace hcomm {
class AicpuTsRoceEndpoint;

class AicpuTsRoceChannel : public Channel {
public:
    explicit AicpuTsRoceChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc);
    ~AicpuTsRoceChannel() override;

    HcclResult Init() override;
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    HcclResult Serialize(std::shared_ptr<hccl::DeviceMem>& out) override;
    HcommChannelKind GetChannelKind() const override;
    const HcommChannelDesc& GetChannelDesc() const override { return channelDesc_; }
    HcclResult Clean() override;
    HcclResult Resume() override;

    // 数据面接口
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override;
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override;
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override;
    HcclResult Write(void* dst, const void* src, uint64_t len) override;
    HcclResult Read(void* dst, const void* src, uint64_t len) override;
    HcclResult ChannelFence() override;

    AicpuTsChannelHelper* GetAicpuTsHelper() override { return &aicpuTsHelper_; }

private:
    AicpuTsChannelHelper aicpuTsHelper_;
    /** Owns res / local+remote RoceMemDetails arrays as separate device allocations for AICPU kernel blob. */
    struct AicpuTsRoceChannelMem {
        hccl::DeviceMem resAlloc{};
        hccl::DeviceMem localAlloc{};
        hccl::DeviceMem remoteAlloc{};
    };

    enum class RoceStatus {
        INIT,              // 初始状态
        SOCKET_CONNECTING, // Socket正在连接
        SOCKET_OK,         // Socket建立连接
        READY,             // Dispatcher + Transport初始化完成
        FAILED             // 建链失败
    };

    HcclResult ParseInputParam();
    HcclResult BuildDataSocket();
    HcclResult BuildClientDataSocket(
        HcclNetDevCtx netDevCtx, const hccl::HcclIpAddress& remoteIp, uint32_t port, const std::string& socketTag);
    HcclResult BuildServerDataSocket(
        AicpuTsRoceEndpoint* roceEp, const hccl::HcclIpAddress& remoteIp, uint32_t port, const std::string& socketTag);
    HcclResult BuildDispatcherAndTransport();
    HcclResult AssignDispatcherCommId();
    HcclResult EnsureDispatcherCtx(u32 devPhyId);
    HcclResult ConfigureMachineParaForTransport();
    void ConfigureTransportParaForRoce();
    HcclResult CreateAndInitTransport(HcclDispatcher dispatcher);
    HcclResult BuildSocketTagName(std::string& outTag) const;
    HcclResult ValidateSerializeParams(u32 qpNum, size_t localMemCount, size_t remoteMemCount) const;
    HcclResult InitSerializeRoceChannelRes(
        HcommRoceChannelRes& res, size_t localMemCount, size_t remoteMemCount, void* localMem, void* remoteMem,
        const std::vector<HcclQpInfoV2>& aiQpInfos, u32 qpNum) const;
    HcclResult BuildSerializeChannelMem(
        AicpuTsRoceChannelMem& bundle, const std::vector<RoceMemDetails>& localMd,
        const std::vector<RoceMemDetails>& remoteMd, const std::vector<HcclQpInfoV2>& aiQpInfos, u32 qpNum) const;

    const char* SocketRoleTag() const noexcept { return isLocalIpClient_ ? "client" : "server"; }
    HcclResult SerializeDrainNotifyInfo(HcommRoceChannelRes& res) const;

    EndpointHandle endpointHandle_{};
    HcommChannelDesc channelDesc_{};

    EndpointDesc localEp_{};
    EndpointDesc remoteEp_{};
    bool isLocalIpClient_{false};
    uint32_t notifyNum_{0};
    RdmaHandle rdmaHandle_{nullptr};

    std::shared_ptr<hccl::HcclSocket> dataSocket_{};
    std::string dispatcherCommId_{};
    DispatcherCtxPtr dispatcherCtx_{nullptr};
    bool ownsDispatcherCtx_{false};

    std::unique_ptr<hccl::NotifyPool> notifyPool_{};
    hccl::MachinePara machinePara_{};
    hccl::TransportPara transportPara_{};
    std::unique_ptr<hccl::Transport> transport_{};

    bool inited_{false};
    RoceStatus roceStatus_{RoceStatus::INIT};
};

} // namespace hcomm

#endif // AICPU_TS_ROCE_CHANNEL_H
