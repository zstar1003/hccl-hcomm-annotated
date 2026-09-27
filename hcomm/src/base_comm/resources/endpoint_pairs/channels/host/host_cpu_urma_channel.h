/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HOST_CPU_URMA_CHANNEL_H
#define HOST_CPU_URMA_CHANNEL_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include "../channel.h"
#include "urma_types.h"

// Orion
#include "../../../../../../src/legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "ub_mem_transport.h"
#include "host_ub_connection.h"
#include "ub_local_notify.h"
#include "aicpu_res_package_helper.h"
#include "hcomm_adapter_hccp.h"

namespace hcomm {

class HostCpuUrmaChannel : public Channel {
public:
    HostCpuUrmaChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc);
    ~HostCpuUrmaChannel() override;

    HcclResult Init() override;
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    const HcommChannelDesc& GetChannelDesc() const override { return channelDesc_; }

    // 数据面接口
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override;
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override;
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override;
    HcclResult Write(void* dst, const void* src, uint64_t len) override;
    HcclResult Read(void* dst, const void* src, uint64_t len) override;
    HcclResult ChannelFence() override;
    HcclResult ChannelDrain() override;

    HcclResult Clean() override;
    HcclResult Resume() override;

private:
    HcclResult ParseInputParam();
    HcclResult StartListen();
    HcclResult BuildSocket();
    HcclResult BuildConnection();
    HcclResult BuildUbMemTransport();
    HcclResult GetLocSeg(const void* addr, const size_t size, u64* seg);
    HcclResult UrmaPostJettySendWr(urma_opcode_t opcode, void* dst, const void* src, uint64_t len);
    HcclResult GetSplitNum(uint64_t len, uint64_t maxJettyWrDataLen, uint64_t& splitNum) const;
    HcclResult
    GetLocalAndRemoteSeg(urma_opcode_t opcode, void* dst, const void* src, uint64_t len, u64& localSeg, u64& remoteSeg);
    HcclResult WaitForWqeCompletion();

private:
    // --------------------- 入参 ---------------------
    EndpointHandle endpointHandle_;
    HcommChannelDesc channelDesc_;

    // --------------------- 转换参数 ---------------------
    EndpointDesc localEp_{};
    EndpointDesc remoteEp_{};

    // --------------------- 具体成员 ---------------------
    Hccl::Socket* socket_{nullptr};
    const Hccl::SocketConfig* socketConfig_{nullptr};
    RdmaHandle rdmaHandle_{nullptr};
    std::unique_ptr<Hccl::UbMemTransport> memTransport_{nullptr};
    Hccl::BaseMemTransport::Attribution attr_{};
    Hccl::BaseMemTransport::CommonLocRes commonRes_{};
    std::vector<std::unique_ptr<Hccl::HostUbConnection>> connections_{};
    std::unique_ptr<Hccl::Socket> serverSocket_{nullptr};

    urma_jfc_t jfc_{};
    urma_jetty_t jetty_{};
    urma_target_jetty_t tjetty_{};
    uint32_t wqeNum_{0};
    bool fenceFlag_{false};

    std::mutex jfcMutex_;
    std::mutex fenceMutex_;
    bool Onetime_{false};
    DevBaseAttr devBaseAttr_{};
    uint32_t devicePhyId_{};
};

} // namespace hcomm

#endif // HOST_CPU_URMA_CHANNEL_H
