/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_P2P_CHANNEL_H
#define AICPU_TS_P2P_CHANNEL_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../channel.h"
#include "aicpu_ts_channel_helper.h"

#include "../../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "p2p_transport.h"
#include "p2p_connection.h"
#include "ipc_local_notify.h"
#include "aicpu_res_package_helper.h"

namespace hcomm {

class AicpuTsP2pChannel : public Channel {
public:
    AicpuTsP2pChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc);
    ~AicpuTsP2pChannel() override;

    HcclResult Init() override;
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    HcommChannelKind GetChannelKind() const override;
    HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum) override;
    const HcommChannelDesc& GetChannelDesc() const override { return channelDesc_; }
    HcclResult H2DResPack(std::vector<char>& buffer);

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
    HcclResult SetModuleDataName(Hccl::ModuleData& module, const std::string& name) const;
    HcclResult ParseInputParam();
    HcclResult BuildAttr();
    HcclResult BuildConnection();
    HcclResult BuildNotify();
    HcclResult BuildP2pMemTransport();
    HcclResult BuildSocket();

    HcclResult PackOpData(std::vector<char>& data);

private:
    EndpointHandle endpointHandle_;
    HcommChannelDesc channelDesc_;

    EndpointDesc localEp_{};
    EndpointDesc remoteEp_{};
    uint32_t notifyNum_{0};

    Hccl::Socket* socket_{nullptr};
    std::unique_ptr<Hccl::P2PTransport> memTransport_{nullptr};
    Hccl::BaseMemTransport::Attribution attr_{};
    Hccl::BaseMemTransport::CommonLocRes commonRes_{};
    std::vector<std::unique_ptr<Hccl::P2PConnection>> connections_{};
    std::vector<std::unique_ptr<Hccl::IpcLocalNotify>> localNotifies_{};
    std::unique_ptr<Hccl::Socket> serverSocket_;
    const Hccl::SocketConfig* socketConfig_{nullptr};
    uint32_t devicePhyId_{};
};

} // namespace hcomm

#endif // AICPU_TS_P2P_CHANNEL_H
