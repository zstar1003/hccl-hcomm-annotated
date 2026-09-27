/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_UB_MEM_CHANNEL_H
#define AIV_UB_MEM_CHANNEL_H

#include <cstdint>
#include <memory>
#include "../channel.h"

// Orion
#include "../../../../../../src/legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "buffer_key.h"
#include "aiv_ub_mem_transport.h"

namespace hcomm {

class AivUbMemChannel : public Channel {
public:
    AivUbMemChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc);

    HcclResult Init() override;
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum) override;
    const HcommChannelDesc& GetChannelDesc() const override { return channelDesc_; }

    HcclResult Clean() override;
    HcclResult Resume() override;

    // 数据面接口
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override;
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override;
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override;
    HcclResult Write(void* dst, const void* src, uint64_t len) override;
    HcclResult Read(void* dst, const void* src, uint64_t len) override;
    HcclResult ChannelFence() override;

private:
    HcclResult ParseInputParam();
    HcclResult BuildTransport();

    // --------------------- 入参 ---------------------
    EndpointHandle endpointHandle_;
    HcommChannelDesc channelDesc_;
    EndpointDesc localEp_{}; // 本端 Endpoint 描述，用于状态日志打印

    // --------------------- 具体成员 ---------------------
    Hccl::Socket* socket_{nullptr};
    std::unique_ptr<AivUbMemTransport> transport_{nullptr};
};

} // namespace hcomm

#endif // AIV_UB_MEM_CHANNEL_H
