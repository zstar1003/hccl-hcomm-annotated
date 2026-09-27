/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_nic_plugin.h"

#include "cpu_urma_endpoint.h"
#include "host_cpu_urma_channel.h"
#include "nic_plugin_ops.h"

namespace {

struct HostUbPluginTraits {
    using EndpointT = hcomm_experimental::CpuUrmaEndpoint;
    using ChannelT = hcomm_experimental::HostCpuUrmaChannel;

    static const char* LogTag() { return "HostUbPlugin"; }

    static HcommResult GetUserRemoteMem(ChannelT* channel, CommMem** remoteMem, char*** memTags, uint32_t* memNum)
    {
        CHK_PTR_NULL(channel);
        return channel->GetUserRemoteMem(remoteMem, memTags, memNum);
    }
};

using PluginOps = hcomm_experimental::NicPluginOps<HostUbPluginTraits>;

const HcommNicPluginInfo kPluginInfo
    = {{HCOMM_NIC_PLUGIN_INFO_VERSION, HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD, sizeof(HcommNicPluginInfo), 0},
       "hcomm_cpu_ub",
       2U,
       // Only the first protocolCount entries are valid. Keep unused slots explicit to avoid HCCS zero-init.
       {COMM_PROTOCOL_UBC_TP, COMM_PROTOCOL_UB_CTP, COMM_PROTOCOL_RESERVED, COMM_PROTOCOL_RESERVED},
       {}};

HcommNicEndpointOps kEndpointOps = PluginOps::EndpointOps();
HcommNicChannelOps kChannelOps = PluginOps::ChannelOps();
} // namespace

HCOMM_EXPERIMENTAL_NIC_PLUGIN_EXPORTS(PluginOps)
