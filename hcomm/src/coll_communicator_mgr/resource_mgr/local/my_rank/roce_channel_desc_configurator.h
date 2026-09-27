/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROCE_CHANNEL_DESC_CONFIGURATOR_H
#define ROCE_CHANNEL_DESC_CONFIGURATOR_H

#include <string>
#include <vector>

#include "hccl/hccl_channel.h"

namespace hccl {

/**
 * @brief Prepares RoCE-specific parameters in HcommChannelDesc.
 *
 * The configurator owns the buffers referenced by HcommChannelDesc::roceAttr.srcPortList. Its lifetime must cover
 * channel creation so that the descriptors always reference valid storage.
 */
class RoceChannelDescConfigurator {
public:
    static constexpr std::size_t HOST_NIC_CONFIG_BUFFER_SIZE = 2048;

    explicit RoceChannelDescConfigurator(uint32_t channelNum);
    RoceChannelDescConfigurator(const RoceChannelDescConfigurator&) = delete;
    RoceChannelDescConfigurator& operator=(const RoceChannelDescConfigurator&) = delete;
    RoceChannelDescConfigurator(RoceChannelDescConfigurator&&) = delete;
    RoceChannelDescConfigurator& operator=(RoceChannelDescConfigurator&&) = delete;

    // 配置 RoCE 源端口，并保证 srcPortList 指向的内存在通道创建期间有效。
    HcclResult FillRoceSrcPortList(const HcclChannelDesc& hcclDesc, uint32_t channelIndex, HcommChannelDesc& hcommDesc);

    // 读取并校验 Host 网卡的多 QP 数量。
    static void ReadHostNicMultiQpCount(uint32_t& qpCount);

private:
    static bool ParseStrictDecimal(const std::string& value, uint32_t minValue, uint32_t maxValue, uint32_t& parsed);
    static void ReadHostNicMultiQpUdpPorts(std::vector<uint16_t>& qpUdpPorts);
    static void FillPortsFromHostRdmaUdpPortsList(std::vector<uint16_t>& ports);
    static HcclResult FillPortsFromMultiQpSrcPortConfig(const HcclChannelDesc& hcclDesc, std::vector<uint16_t>& ports);
    static HcclResult ResolveRoceSrcPorts(const HcclChannelDesc& hcclDesc, std::vector<uint16_t>& ports);

    std::vector<std::vector<uint16_t>> srcPortBuffers_;
};

} // namespace hccl

#endif // ROCE_CHANNEL_DESC_CONFIGURATOR_H
