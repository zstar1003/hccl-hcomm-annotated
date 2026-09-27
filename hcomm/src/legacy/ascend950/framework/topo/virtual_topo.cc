/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "virtual_topo.h"
#include <algorithm>
#include <string>
#include <unordered_map>
#include "internal_exception.h"

namespace Hccl {

LinkData::LinkData(vector<char>& data)
{
    BinaryStream binaryStream(data);
    u32 val;
    binaryStream >> val;
    type = static_cast<PortDeploymentType::Value>(val);
    binaryStream >> val;
    linkProtocol_ = static_cast<LinkProtocol::Value>(val);
    binaryStream >> val;
    direction = static_cast<LinkDirection::Value>(val);
    binaryStream >> localRankId_;
    binaryStream >> remoteRankId_;
    binaryStream >> localPortId_;
    binaryStream >> remotePortId_;
    binaryStream >> readable;
    binaryStream >> writable;
    binaryStream >> hop;
    binaryStream >> localDieId_;
    binaryStream >> portGroupSize;
    binaryStream >> fullmesh;

    u32 offset;
    u32 addrSize;
    binaryStream >> offset;
    binaryStream >> addrSize;

    std::vector<char> locAddr;
    if (data.begin() + offset >= data.end()) {
        THROW<InternalException>("[LinkData][LinkData]invalid offset[%u]", offset);
    }
    if (data.begin() + offset + addrSize >= data.end()) {
        THROW<InternalException>("[LinkData][LinkData]invalid addrSize[%u]", addrSize);
    }
    std::copy(data.begin() + offset, data.begin() + offset + addrSize, std::back_inserter(locAddr));
    std::vector<char> rmtAddr;
    std::copy(data.begin() + offset + addrSize, data.end(), std::back_inserter(rmtAddr));

    localAddr_ = IpAddress(locAddr);
    remoteAddr_ = IpAddress(rmtAddr);
}

std::vector<char> LinkData::GetUniqueId() const
{
    BinaryStream binaryStream;
    u32 val = static_cast<u32>(type);
    binaryStream << val;
    val = static_cast<u32>(linkProtocol_);
    binaryStream << val;
    val = static_cast<u32>(direction);
    binaryStream << val;
    binaryStream << localRankId_;
    binaryStream << remoteRankId_;
    binaryStream << localPortId_;
    binaryStream << remotePortId_;
    binaryStream << readable;
    binaryStream << writable;
    binaryStream << hop;
    binaryStream << localDieId_;
    binaryStream << portGroupSize;
    binaryStream << fullmesh;

    vector<char> result;
    binaryStream.Dump(result);

    auto loc = localAddr_.GetUniqueId();
    auto rmt = remoteAddr_.GetUniqueId();
    HCCL_INFO(
        "[LinkData::%s] localAddr[%s], remoteAddr[%s].", __func__, localAddr_.Describe().c_str(),
        remoteAddr_.Describe().c_str());

    u32 offset = result.size();
    u32 addrSize = loc.size();

    offset += sizeof(offset) + sizeof(addrSize);

    binaryStream << offset;
    binaryStream << addrSize;

    binaryStream.Dump(result);
    result.insert(result.end(), loc.begin(), loc.end());
    result.insert(result.end(), rmt.begin(), rmt.end());
    return result;
}

void LinkData::UpdateIpAddrWithPCIE()
{
#ifndef CCL_KERNEL_AICPU
    if (linkProtocol_ == LinkProtocol::PCIE) {
        // 更新localAddr和remoteAddr为vinc ip
        HrtRaSocketGetVnicIpInfos(localDeviceId_, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, localDeviceId_, localAddr_);
        HrtRaSocketGetVnicIpInfos(localDeviceId_, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, remoteDeviceId_, remoteAddr_);
    }
#endif
}

} // namespace Hccl
