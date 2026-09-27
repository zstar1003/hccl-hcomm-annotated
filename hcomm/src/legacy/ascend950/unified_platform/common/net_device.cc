/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_device.h"

namespace Hccl {
HcclNetDevice::HcclNetDevice(const NetDevInfo& info)
{
    netDevInfo_ = info;
    HCCL_DEBUG("HcclNetDevice created: %p", this);
}

HcclNetDevice::~HcclNetDevice() { HCCL_DEBUG("HcclNetDevice destroyed: %p", this); }

NetDevInfo HcclNetDevice::GetNetDevInfo() const { return netDevInfo_; }
InnerNetDev* HcclNetDevice::GetInnerNetDev() const { return ndev_; }
void HcclNetDevice::SetInnerNetDev(InnerNetDev* value) { ndev_ = value; }

RdmaHandle HcclNetDevice::GetRdmaHandle() const
{
    if (ndev_) {
        return ndev_->getRdmaHandle();
    }
    return nullptr;
}

std::pair<TokenIdHandle, uint32_t> HcclNetDevice::GetTokenIdInfo(const BufferKey<uintptr_t, u64>& bufKey) const
{
    if (ndev_) {
        return ndev_->getTokenIdInfo(bufKey);
    }
    return std::pair<TokenIdHandle, uint32_t>{};
}

void HcclNetDevice::PutTokenIdInfo(const BufferKey<uintptr_t, u64>& bufKey, TokenIdHandle tokenIdHandle) const
{
    if (ndev_) {
        ndev_->putTokenIdInfo(bufKey, tokenIdHandle);
    }
}

bool HcclNetDevice::IsUB() { return netDevInfo_.protoType == LinkProtoType::UB; }

} // namespace Hccl
