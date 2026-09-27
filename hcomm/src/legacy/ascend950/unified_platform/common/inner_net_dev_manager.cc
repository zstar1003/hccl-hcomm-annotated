/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inner_net_dev_manager.h"
#include "log.h"

namespace Hccl {
InnerNetDevManager& InnerNetDevManager::GetInstance()
{
    static InnerNetDevManager instance;
    return instance;
}

HcclResult InnerNetDevManager::AddDevice(const NetDevInfo& info, HcclNetDevice*& device)
{
    device = new (nothrow) HcclNetDevice(info);
    if (device == nullptr) {
        HCCL_ERROR("new HcclNetDevice fail, devId[%u]", info.devId);
        return HCCL_E_PTR;
    }
    std::unique_lock<std::shared_mutex> lock(mtx_);
    InnerNetDev* innerNetDev = nullptr;
    auto it = netDevMap_.find(info);
    if (it == netDevMap_.end()) {
        innerNetDev = new (nothrow) InnerNetDev(info);
        if (innerNetDev == nullptr || !innerNetDev->GetIsValid()) {
            HCCL_ERROR("new InnerNetDev fail, devId[%u]", info.devId);
            if (innerNetDev != nullptr) {
                delete innerNetDev;
                innerNetDev = nullptr;
            }
            delete device;
            device = nullptr;
            return HCCL_E_PARA;
        }
        netDevMap_.insert(std::make_pair(info, std::unique_ptr<InnerNetDev>(innerNetDev)));
        netDevCnt_[info] = 1;
    } else {
        innerNetDev = it->second.get();
        netDevCnt_[info]++;
    }
    device->SetInnerNetDev(innerNetDev);
    return HCCL_SUCCESS;
}

HcclResult InnerNetDevManager::RemoveDevice(const NetDevInfo& info)
{
    std::unique_lock<std::shared_mutex> lock(mtx_);
    auto cntIt = netDevCnt_.find(info);
    if (cntIt == netDevCnt_.end()) {
        HCCL_ERROR("find HcclNetDevice fail, devId[%u]", info.devId);
        return HCCL_E_PTR;
    }

    cntIt->second--;
    if (cntIt->second == 0) {
        netDevMap_.erase(info);
        netDevCnt_.erase(cntIt);
    }
    return HCCL_SUCCESS;
}

HcclResult InnerNetDevManager::DeleteDevice(Hccl::HcclNetDevice* device)
{
    if (device == nullptr) {
        return HCCL_SUCCESS;
    }
    HcclResult ret = RemoveDevice(device->GetNetDevInfo());
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("delete netDev fail, devId[%u]", device->GetNetDevInfo().devId);
        return HCCL_E_PARA;
    }
    delete device;
    device = nullptr;
    return HCCL_SUCCESS;
}
} // namespace Hccl
