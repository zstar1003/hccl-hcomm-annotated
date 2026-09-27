/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_tlv_hdc_manager.h"
#include "orion_adapter_rts.h"
#include "socket_handle_manager.h"

namespace Hccl {

HccpTlvHdcManager::HccpTlvHdcManager() { tlvHandleMap.resize(MAX_DEVICE_NUM); }

HccpTlvHdcManager& HccpTlvHdcManager::GetInstance()
{
    static HccpTlvHdcManager HccpTlvHdcManager;
    return HccpTlvHdcManager;
}

void* HccpTlvHdcManager::GetTlvHandle(s32 deviceLogicId)
{
    if (tlvHandleMap[deviceLogicId] == nullptr) {
        Init(deviceLogicId);
    }
    return tlvHandleMap[deviceLogicId];
}

void HccpTlvHdcManager::Init(s32 deviceLogicId)
{
    std::unique_lock<std::mutex> lock(managerMutex);
    if (instances.count(deviceLogicId) != 0) {
        return;
    }

    HRaTlvInitConfig cfg;
    cfg.phyId = HrtGetDevicePhyIdByUserDevId(deviceLogicId);
    cfg.mode = HrtNetworkMode::HDC;
    cfg.version = 1;
    tlvHandleMap[deviceLogicId] = HrtRaTlvInit(cfg);

    instances.insert(deviceLogicId);
}

void HccpTlvHdcManager::DestroyAll()
{
    std::unique_lock<std::mutex> lock(managerMutex);
    for (auto deviceLogicId : instances) {
        HCCL_INFO("HccpTlvHdcManager deinit");

        void* tlv_handle = tlvHandleMap[deviceLogicId];
        if (tlv_handle == nullptr) {
            continue;
        }
        HrtRaTlvDeInit(tlv_handle); // 要保证DestroyAll只有析构函数调用

        tlvHandleMap[deviceLogicId] = nullptr;
    }
    instances.clear();
}

HccpTlvHdcManager::~HccpTlvHdcManager() { DECTOR_TRY_CATCH("HccpTlvHdcManager", DestroyAll()); }

} // namespace Hccl
