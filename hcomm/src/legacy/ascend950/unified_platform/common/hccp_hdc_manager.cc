/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_hdc_manager.h"
#include "orion_adapter_tsd.h"
#include "orion_adapter_rts.h"

namespace Hccl {

HccpHdcManager& HccpHdcManager::GetInstance()
{
    static HccpHdcManager hccpHdcManager;
    return hccpHdcManager;
}

void HccpHdcManager::Init(u32 deviceLogicId)
{
    std::unique_lock<std::recursive_mutex> lock(managerMutex);
    if (instances.count(deviceLogicId) != 0) {
        return;
    }

    HrtOpenTsdProcess(deviceLogicId);

    HRaInitConfig cfg;
    cfg.phyId = HrtGetDevicePhyIdByUserDevId(deviceLogicId);
    cfg.mode = HrtNetworkMode::HDC;
    HrtRaInit(cfg);

    instances.insert(deviceLogicId);
}

void HccpHdcManager::DeInit(u32 deviceLogicId)
{
    if (destroyed.load()) {
        return;
    }
    HCCL_INFO("[HccpHdcManager::%s] deviceLogicId [%u]", __func__, deviceLogicId);

    std::lock_guard<std::recursive_mutex> lock(managerMutex);
    if (instances.count(deviceLogicId) == 0) {
        HCCL_WARNING("[HccpHdcManager::%s] deviceLogicId[%u] not ra init", __func__, deviceLogicId);
        return;
    }
    instances.erase(deviceLogicId);

    HRaInitConfig cfg;
    cfg.phyId = HrtGetDevicePhyIdByUserDevId(deviceLogicId);
    cfg.mode = HrtNetworkMode::HDC;
    DECTOR_TRY_CATCH("HccpHdcManager", HrtRaDeInit(cfg));
    HCCL_INFO("[HccpHdcManager::%s] devLogicId [%u] ra deinit success.", __func__, deviceLogicId);
}

void HccpHdcManager::DestroyAll()
{
    if (destroyed.load()) {
        return;
    }
    destroyed.store(true);

    // 析构前先注销reset device的回调，防止嵌套调用
    UnregisterDeviceResetCallback();

    std::lock_guard<std::recursive_mutex> lock(managerMutex);
    for (auto deviceLogicId : instances) {
        HCCL_INFO("HccpHdcManager deinit");

        HRaInitConfig cfg;
        cfg.phyId = HrtGetDevicePhyIdByUserDevId(deviceLogicId);
        cfg.mode = HrtNetworkMode::HDC;
        DECTOR_TRY_CATCH("HccpHdcManager", HrtRaDeInit(cfg));
        DECTOR_TRY_CATCH("HccpHdcManager", HrtResetDevice(deviceLogicId));
    }
    instances.clear();
}

void HccpHdcManager::UnregisterDeviceResetCallback() const
{
    aclError ret = aclrtRegDeviceStateCallback("hcomm_res_mgr", nullptr, nullptr);
    if (ret != ACL_SUCCESS) {
        HCCL_WARNING("[HccpHdcManager] aclrtRegDeviceStateCallback unregister failed, ret[%d]", ret);
        return;
    }
    HCCL_INFO("[HccpHdcManager] aclrtRegDeviceStateCallback unregister success");
}

HccpHdcManager::~HccpHdcManager() { DECTOR_TRY_CATCH("HccpHdcManager", DestroyAll()); }

} // namespace Hccl
