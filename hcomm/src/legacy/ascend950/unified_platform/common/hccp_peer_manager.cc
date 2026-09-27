/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_peer_manager.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"

namespace Hccl {

HccpPeerManager& HccpPeerManager::GetInstance()
{
    static HccpPeerManager hccpPeerManager;
    return hccpPeerManager;
}

void HccpPeerManager::Init(s32 deviceLogicId)
{
    std::lock_guard<std::mutex> lock(managerMutex_);

    if (instances_.count(deviceLogicId) != 0) {
        instances_[deviceLogicId].Ref();
        HCCL_INFO(
            "[HccpPeerManager::%s] deviceLogicId[%d] ra has initialized, ref[%d].", __func__, deviceLogicId,
            instances_[deviceLogicId].Count());
        return;
    }
    HRaInitConfig cfg;
    cfg.phyId = HrtGetDevicePhyIdByUserDevId(deviceLogicId);
    cfg.mode = HrtNetworkMode::PEER;
    HrtRaInit(cfg);

    instances_[deviceLogicId].Ref();
    HCCL_INFO("[HccpPeerManager::%s] deviceLogicId[%d] ra init success.", __func__, deviceLogicId);
}

void HccpPeerManager::DeInit(s32 deviceLogicId)
{
    std::lock_guard<std::mutex> lock(managerMutex_);

    if (isDestroy) {
        HCCL_WARNING("[HccpPeerManager::%s] HccpPeerManager has been destroyed", __func__);
        return;
    }

    // 校验是否存在
    if (instances_.count(deviceLogicId) == 0) {
        HCCL_WARNING("[HccpPeerManager::%s] deviceLogicId[%d] not ra init", __func__, deviceLogicId);
        return;
    }

    // 引用计数-1
    instances_[deviceLogicId].Unref();
    u32 count = instances_[deviceLogicId].Count();
    HCCL_INFO("[HccpPeerManager::%s] devLogicId[%d] release one, ref[%u].", __func__, deviceLogicId, count);

    // 若引用计数为0, 则释放资源
    if (count == 0) {
        HRaInitConfig cfg;
        cfg.phyId = HrtGetDevicePhyIdByUserDevId(deviceLogicId);
        cfg.mode = HrtNetworkMode::PEER;
        HrtRaDeInit(cfg);
        instances_.erase(deviceLogicId);
        HCCL_INFO("[HccpPeerManager::%s] devLogicId [%d] ra deinit success.", __func__, deviceLogicId);
    }
}

void HccpPeerManager::DeInitAll()
{
    std::lock_guard<std::mutex> lock(managerMutex_);
    isDestroy = true;

    for (auto const& instance : instances_) {
        u32 count = instance.second.Count();
        CHK_PRT_CONT(
            count != 0, HCCL_WARNING(
                            "[HccpPeerManager::%s] release is not as expected, "
                            "devLogicId[%d] ref[%u]",
                            __func__, instance.first, count));
        HRaInitConfig cfg;
        cfg.phyId = HrtGetDevicePhyIdByUserDevId(instance.first);
        cfg.mode = HrtNetworkMode::PEER;
        HrtRaDeInit(cfg);
        HCCL_INFO("[HccpPeerManager::%s] devLogicId [%d] ra deinit success.", __func__, instance.first);
    }

    instances_.clear();
}

HccpPeerManager::~HccpPeerManager() { DECTOR_TRY_CATCH("HccpPeerManager", DeInitAll()); }
} // namespace Hccl
