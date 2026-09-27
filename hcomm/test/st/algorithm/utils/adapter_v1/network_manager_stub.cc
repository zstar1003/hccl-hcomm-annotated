/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "network_manager_pub.h"

namespace hccl {

NetworkManager* NetworkManager::nmInstance[MAX_DEV_NUM] = {nullptr};
std::atomic<unsigned> NetworkManager::InitTool::initCount(0);

NetworkManager::InitTool::InitTool()
{
    if (initCount.load() == 0) {
        for (u32 i = 0; i < MAX_DEV_NUM; i++) {
            NetworkManager::nmInstance[i] = new NetworkManager;
        }
    }
    ++initCount;
}

NetworkManager::InitTool::~InitTool()
{
    --initCount;
    if (initCount.load() == 0) {
        for (u32 i = 0; i < MAX_DEV_NUM; i++) {
            if (NetworkManager::nmInstance[i] != nullptr) {
                delete NetworkManager::nmInstance[i];
                NetworkManager::nmInstance[i] = nullptr;
            }
        }
    }
}

NetworkManager::NetworkManager()
    : deviceLogicId_(INVALID_INT),
      devicePhyId_(INVALID_UINT),
      isHostUseDevNic_(false),
      notifyType_(NO_USE)
{}

NetworkManager::~NetworkManager() {}

} // namespace hccl
