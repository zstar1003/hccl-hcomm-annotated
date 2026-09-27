/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "root_info_detect_bridge.h"

#include "log.h"

namespace Hccl {
namespace {
    // 槽位位于 hccl_v2，hcomm 的 registrar 在装载期回调这里。
    // 函数内静态可确保跨 SO 注册发生时槽位已经完成构造。
    RootInfoDetectBridge& BridgeSlot()
    {
        static RootInfoDetectBridge bridge{};
        return bridge;
    }
} // namespace

HcclResult RegisterRootInfoDetectBridge(const RootInfoDetectBridge& bridge)
{
    // 只发布完整回调表，避免兼容入口读取到部分可用的 bridge。
    if (bridge.getRootInfo == nullptr || bridge.detectRankTable == nullptr) {
        return HCCL_E_PTR;
    }
    auto& registeredBridge = BridgeSlot();
    // bridge 在装载期固定，禁止运行过程中替换 provider 回调。
    if (registeredBridge.getRootInfo != nullptr || registeredBridge.detectRankTable != nullptr) {
        HCCL_ERROR("[%s] RootInfoDetect bridge has already been registered.", __func__);
        return HCCL_E_INTERNAL;
    }
    registeredBridge = bridge;
    return HCCL_SUCCESS;
}

const RootInfoDetectBridge* GetRootInfoDetectBridge()
{
    const auto& bridge = BridgeSlot();
    // 未完成注册时统一返回 nullptr，由 hccl_v2 兼容入口转换为明确错误。
    if (bridge.getRootInfo == nullptr || bridge.detectRankTable == nullptr) {
        return nullptr;
    }
    return &bridge;
}
} // namespace Hccl
