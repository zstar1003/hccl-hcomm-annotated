/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_graph_builder_bridge.h"

#include "log.h"

namespace Hccl {
namespace {
    // 槽位属于 hccl_v2；使用函数内静态保证 hcomm 装载期注册时不存在跨 SO 初始化顺序依赖。
    RankGraphBuilderBridge& BridgeSlot()
    {
        static RankGraphBuilderBridge bridge{};
        return bridge;
    }
} // namespace

HcclResult RegisterRankGraphBuilderBridge(const RankGraphBuilderBridge& bridge)
{
    // 构建、恢复和所有权接管回调必须同时发布，确保返回对象始终能由 hcomm 侧正确释放。
    if (bridge.buildFromString == nullptr || bridge.buildFromRankTable == nullptr || bridge.recoverBuild == nullptr
        || bridge.adoptRankGraph == nullptr) {
        return HCCL_E_PTR;
    }
    auto& registeredBridge = BridgeSlot();
    // bridge 在动态库装载期完成一次注册，运行阶段不允许替换回调集合。
    if (registeredBridge.buildFromString != nullptr || registeredBridge.buildFromRankTable != nullptr
        || registeredBridge.recoverBuild != nullptr || registeredBridge.adoptRankGraph != nullptr) {
        HCCL_ERROR("[%s] RankGraphBuilder bridge has already been registered.", __func__);
        return HCCL_E_INTERNAL;
    }
    registeredBridge = bridge;
    return HCCL_SUCCESS;
}

const RankGraphBuilderBridge* GetRankGraphBuilderBridge()
{
    const auto& bridge = BridgeSlot();
    // 不向调用方暴露不完整回调表，避免构建成功后缺少匹配的销毁路径。
    if (bridge.buildFromString == nullptr || bridge.buildFromRankTable == nullptr || bridge.recoverBuild == nullptr
        || bridge.adoptRankGraph == nullptr) {
        return nullptr;
    }
    return &bridge;
}
} // namespace Hccl
