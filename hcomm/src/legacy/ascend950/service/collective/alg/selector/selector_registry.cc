/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "selector_registry.h"

#include <string>
#include <iostream>
#include <map>
#include <mutex>
#include <new>

namespace Hccl {

SelectorRegistry* SelectorRegistry::Global()
{
    static SelectorRegistry* globalSelectorRegistry = new (std::nothrow) SelectorRegistry;
    return globalSelectorRegistry;
}

HcclResult SelectorRegistry::Register(u32 priority, BaseSelector* selector)
{
    if (selector == nullptr) {
        HCCL_ERROR("[Algo][Selector] selector is null, register priority %u failed.", priority);
        return HcclResult::HCCL_E_MEMORY;
    }
    const std::lock_guard<std::mutex> lock(mu_);
    if (impls_.count(priority) != 0) {
        HCCL_ERROR("[Algo][Selector] priority %u already registered.", priority);
        return HcclResult::HCCL_E_PARA;
    }

    impls_[priority] = selector;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SelectorRegistry::RegisterByOpType(const OpType opType, u32 priority, BaseSelector* selector)
{
    if (selector == nullptr) {
        HCCL_ERROR("[Algo][Selector] selector is null, register opType %d priority %u failed.", opType, priority);
        return HcclResult::HCCL_E_MEMORY;
    }
    const std::lock_guard<std::mutex> lock(mu_);
    if (opTypeImpls_[opType].count(priority) != 0) {
        HCCL_ERROR("[Algo][Selector] opType %d priority %u already registered.", opType, priority);
        return HcclResult::HCCL_E_PARA;
    }
    opTypeImpls_[opType][priority] = selector;
    return HcclResult::HCCL_SUCCESS;
}

std::map<u32, BaseSelector*> SelectorRegistry::GetSelectorsByOpType(const OpType opType)
{
    if (opTypeImpls_.count(opType) == 0) {
        HCCL_WARNING("[Algo][Selector] opType %d has no selector registered.", opType);
        return {};
    }
    return opTypeImpls_[opType];
}

std::map<u32, BaseSelector*> SelectorRegistry::GetAllSelectors() { return impls_; }

} // namespace Hccl
