/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine_ctx_mgr.h"

namespace hcomm {
EngineCtxMgr::~EngineCtxMgr()
{
    std::lock_guard<std::mutex> lock(mutex_);
    engineCtxs_.clear();
}

HcclResult EngineCtxMgr::AcquireEngineCtx(
    CommEngineType engineType, OpTag opTag, uint32_t ctxSize, EngineCtx** engineCtx, bool* newCreated)
{
    try {
        if (engineCtx == nullptr || newCreated == nullptr) {
            return HcclResult::HCCL_ERROR_INVALID_PARAM;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        std::string key = GenerateCtxKey(engineType, opTag);
        auto it = engineCtxs_.find(key);
        if (it != engineCtxs_.end()) {
            *engineCtx = it->second.get();
            *newCreated = false;
            return HCCL_SUCCESS;
        }

        // 创建新的引擎上下文
        auto newCtx = std::make_unique<EngineCtx>(opTag, engineType, ctxSize);
        if (newCtx == nullptr) {
            return HcclResult::HCCL_ERROR_OUT_OF_MEMORY;
        }

        *engineCtx = newCtx.get();
        *newCreated = true;
        engineCtxs_[key] = std::move(newCtx);

        return HCCL_SUCCESS;
    } catch (const HcclException& e) {
        return e.GetResult();
    } catch (...) {
        return HcclResult::HCCL_ERROR_COMMUNICATION;
    }
}

HcclResult EngineCtxMgr::ReleaseEngineCtx(EngineCtx* engineCtx)
{
    try {
        if (engineCtx == nullptr) {
            return HcclResult::HCCL_ERROR_INVALID_PARAM;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        for (auto it = engineCtxs_.begin(); it != engineCtxs_.end(); ++it) {
            if (it->second.get() == engineCtx) {
                engineCtxs_.erase(it);
                return HCCL_SUCCESS;
            }
        }

        return HcclResult::HCCL_ERROR_INVALID_PARAM;
    } catch (const HcclException& e) {
        return e.GetResult();
    } catch (...) {
        return HcclResult::HCCL_ERROR_COMMUNICATION;
    }
}

EngineCtx* EngineCtxMgr::FindEngineCtx(CommEngineType engineType, OpTag opTag)
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key = GenerateCtxKey(engineType, opTag);
    auto it = engineCtxs_.find(key);
    if (it != engineCtxs_.end()) {
        return it->second.get();
    }

    return nullptr;
}

std::string EngineCtxMgr::GenerateCtxKey(CommEngineType engineType, OpTag opTag)
{
    return std::to_string(static_cast<int>(engineType)) + "_" + std::to_string(opTag);
}
} // namespace hcomm
