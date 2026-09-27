/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGINE_CTX_MGR_H
#define ENGINE_CTX_MGR_H

#include <memory>
#include <unordered_map>
#include <mutex>

namespace hcomm {
/**
 * @note 职责：管理不同通信引擎下的内存Ctx, 比如，AicpuTs通信引擎内可以访问的内存。
 */
class EngineCtxMgr {
public:
    EngineCtxMgr() = default;
    ~EngineCtxMgr();

    // 获取引擎上下文
    HcclResult
    AcquireEngineCtx(CommEngineType engineType, OpTag opTag, uint32_t ctxSize, EngineCtx** engineCtx, bool* newCreated);

    // 释放引擎上下文
    HcclResult ReleaseEngineCtx(EngineCtx* engineCtx);

    // 查找引擎上下文
    EngineCtx* FindEngineCtx(CommEngineType engineType, OpTag opTag);

private:
    std::string GenerateCtxKey(CommEngineType engineType, OpTag opTag);

    std::unordered_map<std::string, void*> engineCtxs_{};
    std::mutex mutex_{};
};
} // namespace hcomm
#endif // ENGINE_CTX_MGR_H
