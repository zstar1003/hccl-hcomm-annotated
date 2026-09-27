/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_ENGINE_RES_H
#define COMM_ENGINE_RES_H

#include <memory>
#include <vector>
#include "threads/thread.h"
#include "engine_ctxs.h"

namespace hcomm {
/**
 * @note 职责：管理同一种通信引擎下的不同资源
 */
class CommEngineRes {
public:
    explicit CommEngineRes(CommEngineType engineType);
    ~CommEngineRes();

    // 申请线程资源
    HcclResult
    AllocateThreads(uint32_t threadNum, uint32_t notifyNumPerThread, std::vector<ThreadHandle>& threadHandles);

    // 释放线程资源
    HcclResult ReleaseThreads(const std::vector<ThreadHandle>& threadHandles);

    // 获取引擎上下文
    HcclResult AcquireEngineCtx(OpTag opTag, uint32_t ctxSize, EngineCtx** engineCtx, bool* newCreated);

    // 释放引擎上下文
    HcclResult ReleaseEngineCtx(EngineCtx* engineCtx);

    // 获取引擎类型
    CommEngineType GetEngineType() const { return engineType_; }

private:
    CommEngineType engineType_{};
    std::vector<std::shared_ptr<Thread>> threads_{};
    std::vector<std::unique_ptr<EngineCtx>> engineCtxs_{};
};
} // namespace hcomm

#endif // COMM_ENGINE_RES_H
