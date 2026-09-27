/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGINE_CTXS_H
#define ENGINE_CTXS_H

#include "hccl/hccl_res.h"
#include "independent_op_context_manager.h"
#include <functional>
#include <unordered_map>
#include <mutex>

namespace hccl {

class EngineCtxs {
public:
    EngineCtxs();
    ~EngineCtxs();
    HcclResult CreateCommEngineCtx(const std::string& tag, CommEngine engine, uint64_t size, void** ctx);
    HcclResult GetCommEngineCtx(const std::string& tag, CommEngine engine, void** ctx, uint64_t* size);
    HcclResult CopyCommEngineCtx(
        const std::string& tag, CommEngine engine, const void* srcCtx, uint64_t size, uint64_t dstCtxOffset);
    HcclResult DestroyEngineCtx(const std::string& tag, CommEngine engine);

private:
    std::unordered_map<std::string, std::unordered_map<CommEngine, HcclMem, CommEngineHash>> contextMap_;
    std::mutex mutex_;
};
} // namespace hccl
#endif // ENGINE_CTXS_H
