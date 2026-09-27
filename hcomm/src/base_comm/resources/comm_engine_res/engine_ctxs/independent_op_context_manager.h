/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDEPENDENT_OP_CONTEXT_MANAGER_H
#define INDEPENDENT_OP_CONTEXT_MANAGER_H

#include "hccl/hccl_res.h"
#include "hccl_mem_defs.h"
#include <functional>
#include <unordered_map>
#include <mutex>
#include <string>

namespace hccl {
struct HcclMemHash {
    static constexpr size_t MEM_ADDR_SHIFT = 1;
    static constexpr size_t MEM_SIZE_SHIFT = 2;
    size_t operator()(const HcclMem& mem) const noexcept
    {
        size_t hashMemType = std::hash<uint32_t>{}(static_cast<int>(mem.type));
        size_t hashMemaddr = std::hash<void*>{}(mem.addr);
        size_t hashMemSize = std::hash<uint64_t>{}(mem.size);
        return hashMemType ^ (hashMemaddr << MEM_ADDR_SHIFT) ^ (hashMemSize << MEM_SIZE_SHIFT);
    }
};

struct CommEngineHash {
    size_t operator()(const CommEngine& engine) const noexcept { return std::hash<int>{}(static_cast<int>(engine)); }
};

class ContextManager {
public:
    ContextManager();
    ~ContextManager();
    HcclResult CreateCommEngineCtx(const std::string& tag, CommEngine engine, uint64_t size, void** ctx);
    HcclResult GetCommEngineCtx(const std::string& tag, CommEngine engine, void** ctx, uint64_t* size);
    HcclResult CopyCommEngineCtx(
        const std::string& tag, CommEngine engine, const void* srcCtx, uint64_t size, uint64_t dstCtxOffset);
    HcclResult DestroyCommEngineCtx(const std::string& tag, CommEngine engine);

private:
    std::unordered_map<std::string, std::unordered_map<CommEngine, HcclMem, CommEngineHash>> contextMap_;
    std::mutex mutex_;
};
} // namespace hccl
#endif // INDEPENDENT_OP_CONTEXT_MANAGER_H
