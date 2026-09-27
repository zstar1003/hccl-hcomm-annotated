/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_MEM_MANAGER_H
#define COMM_MEM_MANAGER_H
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include "hccl/hccl_res.h"
#include "hccl_independent_common.h"
#include "hccl_mem_defs.h"
#include "mem_device_pub.h"
#include "ccl_buffer_manager.h"
#include "rma_buffer_mgr.h"

namespace hccl {

struct HcclMemoryHandle {
    void* addr{nullptr};
    uint64_t size{0};
    HcclMemType memType{HCCL_MEM_TYPE_DEVICE};
    HcclRegMemAttr attr{};
};

class CommMemMgr {
public:
    using Handle = std::shared_ptr<HcclMemoryHandle>;
    using MemKey = hccl::BufferKey<uintptr_t, uint64_t>;
    using Table = hcomm::RmaBufferMgr<MemKey, Handle>;
    CommMemMgr() = default;
    ~CommMemMgr() = default;

    // cclbuffer内存
    void CommSetHcclBufferManager(CCLBufferManager& bufferManager);
    HcclResult GetHcclBuffer(CommBuffer* buffer);

    // 用户注册/反注册内存
    HcclResult CommRegMem(const std::string& memTag, const HcclMem& mem, HcclRegMemAttr attr, void** memHandle);
    HcclResult CommUnregMem(const std::string& memTag, const void* memHandle);
    HcclResult CommGetLocalRegMemByTag(const std::string& tag, std::vector<HcclMem>& memVec);
    HcclResult
    CommGetLocalRegMemByHandles(const HcclMemHandle* memHandles, uint32_t memHandleNum, std::vector<HcclMem>& memVec);

private:
    static inline MemKey MakeKey(void* addr, uint64_t size)
    {
        return MemKey(reinterpret_cast<uintptr_t>(addr), static_cast<uint64_t>(size));
    }
    struct TagRegistry {
        Table table; // 区间树 + ref 语义
    };

    // cclbuffer内存
    std::mutex bufferMutex_;
    CCLBufferManager* bufferManager_{nullptr};

    // 用户绑定内存
    std::mutex memMutex_;
    // 每个 tag 一份 registry
    std::unordered_map<std::string, TagRegistry> tagRegs_;
    // 每个tag n个 HcclMemoryHandle
    std::unordered_map<std::string, std::vector<std::shared_ptr<HcclMemoryHandle>>> opBindings_;
};
} // namespace hccl

#endif
