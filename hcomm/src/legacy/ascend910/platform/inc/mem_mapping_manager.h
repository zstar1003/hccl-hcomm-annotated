/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEM_MAPPING_MANAGER_H
#define MEM_MAPPING_MANAGER_H

#include <atomic>
#include <mutex>
#include "hccl_common.h"
#include "hccl_inner_common.h"
#include "dlhal_function.h"

namespace hccl {
class MemMappingManager {
public:
    ~MemMappingManager();

    static MemMappingManager& GetInstance(s32 deviceLogicID);

    HcclResult GetDevVA(s32 deviceLogicID, void* addr, u64 size, void*& devVA);
    HcclResult ReleaseDevVA(s32 deviceLogicID, void* addr, u64 size);

    // delete copy and move constructors and assign operators
    MemMappingManager(MemMappingManager const&) = delete;            // Copy construct
    MemMappingManager(MemMappingManager&&) = delete;                 // Move construct
    MemMappingManager& operator=(MemMappingManager const&) = delete; // Copy assign
    MemMappingManager& operator=(MemMappingManager&&) = delete;      // Move assign

private:
    struct HostMappingKey {
        u64 addr = 0;
        u64 size = 0;

        HostMappingKey(u64 addr, u64 size) : addr(addr), size(size) {}

        bool operator==(const HostMappingKey& that) const
        {
            return ((this->addr == that.addr) && (this->size == that.size));
        }

        bool operator!=(const HostMappingKey& that) const
        {
            return (this->addr != that.addr) || (this->size != that.size);
        }

        bool operator<(const HostMappingKey& that) const
        {
            return (addr < that.addr) || (addr == that.addr && size < that.size);
        }
    };
    struct HostMappingInfo {
        void* devVA = nullptr;
        Referenced ref;
        drvRegisterTpye registerTpye;
    };
    using HostMappingIter = std::map<MemMappingManager::HostMappingKey, MemMappingManager::HostMappingInfo>::iterator;

    MemMappingManager() {}
    HcclResult MapMem(s32 deviceLogicID, void* addr, u64 size, void*& devVA);
    bool IsRequireMapping(void* addr, u64 size, void*& devVA);
    HostMappingIter SearchMappingMap(u64 userAddr, u64 userSize);

    std::mutex mappedHostToDevMutex_;
    std::map<HostMappingKey, HostMappingInfo> mappedHostToDevMap_;
};
} // namespace hccl
#endif //  MEM_MAPPING_MANAGER_H
