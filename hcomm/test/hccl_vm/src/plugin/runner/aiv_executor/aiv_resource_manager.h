/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_RESOURCE_MANAGER_H
#define AIV_RESOURCE_MANAGER_H

#include <cstdint>
#include <memory>
#include <vector>

#include "sim_common_defs.h"
#include "store_sim_shm_memory_common.h"

namespace sim {
struct OpMemInfoTab;
}

struct AivBufferResource {
    void* realAddr{nullptr};
    uint64_t virtualAddr{0};
    uint64_t size{0};
};

struct AivRankResource {
    AivBufferResource inputBuffer;
    AivBufferResource outputBuffer;
    AivBufferResource cclBuffer;
    AivBufferResource aivCommInfoBuffer;
};

class AivResourceManager {
public:
    static AivResourceManager& GetInstance();

    AivResourceManager(const AivResourceManager&) = delete;
    AivResourceManager& operator=(const AivResourceManager&) = delete;

    ~AivResourceManager();

    HcclSim::HcclVmResult Init(uint32_t rankId, const sim::OpMemInfoTab& opMemInfo, uint32_t rankSize);
    void Reset();

    const std::vector<AivRankResource>& GetAllRankResources() const;
    const AivRankResource* GetRankResource(uint32_t rankId) const;

private:
    AivResourceManager() = default;

    HcclSim::HcclVmResult EnsureInitialized(uint32_t rankSize);
    HcclSim::HcclVmResult MapBuffer(
        uint32_t rankSize, uint32_t rankId, uint8_t bufferType, uint64_t startAddr, uint64_t size,
        bool allowDuplicateSame = false);
    HcclSim::HcclVmResult
    MapOpMemBuffer(uint32_t rankSize, uint32_t rankId, uint8_t bufferType, uint64_t startAddr, uint64_t size);
    HcclSim::HcclVmResult InitAivCommInfoBuffers(uint32_t rankSize);

    std::vector<AivRankResource> rankResources_;
    std::vector<sim::PhyMemBlock> acquiredPhyMemBlocks_;
    std::vector<std::unique_ptr<uint8_t[]>> aivCommInfoBufferOwners_;
};

#endif // AIV_RESOURCE_MANAGER_H
