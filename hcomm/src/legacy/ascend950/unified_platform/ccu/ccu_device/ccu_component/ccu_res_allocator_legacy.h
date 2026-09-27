/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_RES_ALLOCATOR_H
#define HCCL_CCU_RES_ALLOCATOR_H

#include <mutex>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

#include "ccu_device_manager.h"

namespace Hccl {

class CcuResIdAllocator {
public:
    explicit CcuResIdAllocator(const uint32_t capacity) : capacity_(capacity) {};
    CcuResIdAllocator() = default;

    HcclResult Alloc(
        const uint32_t num, const bool consecutive, std::vector<ResInfo>& allocatedResInfos,
        const std::string& dfxInfo = "");
    HcclResult Release(const uint32_t startId, const uint32_t num);
    uint32_t GetConsecutiveRemainSize() const;

    std::string Describe() const;

private:
    uint32_t capacity_{0};
    uint32_t allocatedSize{0};
    std::vector<ResInfo> resInfos{};
    std::mutex innerMutex;

    size_t FindReleaseResIndex(const uint32_t startId) const;
    void AllocResInfo(std::vector<ResInfo> newResInfos);
    void ReleaseResInfo(const size_t resIndex, const uint32_t startId, const uint32_t num);
};

class CcuResAllocator {
public:
    CcuResAllocator(const int32_t devLogicId, const uint8_t dieId);
    CcuResAllocator() = default;

    HcclResult Alloc(const ResType resType, const uint32_t num, const bool consecutive, std::vector<ResInfo>& resInfos);
    HcclResult Release(const ResType resType, const uint32_t startId, const uint32_t num);
    uint32_t GetConsecutiveRemainSize(const ResType resType) const;

    std::string Describe() const;

private:
    int32_t devLogicId_{0};
    uint8_t dieId_{0};
    std::unordered_map<uint8_t, std::unique_ptr<CcuResIdAllocator>> idAllocatorMap;
};

}; // namespace Hccl

#endif
