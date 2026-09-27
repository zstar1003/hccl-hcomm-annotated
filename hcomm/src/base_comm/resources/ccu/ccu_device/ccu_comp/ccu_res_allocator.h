/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_RES_ALLOCATOR_H
#define CCU_RES_ALLOCATOR_H

#include <mutex>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

#include "ccu_dev_mgr_imp.h"

namespace hcomm {

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
    uint32_t allocatedSize_{0};
    std::vector<ResInfo> resInfos_{};
    std::mutex innerMutex_;

    size_t FindReleaseResIndex(const uint32_t startId) const;
    void AllocResInfo(std::vector<ResInfo> newResInfos);
    void ReleaseResInfo(const size_t resIndex, const uint32_t startId, const uint32_t num);
};

class CcuResAllocator {
public:
    CcuResAllocator(const int32_t userDevId, const uint8_t dieId) : userDevId_(userDevId), dieId_(dieId) {};
    CcuResAllocator() = default;

    HcclResult Init();
    HcclResult Alloc(const ResType resType, const uint32_t num, const bool consecutive, std::vector<ResInfo>& resInfos);
    HcclResult Release(const ResType resType, const uint32_t startId, const uint32_t num);
    uint32_t GetConsecutiveRemainSize(const ResType resType) const;

    // 0.5rtt专用
    HcclResult AllocCountXn(const uint32_t num, ResInfo& resInfo);
    HcclResult ReleaseCountXn(const uint32_t startId, const uint32_t num);

    std::string Describe() const;

private:
    int32_t userDevId_{0};
    uint8_t dieId_{0};

    // ccu v2新增countXn管理
    uint32_t xnSpecNum_{0};
    uint32_t countXnSpecNum_{0};
    std::unordered_map<uint8_t, std::unique_ptr<CcuResIdAllocator>> idAllocatorMap_;
};

} // namespace hcomm

#endif // CCU_RES_ALLOCATOR_H
