/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_SUPER_FAST_LOAD_H
#define HCCL_CCU_SUPER_FAST_LOAD_H
#include <cstdlib>
#include <array>
#include <vector>
#include <cstring>
#include <utility>
#include <cstddef>

#include "securec.h"
#include "string_util.h"

#include "types.h"
#include "op_type.h"
#include "orion_adapter_rts.h"
#include "ccu/ccu_task_param.h"
#include "task_param.h"
#include "ccu_ins.h"
#include "acl/acl_rt.h"

namespace Hccl {
using CcuParamsMappingKeyType = std::uint32_t;
constexpr std::size_t CCU_SFL_PARAM_KEY_LEN = 3;
using CcuSFLMappingKey = std::array<CcuParamsMappingKeyType, CCU_SFL_PARAM_KEY_LEN>;
struct CachedCCUParams {
public:
    rtCcuTaskInfo_t* ccuParams{nullptr};
    std::vector<std::size_t> count;
    std::vector<TaskParam> taskParams;
    u64 execId{0};
    std::size_t totalCounts{0};
    CcuInstType insType{};
    bool isSlave{false};

    CachedCCUParams() = default;
    explicit CachedCCUParams(
        std::vector<std::vector<Hccl::CcuTaskParam>>&& ccuInstruction,
        std::vector<std::vector<CcuProfilingInfo>>&& profilingInfo, std::size_t execId, CcuInstType insType,
        bool isSlave, void* comm);

    CachedCCUParams(const CachedCCUParams&) = delete;
    CachedCCUParams& operator=(const CachedCCUParams&) = delete;

    CachedCCUParams(CachedCCUParams&& other) noexcept;
    CachedCCUParams& operator=(CachedCCUParams&& other) noexcept;

    ~CachedCCUParams();

private:
    inline void* aligned_malloc(size_t align, size_t size) const
    {
        void* p = nullptr;
        if (posix_memalign(&p, align, size) != 0) {
            throw std::bad_alloc();
        }
        return p;
    }

    inline void aligned_free(void* ptr) const
    {
        if (ptr) {
            std::free(ptr);
            ptr = nullptr;
        }
    }

    inline bool is_power_of_2(std::size_t x) const { return static_cast<bool>(x) && !static_cast<bool>((x & (x - 1))); }
    inline std::size_t round_up_to(std::size_t size, std::size_t alignment) const
    {
        return alignment != 0 ? (size + alignment - 1) / alignment * alignment : size;
    }
    inline void* alloc_aligned_raw(std::size_t alignment, std::size_t size) const
    {
        if (alignment == 0) {
            alignment = alignof(std::max_align_t);
        }
        if (!is_power_of_2(alignment)) {
            return nullptr;
        };
        return aligned_malloc(alignment, round_up_to(size, alignment));
    }

    rtCcuTaskInfo_t*
    alloc_and_memcpy_aligned(const std::vector<std::vector<rtCcuTaskInfo_t>>& vecs, std::size_t alignment);
};

inline void SuperFastLoad(rtCcuTaskInfo_t* params, aclrtStream const streamPtr, int counts)
{
    for (int i = 0; i < counts; ++i) {
        HrtCcuLaunch(params[i], streamPtr);
    }
}
} // namespace Hccl

namespace std {
constexpr std::size_t NUMBER_SIX = 6;
constexpr std::size_t NUMBER_TWO = 2;
struct ArrayHasher {
    std::size_t
    operator()(const std::array<Hccl::CcuParamsMappingKeyType, Hccl::CCU_SFL_PARAM_KEY_LEN>& ccuParams) const noexcept
    {
        std::size_t hashVal = 0;
        for (auto ccuParam : ccuParams) {
            hashVal ^= std::hash<std::uint32_t>{}(ccuParam) + 0x9e3779b9 + (hashVal << NUMBER_SIX)
                       + (hashVal >> NUMBER_TWO);
        }
        return hashVal;
    }
};

template <>
struct hash<Hccl::OpType> {
    std::size_t operator()(const Hccl::OpType& type) const noexcept { return static_cast<std::size_t>(type); }
};
template <>
struct hash<const Hccl::OpType> {
    std::size_t operator()(const Hccl::OpType& type) const noexcept { return static_cast<std::size_t>(type); }
};
}; // namespace std
#endif // HCCL_CCU_SUPER_FAST_LOAD_H
