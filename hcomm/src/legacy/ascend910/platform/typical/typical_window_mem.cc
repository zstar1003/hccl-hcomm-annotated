/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "typical_window_mem.h"
#include "adapter_rts_common.h"

namespace hccl {
TypicalWindowMem& TypicalWindowMem::GetInstance()
{
    static TypicalWindowMem typicalWindowMem[MAX_MODULE_DEVICE_NUM + 1];
    s32 deviceLogicId = INVALID_INT;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    if (ret == HCCL_SUCCESS && (static_cast<u32>(deviceLogicId) < MAX_MODULE_DEVICE_NUM)) {
        HCCL_INFO("[TypicalWindowMem::GetInstance]deviceLogicID[%d]", deviceLogicId);
        return typicalWindowMem[deviceLogicId];
    }
    HCCL_WARNING("[TypicalWindowMem::GetInstance]deviceLogicID[%d] is invalid, ret[%d]", deviceLogicId, ret);
    return typicalWindowMem[MAX_MODULE_DEVICE_NUM];
}

TypicalWindowMem::TypicalWindowMem() {}

TypicalWindowMem::~TypicalWindowMem() { FreeAllWinowMem(); }

HcclResult TypicalWindowMem::AllocWindowMem(void** ptr, uint64_t size)
{
    HCCL_DEBUG("[TypicalWindowMem][AllocWindowMem]start alloc window mem to [%p] of size[%llu].", ptr, size);
    // Check ptr and size validation.
    CHK_PTR_NULL(ptr);
    CHK_PRT_RET(
        size < 1, HCCL_ERROR("[TypicalWindowMem][AllocWindowMem]size[%lu], cannot alloc window mem less than 1.", size),
        HCCL_E_PARA);

    // Check whether the addr already exist in window mem map.
    std::unique_lock<std::mutex> lockWindowMemMap(windowMemMapMutex_);
    if (*ptr != nullptr) {
        uint64_t inputAddr = reinterpret_cast<uintptr_t>(*ptr);
        auto wmIter = windowMemMap_.find(inputAddr);
        if (wmIter != windowMemMap_.end()) {
            HCCL_ERROR("[TypicalWindowMem][AllocWindowMem]addr[%p] already allocated.", *ptr);
            return HCCL_E_PARA;
        }
        HCCL_WARNING(
            "[TypicalWindowMem][AllocWindowMem]addr[%p] is not nullptr, "
            "we will overwrite it with the new allocated address.",
            *ptr);
    }
    // Alloc window mem.
    CHK_RET(hrtMalloc(ptr, size));
    CHK_PRT_RET(
        (*ptr) == nullptr, HCCL_ERROR("[TypicalWindowMem][AllocWindowMem]In typical notify src buffer, malloc failed."),
        HCCL_E_MEMORY);

    // Add allocated mem into window mem map。
    uint64_t allocatedAddr = reinterpret_cast<uintptr_t>(*ptr);
    windowMemMap_[allocatedAddr] = size;
    HCCL_INFO(
        "[TypicalWindowMem][AllocWindowMem]alloc window memory success, addr[%p], size[%llu]. "
        "please register mr before use",
        *ptr, size);
    return HCCL_SUCCESS;
}

HcclResult TypicalWindowMem::FreeWindowMem(void* ptr)
{
    HCCL_DEBUG(
        "[TypicalWindowMem][FreeWindowMem]start free window mem on [%p], please deregister mr before free.", ptr);
    CHK_PTR_NULL(ptr);
    // Check whether the addr exist in window mem map. Remove allocated mem from window mem map if exists.
    uint64_t addr = reinterpret_cast<uintptr_t>(ptr);
    std::unique_lock<std::mutex> lockWindowMemMap(windowMemMapMutex_);
    auto wmIter = windowMemMap_.find(addr);
    if (wmIter == windowMemMap_.end()) {
        HCCL_ERROR("[TypicalWindowMem][AllocWindowMem]addr[%p] were not allocated or already freed.", ptr);
        return HCCL_E_PARA;
    }
    // Free window mem.
    CHK_RET(hrtFree(ptr));
    windowMemMap_.erase(wmIter);
    ptr = nullptr;
    HCCL_INFO("[TypicalWindowMem][AllocWindowMem]free window memory success, addr[%p].", ptr);
    return HCCL_SUCCESS;
}

HcclResult TypicalWindowMem::FreeAllWinowMem()
{
    std::unique_lock<std::mutex> lockWindowMemMap(windowMemMapMutex_);
    if (!windowMemMap_.empty()) {
        for (auto& wmIter : windowMemMap_) {
            auto ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(wmIter.first));
            if (ptr != nullptr) {
                CHK_RET(hrtFree(ptr));
            }
        }
        windowMemMap_.clear();
    }
    HCCL_INFO("[TypicalWindowMem][FreeAllWinowMem]free all window memory success.");
    return HCCL_SUCCESS;
}
} // namespace hccl
