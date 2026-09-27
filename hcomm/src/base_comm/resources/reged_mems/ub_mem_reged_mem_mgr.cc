/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_mem_reged_mem_mgr.h"
#include <algorithm>
#include "log.h"

namespace hcomm {

UbMemRegedMemMgr::UbMemRegedMemMgr() { localIpcRmaBufferMgr_ = std::make_unique<LocalIpcRmaBufferMgr>(); }

HcclResult UbMemRegedMemMgr::RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle)
{
    HCCL_INFO("[%s] Begin", __func__);
    CHK_PTR_NULL(mem);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(localIpcRmaBufferMgr_);
    std::lock_guard<std::mutex> lock(memMtx_);
    return RegisterMemoryImpl(
        *mem, memTag, memHandle, localIpcRmaBufferMgr_, allRegisteredBuffers_,
        static_cast<std::vector<std::shared_ptr<Hccl::LocalIpcRmaBuffer>>*>(nullptr), "UbMemRegedMemMgr",
        [&](auto& bufPtr, auto& parent) {
            return std::make_shared<Hccl::LocalIpcRmaBuffer>(bufPtr, *parent);
        },
        [&](auto& bufPtr) {
            return std::make_shared<Hccl::LocalIpcRmaBuffer>(bufPtr);
        });
}

HcclResult UbMemRegedMemMgr::UnregisterMemory(void* memHandle)
{
    HCCL_INFO("[%s] Begin", __func__);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(localIpcRmaBufferMgr_);
    std::lock_guard<std::mutex> lock(memMtx_);
    return UnregisterMemoryImpl(
        memHandle, localIpcRmaBufferMgr_, allRegisteredBuffers_,
        static_cast<std::vector<std::shared_ptr<Hccl::LocalIpcRmaBuffer>>*>(nullptr),
        [](auto* b) {
            return b->GetIpcPtr();
        },
        [](auto a, auto b) {
            return a == b;
        });
}

HcclResult UbMemRegedMemMgr::MemoryExport(
    [[maybe_unused]] const EndpointDesc& endpointDesc, [[maybe_unused]] void* memHandle,
    [[maybe_unused]] void** memDesc, [[maybe_unused]] uint32_t* memDescLen)
{
    HCCL_INFO("UbMemRegedMemMgr MemoryExport is not supported.");
    return HCCL_SUCCESS;
}

HcclResult UbMemRegedMemMgr::MemoryImport(
    [[maybe_unused]] const void* memDesc, [[maybe_unused]] uint32_t descLen, [[maybe_unused]] HcommMem* outMem)
{
    HCCL_INFO("UbMemRegedMemMgr MemoryImport is not supported.");
    return HCCL_SUCCESS;
}

HcclResult UbMemRegedMemMgr::MemoryUnimport([[maybe_unused]] const void* memDesc, [[maybe_unused]] uint32_t descLen)
{
    HCCL_INFO("UbMemRegedMemMgr MemoryUnimport is not supported.");
    return HCCL_SUCCESS;
}

HcclResult
UbMemRegedMemMgr::GetAllMemHandles([[maybe_unused]] void** memHandles, [[maybe_unused]] uint32_t* memHandleNum)
{
    HCCL_INFO("UbMemRegedMemMgr GetAllMemHandles is not supported.");
    return HCCL_SUCCESS;
}
} // namespace hcomm
