/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_symmetric_memory.h"
#include "symmetric_memory/symmetric_memory.h"
#include "hcomm_team_entity_defs.h"

#ifdef CCL_KERNEL_AICPU
namespace hccl {

class SymmetricMemory::SimpleVaAllocator {
public:
    // 不需要任何成员，只要让编译器觉得它是个完整的类就行
    SimpleVaAllocator() {}
    ~SimpleVaAllocator() {}
};

SymmetricMemory::~SymmetricMemory() {}
} // namespace hccl
#endif // CCL_KERNEL_AICPU

using namespace hccl;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

static bool IsHcommWindow(const HcommWindow* window)
{
    return window->header.magicWord == HCOMM_WINDOW_MAGIC_WORD && window->header.version == HCOMM_WINDOW_VERSION
           && window->header.size == sizeof(HcommWindow);
}

static SymmetricWindow* GetLegacyWindow(HcclCommSymWindow winHandle)
{
    HcommWindow* window = static_cast<HcommWindow*>(winHandle);
    if (!IsHcommWindow(window)) {
        return static_cast<SymmetricWindow*>(winHandle);
    }
    return reinterpret_cast<SymmetricWindow*>(window->legacySymWindow);
}

static HcclResult GetLegacyPeerPointer(SymmetricWindow* symWin, size_t offset, uint32_t peerRank, void** ptr)
{
    CHK_PTR_NULL(symWin);
    if (symWin->mode == SymmetricMemoryMode::URMA) {
        HCCL_ERROR("[%s] PeerPointer does not support URMA mode", __func__);
        *ptr = nullptr;
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        peerRank >= symWin->rankSize,
        HCCL_ERROR("[%s] peerRank[%u] exceeds rankSize[%u]", __func__, peerRank, symWin->rankSize), HCCL_E_PARA);
    CHK_PRT_RET(
        offset >= symWin->userSize,
        HCCL_ERROR("[%s] offset[%zu] exceeds userSize[%zu]", __func__, offset, symWin->userSize), HCCL_E_PARA);
    CHK_PRT_RET(
        symWin->baseVa == nullptr || symWin->stride == 0 || peerRank > (SIZE_MAX - offset) / symWin->stride,
        HCCL_ERROR("[%s] invalid legacy symmetric VA layout", __func__), HCCL_E_PARA);
    size_t peerOffset = static_cast<size_t>(peerRank) * symWin->stride + offset;
    uintptr_t base = reinterpret_cast<uintptr_t>(symWin->baseVa);
    CHK_PRT_RET(peerOffset > UINTPTR_MAX - base, HCCL_ERROR("[%s] peer address overflows", __func__), HCCL_E_PARA);
    *ptr = reinterpret_cast<void*>(base + peerOffset);
    return HCCL_SUCCESS;
}

static HcclResult GetUbMemPeerPointer(const HcommWindow* window, size_t offset, uint32_t lsaMemberId, void** ptr)
{
    CHK_PTR_NULL(window);
    CHK_PTR_NULL(ptr);
    *ptr = nullptr;
    CHK_PRT_RET(
        window->lsaWin.baseVa == 0 || window->lsaWin.stride == 0 || window->lsaWin.userSize == 0
            || offset >= window->lsaWin.userSize,
        HCCL_ERROR("[%s] invalid UB Memory window layout or offset", __func__), HCCL_E_PARA);
    uintptr_t base = static_cast<uintptr_t>(window->lsaWin.baseVa);
    CHK_PRT_RET(
        offset > UINTPTR_MAX - base || lsaMemberId > (UINTPTR_MAX - base - offset) / window->lsaWin.stride,
        HCCL_ERROR("[%s] peer address overflows", __func__), HCCL_E_PARA);
    *ptr = reinterpret_cast<void*>(base + static_cast<uint64_t>(lsaMemberId) * window->lsaWin.stride + offset);
    return HCCL_SUCCESS;
}

static HcclResult GetLegacyRemoteAddr(SymmetricWindow* symWin, size_t offset, uint32_t peerRank, void** ptr)
{
    CHK_PTR_NULL(symWin);
    if (symWin->mode != SymmetricMemoryMode::URMA) {
        HCCL_ERROR("[%s] only URMA mode is supported", __func__);
        *ptr = nullptr;
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PTR_NULL(symWin->remoteMems);
    CHK_PRT_RET(
        peerRank >= symWin->remoteMemNum,
        HCCL_ERROR("[%s] peerRank[%u] exceeds remoteMemNum[%u]", __func__, peerRank, symWin->remoteMemNum),
        HCCL_E_PARA);
    CommMem& remoteMem = symWin->remoteMems[peerRank];
    CHK_PRT_RET(
        remoteMem.addr == nullptr || remoteMem.type == COMM_MEM_TYPE_INVALID,
        HCCL_ERROR("[%s] invalid remote memory, peerRank[%u], addr[%p]", __func__, peerRank, remoteMem.addr),
        HCCL_E_PARA);
    CHK_PRT_RET(
        offset >= remoteMem.size,
        HCCL_ERROR("[%s] offset[%zu] exceeds remote size[%llu]", __func__, offset, remoteMem.size), HCCL_E_PARA);
    uintptr_t remoteBase = reinterpret_cast<uintptr_t>(remoteMem.addr);
    CHK_PRT_RET(offset > UINTPTR_MAX - remoteBase, HCCL_ERROR("[%s] remote address overflows", __func__), HCCL_E_PARA);
    *ptr = reinterpret_cast<void*>(remoteBase + offset);
    return HCCL_SUCCESS;
}

HcclResult HcclSymWinGetPeerPointer(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    CHK_PTR_NULL(winHandle);
    CHK_PTR_NULL(ptr);
    *ptr = nullptr;
    HcommWindow* window = static_cast<HcommWindow*>(winHandle);
    if (!IsHcommWindow(window)) {
        return GetLegacyPeerPointer(static_cast<SymmetricWindow*>(winHandle), offset, peerRank, ptr);
    }
    if (window->lsaWin.baseVa != 0 && window->lsaWin.stride != 0 && window->lsaWin.userSize != 0) {
        return GetUbMemPeerPointer(window, offset, peerRank, ptr);
    }
    return GetLegacyPeerPointer(reinterpret_cast<SymmetricWindow*>(window->legacySymWindow), offset, peerRank, ptr);
}

HcclResult HcclSymWinGetRemoteAddr(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    CHK_PTR_NULL(winHandle);
    CHK_PTR_NULL(ptr);
    *ptr = nullptr;
    SymmetricWindow* symWin = GetLegacyWindow(winHandle);
    CHK_PRT_RET(
        symWin == nullptr, HCCL_ERROR("[%s] UB Memory window does not provide URMA remote address", __func__),
        HCCL_E_NOT_SUPPORT);
    return GetLegacyRemoteAddr(symWin, offset, peerRank, ptr);
}

#ifdef __cplusplus
}
#endif // __cplusplus
