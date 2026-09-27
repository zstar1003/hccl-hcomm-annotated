/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REGED_MEM_MGR_H
#define REGED_MEM_MGR_H

#include "cast_utils.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>
#include "hcomm_c_adpt.h"
#include "log.h"
#include "buffer_key.h"
#include "buffer.h"

using RdmaHandle = void*;

namespace hcomm {
/**
 * @note 职责：用于通信设备EndPoint的注册内存信息管理，支持基于RmaBufferMgr类的重叠内存的检测报错等。
 */
class RegedMemMgr {
public:
    RegedMemMgr() = default;
    virtual ~RegedMemMgr() = default;

    // 注册内存
    virtual HcclResult RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle) = 0;

    // 注销内存
    virtual HcclResult UnregisterMemory(void* memHandle) = 0;

    // 导出指定内存描述，用于交换
    virtual HcclResult
    MemoryExport(const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen)
        = 0;

    // 基于内存描述，导入获得内存
    virtual HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) = 0;

    // 关闭内存
    virtual HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) = 0;

    virtual HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) = 0;

    // 不含数据成员——rdmaHandle_/memMtx_ 已移至各派生类 private
    // protected 静态模板 helper 保留在 RegedMemMgr 基类中（RegisterMemoryImpl 等），各派生类直接调用

protected:
    template <typename RmaBuffer>
    using RegedBufferEntry = std::pair<std::shared_ptr<RmaBuffer>, bool>;

    static HcclResult ValidateMemParams(HcommMem mem, void* const* memHandle)
    {
        CHK_PTR_NULL(memHandle);
        CHK_PTR_NULL(mem.addr);
        CHK_PRT_RET(mem.size == 0, HCCL_ERROR("[%s] mem size is zero", __func__), HCCL_E_PARA);
        CHK_PRT_RET(
            mem.type == COMM_MEM_TYPE_INVALID, HCCL_ERROR("[%s] invalid mem type [%d]", __func__, mem.type),
            HCCL_E_PARA);
        return HCCL_SUCCESS;
    }

    // MemoryExport: 从allBuffers中校验memHandle并获取buffer指针
    template <typename RmaBuffer>
    static HcclResult ValidateMemExportHandle(
        void* memHandle, const std::vector<RegedBufferEntry<RmaBuffer>>& allBuffers, RmaBuffer*& outBuffer)
    {
        auto it = std::find_if(allBuffers.begin(), allBuffers.end(), [memHandle](const auto& entry) {
            return entry.first != nullptr && entry.first.get() == memHandle;
        });
        if (it == allBuffers.end()) {
            HCCL_ERROR("[RegedMemMgr][MemoryExport] memHandle[%p] is not registered.", memHandle);
            return HCCL_E_NOT_FOUND;
        }
        outBuffer = it->first.get();
        return HCCL_SUCCESS;
    }

    // ---- 底层：tree 操作 ----

    // 用实际注册后的addr/size做key，对buffer增加引用计数
    template <typename Mgr, typename BufferPtr>
    static HcclResult AddBuffer(Mgr& mgr, const BufferPtr& buffer)
    {
        hccl::BufferKey<uintptr_t, u64> actualRegKey(
            ReinterpretAs<uintptr_t>(buffer->GetAddr()), static_cast<uint64_t>(buffer->GetSize()));
        EXCEPTION_CATCH((void)mgr->AddWithoutCheck(actualRegKey, buffer), return HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    // UnregisterMemory: IsAlias为true时，通过硬件句柄定位父buffer
    template <typename Mgr, typename BufferPtr, typename BufferVec, typename HwHandleFn, typename EqualFn>
    static BufferPtr ResolveAliasParent(
        Mgr& mgr, const hccl::BufferKey<uintptr_t, u64>& ownKey, BufferPtr buffer, BufferVec& allBuffers,
        HwHandleFn&& hwHandleGetter, EqualFn&& tokenEqual)
    {
        auto token = hwHandleGetter(buffer);
        auto findResult = mgr->Find(ownKey);
        if (findResult.first && tokenEqual(hwHandleGetter(findResult.second.get()), token)) {
            return findResult.second.get();
        }
        for (auto& entry : allBuffers) {
            const auto& ptr = entry.first;
            if (ptr.get() == buffer) {
                continue;
            }
            if (tokenEqual(hwHandleGetter(ptr.get()), token) && !ptr->IsAlias()) {
                return ptr.get();
            }
        }
        return nullptr;
    }

    // ---- 中层：注册/注销 核心逻辑 ----

    // Find命中则基于父buffer构造别名，未命中则构造新buffer并注册
    template <typename Mgr, typename FindResult, typename BufferPtr, typename MakeAlias, typename MakeNew>
    static HcclResult
    RegisterOrAlias(Mgr& mgr, const FindResult& findPair, BufferPtr& buffer, MakeAlias&& makeAlias, MakeNew&& makeNew)
    {
        if (findPair.first) {
            auto parentBuffer = findPair.second;
            EXCEPTION_CATCH((buffer = makeAlias(parentBuffer)), return HCCL_E_PTR);
            CHK_RET(AddBuffer(mgr, parentBuffer));
        } else {
            EXCEPTION_CATCH((buffer = makeNew()), return HCCL_E_PTR);
            CHK_RET(AddBuffer(mgr, buffer));
        }
        return HCCL_SUCCESS;
    }

    // ---- 上层：RegisterMemory / UnregisterMemory 通用实现 ----

    // 构造基类Buffer → RegisterOrAlias → 写回memHandle
    // makeAlias(bufPtr, parent) — 基于父buffer构造别名
    // makeNew(bufPtr)          — 构造新buffer
    // outRecords               — 可选的记录向量，注册成功时追加
    template <typename Mgr, typename RmaBuffer, typename MakeAlias, typename MakeNew>
    static HcclResult RegisterMemoryImpl(
        HcommMem mem, const char* memTag, void** memHandle, Mgr& mgr,
        std::vector<RegedBufferEntry<RmaBuffer>>& allBuffers, std::vector<std::shared_ptr<RmaBuffer>>* outRecords,
        const char* logTag, MakeAlias&& makeAlias, MakeNew&& makeNew)
    {
        CHK_RET(ValidateMemParams(mem, memHandle));

        hccl::BufferKey<uintptr_t, u64> tempKey(ReinterpretAs<uintptr_t>(mem.addr), mem.size);
        auto findPair = mgr->Find(tempKey);

        std::shared_ptr<Hccl::Buffer> localBufferPtr = nullptr;
        EXCEPTION_CATCH(
            (localBufferPtr = std::make_shared<Hccl::Buffer>(
                 ReinterpretAs<uintptr_t>(mem.addr), mem.size,
                 (mem.type == COMM_MEM_TYPE_HOST) ? HcclMemType::HCCL_MEM_TYPE_HOST : HcclMemType::HCCL_MEM_TYPE_DEVICE,
                 memTag)),
            return HCCL_E_PTR);

        std::shared_ptr<RmaBuffer> rmaBuffer;
        CHK_RET(RegisterOrAlias(
            mgr, findPair, rmaBuffer,
            [&localBufferPtr, &makeAlias](auto& parent) {
                return makeAlias(localBufferPtr, parent);
            },
            [&localBufferPtr, &makeNew]() {
                return makeNew(localBufferPtr);
            }));

        HCCL_INFO("[%s][RegisterMemory] success, key {%p, %llu}", logTag, mem.addr, mem.size);
        *memHandle = static_cast<void*>(rmaBuffer.get());
        allBuffers.emplace_back(rmaBuffer, false);
        if (outRecords != nullptr) {
            outRecords->push_back(rmaBuffer);
        }
        return HCCL_SUCCESS;
    }

    // IsAlias → ResolveAliasParent → Del → erase
    // hwHandleGetter(buffer) — 获取硬件句柄用于ResolveAliasParent
    // tokenEqual(lhs, rhs)   — 比较两个句柄是否相等
    // outRecords             — 可选的记录向量，注销成功时移除
    template <typename Mgr, typename RmaBuffer, typename HwHandleFn, typename EqualFn>
    static HcclResult UnregisterMemoryImpl(
        void* memHandle, Mgr& mgr, std::vector<RegedBufferEntry<RmaBuffer>>& allBuffers,
        std::vector<std::shared_ptr<RmaBuffer>>* outRecords, HwHandleFn&& hwHandleGetter, EqualFn&& tokenEqual)
    {
        CHK_PTR_NULL(memHandle);
        RmaBuffer* buffer = static_cast<RmaBuffer*>(memHandle);
        CHK_PTR_NULL(buffer);
        auto bufferInfo = buffer->GetBufferInfo();

        hccl::BufferKey<uintptr_t, u64> ownKey(bufferInfo.first, bufferInfo.second);
        RmaBuffer* refBuffer = buffer;
        if (buffer->IsAlias()) {
            refBuffer = ResolveAliasParent(
                mgr, ownKey, buffer, allBuffers, std::forward<HwHandleFn>(hwHandleGetter),
                std::forward<EqualFn>(tokenEqual));
            if (refBuffer == nullptr) {
                HCCL_ERROR("[UnregisterMemory] alias parent not found");
                return HCCL_E_NOT_FOUND;
            }
        }

        auto refBufferInfo = refBuffer->GetBufferInfo();
        hccl::BufferKey<uintptr_t, u64> tempKey(refBufferInfo.first, refBufferInfo.second);
        // Del returns false when ref remains nonzero; local unregister still succeeds after erasing this handle.
        EXCEPTION_CATCH((void)mgr->Del(tempKey), return HCCL_E_NOT_FOUND);
        // IsInTree判断tree中是否还有该key的引用
        auto it
            = std::find_if(allBuffers.begin(), allBuffers.end(), [buffer](const RegedBufferEntry<RmaBuffer>& entry) {
                  return entry.first.get() == buffer;
              });
        if (it != allBuffers.end()) {
            auto bufferPtr = it->first;
            if (!mgr->IsInTree(ownKey)) {
                allBuffers.erase(it);
            } else {
                it->second = true;
            }
            if (outRecords != nullptr) {
                outRecords->erase(std::remove(outRecords->begin(), outRecords->end(), bufferPtr), outRecords->end());
            }
        }
        return HCCL_SUCCESS;
    }
};
} // namespace hcomm
#endif // REGED_MEM_MGR_H
