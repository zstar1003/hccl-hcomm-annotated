/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_mem_manager.h"
namespace hccl {

void CommMemMgr::CommSetHcclBufferManager(CCLBufferManager& bufferManager) { bufferManager_ = &bufferManager; }

HcclResult CommMemMgr::GetHcclBuffer(CommBuffer* buffer)
{
    CHK_PTR_NULL(buffer);
    CHK_PTR_NULL(bufferManager_);
    std::lock_guard<std::mutex> lock(bufferMutex_);
    void* temp = nullptr;
    uint64_t tempSize = 0;
    HcclResult ret = bufferManager_->GetIndependentOpCCLbuffer(temp, tempSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GetHcclBuffer] GetHcclBuffer failed"), ret);
    buffer->addr = temp;
    buffer->size = tempSize;
    return HCCL_SUCCESS;
}

// 绑定：opTag -> 句柄（幂等）
HcclResult CommMemMgr::CommRegMem(const std::string& memTag, const HcclMem& mem, HcclRegMemAttr attr, void** memHandle)
{
    CHK_PRT_RET(
        memHandle == nullptr, HCCL_ERROR("[CommRegMem] memHandle is null. tag[%s]", memTag.c_str()), HCCL_E_PARA);
    CHK_PRT_RET(
        mem.addr == nullptr || mem.size == 0,
        HCCL_ERROR("[CommRegMem] invalid mem. addr[%p] size[%llu]", mem.addr, mem.size), HCCL_E_PARA);

    // 组装句柄（仅域内管理，无进程级注册）
    Handle h;
    EXCEPTION_CATCH(h = std::make_shared<HcclMemoryHandle>(), return HCCL_E_PTR);
    h->addr = mem.addr;
    h->size = static_cast<uint64_t>(mem.size);
    h->memType = static_cast<HcclMemType>(mem.type);
    h->attr = attr;

    const auto key = MakeKey(mem.addr, static_cast<size_t>(mem.size));

    std::lock_guard<std::mutex> addLock(memMutex_);
    auto& reg = tagRegs_[memTag];

    // 同tag内做区间冲突/幂等复用
    auto res = reg.table.Add(key, h);
    if (!res.second) {
        // 只能用 Find 的返回值来判定：
        // - 等于(全集命中)：Find(key).first == true（允许，Add 内已 ref）
        // - 子集/超集/交集：Find(key).first 可能为 true(子) 或 false(交/超/空)，但都属于冲突！
        auto f = reg.table.Find(key);
        if (!f.first || !(f.second && f.second->addr == mem.addr && f.second->size == mem.size)) {
            HCCL_ERROR("[CommRegMem] overlap in tag[%s], key=%s", memTag.c_str(), key.ToString().c_str());
            return HCCL_E_PARA;
        }
        // HcclRegMemAttr不同时更新
        if (f.second->attr.value != attr.value) {
            HCCL_WARNING("[CommRegMem] inconsistent attr for same mem. tag[%s]", memTag.c_str());
            f.second->attr.value = attr.value;
        }
        // 复用已有句柄：直接用 Find 返回的 buffer，避免解引用 res.first（可能是 end()）
        h = f.second;
    }

    // 幂等加入绑定列表（同memHandle不重复）
    auto& vec = opBindings_[memTag];
    bool exists = std::any_of(vec.begin(), vec.end(), [&h](const Handle& x) {
        return x && (x.get() == h.get());
    });
    if (!exists)
        vec.emplace_back(h);

    *memHandle = h.get();
    HCCL_INFO("[CommRegMem] ok. tag[%s] memHandle[%p] size[%llu]", memTag.c_str(), *memHandle, h->size);
    return HCCL_SUCCESS;
}

// 解绑：在该通信域实例内，移除“指定算子(memTag)”下的该句柄
HcclResult CommMemMgr::CommUnregMem(const std::string& memTag, const void* memHandle)
{
    CHK_PRT_RET(memHandle == nullptr, HCCL_ERROR("[CommUnregMem] memHandle is null"), HCCL_E_PARA);
    CHK_PRT_RET(memTag.empty(), HCCL_ERROR("[CommUnregMem] memTag is null or empty"), HCCL_E_PARA);

    std::lock_guard<std::mutex> addLock(memMutex_);

    auto itTag = opBindings_.find(memTag);
    CHK_PRT_RET(
        itTag == opBindings_.end(), HCCL_WARNING("[CommUnregMem] tag[%s] not found in bindings", memTag.c_str()),
        HCCL_E_NOT_FOUND);

    auto& vec = itTag->second;          // vector<Handle> under this tag
    auto& reg = tagRegs_[itTag->first]; // TagRegistry for this tag
    size_t unboundCount = 0;            // 本次解绑命中的句柄个数（即便 Del 未真正擦除也计数）
    size_t erasedCount = 0;             // RmaBufferMgr::Del 返回 true 的次数（ref 归零而“擦除”）

    vec.erase(
        std::remove_if(
            vec.begin(), vec.end(),
            [&](const Handle& h) {
                if (!h || h.get() != memHandle)
                    return false;
                const auto key = MakeKey(h->addr, static_cast<size_t>(h->size));
                try {
                    if (reg.table.Del(key)) {
                        ++erasedCount; // 该 key 的引用归零并从表中移除
                    }
                } catch (const std::out_of_range&) {
                    HCCL_ERROR(
                        "[CommUnregMem] tag[%s] key not found on Del (maybe already removed)", itTag->first.c_str());
                }

                ++unboundCount; // 从绑定列表移除，无论 Del 是否真正擦除
                return true;    // erase-remove：删除该 handle
            }),
        vec.end());

    // 若该 tag 已无绑定，可按需清理映射条目（以及空表）
    if (vec.empty()) {
        opBindings_.erase(itTag);
        if (reg.table.size() == 0) {
            tagRegs_.erase(std::string(memTag));
        }
    }

    CHK_PRT_RET(
        unboundCount == 0, HCCL_WARNING("[CommUnregMem] tag[%s] memHandle[%p] not found", memTag.c_str(), memHandle),
        HCCL_E_NOT_FOUND);

    HCCL_INFO(
        "[CommUnregMem] tag[%s] memHandle[%p] unbound=%zu, erased=%zu", memTag.c_str(), memHandle, unboundCount,
        erasedCount);
    return HCCL_SUCCESS;
}

HcclResult CommMemMgr::CommGetLocalRegMemByTag(const std::string& tag, std::vector<HcclMem>& memVec)
{
    std::lock_guard<std::mutex> lock(memMutex_);
    auto it = opBindings_.find(tag);
    if (it == opBindings_.end()) {
        HCCL_INFO("[CommMemMgr] tag[%s] key not found", tag.c_str());
        return HCCL_SUCCESS;
    }

    const auto& vec = it->second;
    memVec.reserve(vec.size());
    for (const auto& handle : vec) {
        HcclMem mem;
        mem.addr = handle->addr;
        mem.size = handle->size;
        mem.type = handle->memType;
        memVec.push_back(mem);
    }
    return HCCL_SUCCESS;
}

HcclResult CommMemMgr::CommGetLocalRegMemByHandles(
    const HcclMemHandle* memHandles, uint32_t memHandleNum, std::vector<HcclMem>& memVec)
{
    if (memHandleNum == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(memHandles);

    std::lock_guard<std::mutex> lock(memMutex_);
    memVec.clear();
    memVec.reserve(memHandleNum);

    for (uint32_t i = 0; i < memHandleNum; ++i) {
        bool found = false;
        for (const auto& tagBinding : opBindings_) {
            for (const auto& handle : tagBinding.second) {
                if (handle != nullptr && handle.get() == memHandles[i]) {
                    HcclMem mem;
                    mem.addr = handle->addr;
                    mem.size = handle->size;
                    mem.type = handle->memType;
                    memVec.push_back(mem);
                    found = true;
                    break;
                }
            }
            if (found) {
                break;
            }
        }
        if (!found) {
            HCCL_ERROR("[CommGetLocalRegMemByHandles] memHandle[%p] not found in any tag", memHandles[i]);
            return HCCL_E_NOT_FOUND;
        }
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
