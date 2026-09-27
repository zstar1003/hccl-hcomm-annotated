/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_mems.h"
#include <cstdlib>
#include <algorithm>
#include "orion_adapter_rts.h"

namespace hccl {

CommMemType ConvertHcclToCommMemType(HcclMemType hcclType)
{
    switch (hcclType) {
        case HCCL_MEM_TYPE_DEVICE:
            return COMM_MEM_TYPE_DEVICE;
        case HCCL_MEM_TYPE_HOST:
            return COMM_MEM_TYPE_HOST;
        default:
            return COMM_MEM_TYPE_INVALID;
    }
}

HcclMemType ConvertCommToHcclMemType(CommMemType commType)
{
    switch (commType) {
        case COMM_MEM_TYPE_DEVICE:
            return HCCL_MEM_TYPE_DEVICE;
        case COMM_MEM_TYPE_HOST:
            return HCCL_MEM_TYPE_HOST;
        default:
            return HCCL_MEM_TYPE_NUM;
    }
}

CommMems::CommMems(uint64_t bufferSize) : bufferSize_(bufferSize)
{
    cclMemInfo_.mem.addr = nullptr;
    cclMemInfo_.mem.size = 0;
    cclMemInfo_.mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
}

HcclResult CommMems::Add([[maybe_unused]] void* addr, [[maybe_unused]] uint64_t len) { return HCCL_SUCCESS; }

HcclResult CommMems::GetHcclBuffer(void*& addr, uint64_t& len)
{
    addr = reinterpret_cast<void*>(cclMemInfo_.mem.addr);
    len = static_cast<uint64_t>(cclMemInfo_.mem.size);
    return HCCL_SUCCESS;
}

HcclResult CommMems::HcclBufferMemset(void*& addr, uint64_t& len, bool clearFlag) const
{
    if (!clearFlag) {
        HCCL_DEBUG("[CommMems][HcclBufferMemset] clearFlag[%d] is false, skip memset.", clearFlag);
        return HCCL_SUCCESS;
    }

    if (addr != nullptr && len > 0) {
        EXCEPTION_CATCH(Hccl::HrtMemset(addr, len, len), return HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[CommMems][HcclBufferMemset] buffer[%p] is null or size[%llu] is 0, skip.", addr, len);
    return HCCL_E_PARA;
}

HcclResult CommMems::Init(HcclMem cclBuffer)
{
    cclMemInfo_.mem.addr = cclBuffer.addr;
    cclMemInfo_.mem.size = cclBuffer.size;
    cclMemInfo_.mem.type = ConvertHcclToCommMemType(cclBuffer.type);
    std::string memTag = "HcclBuffer";
    errno_t sRet = strncpy_s(cclMemInfo_.memTag, HCOMM_RES_TAG_MAX_LEN, memTag.c_str(), memTag.size());
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[CommMems][Init] strncpy_s failed, return [%d].", sRet), HCCL_E_MEMORY);
    HCCL_INFO("[CommMems][Init] addr[%p] size[%llu] memType[%u]", cclBuffer.addr, cclBuffer.size, cclBuffer.type);
    memVersion_++;
    return HCCL_SUCCESS;
}

HcclResult CommMems::CommRegMem(const std::string& memTag, const CommMem& mem, void** memHandle)
{
    CHK_PRT_RET(
        memHandle == nullptr, HCCL_ERROR("[CommRegMem] memHandle is null. tag[%s]", memTag.c_str()), HCCL_E_PARA);
    CHK_PRT_RET(
        mem.addr == nullptr || mem.size == 0,
        HCCL_ERROR("[CommRegMem] invalid mem. addr[%p] size[%llu]", mem.addr, (unsigned long long)mem.size),
        HCCL_E_PARA);
    if (UNLIKELY(memTag.size() >= HCOMM_RES_TAG_MAX_LEN)) {
        HCCL_ERROR("[CommRegMem] memTag.size()[%zu] exceeds limit[%u]", memTag.size(), HCOMM_RES_TAG_MAX_LEN);
        return HCCL_E_PARA;
    }

    // 组装句柄（仅域内管理，无进程级注册）
    Handle h;
    EXCEPTION_CATCH(h = std::make_shared<CommMemInfo>(), return HCCL_E_PTR);
    h->mem.addr = mem.addr;
    h->mem.size = mem.size;
    h->mem.type = mem.type;
    errno_t sRet = strncpy_s(h->memTag, HCOMM_RES_TAG_MAX_LEN, memTag.c_str(), memTag.size());
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[CommRegMem] strncpy_s failed, return [%d].", sRet), HCCL_E_MEMORY);

    const auto key = MakeKey(mem.addr, static_cast<size_t>(mem.size));

    std::lock_guard<std::mutex> addLock(memMutex_);

    auto opIt = opBindings_.find(memTag);
    if (opIt != opBindings_.end()) {
        HCCL_ERROR(
            "[CommRegMem] memTag[%s] already registered: old addr[%p] size[%llu], new addr[%p] size[%llu].",
            memTag.c_str(), opIt->second->mem.addr, static_cast<unsigned long long>(opIt->second->mem.size), mem.addr,
            static_cast<unsigned long long>(mem.size));
        return HCCL_E_PARA;
    }

    auto& reg = tagRegs_[memTag];

    // 同tag内做区间冲突/幂等复用
    reg.table.AddWithoutCheck(key, h);

    // 加入绑定map
    opBindings_.emplace(memTag, h);

    memVersion_++;
    *memHandle = h.get();
    HCCL_INFO(
        "[CommRegMem] ok. tag[%s] memHandle[%p] size[%llu]", memTag.c_str(), *memHandle,
        static_cast<unsigned long long>(h->mem.size));
    return HCCL_SUCCESS;
}

HcclResult CommMems::CommUnregMem(const std::string& memTag, const void* memHandle) // 待确认是否要解注册
{
    CHK_PRT_RET(memHandle == nullptr, HCCL_ERROR("[CommUnregMem] memHandle is null"), HCCL_E_PARA);
    CHK_PRT_RET(memTag.empty(), HCCL_ERROR("[CommUnregMem] memTag is null or empty"), HCCL_E_PARA);

    std::lock_guard<std::mutex> addLock(memMutex_);

    auto itTag = opBindings_.find(memTag);
    CHK_PRT_RET(
        itTag == opBindings_.end(), HCCL_WARNING("[CommUnregMem] tag[%s] not found in bindings", memTag.c_str()),
        HCCL_E_NOT_FOUND);

    auto& h = itTag->second;            // Handle under this tag
    auto& reg = tagRegs_[itTag->first]; // TagRegistry for this tag
    size_t unboundCount = 0;            // 本次解绑命中的句柄个数（即便 Del 未真正擦除也计数）
    size_t erasedCount = 0;             // RmaBufferMgr::Del 返回 true 的次数（ref 归零而“擦除”）

    if (h.get() == memHandle) {
        const auto key = MakeKey(h->mem.addr, static_cast<size_t>(h->mem.size));
        try {
            if (reg.table.Del(key)) {
                ++erasedCount; // 该 key 的引用归零并从表中移除
            }
        } catch (const std::out_of_range&) {
            HCCL_ERROR("[CommUnregMem] tag[%s] key not found on Del (maybe already removed)", itTag->first.c_str());
        }
        ++unboundCount; // 从绑定列表移除，无论 Del 是否真正擦除
        opBindings_.erase(itTag);
        memVersion_++;
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

HcclResult CommMems::GetTagsFromHandles(void** memHandles, uint32_t memHandleNum, std::vector<std::string>& memTags)
{
    memTags.push_back("HcclBuffer");

    if (memHandles == nullptr) {
        // cclBuffer 已入 memTags，不因用户内存异常阻断注册流程，上游仍可获得 cclBuffer handle
        HCCL_INFO("[CommMems] memHandles is null, skip user memory");
        return HCCL_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(memMutex_);
    for (uint32_t i = 0; i < memHandleNum; i++) {
        auto* handle = static_cast<CommMemInfo*>(memHandles[i]);
        if (handle == nullptr) {
            HCCL_ERROR("[CommMems] memHandle[%p] not found", handle);
            return HCCL_E_NOT_FOUND;
        }
        memTags.push_back(handle->memTag);
    }
    return HCCL_SUCCESS;
}

HcclResult CommMems::GetAllMemory(std::vector<HcclMem>& memVec, std::vector<std::string>& memTags, uint64_t& version)
{
    // cclBuffer
    HcclMem cclMem;
    cclMem.size = cclMemInfo_.mem.size;
    cclMem.type = ConvertCommToHcclMemType(cclMemInfo_.mem.type);
    cclMem.addr = cclMemInfo_.mem.addr;
    memVec.push_back(cclMem);
    memTags.push_back("HcclBuffer");

    // 遍历 opBindings_ 全量
    std::lock_guard<std::mutex> lock(memMutex_);
    for (auto& tagHandle : opBindings_) {
        auto& memInfo = tagHandle.second;
        HcclMem mem;
        mem.addr = memInfo->mem.addr;
        mem.size = memInfo->mem.size;
        mem.type = ConvertCommToHcclMemType(memInfo->mem.type);
        memVec.push_back(mem);
        memTags.push_back(tagHandle.first);
    }

    version = memVersion_;
    HCCL_INFO("[CommMems][%s] total memory count[%zu] version[%llu]", __func__, memVec.size(), version);
    return HCCL_SUCCESS;
}

} // namespace hccl
