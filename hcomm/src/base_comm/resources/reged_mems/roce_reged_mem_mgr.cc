/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "endpoint_pair.h"
#include "log.h"
#include "roce_reged_mem_mgr.h"
#include "exchange_rdma_buffer_dto.h"
#include "local_rdma_rma_buffer.h"
#include "hccl_one_sided_data.h"
#include "acl/acl_rt.h"

namespace hcomm {

RoceRegedMemMgr::RoceRegedMemMgr(RdmaHandle rdmaHandle, bool useAllocMemBase)
    : rdmaHandle_(rdmaHandle),
      useAllocMemBase_(useAllocMemBase)
{
    localRdmaRmaBufferMgr_ = std::make_unique<LocalRdmaRmaBufferMgr>();
    HCCL_INFO(
        "[RoceRegedMemMgr] construct this[%p] rdmaHandle[%p] useAllocMemBase[%d]", this, rdmaHandle_,
        static_cast<int>(useAllocMemBase_));
}

RoceRegedMemMgr::~RoceRegedMemMgr()
{
    HCCL_INFO(
        "[RoceRegedMemMgr] destroy this[%p] rdmaHandle[%p] useAllocMemBase[%d] "
        "allocToMr[%zu] handleMap[%zu] buffers[%zu] records[%zu]",
        this, rdmaHandle_, static_cast<int>(useAllocMemBase_), allocToMrMap_.size(), handleToAllocKey_.size(),
        allRegisteredBuffers_.size(), handlesRecords_.size());
}

HcclResult RoceRegedMemMgr::GetMemAllocAddrRange(const HcommMem& mem, MemKey& allocKey)
{
    void* basePtr = nullptr;
    size_t rangeSize = 0;
    const aclError aclRet = aclrtMemGetAddressRange(mem.addr, &basePtr, &rangeSize);
    if (aclRet != ACL_SUCCESS || basePtr == nullptr || rangeSize == 0) {
        HCCL_ERROR(
            "[RoceRegedMemMgr][GetMemAllocAddrRange] aclrtMemGetAddressRange failed, "
            "ret[%d] base[%p] size[%zu], user {%p, %llu}",
            static_cast<int>(aclRet), basePtr, rangeSize, mem.addr, mem.size);
        return HCCL_E_MEMORY;
    }

    const uintptr_t base = ReinterpretAs<uintptr_t>(basePtr);
    const uintptr_t userStart = ReinterpretAs<uintptr_t>(mem.addr);
    const uintptr_t userEnd = userStart + static_cast<uintptr_t>(mem.size);
    const uintptr_t baseEnd = base + static_cast<uintptr_t>(rangeSize);
    if (userStart < base || userEnd < userStart || userEnd > baseEnd) {
        HCCL_ERROR(
            "[RoceRegedMemMgr][GetMemAllocAddrRange] user window {%p, %llu} exceeds alloc {%p, %zu}", mem.addr,
            mem.size, basePtr, rangeSize);
        return HCCL_E_PARA;
    }

    allocKey = MemKey(base, static_cast<uint64_t>(rangeSize));
    HCCL_INFO(
        "[RoceRegedMemMgr][GetMemAllocAddrRange] user {%p, %llu} -> alloc {%p, %llu}", mem.addr, mem.size,
        ReinterpretAs<void*>(allocKey.Addr()), allocKey.Size());
    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle)
{
    HCCL_INFO("[%s] Begin this[%p]", __FUNCTION__, this);
    CHK_PTR_NULL(mem);
    CHK_PTR_NULL(memHandle);
    std::lock_guard<std::mutex> lock(memMtx_);
    CHK_RET(ValidateMemParams(*mem, memHandle));

    if (mem->type == COMM_MEM_TYPE_DEVICE && useAllocMemBase_) {
        // BufferKey 无默认构造；占位后由 GetMemAllocAddrRange 全量覆盖（失败硬返回，不读初值）
        MemKey allocKey(0, 0);
        CHK_RET(GetMemAllocAddrRange(*mem, allocKey));
        return RegisterByMemAllocAddrRange(*mem, memTag, memHandle, allocKey);
    }

    CHK_PTR_NULL(localRdmaRmaBufferMgr_);
    const HcclResult ret = RegisterMemoryImpl(
        *mem, memTag, memHandle, localRdmaRmaBufferMgr_, allRegisteredBuffers_, &handlesRecords_, "RoceRegedMemMgr",
        [&](auto& bufPtr, auto& parent) {
            return std::make_shared<Hccl::LocalRdmaRmaBuffer>(
                bufPtr, rdmaHandle_, parent->GetLkey(), parent->GetRkey(), parent->GetMrHandle());
        },
        [&](auto& bufPtr) {
            return std::make_shared<Hccl::LocalRdmaRmaBuffer>(bufPtr, rdmaHandle_);
        });
    if (ret == HCCL_SUCCESS) {
        HCCL_INFO(
            "[RoceRegedMemMgr][Register] this[%p] path[legacy] handle[%p] user {%p, %llu}", this, *memHandle, mem->addr,
            mem->size);
    }
    return ret;
}

HcclResult RoceRegedMemMgr::UnregisterMemory(void* memHandle)
{
    HCCL_INFO("[%s] Begin this[%p] handle[%p]", __FUNCTION__, this, memHandle);
    CHK_PTR_NULL(memHandle);
    std::lock_guard<std::mutex> lock(memMtx_);

    // MemAlloc 路径以 handleToAllocKey_ 为准（比再判 useAllocMemBase+DEVICE 更准确）
    auto* buffer = static_cast<Hccl::LocalRdmaRmaBuffer*>(memHandle);
    CHK_PTR_NULL(buffer);
    auto handleIt = handleToAllocKey_.find(buffer);
    if (handleIt != handleToAllocKey_.end()) {
        return UnregisterByMemAllocAddrRange(buffer, handleIt->second);
    }

    CHK_PTR_NULL(localRdmaRmaBufferMgr_);
    const HcclResult ret = UnregisterMemoryImpl(
        memHandle, localRdmaRmaBufferMgr_, allRegisteredBuffers_, &handlesRecords_,
        [](auto* b) {
            return b->GetLkey();
        },
        [](auto a, auto b) {
            return a == b;
        });
    if (ret == HCCL_SUCCESS) {
        HCCL_INFO("[RoceRegedMemMgr][Unregister] this[%p] path[legacy] handle[%p]", this, memHandle);
    }
    return ret;
}

HcclResult
RoceRegedMemMgr::AcquireAllocMr(const MemKey& allocKey, HcclMemType memType, const char* memTag, bool& createdMr)
{
    AllocMrEntry& slot = allocToMrMap_[allocKey];
    createdMr = false;
    if (slot.ref == 0) {
        std::shared_ptr<Hccl::Buffer> allocBuf;
        EXCEPTION_CATCH(
            (allocBuf = std::make_shared<Hccl::Buffer>(allocKey.Addr(), allocKey.Size(), memType, memTag)), {
                allocToMrMap_.erase(allocKey);
                return HCCL_E_PTR;
            });
        EXCEPTION_CATCH((slot.mr = std::make_shared<Hccl::LocalRdmaRmaBuffer>(allocBuf, rdmaHandle_)), {
            allocToMrMap_.erase(allocKey);
            return HCCL_E_PTR;
        });
        createdMr = true;
    }
    slot.ref++;
    HCCL_INFO(
        "[RoceRegedMemMgr][AcquireAllocMr] this[%p] alloc {%p, %llu} allocRef[%llu] createdMr[%d]", this,
        ReinterpretAs<void*>(allocKey.Addr()), allocKey.Size(), slot.ref, static_cast<int>(createdMr));
    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::ReleaseAllocMrRef(const MemKey& allocKey, uint64_t& allocRefAfter)
{
    auto allocIt = allocToMrMap_.find(allocKey);
    if (allocIt == allocToMrMap_.end() || allocIt->second.ref == 0) {
        HCCL_ERROR(
            "[RoceRegedMemMgr][ReleaseAllocMrRef] this[%p] alloc MR not found, key {%p, %llu}", this,
            ReinterpretAs<void*>(allocKey.Addr()), allocKey.Size());
        return HCCL_E_NOT_FOUND;
    }
    allocRefAfter = --(allocIt->second.ref);
    HCCL_INFO(
        "[RoceRegedMemMgr][ReleaseAllocMrRef] this[%p] alloc {%p, %llu} allocRefAfter[%llu]", this,
        ReinterpretAs<void*>(allocKey.Addr()), allocKey.Size(), allocRefAfter);
    if (allocRefAfter == 0) {
        allocIt->second.mr.reset();
        allocToMrMap_.erase(allocIt);
    }
    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::PublishMemHandle(
    const HcommMem& mem, const char* memTag, const MemKey& userKey, const MemKey& allocKey, bool createdMr,
    void** memHandle)
{
    auto allocIt = allocToMrMap_.find(allocKey);
    if (allocIt == allocToMrMap_.end() || allocIt->second.mr == nullptr) {
        HCCL_ERROR(
            "[RoceRegedMemMgr][PublishMemHandle] alloc MR missing, key {%p, %llu}",
            ReinterpretAs<void*>(allocKey.Addr()), allocKey.Size());
        return HCCL_E_INTERNAL;
    }
    AllocMrEntry& slot = allocIt->second;

    // 首次且用户窗==alloc：返回 MR 本体；否则 alias（仅 alias 路径构造 userBuf）
    std::shared_ptr<Hccl::LocalRdmaRmaBuffer> rmaBuffer;
    if (createdMr && userKey == allocKey) {
        rmaBuffer = slot.mr;
    } else {
        std::shared_ptr<Hccl::Buffer> userBuf;
        EXCEPTION_CATCH(
            (userBuf = std::make_shared<Hccl::Buffer>(
                 ReinterpretAs<uintptr_t>(mem.addr), mem.size, static_cast<HcclMemType>(mem.type), memTag)),
            return HCCL_E_PTR);
        EXCEPTION_CATCH(
            (rmaBuffer = std::make_shared<Hccl::LocalRdmaRmaBuffer>(
                 userBuf, rdmaHandle_, slot.mr->GetLkey(), slot.mr->GetRkey(), slot.mr->GetMrHandle())),
            return HCCL_E_PTR);
    }

    handleToAllocKey_.insert_or_assign(rmaBuffer.get(), allocKey);
    *memHandle = static_cast<void*>(rmaBuffer.get());
    allRegisteredBuffers_.emplace_back(rmaBuffer, false);
    handlesRecords_.push_back(rmaBuffer);
    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::UnpublishMemHandle(Hccl::LocalRdmaRmaBuffer* buffer)
{
    auto bufIt = std::find_if(allRegisteredBuffers_.begin(), allRegisteredBuffers_.end(), [buffer](const auto& entry) {
        return entry.first.get() == buffer;
    });
    if (bufIt == allRegisteredBuffers_.end()) {
        HCCL_ERROR(
            "[RoceRegedMemMgr][UnpublishMemHandle] this[%p] buffer %p not in allRegisteredBuffers_", this, buffer);
        handleToAllocKey_.erase(buffer);
        return HCCL_E_NOT_FOUND;
    }

    handlesRecords_.erase(
        std::remove(handlesRecords_.begin(), handlesRecords_.end(), bufIt->first), handlesRecords_.end());
    allRegisteredBuffers_.erase(bufIt);
    handleToAllocKey_.erase(buffer);
    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::RegisterByMemAllocAddrRange(
    const HcommMem& mem, const char* memTag, void** memHandle, const MemKey& allocKey)
{
    const MemKey userKey(ReinterpretAs<uintptr_t>(mem.addr), mem.size);

    bool createdMr = false;
    CHK_RET(AcquireAllocMr(allocKey, static_cast<HcclMemType>(mem.type), memTag, createdMr));

    const HcclResult pubRet = PublishMemHandle(mem, memTag, userKey, allocKey, createdMr, memHandle);
    if (pubRet != HCCL_SUCCESS) {
        uint64_t unused = 0;
        (void)ReleaseAllocMrRef(allocKey, unused);
        return pubRet;
    }

    const uint64_t allocRef = allocToMrMap_.at(allocKey).ref;
    HCCL_INFO(
        "[RoceRegedMemMgr][Register] this[%p] path[MemAlloc] handle[%p] user {%p, %llu} alloc {%p, %llu} "
        "allocRef[%llu] isAlias[%d]",
        this, *memHandle, mem.addr, mem.size, ReinterpretAs<void*>(allocKey.Addr()), allocKey.Size(), allocRef,
        static_cast<int>(static_cast<Hccl::LocalRdmaRmaBuffer*>(*memHandle)->IsAlias()));
    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::UnregisterByMemAllocAddrRange(Hccl::LocalRdmaRmaBuffer* buffer, const MemKey& allocKey)
{
    CHK_PTR_NULL(buffer);
    const MemKey userKey(buffer->GetBufferInfo().first, buffer->GetBufferInfo().second);

    // 先卸本 handle 账本；失败则不减 alloc 引用，避免半成功
    CHK_RET(UnpublishMemHandle(buffer));

    uint64_t allocRefAfter = 0;
    CHK_RET(ReleaseAllocMrRef(allocKey, allocRefAfter));

    HCCL_INFO(
        "[RoceRegedMemMgr][Unregister] this[%p] path[MemAlloc] handle[%p] user {%p, %llu} alloc {%p, %llu} "
        "allocRefAfter[%llu]",
        this, buffer, ReinterpretAs<void*>(userKey.Addr()), userKey.Size(), ReinterpretAs<void*>(allocKey.Addr()),
        allocKey.Size(), allocRefAfter);
    return HCCL_SUCCESS;
}

HcclResult
RoceRegedMemMgr::GetMemDesc(const EndpointDesc endpointDesc, Hccl::LocalRdmaRmaBuffer* localRdmaRmaBuffer) const
{
    auto dto = localRdmaRmaBuffer->GetExchangeDto();
    Hccl::BinaryStream localRdmaRmaBufferStream;
    dto->Serialize(localRdmaRmaBufferStream);
    std::vector<char> tempLocalMemDesc;
    localRdmaRmaBufferStream.Dump(tempLocalMemDesc);
    HCCL_DEBUG("[RoceRegedMemMgr][GetMemDesc] [%s] dump data size [%u]", __func__, tempLocalMemDesc.size());
    if (tempLocalMemDesc.empty()) {
        HCCL_ERROR("[RoceRegedMemMgr][GetMemDesc] [%s] tempLocalMemDesc export failed.", __func__);
        return HCCL_E_INTERNAL;
    }

    std::vector<char> tempLocalEndpointDesc;
    tempLocalEndpointDesc.resize(sizeof(EndpointDesc));
    if (memcpy_s(tempLocalEndpointDesc.data(), sizeof(EndpointDesc), &endpointDesc, sizeof(EndpointDesc)) != EOK) {
        HCCL_ERROR("[RoceRegedMemMgr][GetMemDesc] [%s] endpointDesc memcpy_s failed.", __func__);
        return HCCL_E_INTERNAL;
    }

    tempLocalMemDesc.insert(tempLocalMemDesc.end(), tempLocalEndpointDesc.begin(), tempLocalEndpointDesc.end());
    localRdmaRmaBuffer->Desc = std::move(tempLocalMemDesc);
    return HCCL_SUCCESS;
}

HcclResult
RoceRegedMemMgr::MemoryExport(const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(memDescLen);
    std::lock_guard<std::mutex> lock(memMtx_);

    Hccl::LocalRdmaRmaBuffer* localRdmaRmaBuffer = nullptr;
    CHK_RET(ValidateMemExportHandle(memHandle, allRegisteredBuffers_, localRdmaRmaBuffer));
    CHK_RET(GetMemDesc(endpointDesc, localRdmaRmaBuffer));

    *memDescLen = static_cast<uint32_t>(localRdmaRmaBuffer->Desc.size());
    *memDesc = static_cast<void*>(localRdmaRmaBuffer->Desc.data());
    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::GetParamsFromMemDesc(
    const void* memDesc, uint32_t descLen, EndpointDesc& endpointDesc, Hccl::ExchangeRdmaBufferDto& dto) const
{
    const char* description = static_cast<const char*>(memDesc);

    if (descLen < sizeof(EndpointDesc)) {
        HCCL_ERROR(
            "[RoceRegedMemMgr][GetParamsFromMemDesc] [%s] descLen[%u] is too small. aim size:[%llu]", __func__, descLen,
            sizeof(EndpointDesc));
        return HCCL_E_INTERNAL;
    }
    if (memcpy_s(
            &endpointDesc, sizeof(EndpointDesc), description + descLen - sizeof(EndpointDesc), sizeof(EndpointDesc))
        != EOK) {
        HCCL_ERROR(
            "[RoceRegedMemMgr][GetParamsFromMemDesc] [%s] endpointDesc copy error. aim size:[%llu]", __func__,
            sizeof(EndpointDesc));
        return HCCL_E_INTERNAL;
    }

    std::vector<char> tempDesc{};
    tempDesc.resize(TRANSPORT_EMD_ESC_SIZE);
    tempDesc.assign(description, description + descLen - sizeof(EndpointDesc));
    Hccl::BinaryStream remoteRdmaRmaBufferStream(tempDesc);
    dto.Deserialize(remoteRdmaRmaBufferStream);
    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(outMem);
    std::lock_guard<std::mutex> lock(memMtx_);

    EndpointDesc endpointDesc;
    Hccl::ExchangeRdmaBufferDto dto;
    CHK_RET(GetParamsFromMemDesc(memDesc, descLen, endpointDesc, dto));

    std::shared_ptr<Hccl::RemoteRdmaRmaBuffer> remoteRdmaRmaBuffer;
    EXCEPTION_CATCH(remoteRdmaRmaBuffer = std::make_shared<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle_, dto),
                    return HCCL_E_PTR;);

    hccl::BufferKey<uintptr_t, u64> tempKey(static_cast<uintptr_t>(dto.addr), dto.size);
    if (remoteRdmaRmaBufferMgrs_.find(endpointDesc) == remoteRdmaRmaBufferMgrs_.end()) {
        std::unique_ptr<RemoteRdmaRmaBufferMgr> remoteRdmaRmaBufferMgr;
        EXCEPTION_CATCH((remoteRdmaRmaBufferMgr = std::make_unique<RemoteRdmaRmaBufferMgr>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(remoteRdmaRmaBufferMgr);
        remoteRdmaRmaBufferMgrs_[endpointDesc] = std::move(remoteRdmaRmaBufferMgr);
        HCCL_INFO("remoteRdmaRmaBufferMgrs_ add remoteRdmaRmaBufferMgr successfully!");
    }

    auto resultPair = remoteRdmaRmaBufferMgrs_[endpointDesc]->Add(tempKey, remoteRdmaRmaBuffer);
    if (!resultPair.second) {
        HCCL_ERROR("[RoceRegedMemMgr][MemoryImport] This memDesc has already been imported!");
        return HCCL_E_AGAIN;
    }

    outMem->addr = ReinterpretAs<void*>(remoteRdmaRmaBuffer->GetAddr());
    outMem->size = remoteRdmaRmaBuffer->GetSize();
    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::MemoryUnimport(const void* memDesc, uint32_t descLen)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memDesc);
    std::lock_guard<std::mutex> lock(memMtx_);

    EndpointDesc endpointDesc;
    Hccl::ExchangeRdmaBufferDto dto;
    CHK_RET(GetParamsFromMemDesc(memDesc, descLen, endpointDesc, dto));

    if (remoteRdmaRmaBufferMgrs_.find(endpointDesc) == remoteRdmaRmaBufferMgrs_.end()) {
        HCCL_ERROR("[RoceRegedMemMgr][MemoryUnimport] Remote buffer manager Not Found.");
        return HCCL_E_NOT_FOUND;
    }

    HCCL_INFO("[MemoryUnimport][Rdma] MemoryUnimport");
    hccl::BufferKey<uintptr_t, u64> tempKey(static_cast<uintptr_t>(dto.addr), dto.size);

    bool resultPair = false;
    EXCEPTION_CATCH(resultPair = remoteRdmaRmaBufferMgrs_[endpointDesc]->Del(tempKey), return HCCL_E_NOT_FOUND);
    if (!resultPair) {
        HCCL_INFO("[RoceRegedMemMgr][[MemoryUnimport] Memory reference count is larger than 0"
                  "(used by other RemoteRank).");
        return HCCL_E_AGAIN;
    }
    if (!remoteRdmaRmaBufferMgrs_[endpointDesc]->size()) {
        remoteRdmaRmaBufferMgrs_.erase(endpointDesc);
    }
    return HCCL_SUCCESS;
}

HcclResult RoceRegedMemMgr::GetAllMemHandles(void** memHandles, uint32_t* memHandleNum)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    std::lock_guard<std::mutex> lock(memMtx_);
    CHK_PTR_NULL(memHandles);
    CHK_PTR_NULL(memHandleNum);
    *memHandleNum = static_cast<uint32_t>(handlesRecords_.size());
    *memHandles = handlesRecords_.empty() ? nullptr : static_cast<void*>(handlesRecords_.data());
    HCCL_INFO("[RoceRegedMemMgr][GetAllMemHandles] memHandleNum[%u]", *memHandleNum);
    return HCCL_SUCCESS;
}

} // namespace hcomm
