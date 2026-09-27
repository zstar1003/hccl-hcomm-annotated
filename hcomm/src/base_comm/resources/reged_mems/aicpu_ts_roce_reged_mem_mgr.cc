/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "aicpu_ts_roce_reged_mem_mgr.h"
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include "securec.h"
#include "adapter_hccp.h"
#include "hccl_network.h"
#include "log.h"
#include "rma_buffer.h"

namespace {
using LocalRdmaRmaBufferMgr = hccl::NetDevContext::LocalRdmaRmaBufferMgr;
struct LocalBufferMgrCtx {
    std::shared_ptr<std::mutex> mu;
    std::shared_ptr<LocalRdmaRmaBufferMgr> mgr;
};

std::mutex g_phyLocalRdmaBundleMapMu;
std::unordered_map<s32, std::shared_ptr<LocalBufferMgrCtx>> g_phyIdToLocalBufferMgrCtx;

std::shared_ptr<LocalBufferMgrCtx> GetOrCreateLocalBufferMgr(s32 devicePhyId)
{
    std::lock_guard<std::mutex> mapLock(g_phyLocalRdmaBundleMapMu);
    std::shared_ptr<LocalBufferMgrCtx>& slot = g_phyIdToLocalBufferMgrCtx[devicePhyId];
    if (slot == nullptr) {
        std::shared_ptr<LocalRdmaRmaBufferMgr> mgr;
        EXCEPTION_CATCH((mgr = std::make_shared<LocalRdmaRmaBufferMgr>()), return nullptr);
        slot = std::make_shared<LocalBufferMgrCtx>();
        slot->mu = std::make_shared<std::mutex>();
        slot->mgr = std::move(mgr);
    }
    return slot;
}
} // namespace

namespace hcomm {
AicpuTsRoceRegedMemMgr::AicpuTsRoceRegedMemMgr(HcclNetDev netDev, RdmaHandle rdmaHandle)
    : netDev_(netDev),
      rdmaHandle_(rdmaHandle)
{
    if (netDev_ != nullptr) {
        auto* netDevCtx = static_cast<hccl::NetDevContext*>(netDev_);
        const s32 phyId = netDevCtx->GetPhyId();
        std::shared_ptr<LocalBufferMgrCtx> ctx = GetOrCreateLocalBufferMgr(phyId);
        if (ctx != nullptr) {
            localRdmaRmaBufferMgr_ = ctx->mgr;
        }
        HCCL_INFO(
            "[AicpuTsRoceRegedMemMgr] ctor netDev[%p] phyId[%d] process-local localRdmaRmaBufferMgr[%p] (not NetDev "
            "embed)",
            static_cast<void*>(netDev_), static_cast<int>(phyId), static_cast<void*>(localRdmaRmaBufferMgr_.get()));
    } else {
        HCCL_INFO("[AicpuTsRoceRegedMemMgr] ctor netDev is null, local mgr unset");
    }
}

void AicpuTsRoceRegedMemMgr::TrackRegisteredBuffer(const std::shared_ptr<hccl::LocalRdmaRmaBuffer>& localBuffer)
{
    void* const handlePtr = static_cast<void*>(localBuffer.get());
    const bool alreadyListed
        = std::any_of(allRegisteredBuffers_.begin(), allRegisteredBuffers_.end(), [handlePtr](const auto& entry) {
              return static_cast<void*>(entry.first.get()) == handlePtr;
          });
    if (alreadyListed) {
        return;
    }
    allRegisteredBuffers_.emplace_back(localBuffer, false);
    HcclBuf rec{};
    rec.addr = localBuffer->GetAddr();
    rec.len = localBuffer->GetSize();
    auto* rma = dynamic_cast<hccl::RmaBuffer*>(localBuffer.get());
    if (rma == nullptr) {
        HCCL_ERROR("[AicpuTsRoceRegedMemMgr][TrackRegisteredBuffer] rma is nullptr");
        return;
    }
    rec.handle = static_cast<void*>(rma);
    hcclBufRecords_.push_back(rec);
}

HcclResult AicpuTsRoceRegedMemMgr::RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle)
{
    (void)memTag;
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(mem);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(netDev_);
    CHK_PTR_NULL(localRdmaRmaBufferMgr_);

    auto* netDevCtx = static_cast<hccl::NetDevContext*>(netDev_);

    std::shared_ptr<LocalBufferMgrCtx> ctx = GetOrCreateLocalBufferMgr(netDevCtx->GetPhyId());
    CHK_PTR_NULL(ctx);
    std::lock_guard<std::mutex> phyLocalLock(*ctx->mu);

    hccl::RmaMemType memType = static_cast<hccl::RmaMemType>(mem->type);
    hccl::BufferKey<uintptr_t, u64> tempKey(ReinterpretAs<uintptr_t>(mem->addr), static_cast<u64>(mem->size));
    auto findPair = localRdmaRmaBufferMgr_->Find(tempKey);

    std::shared_ptr<hccl::LocalRdmaRmaBuffer> localRdmaRmaBuffer;
    if (findPair.first) {
        auto parentBuffer = findPair.second;
        EXCEPTION_CATCH(
            (localRdmaRmaBuffer = std::make_shared<hccl::LocalRdmaRmaBuffer>(
                 netDevCtx, mem->addr, static_cast<u64>(mem->size), memType, *parentBuffer)),
            return HCCL_E_PTR);
        CHK_RET(AddBuffer(localRdmaRmaBufferMgr_, parentBuffer));
        HCCL_INFO("[AicpuTsRoceRegedMemMgr][RegisterMemory] alias created, key {%p, %llu}", mem->addr, mem->size);
    } else {
        EXCEPTION_CATCH(
            (localRdmaRmaBuffer
             = std::make_shared<hccl::LocalRdmaRmaBuffer>(netDevCtx, mem->addr, static_cast<u64>(mem->size), memType)),
            return HCCL_E_PTR);

        HcclResult ret = localRdmaRmaBuffer->Init();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[AicpuTsRoceRegedMemMgr][RegisterMemory] Init failed, ret[%d]", ret);
            return ret;
        }

        CHK_RET(AddBuffer(localRdmaRmaBufferMgr_, localRdmaRmaBuffer));
        HCCL_INFO("[AicpuTsRoceRegedMemMgr][RegisterMemory] success, key {%p, %llu}", mem->addr, mem->size);
    }

    *memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    TrackRegisteredBuffer(localRdmaRmaBuffer);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceRegedMemMgr::UnregisterMemory(void* memHandle)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(netDev_);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(localRdmaRmaBufferMgr_);

    auto* netDevCtx = static_cast<hccl::NetDevContext*>(netDev_);
    std::shared_ptr<LocalBufferMgrCtx> ctx = GetOrCreateLocalBufferMgr(netDevCtx->GetPhyId());
    CHK_PTR_NULL(ctx);
    std::lock_guard<std::mutex> phyLocalLock(*ctx->mu);

    auto* buffer = static_cast<hccl::LocalRdmaRmaBuffer*>(memHandle);

    // IsAlias() 直接区分父子buffer：
    //   - 父buffer (IsAlias()=false): 自己的key在tree中 → Del(ownKey)
    //   - 子buffer (IsAlias()=true):  自己的key不在tree中 → 通过GetKey找父key做Del
    hccl::BufferKey<uintptr_t, u64> ownKey(ReinterpretAs<uintptr_t>(buffer->GetAddr()), buffer->GetSize());
    hccl::LocalRdmaRmaBuffer* refBuffer = buffer;
    if (buffer->IsAlias()) {
        refBuffer = ResolveAliasParent(
            localRdmaRmaBufferMgr_, ownKey, buffer, allRegisteredBuffers_,
            [](auto* b) {
                return b->GetKey();
            },
            [](auto a, auto b) {
                return a == b;
            });
        if (refBuffer == nullptr) {
            HCCL_ERROR("[AicpuTsRoceRegedMemMgr][UnregisterMemory] alias parent not found");
            return HCCL_E_NOT_FOUND;
        }
    }

    hccl::BufferKey<uintptr_t, u64> tempKey(ReinterpretAs<uintptr_t>(refBuffer->GetAddr()), refBuffer->GetSize());

    bool delOk = false;
    EXCEPTION_CATCH(delOk = localRdmaRmaBufferMgr_->Del(tempKey), return HCCL_E_NOT_FOUND);
    if (!delOk) {
        HCCL_INFO("[AicpuTsRoceRegedMemMgr][UnregisterMemory] ref count > 0");
    }

    CleanupBufferRecords(memHandle, buffer, ownKey);
    HCCL_INFO(
        "[AicpuTsRoceRegedMemMgr][UnregisterMemory] success, memHandle[%p] key {%p, %llu}", memHandle,
        buffer->GetAddr(), static_cast<unsigned long long>(buffer->GetSize()));
    return HCCL_SUCCESS;
}

void AicpuTsRoceRegedMemMgr::CleanupBufferRecords(
    const void* memHandle, hccl::LocalRdmaRmaBuffer* buffer, const hccl::BufferKey<uintptr_t, u64>& ownKey)
{
    exportDescByBuffer_.erase(buffer);

    auto it = std::find_if(allRegisteredBuffers_.begin(), allRegisteredBuffers_.end(), [memHandle](const auto& entry) {
        return entry.first.get() == memHandle;
    });
    if (it != allRegisteredBuffers_.end()) {
        // IsInTree判断tree中是否还有该key的引用
        if (!localRdmaRmaBufferMgr_->IsInTree(ownKey)) {
            allRegisteredBuffers_.erase(it);
        } else {
            it->second = true;
        }
    }

    hcclBufRecords_.erase(
        std::remove_if(
            hcclBufRecords_.begin(), hcclBufRecords_.end(),
            [memHandle](const HcclBuf& b) {
                return b.handle == memHandle;
            }),
        hcclBufRecords_.end());
}

HcclResult AicpuTsRoceRegedMemMgr::MemoryExport(
    const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(memDescLen);
    CHK_PTR_NULL(netDev_);
    CHK_PTR_NULL(localRdmaRmaBufferMgr_);

    auto* netDevCtx = static_cast<hccl::NetDevContext*>(netDev_);
    std::shared_ptr<LocalBufferMgrCtx> ctx = GetOrCreateLocalBufferMgr(netDevCtx->GetPhyId());
    CHK_PTR_NULL(ctx);
    std::lock_guard<std::mutex> phyLocalLock(*ctx->mu);

    hccl::LocalRdmaRmaBuffer* buf = nullptr;
    CHK_RET(ValidateMemExportHandle(memHandle, allRegisteredBuffers_, buf));
    std::string& ser = buf->Serialize();
    if (ser.empty()) {
        HCCL_ERROR("[AicpuTsRoceRegedMemMgr][MemoryExport] Serialize empty");
        return HCCL_E_INTERNAL;
    }

    std::vector<char>& blob = exportDescByBuffer_[buf];
    blob.clear();
    blob.reserve(ser.size() + sizeof(EndpointDesc));
    blob.insert(blob.end(), ser.begin(), ser.end());

    std::vector<char> ep(sizeof(EndpointDesc));
    if (memcpy_s(ep.data(), sizeof(EndpointDesc), &endpointDesc, sizeof(EndpointDesc)) != EOK) {
        HCCL_ERROR("[AicpuTsRoceRegedMemMgr][MemoryExport] endpointDesc memcpy_s failed");
        return HCCL_E_INTERNAL;
    }
    blob.insert(blob.end(), ep.begin(), ep.end());

    *memDesc = static_cast<void*>(blob.data());
    *memDescLen = static_cast<uint32_t>(blob.size());
    HCCL_INFO(
        "[AicpuTsRoceRegedMemMgr][MemoryExport] success memHandle[%p] memDescLen[%u] rdmaSerLen[%zu]", memHandle,
        *memDescLen, ser.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceRegedMemMgr::GetParamsFromMemDesc(
    const void* memDesc, uint32_t descLen, EndpointDesc& endpointDesc, std::string& rdmaBlob) const
{
    CHK_PTR_NULL(memDesc);
    if (descLen < sizeof(EndpointDesc)) {
        HCCL_ERROR("[AicpuTsRoceRegedMemMgr][GetParamsFromMemDesc] descLen[%u] too small", descLen);
        return HCCL_E_PARA;
    }
    const auto* base = static_cast<const char*>(memDesc);
    if (memcpy_s(&endpointDesc, sizeof(EndpointDesc), base + descLen - sizeof(EndpointDesc), sizeof(EndpointDesc))
        != EOK) {
        HCCL_ERROR("[AicpuTsRoceRegedMemMgr][GetParamsFromMemDesc] endpointDesc copy failed");
        return HCCL_E_INTERNAL;
    }
    rdmaBlob.assign(base, base + descLen - sizeof(EndpointDesc));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceRegedMemMgr::MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(outMem);

    EndpointDesc endpointDesc{};
    std::string rdmaBlob;
    CHK_RET(GetParamsFromMemDesc(memDesc, descLen, endpointDesc, rdmaBlob));

    std::shared_ptr<hccl::RemoteRdmaRmaBuffer> remoteBuf;
    EXCEPTION_CATCH(remoteBuf = std::make_shared<hccl::RemoteRdmaRmaBuffer>(), return HCCL_E_PTR);
    CHK_RET(remoteBuf->Deserialize(rdmaBlob));

    if (remoteRdmaRmaBufferMgrs_.find(endpointDesc) == remoteRdmaRmaBufferMgrs_.end()) {
        std::unique_ptr<RemoteRdmaRmaBufferMgr> mgr;
        EXCEPTION_CATCH(mgr = std::make_unique<RemoteRdmaRmaBufferMgr>(), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(mgr);
        remoteRdmaRmaBufferMgrs_[endpointDesc] = std::move(mgr);
        HCCL_INFO(
            "[AicpuTsRoceRegedMemMgr][MemoryImport] created RemoteRdmaRmaBufferMgr for new endpoint, mgrCnt[%zu]",
            remoteRdmaRmaBufferMgrs_.size());
    }

    hccl::BufferKey<uintptr_t, u64> tempKey(ReinterpretAs<uintptr_t>(remoteBuf->GetAddr()), remoteBuf->GetSize());
    auto resultPair = remoteRdmaRmaBufferMgrs_[endpointDesc]->Add(tempKey, remoteBuf);
    if (!resultPair.second) {
        HCCL_ERROR("[AicpuTsRoceRegedMemMgr][MemoryImport] memDesc already imported");
        return HCCL_E_AGAIN;
    }

    outMem->addr = remoteBuf->GetAddr();
    outMem->size = remoteBuf->GetSize();
    outMem->type = COMM_MEM_TYPE_DEVICE;
    HCCL_INFO(
        "[AicpuTsRoceRegedMemMgr][MemoryImport] success descLen[%u] outMem addr[%p] size[%llu], mgrCnt[%zu]", descLen,
        outMem->addr, static_cast<unsigned long long>(outMem->size), remoteRdmaRmaBufferMgrs_.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceRegedMemMgr::MemoryUnimport(const void* memDesc, uint32_t descLen)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memDesc);

    EndpointDesc endpointDesc{};
    std::string rdmaBlob;
    CHK_RET(GetParamsFromMemDesc(memDesc, descLen, endpointDesc, rdmaBlob));

    auto mgrIt = remoteRdmaRmaBufferMgrs_.find(endpointDesc);
    if (mgrIt == remoteRdmaRmaBufferMgrs_.end()) {
        HCCL_ERROR("[AicpuTsRoceRegedMemMgr][MemoryUnimport] remote mgr not found");
        return HCCL_E_NOT_FOUND;
    }

    std::shared_ptr<hccl::RemoteRdmaRmaBuffer> probe;
    EXCEPTION_CATCH(probe = std::make_shared<hccl::RemoteRdmaRmaBuffer>(), return HCCL_E_PTR);
    CHK_RET(probe->Deserialize(rdmaBlob));

    hccl::BufferKey<uintptr_t, u64> tempKey(ReinterpretAs<uintptr_t>(probe->GetAddr()), probe->GetSize());
    bool delOk = false;
    EXCEPTION_CATCH(delOk = mgrIt->second->Del(tempKey), return HCCL_E_NOT_FOUND);
    if (!delOk) {
        HCCL_INFO("[AicpuTsRoceRegedMemMgr][MemoryUnimport] ref count > 0");
        return HCCL_E_AGAIN;
    }
    if (mgrIt->second->size() == 0) {
        remoteRdmaRmaBufferMgrs_.erase(mgrIt);
        HCCL_INFO(
            "[AicpuTsRoceRegedMemMgr][MemoryUnimport] erased empty remote mgr, descLen[%u] remainingMgrCnt[%zu]",
            descLen, remoteRdmaRmaBufferMgrs_.size());
    }
    HCCL_INFO(
        "[AicpuTsRoceRegedMemMgr][MemoryUnimport] success descLen[%u] key {%p, %llu}", descLen, probe->GetAddr(),
        static_cast<unsigned long long>(probe->GetSize()));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceRegedMemMgr::GetAllMemHandles(void** memHandles, uint32_t* memHandleNum)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memHandleNum);

    *memHandleNum = static_cast<uint32_t>(hcclBufRecords_.size());
    if (*memHandleNum == 0U) {
        *memHandles = nullptr;
        HCCL_INFO("[AicpuTsRoceRegedMemMgr][GetAllMemHandles] no records, memHandleNum[0]");
        return HCCL_SUCCESS;
    }
    *memHandles = static_cast<void*>(hcclBufRecords_.data());
    HCCL_INFO(
        "[AicpuTsRoceRegedMemMgr][GetAllMemHandles] memHandleNum[%u] hcclBufRecords[%p]", *memHandleNum,
        static_cast<void*>(hcclBufRecords_.data()));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceRegedMemMgr::GatherLocalMemDetails(std::vector<RoceMemDetails>& localOut) const
{
    localOut.reserve(allRegisteredBuffers_.size());
    for (const auto& entry : allRegisteredBuffers_) {
        if (entry.second) {
            continue;
        }
        const auto& buf = entry.first;
        if (buf == nullptr) {
            continue;
        }
        auto* rma = static_cast<hccl::RmaBuffer*>(buf.get());
        CHK_PTR_NULL(rma->GetDevAddr());
        RoceMemDetails r{};
        r.addr = static_cast<u64>(ReinterpretAs<uintptr_t>(rma->GetAddr()));
        r.devAddr = static_cast<u64>(ReinterpretAs<uintptr_t>(rma->GetDevAddr()));
        r.size = buf->GetSize();
        r.key = buf->GetKey();
        localOut.push_back(r);
        HCCL_INFO(
            "[AicpuTsRoceRegedMemMgr][GetAllMemDetails][local][%zu] addr[0x%llx] devAddr[0x%llx] size[%llu] key[%u]",
            localOut.size() - 1U, static_cast<unsigned long long>(r.addr), static_cast<unsigned long long>(r.devAddr),
            static_cast<unsigned long long>(r.size), r.key);
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceRegedMemMgr::AppendLocalNotifyMemDetails(std::vector<RoceMemDetails>& localOut) const
{
    CHK_PTR_NULL(rdmaHandle_);
    CHK_PTR_NULL(netDev_);
    auto* netCtx = static_cast<hccl::NetDevContext*>(netDev_);
    struct MrInfoT mrInfo {};
    CHK_RET(HrtRaGetNotifyMrInfo(static_cast<u32>(netCtx->GetPhyId()), rdmaHandle_, &mrInfo));
    CHK_PTR_NULL(mrInfo.addr);
    RoceMemDetails notifyMd{};
    notifyMd.addr = static_cast<u64>(ReinterpretAs<uintptr_t>(mrInfo.addr));
    notifyMd.devAddr = notifyMd.addr;
    notifyMd.size = static_cast<u64>(mrInfo.size);
    notifyMd.key = mrInfo.lkey;
    localOut.push_back(notifyMd);
    HCCL_INFO(
        "[AicpuTsRoceRegedMemMgr][GetAllMemDetails][local][notify] addr[0x%llx] devAddr[0x%llx] size[%llu] key[%u]",
        static_cast<unsigned long long>(notifyMd.addr), static_cast<unsigned long long>(notifyMd.devAddr),
        static_cast<unsigned long long>(notifyMd.size), notifyMd.key);
    return HCCL_SUCCESS;
}

void AicpuTsRoceRegedMemMgr::GatherRemoteMemDetails(std::vector<RoceMemDetails>& remoteOut) const
{
    for (const auto& epMgr : remoteRdmaRmaBufferMgrs_) {
        const auto& mgr = epMgr.second;
        if (mgr == nullptr) {
            continue;
        }
        mgr->ForEach(
            [&remoteOut](const hccl::BufferKey<uintptr_t, u64>&, const std::shared_ptr<hccl::RemoteRdmaRmaBuffer>& rb) {
                if (rb == nullptr) {
                    return;
                }
                auto* rma = static_cast<hccl::RmaBuffer*>(rb.get());
                if (rma->GetDevAddr() == nullptr) {
                    return;
                }
                RoceMemDetails r{};
                r.addr = static_cast<u64>(ReinterpretAs<uintptr_t>(rma->GetAddr()));
                r.devAddr = static_cast<u64>(ReinterpretAs<uintptr_t>(rma->GetDevAddr()));
                r.size = rb->GetSize();
                r.key = rb->GetKey();
                remoteOut.push_back(r);
                HCCL_INFO(
                    "[AicpuTsRoceRegedMemMgr][GetAllMemDetails][remote][%zu] addr[0x%llx] devAddr[0x%llx] size[%llu] "
                    "key[%u]",
                    remoteOut.size() - 1U, static_cast<unsigned long long>(r.addr),
                    static_cast<unsigned long long>(r.devAddr), static_cast<unsigned long long>(r.size), r.key);
            });
    }
}

HcclResult AicpuTsRoceRegedMemMgr::GetAllMemDetails(
    std::vector<RoceMemDetails>& localOut, std::vector<RoceMemDetails>& remoteOut) const
{
    localOut.clear();
    remoteOut.clear();
    CHK_RET(GatherLocalMemDetails(localOut));
    CHK_RET(AppendLocalNotifyMemDetails(localOut));
    GatherRemoteMemDetails(remoteOut);
    HCCL_INFO(
        "[AicpuTsRoceRegedMemMgr][GetAllMemDetails] summary localCnt[%zu] remoteCnt[%zu] remoteMgrCnt[%zu]",
        localOut.size(), remoteOut.size(), remoteRdmaRmaBufferMgrs_.size());
    return HCCL_SUCCESS;
}

} // namespace hcomm
