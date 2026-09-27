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
#include "log.h"
#include "hccs_reged_mem_mgr.h"
// for hccl_network.h
#include "hccl_network.h"
#include "adapter_rts.h"

using namespace hccl;
namespace hcomm {

HccsRegedMemMgr::HccsRegedMemMgr(HcclNetDevCtx netDevCtx) { netDevCtx_ = netDevCtx; }

HccsRegedMemMgr::~HccsRegedMemMgr() { allRegisteredBuffers_.clear(); }

HcclResult HccsRegedMemMgr::RegisterMemory(const HcommMem* mem, [[maybe_unused]] const char* memTag, void** memHandle)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(mem);
    CHK_RET(ValidateMemParams(*mem, memHandle));
    HCCL_INFO("[%s] addr[%p] size[%u] start", __FUNCTION__, mem->addr, mem->size);

    NetDevContext* netDevCtx = static_cast<NetDevContext*>(netDevCtx_);
    std::shared_ptr<LocalIpcRmaBufferMgr> localIpcRmaBufferMgr = netDevCtx->GetlocalIpcRmaBufferMgr();
    CHK_PTR_NULL(localIpcRmaBufferMgr);

    std::shared_ptr<hccl::LocalIpcRmaBuffer> localIpcRmaBuffer = nullptr;
    // LocalIpcRmaBuffer构造函数存在注册动作，在调用该构造函数前需检查是否注册过
    hccl::BufferKey<uintptr_t, u64> memKey(ReinterpretAs<uintptr_t>(mem->addr), mem->size);
    auto findPair = localIpcRmaBufferMgr->Find(memKey);
    if (findPair.first) {
        auto parentBuffer = findPair.second;
        EXCEPTION_CATCH(
            (localIpcRmaBuffer = std::make_shared<hccl::LocalIpcRmaBuffer>(
                 netDevCtx_, mem->addr, mem->size, static_cast<RmaMemType>(mem->type), *parentBuffer)),
            return HCCL_E_PTR);
        CHK_RET(AddBuffer(localIpcRmaBufferMgr, parentBuffer));
        HCCL_INFO("[HccsRegedMemMgr][RegisterMemory] alias created, key {%p, %llu}", mem->addr, mem->size);
    } else {
        EXCEPTION_CATCH(
            (localIpcRmaBuffer = std::make_shared<hccl::LocalIpcRmaBuffer>(
                 netDevCtx_, mem->addr, mem->size, static_cast<RmaMemType>(mem->type))),
            return HCCL_E_PTR);

        HcclResult ret = localIpcRmaBuffer->Init();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HccsRegedMemMgr][RegisterMemory]localbuffer init failed %d.", ret);
            return ret;
        }

        CHK_RET(AddBuffer(localIpcRmaBufferMgr, localIpcRmaBuffer));
        HCCL_INFO("[HccsRegedMemMgr][RegisterMemory]Register memory success! Add key {%p, %llu}", mem->addr, mem->size);
    }

    *memHandle = static_cast<void*>(localIpcRmaBuffer.get());
    allRegisteredBuffers_.emplace_back(localIpcRmaBuffer, false);
    handlesRecords_.push_back(localIpcRmaBuffer);

    HCCL_INFO(
        "[%s] addr[%p] size[%u] memHandle[%p] allRegisteredBuffers_.size[%d]done", __FUNCTION__, mem->addr, mem->size,
        *memHandle, allRegisteredBuffers_.size());
    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::UnregisterMemory(void* memHandle)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memHandle);

    NetDevContext* netDevCtx = static_cast<NetDevContext*>(netDevCtx_);
    std::shared_ptr<LocalIpcRmaBufferMgr> localIpcRmaBufferMgr = netDevCtx->GetlocalIpcRmaBufferMgr();
    CHK_PTR_NULL(localIpcRmaBufferMgr);

    hccl::LocalIpcRmaBuffer* buffer = static_cast<hccl::LocalIpcRmaBuffer*>(memHandle);
    void* addr = buffer->GetAddr();
    uint64_t size = buffer->GetSize();
    HCCL_INFO("[%s] addr[%p] size[%u] memHandle[%p] start", __FUNCTION__, addr, size, memHandle);

    // IsAlias() 直接区分父子buffer：
    //   - 父buffer (IsAlias()=false): 自己的key在tree中 → Del(ownKey)
    //   - 子buffer (IsAlias()=true):  自己的key不在tree中 → 通过Find找父key做Del
    hccl::BufferKey<uintptr_t, u64> ownKey(ReinterpretAs<uintptr_t>(addr), size);
    hccl::LocalIpcRmaBuffer* refBuffer = buffer;

    if (buffer->IsAlias()) {
        auto findResult = localIpcRmaBufferMgr->Find(ownKey);
        if (findResult.first) {
            refBuffer = findResult.second.get();
        } else {
            HCCL_ERROR("[HccsRegedMemMgr][UnregisterMemory] alias parent not found");
            return HCCL_E_NOT_FOUND;
        }
    }

    auto refBufferInfo = std::make_pair(refBuffer->GetAddr(), refBuffer->GetSize());
    hccl::BufferKey<uintptr_t, u64> memKey(ReinterpretAs<uintptr_t>(refBufferInfo.first), refBufferInfo.second);
    bool resultPair = false;
    EXCEPTION_CATCH(resultPair = localIpcRmaBufferMgr->Del(memKey), return HCCL_E_NOT_FOUND);
    // 无论tree中是否删除（ref是否归零），当前handle都要从allBuffers移除
    bool found = false;
    for (auto it = allRegisteredBuffers_.begin(); it != allRegisteredBuffers_.end(); it++) {
        if (it->first.get() == buffer) {
            HCCL_INFO("[%s] addr[%p] size[%u] memHandle[%p]", __FUNCTION__, addr, size, memHandle);
            // IsInTree判断tree中是否还有该key的引用
            handlesRecords_.erase(
                std::remove(handlesRecords_.begin(), handlesRecords_.end(), it->first), handlesRecords_.end());
            if (!localIpcRmaBufferMgr->IsInTree(ownKey)) {
                it = allRegisteredBuffers_.erase(it);
            } else {
                it->second = true;
            }
            found = true;
            break;
        }
    }
    if (!found) {
        HCCL_ERROR(
            "[%s] addr[%p] size[%zu] memHandle[%p] allRegisteredBuffers_.size[%zu] Not Found", __FUNCTION__, addr,
            static_cast<size_t>(size), memHandle, allRegisteredBuffers_.size());
        return HCCL_E_NOT_FOUND;
    }
    if (!resultPair) {
        HCCL_INFO(
            "[HccsRegedMemMgr][UnregisterMemory]Memory reference count is larger than 0, do not deregister memory.");
        return HCCL_SUCCESS;
    }
    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::SerializeToMemDesc(
    const EndpointDesc& endpointDesc, hccl::LocalIpcRmaBuffer* localIpcRmaBuffer, void** memDesc,
    uint32_t* descLen) const
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(descLen);

    std::string& ipcRmaBufferDesc = localIpcRmaBuffer->Serialize();
    HCCL_INFO("[%s] ipcRmaBufferDesc.len[%u]", __FUNCTION__, ipcRmaBufferDesc.length());

    ipcRmaBufferDesc.resize(ipcRmaBufferDesc.length() + sizeof(EndpointDesc));
    // put the EndpointDesc at the end of the Serialize-ed buf
    if (memcpy_s(
            &ipcRmaBufferDesc[0] + (ipcRmaBufferDesc.length() - sizeof(EndpointDesc)), sizeof(EndpointDesc),
            &endpointDesc, sizeof(EndpointDesc))
        != EOK) {
        HCCL_ERROR("[RoceRegedMemMgr][SerializeToMemDesc] [%s] endpointDesc memcpy_s failed.", __func__);
        return HCCL_E_INTERNAL;
    }

    *descLen = static_cast<uint32_t>(ipcRmaBufferDesc.length());
    *memDesc = const_cast<char*>(ipcRmaBufferDesc.c_str());
    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::MakeRemoteIpcRmaBuffer(
    std::string& ipcRmaBufferDesc, std::shared_ptr<hccl::RemoteIpcRmaBuffer>& remoteIpcRmaBuffer)
{
    HCCL_INFO("[HccsRegedMemMgr][%s] start", __FUNCTION__);
    EXCEPTION_CATCH(remoteIpcRmaBuffer = std::make_shared<hccl::RemoteIpcRmaBuffer>(netDevCtx_), return HCCL_E_PTR;);
    CHK_PTR_NULL(remoteIpcRmaBuffer);

    HcclResult ret = remoteIpcRmaBuffer->Deserialize(ipcRmaBufferDesc);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HccsRegedMemMgr][MemoryImport]RemoteIpcRmaBuffer Deserialize failed.");
        return ret;
    }

    HCCL_INFO(
        "[%s] addr[%p] size[%lu] done", __FUNCTION__, remoteIpcRmaBuffer->GetAddr(), remoteIpcRmaBuffer->GetSize());

    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::DeSerializeFromMemDesc(
    const void* memDesc, uint32_t descLen, EndpointDesc& endpointDesc,
    std::shared_ptr<hccl::RemoteIpcRmaBuffer>& remoteIpcRmaBuffer)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memDesc);

    const char* description = static_cast<const char*>(memDesc);
    HCCL_INFO("[%s] descLen[%u] memDesc[%s]", __FUNCTION__, descLen, description);

    if (descLen <= sizeof(EndpointDesc)) {
        HCCL_ERROR(
            "[HccsRegedMemMgr][DeSerializeFromMemDesc] [%s] descLen :%u too small error. need more than size:[%llu]",
            __func__, sizeof(EndpointDesc));
        return HCCL_E_INTERNAL;
    }

    uint32_t ipcRmaBufferDescLen = descLen - sizeof(EndpointDesc);
    if (memcpy_s(&endpointDesc, sizeof(EndpointDesc), description + ipcRmaBufferDescLen, sizeof(EndpointDesc)) != EOK) {
        HCCL_ERROR(
            "[HccsRegedMemMgr][DeSerializeFromMemDesc] [%s] endpointDesc copy error. aim size:[%llu]", __func__,
            sizeof(EndpointDesc));
        return HCCL_E_INTERNAL;
    }

    std::string ipcRmaBufferDesc;
    ipcRmaBufferDesc.resize(ipcRmaBufferDescLen);
    if (memcpy_s(&ipcRmaBufferDesc[0], ipcRmaBufferDescLen, description, ipcRmaBufferDescLen) != EOK) {
        HCCL_ERROR(
            "[HccsRegedMemMgr][DeSerializeFromMemDesc] [%s] ipcRmaBufferDesc copy error. aim size:[%llu]", __func__,
            ipcRmaBufferDescLen);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(MakeRemoteIpcRmaBuffer(ipcRmaBufferDesc, remoteIpcRmaBuffer));
    return HCCL_SUCCESS;
}

HcclResult
HccsRegedMemMgr::MemoryExport(const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(memDescLen);

    hccl::LocalIpcRmaBuffer* localIpcRmaBuffer = nullptr;
    CHK_RET(ValidateMemExportHandle(memHandle, allRegisteredBuffers_, localIpcRmaBuffer));
    CHK_RET(SerializeToMemDesc(endpointDesc, localIpcRmaBuffer, memDesc, memDescLen));
    HCCL_INFO("[%s] memDesc[%p] descLen[%u]", __FUNCTION__, *memDesc, *memDescLen);
    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::AddMem(
    hccl::BufferKey<uintptr_t, u64>& memKey, std::shared_ptr<hccl::RemoteIpcRmaBuffer>& remoteIpcRmaBuffer)
{
    CHK_PTR_NULL(remoteIpcRmaBuffer);
    HCCL_INFO(
        "[HccsRegedMemMgr][%s] addr[%p], size[%lu] start", __FUNCTION__, ReinterpretAs<void*>(memKey.Addr()),
        memKey.Size());

    auto resultPair = remoteIpcRmaBufferMgr_.Add(memKey, remoteIpcRmaBuffer);
    if (!resultPair.second) {
        HCCL_ERROR(
            "[HccsRegedMemMgr][%s] addr[%p], size[%lu] has already been imported!", __FUNCTION__,
            ReinterpretAs<void*>(memKey.Addr()), memKey.Size());
        return HCCL_E_AGAIN;
    }

    HCCL_INFO(
        "[HccsRegedMemMgr][%s] addr[%p], size[%lu] done", __FUNCTION__, ReinterpretAs<void*>(memKey.Addr()),
        memKey.Size());
    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::DeleteMem(hccl::BufferKey<uintptr_t, u64>& memKey)
{
    HCCL_INFO(
        "[HccsRegedMemMgr][%s] addr[%p], size[%lu] start", __FUNCTION__, ReinterpretAs<void*>(memKey.Addr()),
        memKey.Size());

    bool delResultPair = false;
    EXCEPTION_CATCH(delResultPair = remoteIpcRmaBufferMgr_.Del(memKey), return HCCL_E_NOT_FOUND);
    // 计数器大于1时，返回false，说明框架层有其它设备在使用这段endpointDesc，返回HCCL_SUCCESS
    if (!delResultPair) {
        HCCL_INFO(
            "[HccsRegedMemMgr][%s] addr[%p], size[%lu] reference count is larger than 0", __FUNCTION__,
            ReinterpretAs<void*>(memKey.Addr()), memKey.Size());
        return HCCL_SUCCESS;
    }

    HCCL_INFO(
        "[HccsRegedMemMgr][%s] addr[%p], size[%lu] done", __FUNCTION__, ReinterpretAs<void*>(memKey.Addr()),
        memKey.Size());

    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem)
{
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(outMem);
    HCCL_INFO("[%s] memDesc[%p] descLen[%u] start", __FUNCTION__, memDesc, descLen);

    EndpointDesc endpointDesc;
    std::shared_ptr<hccl::RemoteIpcRmaBuffer> remoteIpcRmaBuffer = nullptr;
    CHK_RET(DeSerializeFromMemDesc(memDesc, descLen, endpointDesc, remoteIpcRmaBuffer));
    hccl::BufferKey<uintptr_t, u64> memKey(
        ReinterpretAs<uintptr_t>(remoteIpcRmaBuffer->GetAddr()), remoteIpcRmaBuffer->GetSize());

    auto resultPair = remoteIpcRmaBufferMgr_.Find(memKey);
    if (!resultPair.first) {
        HCCL_INFO(
            "[HccsRegedMemMgr][%s] addr[%p], size[%lu] has Not Found!", __FUNCTION__, memKey.Addr(), memKey.Size());
        CHK_RET(AddMem(memKey, remoteIpcRmaBuffer));
    }

    outMem->addr = ReinterpretAs<void*>(remoteIpcRmaBuffer->GetAddr());
    outMem->size = remoteIpcRmaBuffer->GetSize();
    outMem->type = static_cast<CommMemType>(remoteIpcRmaBuffer->GetMemType());

    HCCL_INFO(
        "[%s]memDesc[%p] descLen[%u] addr[%p] size[%lu], type[%u] done", __FUNCTION__, memDesc, descLen, outMem->addr,
        outMem->size, outMem->type);

    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::MemoryUnimport(const void* memDesc, uint32_t descLen)
{
    CHK_PTR_NULL(memDesc);
    HCCL_INFO("[%s] memDesc[%p] descLen[%u] start", __FUNCTION__, memDesc, descLen);

    EndpointDesc endpointDesc;
    std::shared_ptr<hccl::RemoteIpcRmaBuffer> remoteIpcRmaBufferTmp = nullptr;
    CHK_RET(DeSerializeFromMemDesc(memDesc, descLen, endpointDesc, remoteIpcRmaBufferTmp));
    hccl::BufferKey<uintptr_t, u64> memKey(
        ReinterpretAs<uintptr_t>(remoteIpcRmaBufferTmp->GetAddr()), remoteIpcRmaBufferTmp->GetSize());

    std::shared_ptr<hccl::RemoteIpcRmaBuffer> remoteIpcRmaBuffer = nullptr;
    auto resultPair = remoteIpcRmaBufferMgr_.Find(memKey);
    if (!resultPair.first) {
        HCCL_ERROR(
            "[HccsRegedMemMgr][%s] addr[%p], size[%lu] has Not Found!", __FUNCTION__, memKey.Addr(), memKey.Size());
        return HCCL_E_NOT_FOUND;
    }
    remoteIpcRmaBuffer = resultPair.second;
    CHK_PTR_NULL(remoteIpcRmaBuffer);

    if (remoteIpcRmaBuffer->IsOpened()) {
        HCCL_INFO(
            "[HccsRegedMemMgr][DeleteMemDesc] memDesc[%p] descLen[%u] addr[%p] size[%lu] need to close first", memDesc,
            descLen, ReinterpretAs<void*>(memKey.Addr()), memKey.Size());
        return HCCL_SUCCESS;
    }

    CHK_RET(DeleteMem(memKey));

    HCCL_INFO(
        "[%s] memDesc[%p] descLen[%u] addr[%p] size[%lu] done", __FUNCTION__, memDesc, descLen,
        ReinterpretAs<void*>(memKey.Addr()), memKey.Size());

    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::MemoryGrant(const HcommMemGrantInfo* remoteGrantInfo)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(remoteGrantInfo);

    HCCL_INFO(
        "[HccsRegedMemMgr][MemoryGrant]Grant remotePid:%d, remoteSdid:%u", remoteGrantInfo->pid, remoteGrantInfo->sdid);
    for (auto it = allRegisteredBuffers_.begin(); it != allRegisteredBuffers_.end(); it++) {
        if (it->second) {
            continue;
        }
        std::shared_ptr<hccl::LocalIpcRmaBuffer> localIpcRmaBuffer = it->first;
        CHK_PTR_NULL(localIpcRmaBuffer);

        HcclResult ret = localIpcRmaBuffer->Grant(remoteGrantInfo->pid, remoteGrantInfo->sdid);
        CHK_PRT_RET(
            (ret != HCCL_SUCCESS),
            HCCL_ERROR(
                "[HccsRegedMemMgr][MemoryGrant]Grant remotePid:%d, remoteSdid:%u error", remoteGrantInfo->pid,
                remoteGrantInfo->sdid),
            ret);
        HCCL_INFO(
            "[HccsRegedMemMgr][MemoryGrant]Grant remotePid:%d, remoteSdid:%u addr [%p] done", remoteGrantInfo->pid,
            remoteGrantInfo->sdid, localIpcRmaBuffer->GetAddr());
    }

    HCCL_INFO(
        "[HccsRegedMemMgr][MemoryGrant]Grant remotePid:%d, remoteSdid:%u done", remoteGrantInfo->pid,
        remoteGrantInfo->sdid);
    return HCCL_SUCCESS;
}

HcclResult
HccsRegedMemMgr::MemoryEnableP2P(const EndpointDesc& localEndpointDesc, const EndpointDesc& remoteEndpointDesc) const
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    if (localEndpointDesc.loc.device.serverIdx == remoteEndpointDesc.loc.device.serverIdx) {
        u32 deviceLogicId;
        CHK_RET(hrtGetDeviceIndexByPhyId(localEndpointDesc.loc.device.devPhyId, deviceLogicId));
        HCCL_INFO(
            "Need do hrtEnableP2P for device[%u] with deviceLogicId[%u]", remoteEndpointDesc.loc.device.devPhyId,
            deviceLogicId);
        CHK_RET(hrtEnableP2P(deviceLogicId, remoteEndpointDesc.loc.device.devPhyId));
    }
    return HCCL_SUCCESS;
}

HcclResult
HccsRegedMemMgr::MemoryDisableP2P(const EndpointDesc& localEndpointDesc, const EndpointDesc& remoteEndpointDesc) const
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    if (localEndpointDesc.loc.device.serverIdx == remoteEndpointDesc.loc.device.serverIdx) {
        u32 deviceLogicId;
        CHK_RET(hrtGetDeviceIndexByPhyId(localEndpointDesc.loc.device.devPhyId, deviceLogicId));
        HCCL_INFO(
            "Need do hrtDisableP2P for device[%u] with deviceLogicId[%u]", remoteEndpointDesc.loc.device.devPhyId,
            deviceLogicId);
        CHK_RET(hrtDisableP2P(deviceLogicId, remoteEndpointDesc.loc.device.devPhyId));
    }
    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::MemoryOpenRemoteIpc()
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    for (auto it = remoteIpcRmaBufferMgr_.Begin(); it != remoteIpcRmaBufferMgr_.End();) {
        const std::shared_ptr<hccl::RemoteIpcRmaBuffer>& remoteIpcRmaBuffer = it->second.buffer;
        HcclResult openRet = remoteIpcRmaBuffer->Open();
        if (openRet != HCCL_SUCCESS) {
            HCCL_ERROR("[HccsRegedMemMgr][MemoryOpenRemoteIpc]RemoteIpcRmaBuffer Open failed.");
            for (auto it2 = remoteIpcRmaBufferMgr_.Begin(); it2 != it;) {
                const std::shared_ptr<hccl::RemoteIpcRmaBuffer>& remoteIpcRmaBuffer2 = it2->second.buffer;
                (void)remoteIpcRmaBuffer2->Close();
                it2 = remoteIpcRmaBufferMgr_.Next(it2);
            }
            return openRet;
        }
        it = remoteIpcRmaBufferMgr_.Next(it);
    }

    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::MemoryCloseRemoteIpc()
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    for (auto it = remoteIpcRmaBufferMgr_.Begin(); it != remoteIpcRmaBufferMgr_.End();) {
        const std::shared_ptr<hccl::RemoteIpcRmaBuffer>& remoteIpcRmaBuffer = it->second.buffer;
        (void)remoteIpcRmaBuffer->Close();
        it = remoteIpcRmaBufferMgr_.Next(it);
    }

    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::GetAllMemHandles(void** memHandles, uint32_t* memHandleNum)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memHandles);
    CHK_PTR_NULL(memHandleNum);
    *memHandleNum = static_cast<uint32_t>(handlesRecords_.size());
    *memHandles = handlesRecords_.empty() ? nullptr : static_cast<void*>(handlesRecords_.data());
    HCCL_INFO("[HccsRegedMemMgr][GetAllMemHandles] memHandleNum[%u]", *memHandleNum);
    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::GetRemoteIpcRmaBuffer(std::vector<CommMem>& remoteIpcRmaBufferVec)
{
    CommMem mem;
    for (auto it = remoteIpcRmaBufferMgr_.Begin(); it != remoteIpcRmaBufferMgr_.End();) {
        const std::shared_ptr<hccl::RemoteIpcRmaBuffer>& remoteIpcRmaBuffer = it->second.buffer;
        mem.addr = remoteIpcRmaBuffer->GetAddr();
        mem.size = remoteIpcRmaBuffer->GetSize();
        mem.type = remoteIpcRmaBuffer->GetMemType() == RmaMemType::DEVICE ? CommMemType::COMM_MEM_TYPE_DEVICE :
                                                                            CommMemType::COMM_MEM_TYPE_HOST;
        remoteIpcRmaBufferVec.emplace_back(mem);
        HCCL_INFO(
            "[HccsRegedMemMgr][GetRemoteIpcRmaBuffer]remote addr:%p, size[%lu], type[%u]", mem.addr, mem.size,
            static_cast<u32>(mem.type));
        it = remoteIpcRmaBufferMgr_.Next(it);
    }
    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::GetRemoteIpcRmaBufferEx(std::vector<HcclMemEx>& remoteIpcRmaBufferVecEx)
{
    HcclMemEx mem;
    for (auto it = remoteIpcRmaBufferMgr_.Begin(); it != remoteIpcRmaBufferMgr_.End();) {
        const std::shared_ptr<hccl::RemoteIpcRmaBuffer>& remoteIpcRmaBuffer = it->second.buffer;
        mem.addr = remoteIpcRmaBuffer->GetAddr();
        mem.size = remoteIpcRmaBuffer->GetSize();
        mem.type = remoteIpcRmaBuffer->GetMemType() == RmaMemType::DEVICE ? HcclMemType::HCCL_MEM_TYPE_DEVICE :
                                                                            HcclMemType::HCCL_MEM_TYPE_HOST;
        mem.devAddr = remoteIpcRmaBuffer->GetDevAddr();
        remoteIpcRmaBufferVecEx.emplace_back(mem);
        HCCL_INFO(
            "[HccsRegedMemMgr][GetRemoteIpcRmaBufferEx]remote addr:%p, size[%lu], type[%u], devAddr[%p]", mem.addr,
            mem.size, static_cast<u32>(mem.type), mem.devAddr);
        it = remoteIpcRmaBufferMgr_.Next(it);
    }
    return HCCL_SUCCESS;
}

HcclResult HccsRegedMemMgr::GetLocalIpcRmaBufferEx(std::vector<HcclMemEx>& localIpcRmaBufferVecEx) const
{
    NetDevContext* netDevCtx = static_cast<NetDevContext*>(netDevCtx_);
    const std::shared_ptr<LocalIpcRmaBufferMgr>& localIpcRmaBufferMgr = netDevCtx->GetlocalIpcRmaBufferMgr();
    CHK_PTR_NULL(localIpcRmaBufferMgr);

    HcclMemEx mem;
    for (auto it = localIpcRmaBufferMgr->Begin(); it != localIpcRmaBufferMgr->End();) {
        const std::shared_ptr<hccl::LocalIpcRmaBuffer>& localIpcRmaBuffer = it->second.buffer;
        mem.addr = localIpcRmaBuffer->GetAddr();
        mem.size = localIpcRmaBuffer->GetSize();
        mem.type = localIpcRmaBuffer->GetMemType() == RmaMemType::DEVICE ? HcclMemType::HCCL_MEM_TYPE_DEVICE :
                                                                           HcclMemType::HCCL_MEM_TYPE_HOST;
        mem.devAddr = localIpcRmaBuffer->GetDevAddr();
        localIpcRmaBufferVecEx.emplace_back(mem);
        HCCL_INFO(
            "[HccsRegedMemMgr][GetLocalIpcRmaBufferEx]local addr:%p, size[%lu], type[%u], devAddr[%p]", mem.addr,
            mem.size, static_cast<u32>(mem.type), mem.devAddr);
        it = localIpcRmaBufferMgr->Next(it);
    }
    return HCCL_SUCCESS;
}
} // namespace hcomm
