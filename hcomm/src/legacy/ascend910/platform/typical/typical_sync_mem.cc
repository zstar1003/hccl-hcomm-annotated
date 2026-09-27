/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "typical_sync_mem.h"
#include "adapter_rts_common.h"
#include "adapter_rts.h"
#include "rdma_resource_manager.h"

namespace hccl {
TypicalSyncMem& TypicalSyncMem::GetInstance()
{
    static TypicalSyncMem typicalSyncMem[MAX_MODULE_DEVICE_NUM + 1];
    s32 deviceLogicId = INVALID_INT;
    s32 ret = hrtGetDevice(&deviceLogicId);
    if (ret == HCCL_SUCCESS && (static_cast<u32>(deviceLogicId) < MAX_MODULE_DEVICE_NUM)) {
        HCCL_INFO("[TypicalSyncMem::GetInstance]deviceLogicID[%d]", deviceLogicId);
        return typicalSyncMem[deviceLogicId];
    }
    HCCL_WARNING("[TypicalSyncMem::GetInstance]deviceLogicID[%d] is invalid, ret[%d].", deviceLogicId, ret);
    return typicalSyncMem[MAX_MODULE_DEVICE_NUM];
}

TypicalSyncMem::TypicalSyncMem() {}

TypicalSyncMem::~TypicalSyncMem()
{
    (void)FreeAllSyncMem();
    (void)DeInitNotifySrcMem();
}

HcclResult TypicalSyncMem::InitNotifySrcMem()
{
    CHK_RET(RdmaResourceManager::GetInstance().GetRdmaHandle(rdmaHandle_));
    CHK_PTR_NULL(rdmaHandle_);
    HCCL_DEBUG("[TypicalSyncMem][InitNotifySrcMem]start init notify source mem.");
    u32 notifyVaule = 1; // notify值写1表示record
    u32 notifySize = 0;
    CHK_RET(hrtGetNotifySize(notifySize));

    CHK_RET(DeviceMem::alloc(srcDevMem_, notifySize));
    HCCL_DEBUG(
        "[TypicalSyncMem][InitNotifySrcMem]Create notify src buffer[%p], size[%u].", srcDevMem_.ptr(), notifySize);

    CHK_RET(hrtMemSyncCopy(
        srcDevMem_.ptr(), notifySize, &notifyVaule, notifySize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    notifySrcMrInfo_.addr = srcDevMem_.ptr();
    notifySrcMrInfo_.size = notifySize;
    notifySrcMrInfo_.access = RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_WRITE;
    notifySrcMrHandle_ = nullptr;
    CHK_RET(hrtRaRegGlobalMr(rdmaHandle_, notifySrcMrInfo_, notifySrcMrHandle_));

    HCCL_INFO(
        "[TypicalSyncMem][InitNotifySrcMem]Init notifySrcMem_=%p success, mr lkey is [%u].", notifySrcMrInfo_.addr,
        notifySrcMrInfo_.lkey);
    return HCCL_SUCCESS;
}

HcclResult TypicalSyncMem::DeInitNotifySrcMem()
{
    if (notifySrcMrHandle_ == nullptr) {
        HCCL_INFO("[TypicalSyncMem][InitNotifySrcMem] NotifySrcMem has been DeInit.");
        return HCCL_SUCCESS;
    }
    CHK_RET(RdmaResourceManager::GetInstance().GetRdmaHandle(rdmaHandle_));
    CHK_PTR_NULL(rdmaHandle_);
    HCCL_INFO("[TypicalSyncMem][InitNotifySrcMem] DeRegister notifySrcMem_=%p.", notifySrcMrInfo_.addr);
    CHK_RET(hrtRaDeRegGlobalMr(rdmaHandle_, notifySrcMrHandle_));
    notifySrcMrHandle_ = nullptr;
    return HCCL_SUCCESS;
}

HcclResult TypicalSyncMem::AllocSyncMem(int32_t** ptr)
{
    HCCL_DEBUG("[TypicalSyncMem][AllocSyncMem]start alloc sync mem on [%p].", ptr);
    std::unique_lock<std::mutex> lockSyncMemMap(syncMemMapMutex_);
    if (syncMemMap_.empty()) {
        HCCL_INFO("[TypicalSyncMem][AllocSyncMem] syncMem has not inited. Start to init notify src mem.");
        CHK_RET(InitNotifySrcMem());
    }
    CHK_PTR_NULL(ptr);
    CHK_RET(RdmaResourceManager::GetInstance().GetRdmaHandle(rdmaHandle_));
    CHK_PTR_NULL(rdmaHandle_);

    u64 offset = 0;
    u64 notifyBaseVa = 0;
    u64 notifyTotalSize = 0;

    // Create an empty notify and get it's handle
    HcclRtSignal notify = nullptr;
    CHK_RET(CreateEmptyNotify(notify));
    HCCL_DEBUG("[TypicalSyncMem][AllocSyncMem]create an empty notify success.");

    // Get the base virtual address and the size of notify register.
    u64 notifyBaseVaTmp = 0;
    notifyBaseVaTmp = notifyBaseVa;
    CHK_RET(HrtRaGetNotifyBaseAddr(rdmaHandle_, &notifyBaseVa, &notifyTotalSize));

    CHK_PRT_RET(
        ((notifyBaseVaTmp != 0) && (notifyBaseVaTmp != notifyBaseVa)),
        HCCL_ERROR("[TypicalSyncMem][AllocSyncMem]get base addr failed, notify base va has changed."), HCCL_E_INTERNAL);

    // Get the offset to the base address for the created notify,
    // which is same for both physical address and virtual address.
    // Here we use physical address to calculate the offset.
    CHK_RET(hrtNotifyGetOffset(notify, offset));

    // notify寄存器的虚拟地址与物理地址偏移相同，所以虚拟地址为虚拟基地址加偏移
    u64 notifyVa = notifyBaseVa + offset;

    HCCL_INFO(
        "[TypicalSyncMem][AllocSyncMem]notifyBaseVa=0x%llx,"
        "notifyTotalSize=0x%x, offset=0x%llx, notifyVa=0x%llx notify=%p.",
        notifyBaseVa, notifyTotalSize, offset, notifyVa, notify);
    // Store the notifyVa to set
    syncMemMap_[notifyVa] = notify;
    // Assign the notify virtual address to *ptr.
    *ptr = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(notifyVa));
    HCCL_RUN_INFO(
        "[TypicalSyncMem][AllocSyncMem]alloc an empty sync mem success, notifyVa[%p]. "
        "please register mr before use.",
        *ptr);
    return HCCL_SUCCESS;
}

HcclResult TypicalSyncMem::FreeSyncMem(int32_t* ptr)
{
    HCCL_DEBUG("[TypicalSyncMem][FreeSyncMem]start free sync mem[%p], please deregister mr before free.", ptr);
    CHK_PTR_NULL(ptr);
    u64 notifyVa = reinterpret_cast<uintptr_t>(ptr);
    std::unique_lock<std::mutex> lockSyncMemMap(syncMemMapMutex_);
    auto smIter = syncMemMap_.find(notifyVa);
    if (smIter == syncMemMap_.end()) {
        HCCL_WARNING("[TypicalSyncMem][FreeSyncMem]No notifyVa match the given ptr[%p] in sync mem map.", ptr);
        return HCCL_SUCCESS;
    }
    CHK_RET(DestroyNotify(syncMemMap_[notifyVa]));
    syncMemMap_.erase(smIter);
    if (syncMemMap_.empty()) {
        HCCL_INFO("[TypicalSyncMem][FreeSyncMem] syncMem all deinit. Start to deinit notify src mem.");
        CHK_RET(DeInitNotifySrcMem());
    }
    HCCL_INFO("[TypicalSyncMem][FreeSyncMem] Free [%p] success.", ptr);
    return HCCL_SUCCESS;
}

HcclResult TypicalSyncMem::GetNotifyHandle(u64 notifyVa, HcclRtNotify& notifyHandle)
{
    std::unique_lock<std::mutex> lockSyncMemMap(syncMemMapMutex_);
    auto smIter = syncMemMap_.find(notifyVa);
    if (smIter != syncMemMap_.end()) {
        notifyHandle = smIter->second;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[TypicalSyncMem][GetNotifyHandle]invalid notifyVa[%llu].", notifyVa);
    return HCCL_E_PARA;
}

HcclResult TypicalSyncMem::GetNotifySrcMem(struct MrInfoT& mrInfo)
{
    CHK_PTR_NULL(notifySrcMrInfo_.addr);
    mrInfo.addr = notifySrcMrInfo_.addr;
    mrInfo.size = notifySrcMrInfo_.size;
    mrInfo.access = notifySrcMrInfo_.access;
    mrInfo.lkey = notifySrcMrInfo_.lkey;
    return HCCL_SUCCESS;
}

HcclResult TypicalSyncMem::CreateEmptyNotify(HcclRtNotify& notifyHandle)
{
    s32 deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));
    HcclResult ret = hrtNotifyCreate(deviceId, &notifyHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TypicalSyncMem][CreateNotify]errNo[0x%016llx] Notify create failed. return[%d], deviceLogicId[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, deviceId),
        HCCL_E_RUNTIME);
    CHK_PRT_RET(
        notifyHandle == nullptr,
        HCCL_ERROR(
            "[TypicalSyncMem][CreateNotify]errNo[0x%016llx] Notify create failed. notifyHandle is NULL",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME)),
        HCCL_E_RUNTIME);

    HCCL_INFO(
        "[TypicalSyncMem][CreateNotify]create notify success, deviceId[%d], notify handle[%p].", deviceId,
        notifyHandle);
    return HCCL_SUCCESS;
}

HcclResult TypicalSyncMem::DestroyNotify(HcclRtNotify notifyHandle)
{
    HCCL_DEBUG("[TypicalSyncMem][DestroyNotify]start destroy notify[%p].", notifyHandle);
    CHK_PTR_NULL(notifyHandle);
    HcclResult ret = hrtNotifyDestroy(notifyHandle);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[TypicalSyncMem][DestroyNotify]errNo[0x%016llx] rt notify destroy fail, return[%d].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    HCCL_INFO("[TypicalSyncMem][DestroyNotify]destroy notify success.");
    return HCCL_SUCCESS;
}

HcclResult TypicalSyncMem::FreeAllSyncMem()
{
    std::unique_lock<std::mutex> lockSyncMemMap(syncMemMapMutex_);
    if (!syncMemMap_.empty()) {
        for (auto& smIter : syncMemMap_) {
            if (smIter.second != nullptr) {
                CHK_RET(DestroyNotify(smIter.second));
            }
        }
        syncMemMap_.clear();
    }
    HCCL_INFO("[TypicalSyncMem][FreeAllSyncMem]free all sync memory success.");
    return HCCL_SUCCESS;
}
} // namespace hccl
