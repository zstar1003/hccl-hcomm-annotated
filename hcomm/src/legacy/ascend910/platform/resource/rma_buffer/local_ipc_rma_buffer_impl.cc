/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_ipc_rma_buffer_impl.h"
#include "adapter_rts.h"
#include "hccl_network.h"
#include "mem_mapping_manager.h"

namespace hccl {
namespace {
    void* GetAliasDevAddr(void* addr, const LocalIpcRmaBufferImpl& parent)
    {
        const uintptr_t childAddr = reinterpret_cast<uintptr_t>(addr);
        const uintptr_t parentAddr = reinterpret_cast<uintptr_t>(parent.GetAddr());
        const uintptr_t parentDevAddr = reinterpret_cast<uintptr_t>(parent.GetDevAddr());
        return reinterpret_cast<void*>(parentDevAddr + (childAddr - parentAddr));
    }
} // namespace

LocalIpcRmaBufferImpl::LocalIpcRmaBufferImpl(
    const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType)
    : RmaBuffer(netDevCtx, addr, size, memType, RmaType::IPC_RMA)
{}

LocalIpcRmaBufferImpl::LocalIpcRmaBufferImpl(
    const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType, const LocalIpcRmaBufferImpl& parent)
    : RmaBuffer(netDevCtx, addr, size, memType, RmaType::IPC_RMA, true),
      deviceLogicId(parent.deviceLogicId),
      memName(parent.memName),
      memOffset(parent.memOffset),
      initialized_(true)
{
    devAddr = GetAliasDevAddr(addr, parent);
    HCCL_INFO("[LocalIpcRmaBufferImpl] alias constructor, ipcName[%s]", memName.ipcName);
}

LocalIpcRmaBufferImpl::~LocalIpcRmaBufferImpl()
{
    if (!isAlias_) {
        HcclResult res = Destroy();
        if (res != HCCL_SUCCESS) {
            HCCL_ERROR("[LocalIpcRmaBufferImpl][~LocalIpcRmaBufferImpl]failed, ret[%d]", res);
        }
    }
}

HcclResult LocalIpcRmaBufferImpl::Init()
{
    CHK_PTR_NULL(netDevCtx);
    deviceLogicId = (static_cast<NetDevContext*>(netDevCtx))->GetLogicId();

    // host内存地址映射
    devAddr = addr;
    if (memType == RmaMemType::HOST) {
        CHK_RET(MemMappingManager::GetInstance(deviceLogicId).GetDevVA(deviceLogicId, addr, size, devAddr));
    } else {
        // 设置ipc mem name
        HCCL_INFO("[LocalIpcRmaBufferImpl][Init]ipc set mem name");
        HcclResult ret = MemNameRepository::GetInstance(deviceLogicId)
                             ->SetIpcMem(devAddr, size, memName.ipcName, HCCL_IPC_MEM_NAME_LEN);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[LocalIpcRmaBufferImpl][Init]errNo[0x%016llx], get para mem name failed. "
                "mem addr[%p] deviceLogicId[%d]",
                HCCL_ERROR_CODE(ret), devAddr, deviceLogicId),
            ret);
    }
    HCCL_DEBUG(
        "[LocalIpcRmaBufferImpl][Init]addr[%p], size[%llu], devAddr[%p], memType[%d]", addr, size, devAddr, memType);
    initialized_ = true;
    return HCCL_SUCCESS;
}

std::string& LocalIpcRmaBufferImpl::Serialize()
{
    if (!serializeStr_.empty()) {
        return serializeStr_;
    }
    // 序列化信息
    std::ostringstream oss;
    u8 type{static_cast<u8>(rmaType)};
    oss.write(reinterpret_cast<const char_t*>(&type), sizeof(type));
    oss.write(reinterpret_cast<const char_t*>(&addr), sizeof(addr));
    oss.write(reinterpret_cast<const char_t*>(&size), sizeof(size));
    oss.write(reinterpret_cast<const char_t*>(&devAddr), sizeof(devAddr));
    oss.write(reinterpret_cast<const char_t*>(&memType), sizeof(memType));
    oss.write(reinterpret_cast<const char_t*>(&memName.ipcName), sizeof(memName.ipcName));
    oss.write(reinterpret_cast<const char_t*>(&memOffset), sizeof(memOffset));
    HCCL_DEBUG(
        "[LocalIpcRmaBufferImpl][Serialize] addr[%p], size[%llu], devAddr[%p], memType[%d], ipcName[%s], "
        "memOffset[%llu]",
        reinterpret_cast<void*>(addr), size, reinterpret_cast<void*>(devAddr), memType, memName.ipcName, memOffset);

    serializeStr_ = oss.str();
    return serializeStr_;
}

constexpr s32 IPC_NOTIFY_PID_ARRAY_SIZE = 1;
HcclResult LocalIpcRmaBufferImpl::Grant(u32 remotePid, u32 remoteSdid)
{
    if (memType == RmaMemType::HOST) {
        HCCL_DEBUG("[LocalIpcRmaBufferImpl][Grant]memType is [%d].", memType);
        return HCCL_SUCCESS;
    }

    HCCL_DEBUG("[LocalIpcRmaBufferImpl][Grant]ipcName[%s], pid[%u], sdid[%u]", memName.ipcName, remotePid, remoteSdid);
    s32 peerPid = static_cast<s32>(remotePid);
    s32 peerSdid = static_cast<s32>(remoteSdid);
    if (peerSdid != INVALID_INT) {
        CHK_RET(hrtSetIpcMemorySuperPodPid(memName.ipcName, peerSdid, &peerPid, IPC_NOTIFY_PID_ARRAY_SIZE));
    } else {
        CHK_RET(hrtIpcSetMemoryPid(memName.ipcName, &peerPid, IPC_NOTIFY_PID_ARRAY_SIZE));
    }
    return HCCL_SUCCESS;
}

HcclResult LocalIpcRmaBufferImpl::Destroy()
{
    if (addr != nullptr && initialized_) {
        // host内存解映射
        HcclResult ret = HCCL_SUCCESS;
        if (memType == RmaMemType::HOST) {
            ret = MemMappingManager::GetInstance(deviceLogicId).ReleaseDevVA(deviceLogicId, addr, size);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[LocalIpcRmaBufferImpl][Destroy]release dev va failed, "
                    "ret[%d], dev[%d], ptr[%p], size[%llu]",
                    ret, deviceLogicId, addr, size);
            }
        } else {
            // 销毁ipc mem name
            MemNameRepository::GetInstance(deviceLogicId)->DestroyIpcMem(devAddr, size);
            HCCL_INFO(
                "[LocalIpcRmaBufferImpl][Destroy]ipc destroy mem name. "
                "mem addr[%p] deviceLogicId[%d]",
                devAddr, deviceLogicId);
        }

        addr = nullptr;
        size = 0;
        initialized_ = false;
        return ret;
    }

    return HCCL_SUCCESS;
}
} // namespace hccl
