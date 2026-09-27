/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_rts.h"

#include "log.h"
#include "sal_pub.h"
#include "mem_name_repository.h"

namespace hccl {
MemNameRepository::~MemNameRepository() { ClearMemNameRepository(); }

MemNameRepository* MemNameRepository::GetInstance(s32 deviceLogicID)
{
    static MemNameRepository instances[MAX_DEV_NUM_IPC_MEM];
    if (deviceLogicID == HOST_DEVICE_ID) {
        return &instances[0];
    }

    if (static_cast<u32>(deviceLogicID) >= MAX_DEV_NUM_IPC_MEM || deviceLogicID < 0) {
        HCCL_WARNING("[Get][Instance]deviceLogicID[%d] is invalid", deviceLogicID);
        return &instances[0];
    }
    return &instances[deviceLogicID];
}

HcclResult MemNameRepository::SetDeviceUnavailable(bool unavailable)
{
    std::unique_lock<std::mutex> lock(memMutex_);
    unavailable_ = unavailable;
    HCCL_RUN_INFO("SetDeviceUnavailable unavailable[%d]", unavailable);
    return HCCL_SUCCESS;
}

HcclResult MemNameRepository::SetIpcMem(void* ptr, u64 size, u8* name, u32 nameLen, u64& offset, bool isSioToHccs)
{
    CHK_PTR_NULL(name);
    CHK_PTR_NULL(ptr);

    HcclResult ret;
    std::unique_lock<std::mutex> lock(memMutex_);
    IpcMemInfo ipcMemInfo = {};

    ipcMemInfo.ptr = ptr;
    ipcMemInfo.size = size;
    ipcMemInfo.isSioToHccs = isSioToHccs;
    IpcMemInfo preIpcMemInfo = ipcMemInfo;

    // 记录页表大小
    ret = hrtDevMemAlignWithPage(ipcMemInfo.ptr, ipcMemInfo.size);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Set][IpcMem]errNo[0x%016llx] Set ptr and offset error. ptr[%p] size[%llu Byte]", HCCL_ERROR_CODE(ret),
            ipcMemInfo.ptr, ipcMemInfo.size),
        ret);
    alignPtrMap_.insert(std::make_pair(preIpcMemInfo, ipcMemInfo));

    // 在SetNameMap中查找MemName,若未找到则插入
    ret = FindIpcMem(ipcMemInfo, name, nameLen);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Find][IpcMem]errNo[0x%016llx] In link base, sal Find ipc memory error. ptr[%p] size[%llu Byte]",
            HCCL_ERROR_CODE(ret), ipcMemInfo.ptr, ipcMemInfo.size),
        ret);

    offset = reinterpret_cast<u64>(ptr) - reinterpret_cast<u64>(ipcMemInfo.ptr);

    return HCCL_SUCCESS;
}

HcclResult MemNameRepository::SetIpcMem(void* ptr, u64 size, u8* name, u32 nameLen)
{
    CHK_PTR_NULL(name);
    CHK_PTR_NULL(ptr);

    HcclResult ret;
    std::unique_lock<std::mutex> lock(memMutex_);
    IpcMemInfo ipcMemInfo = {};

    ipcMemInfo.ptr = ptr;
    ipcMemInfo.size = size;
    ipcMemInfo.isSioToHccs = false;
    alignPtrMap_.insert(std::make_pair(ipcMemInfo, ipcMemInfo));

    // 在SetNameMap中查找memName,若未找到则插入
    ret = FindIpcMem(ipcMemInfo, name, nameLen);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Find][IpcMem]errNo[0x%016llx] In link base, sal Find ipc memory error. ptr[%p] size[%llu Byte]",
            HCCL_ERROR_CODE(ret), ipcMemInfo.ptr, ipcMemInfo.size),
        ret);

    return HCCL_SUCCESS;
}

HcclResult MemNameRepository::SetIpcMem(
    void* ptr, u64 size, u8* name, u32 nameLen, u64& offset, s32 pid, s32 sdid, bool isSioToHccs)
{
    HCCL_DEBUG(
        "SetIpcMem para: ptr[%p], size[%llu Byte], name[%d], nameLen[%u], pid[%d], sdid[%016llx], isSioToHccs[%d]", ptr,
        size, name, nameLen, pid, sdid, isSioToHccs);

    CHK_RET(SetIpcMem(ptr, size, name, nameLen, offset, isSioToHccs));

    /* 不管任何情况，都需设置PID 的白名单 */
    if (sdid != INVALID_INT) {
        CHK_RET(hrtSetIpcMemorySuperPodPid(name, sdid, &pid, HCCL_IPC_PID_ARRAY_SIZE));
    } else {
        CHK_RET(hrtIpcSetMemoryPid(name, &pid, HCCL_IPC_PID_ARRAY_SIZE));
    }
    return HCCL_SUCCESS;
}

// 在SetNameMap中查找MemName,若未找到则插入
HcclResult MemNameRepository::FindIpcMem(IpcMemInfo& ipcMemInfo, u8* name, u32 nameLen)
{
    s32 sret;
    HcclResult ret;
    auto iter = setNameMap_.find(ipcMemInfo);
    if (iter == setNameMap_.end()) {
        ret = hrtIpcSetMemoryName(ipcMemInfo.ptr, name, ipcMemInfo.size, nameLen);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Set][IpcMem]errNo[0x%016llx] In link base, sal set ipc memory error. ptr[%p] size[%llu Byte]",
                HCCL_ERROR_CODE(ret), ipcMemInfo.ptr, ipcMemInfo.size),
            ret);

        SecIpcName_t memName;
        sret = memcpy_s(memName.ipcName, HCCL_IPC_MEM_NAME_LEN, name, nameLen);
        if (sret != EOK) {
            HCCL_ERROR(
                "[Set][IpcMem]errNo[0x%016llx] In SecIpcName, memcpy_s failed. errorno[%d], params:"
                "dest len[%u], src len[%u]",
                HCCL_ERROR_CODE(HCCL_E_SYSCALL), sret, HCCL_IPC_MEM_NAME_LEN, nameLen);
            return HCCL_E_SYSCALL;
        }
        setNameMap_.insert(std::make_pair(ipcMemInfo, memName)); // 记录mem name
    } else {
        SecIpcName_t memName = iter->second;
        sret = memcpy_s(name, HCCL_IPC_MEM_NAME_LEN, memName.ipcName, nameLen);
        if (sret != EOK) {
            HCCL_ERROR(
                "[Set][IpcMem]errNo[0x%016llx] In SecIpcName, memcpy_s failed. errorno[%d], params:"
                "dest len[%u], src len[%u]",
                HCCL_ERROR_CODE(HCCL_E_SYSCALL), sret, HCCL_IPC_MEM_NAME_LEN, nameLen);
            return HCCL_E_SYSCALL;
        }
        HCCL_INFO("SetIpcMem: name[%s] has opened, skip.", memName.ipcName);
    }
    setNameMapRef_[ipcMemInfo].Ref();

    return HCCL_SUCCESS;
}

HcclResult MemNameRepository::OpenIpcMem(
    void** ptr, u64 size, const u8* name, u32 nameLen, u64 offset, bool& isOpened, bool isSioToHccs)
{
    CHK_PTR_NULL(name);
    CHK_PTR_NULL(ptr);

    HcclResult ret;
    IpcMemInfo ipcMemInfo = {};

    std::unique_lock<std::mutex> lock(memMutex_);
    auto iter = openedNameMap_.begin();
    while (iter != openedNameMap_.end()) {
        SecIpcName_t memName = iter->second;
        if (strncmp(
                reinterpret_cast<char*>(memName.ipcName), reinterpret_cast<char*>(const_cast<u8*>(name)),
                HCCL_IPC_MEM_NAME_LEN)
            == 0) {
            // 找到相同ipc 名字,跳出循环
            *ptr = iter->first.ptr;
            ipcMemInfo.ptr = *ptr;
            ipcMemInfo.size = iter->first.size;
            ipcMemInfo.isSioToHccs = iter->first.isSioToHccs;
            HCCL_INFO("OpenIpcMem: name[%s] has opened, skip.", memName.ipcName);
            isOpened = true;
            break;
        } else {
            iter++;
        }
    }
    if (iter == openedNameMap_.end()) {
        /* 未找到相同IPC name , 调用open memory打开IPC */
        ret = hrtIpcOpenMemory(ptr, name);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Open][IpcMem]errNo[0x%016llx] In mem repository, ipc open memory ptr[%p] offset[%llu]"
                " name[%s] local pid[%d]",
                ret, ptr, offset, name, SalGetPid()),
            ret);

        SecIpcName_t memName;
        s32 sret = memcpy_s(memName.ipcName, HCCL_IPC_MEM_NAME_LEN, name, nameLen);
        if (sret != EOK) {
            HCCL_ERROR(
                "[Open][IpcMem]errNo[0x%016llx] In SecIpcName, memset_s failed. errno[%d], params:"
                "dest len[%u], src len[%u]",
                HCCL_ERROR_CODE(HCCL_E_SYSCALL), sret, HCCL_IPC_MEM_NAME_LEN, nameLen);
            return HCCL_E_SYSCALL;
        }
        ipcMemInfo.ptr = *ptr;
        ipcMemInfo.size = size;
        ipcMemInfo.isSioToHccs = isSioToHccs;
        openedNameMap_.insert(std::make_pair(ipcMemInfo, memName)); // 记录mem name
        isOpened = false;
    }
    openedNameMapRef_[ipcMemInfo].Ref();

    HCCL_DEBUG(
        "OpenIpcMem: name[%s] ptr[%p] alignPtr[%p] offset[%llu] size[%llu Byte].", name,
        (reinterpret_cast<char*>(*ptr) + offset), *ptr, offset, size);

    *ptr = reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(*ptr) + offset);

    return HCCL_SUCCESS;
}

void MemNameRepository::CloseIpcMem(const u8* name)
{
    HcclResult ret;
    std::unique_lock<std::mutex> lock(memMutex_);

    if (name == nullptr) {
        HCCL_WARNING("In mem repository, destroy null ipc ptr");
        return;
    }

    if (unavailable_) {
        ClearMemNameRepositoryImpl();
        unavailable_ = false;
        HCCL_RUN_INFO("CloseIpMem unavailable_[%d]", unavailable_);
        return;
    }

    auto iter = openedNameMap_.begin();
    while (iter != openedNameMap_.end()) {
        SecIpcName_t memName = iter->second;
        if (strncmp(
                reinterpret_cast<char*>(memName.ipcName), reinterpret_cast<const char*>(name), HCCL_IPC_MEM_NAME_LEN)
            == 0) {
            if (openedNameMapRef_[iter->first].Unref() == 0) {
                // 找到相同ipc 名字, 并且引用计数减为0再close
                ret = hrtIpcDestroyMemoryName(memName.ipcName);
                if (ret != HCCL_SUCCESS) {
                    HCCL_WARNING("In mem repository, ipc close memory ret[%d] ", ret);
                }
                openedNameMapRef_.erase(iter->first);
                openedNameMap_.erase(iter);
            }
            break;
        } else {
            iter++;
        }
    }
}

void MemNameRepository::DestroyIpcMem(void* ptr, u64 size, bool isSioToHccs)
{
    HcclResult ret;
    std::unique_lock<std::mutex> lock(memMutex_);

    if (ptr == nullptr) {
        HCCL_WARNING("In mem repository, destroy null ipc ptr");
        return;
    }

    if (unavailable_) {
        ClearMemNameRepositoryImpl();
        unavailable_ = false;
        HCCL_RUN_INFO("DestroyIpcMem unavailable_[%d]", unavailable_);
        return;
    }

    IpcMemInfo ipcMemInfo = {};
    ipcMemInfo.ptr = ptr;
    ipcMemInfo.size = size;
    ipcMemInfo.isSioToHccs = isSioToHccs;

    auto it = alignPtrMap_.find(ipcMemInfo);
    if (it == alignPtrMap_.end()) {
        HCCL_WARNING("Unapplied Memory ptr");
        ptr = nullptr;
        return;
    } else {
        ipcMemInfo = it->second;
    }

    auto iter = setNameMap_.find(ipcMemInfo);
    if (iter == setNameMap_.end()) {
        // 说明已经销毁该IPC name
        ptr = nullptr;
        return;
    } else {
        if (setNameMapRef_[ipcMemInfo].Unref() == 0) {
            // 找到相同ipc 名字, 并且引用计数减为0再destroy
            SecIpcName_t memName = iter->second;
            ret = hrtIpcDestroyMemoryName(memName.ipcName);
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING("In mem repository, sal destroy ipc memory name ret[%d]", ret);
            }
            setNameMapRef_.erase(ipcMemInfo);
            setNameMap_.erase(iter);
        }
        ptr = nullptr;
    }
}

void MemNameRepository::ClearMemNameRepositoryImpl()
{
    setNameMap_.clear();
    openedNameMap_.clear();
    setNameMapRef_.clear();
    openedNameMapRef_.clear();
    alignPtrMap_.clear();
}

void MemNameRepository::ClearMemNameRepository()
{
    std::unique_lock<std::mutex> lock(memMutex_);
    ClearMemNameRepositoryImpl();
}
} // namespace hccl
