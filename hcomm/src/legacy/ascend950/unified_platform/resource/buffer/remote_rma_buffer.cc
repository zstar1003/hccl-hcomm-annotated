/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "remote_rma_buffer.h"
#include "null_ptr_exception.h"
#include "exchange_ub_buffer_dto.h"
#include "exchange_ipc_buffer_dto.h"
#include "exchange_rdma_buffer_dto.h"
#include "rdma_handle_manager.h"
namespace Hccl {
RemoteIpcRmaBuffer::RemoteIpcRmaBuffer() : RemoteRmaBuffer(RmaType::IPC), isOpened(false) {}

RemoteIpcRmaBuffer::RemoteIpcRmaBuffer(const Serializable& rmtDto) : RemoteRmaBuffer(RmaType::IPC), isOpened(false)
{
    const auto& dto = dynamic_cast<const ExchangeIpcBufferDto&>(rmtDto);
    remotePid = dto.pid;
    ipcAddr = dto.addr;
    ipcOffset = dto.offset;
    size = dto.size;
    memInfo = dto.memInfo;
    (void)memcpy_s(ipcName, RTS_IPC_MEM_NAME_LEN, dto.name, RTS_IPC_MEM_NAME_LEN);
    HCCL_INFO(
        "[RemoteIpcRmaBuffer][RemoteIpcRmaBuffer]ipcAddr[%llu] ipcOffset[%llu] ipcName[%s] memInfo[%s]", ipcAddr,
        ipcOffset, ipcName, memInfo.c_str());
    myPid = HrtDeviceGetBareTgid();
    if (myPid == remotePid) {
        HCCL_INFO("RemoteIpcRmaBuffer: myPid is equal to remotePid, do not need to open memory");
        HrtMemPrefetchToDevice(reinterpret_cast<void*>(ipcAddr + ipcOffset), size);
        addr = ipcAddr + ipcOffset;
    } else {
        HCCL_INFO("RemoteIpcRmaBuffer: open memory.");
        ipcPtr = HrtIpcOpenMemory(ipcName);
        addr = reinterpret_cast<uintptr_t>(ipcPtr) + ipcOffset;
        isOpened = true;
    }
}

RemoteIpcRmaBuffer::RemoteIpcRmaBuffer(const Serializable& rmtDto, const string tag)
    : RemoteRmaBuffer(RmaType::IPC),
      isOpened(true)
{
    const auto& dto = dynamic_cast<const ExchangeIpcBufferDto&>(rmtDto);
    HCCL_INFO("[RemoteIpcRmaBuffer][RemoteIpcRmaBuffer] dtoName[%s]", dto.name);
    ipcAddr = dto.addr;
    ipcOffset = dto.offset;
    size = dto.size;
    memInfo = dto.memInfo;
    (void)memcpy_s(ipcName, RTS_IPC_MEM_NAME_LEN, dto.name, RTS_IPC_MEM_NAME_LEN);
    HCCL_INFO(
        "[RemoteIpcRmaBuffer][RemoteIpcRmaBuffer] tag[%s] ipcAddr[%llu] ipcOffset[%llu] ipcName[%s] memInfo[%s]",
        tag.c_str(), ipcAddr, ipcOffset, ipcName, memInfo.c_str());
    ipcPtr = HrtIpcOpenMemory(ipcName);
    addr = reinterpret_cast<uintptr_t>(ipcPtr) + ipcOffset;
    isOpened = true;
}

void RemoteIpcRmaBuffer::Close() const
{
    if (isOpened) {
        HrtIpcCloseMemory(ipcName);
    }
}

RemoteIpcRmaBuffer::~RemoteIpcRmaBuffer() { DECTOR_TRY_CATCH("RemoteIpcRmaBuffer", Close()); }

string RemoteIpcRmaBuffer::Describe() const
{
    return StringFormat(
        "RemoteIpcRmaBuffer[addr=0x%llx, size=0x%llx, myPid=%u, "
        "remotePid=%u, ipcAddr=0x%llx, ipcOffset=0x%llx, ipcPtr=%p, ipcName=%s, "
        "isOpened=%d, pathMode=%u]",
        addr, size, myPid, remotePid, ipcAddr, ipcOffset, ipcPtr, ipcName, isOpened, pathMode_);
}

RemoteRdmaRmaBuffer::RemoteRdmaRmaBuffer(RdmaHandle rdmaHandle)
    : RemoteRmaBuffer(RmaType::RDMA),
      rdmaHandle(rdmaHandle),
      keyValidLen(RDMA_MEM_KEY_LEN_ROCE)
{
    if (rdmaHandle == nullptr) { // 使用rdmaHandle调用 HCCP 新接口 import/unimport 接口，获取和销毁key
        THROW<NullPtrException>("RemoteRdmaRmaBuffer's rdmaHandle is nullptr");
    }
    // 待修改: 利用 rdmaHandle 从 HCCP 新接口获取keyValidLen, 暂定固定值 ROCE
}

RemoteRdmaRmaBuffer::RemoteRdmaRmaBuffer(RdmaHandle rdmaHandle, const Serializable& rmtDto)
    : RemoteRmaBuffer(RmaType::RDMA),
      rdmaHandle(rdmaHandle)
{
    auto dto = dynamic_cast<const ExchangeRdmaBufferDto&>(rmtDto);
    addr = dto.addr;
    size = dto.size;
    rkey = dto.rkey;
    memInfo = dto.memInfo;
    HCCL_INFO("[RemoteRdmaRmaBuffer]addr = 0x%llx; size = 0x%llx; memInfo = %s", addr, size, memInfo.c_str());
}

RemoteRdmaRmaBuffer::~RemoteRdmaRmaBuffer()
{
    // 待修改:  使用rdmaHandle调用 HCCP 新接口 unimport 接口，销毁key
}

string RemoteRdmaRmaBuffer::Describe() const
{
    return StringFormat("RemoteRdmaRmaBuffer[addr=0x%llx, size=0x%llx]", addr, size);
}

RemoteUbRmaBuffer::RemoteUbRmaBuffer(RdmaHandle rdmaHandle) : RemoteRmaBuffer(RmaType::UB), rdmaHandle(rdmaHandle)
{
    if (rdmaHandle == nullptr) {
        THROW<NullPtrException>("RemoteUbRmaBuffer's rdmaHandle is nullptr");
    }
}

RemoteUbRmaBuffer::~RemoteUbRmaBuffer()
{
    if (memHandle != 0) {
        if (rdmaHandle == nullptr || !RdmaHandleManager::GetInstance().IsHandleValid(rdmaHandle)) {
            HCCL_WARNING(
                "[RemoteUbRmaBuffer][%s] skip HrtRaUbRemoteMemUnimport, "
                "rdmaHandle=%p invalid (DeInit/DestroyAll done), memHandle=0x%llx",
                __func__, rdmaHandle, static_cast<unsigned long long>(memHandle));
        } else {
            DECTOR_TRY_CATCH("RemoteUbRmaBuffer", HrtRaUbRemoteMemUnimport(rdmaHandle, memHandle));
        }
        memHandle = 0;
    }
}

RemoteUbRmaBuffer::RemoteUbRmaBuffer(
    uintptr_t addr, u64 size, u32 tokenId, u32 tokenValue, HcclMemType memType, const std::string& memInfo)
    : RemoteRmaBuffer(RmaType::UB),
      tokenId(tokenId),
      tokenValue(tokenValue)
{
    this->addr = addr;
    this->size = size;
    this->memType = memType;
    this->memInfo = memInfo;
    HCCL_INFO("[RemoteUbRmaBuffer]addr = 0x%llx; size = 0x%llx; memInfo = %s", addr, size, memInfo.c_str());
}

RemoteUbRmaBuffer::RemoteUbRmaBuffer(RdmaHandle rdmaHandle1, const Serializable& rmtDto)
    : RemoteRmaBuffer(RmaType::UB),
      rdmaHandle(rdmaHandle1)
{ // 从 DTO 取得数据，然后生成 memHandle
    auto dto = dynamic_cast<const ExchangeUbBufferDto&>(rmtDto);
    memcpy_s(key, HRT_UB_MEM_KEY_MAX_LEN, dto.key, HRT_UB_MEM_KEY_MAX_LEN);
    addr = dto.addr;
    size = dto.size;
    memType = dto.memType;
    memInfo = dto.memInfo;
    tokenId = dto.tokenId;
    tokenValue = dto.tokenValue;
    keySize = dto.keySize;
    notifyId = dto.notifyId;

    if (keySize != 0) {
        auto res = HrtRaUbRemoteMemImport(rdmaHandle1, key, keySize, tokenValue);
        memHandle = res.handle;
        segVa = res.targetSegVa;
    } else {
        HCCL_INFO("[RemoteUbRmaBuffer] key is 0, do not need to import memory");
        memHandle = 0;
    }
    HCCL_INFO("Construct RemoteUbRmaBuffer:%s", Describe().c_str());
}

string RemoteUbRmaBuffer::Describe() const
{
    return StringFormat(
        "RemoteUbRmaBuffer[rdmaHandle=%p, addr=0x%llx, size=0x%llx, memHandle=%p segVa=%llu, notifyId=%u]", rdmaHandle,
        addr, size, memHandle, segVa, notifyId);
}

} // namespace Hccl
