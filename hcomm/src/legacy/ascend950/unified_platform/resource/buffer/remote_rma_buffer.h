/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_REMOTE_RMA_BUFFER_H
#define HCCLV2_REMOTE_RMA_BUFFER_H

#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "rma_type.h"
#include "serializable.h"
#include <hcomm_res_defs.h>
#include "hccl_mem_defs.h"
namespace Hccl {

class RemoteRmaBuffer {
public:
    explicit RemoteRmaBuffer(const RmaType rmaType) : rmaType(rmaType) {}

    virtual ~RemoteRmaBuffer() = default;

    RmaType GetRmaType() const { return rmaType; }

    inline uintptr_t GetAddr() const { return addr; }

    inline u64 GetSize() const { return size; }

    inline HcclMemType GetMemType() const { return memType; }

    inline const std::string GetMemInfo() const { return memInfo; }

    u64 GetMemHandle() const { return memHandle; }

    virtual std::string Describe() const = 0;

protected:
    uintptr_t addr{0};
    u64 size{0};
    RmaType rmaType;
    HcclMemType memType;
    std::string memInfo;
    u64 memHandle{0};
};

class RemoteIpcRmaBuffer : public RemoteRmaBuffer {
public:
    RemoteIpcRmaBuffer();

    explicit RemoteIpcRmaBuffer(const Serializable& rmtDto);

    RemoteIpcRmaBuffer(const Serializable& rmtDto, const std::string tag);

    ~RemoteIpcRmaBuffer() override;

    RemoteIpcRmaBuffer(const RemoteIpcRmaBuffer& that) = delete;

    RemoteIpcRmaBuffer& operator=(const RemoteIpcRmaBuffer& that) = delete;

    std::string Describe() const override;

    void SetPathMode(uint8_t pathMode) { pathMode_ = pathMode; }

private:
    void Close() const;

    uint8_t pathMode_{0};

    char ipcName[RTS_IPC_MEM_NAME_LEN]{0};
    u64 ipcAddr{0};
    u64 ipcOffset{0};
    void* ipcPtr{};
    u32 remotePid{0};
    u32 myPid{0};
    bool isOpened;
};

class RemoteRdmaRmaBuffer : public RemoteRmaBuffer {
public:
    explicit RemoteRdmaRmaBuffer(RdmaHandle rdmaHandle);

    RemoteRdmaRmaBuffer(RdmaHandle rdmaHandle, const Serializable& rmtDto);

    ~RemoteRdmaRmaBuffer() override;

    RemoteRdmaRmaBuffer(const RemoteRdmaRmaBuffer& that) = delete;

    RemoteRdmaRmaBuffer& operator=(const RemoteRdmaRmaBuffer& that) = delete;

    std::string Describe() const override;

    const u8* GetKey() const { return key; }

    u32 GetRkey() const { return rkey; }

private:
    RdmaHandle rdmaHandle{nullptr};
    u8 exchangedKey[RDMA_MEM_KEY_MAX_LEN]{0};
    u8 key[RDMA_MEM_KEY_MAX_LEN]{0};
    u32 keyValidLen{0};
    u32 rkey{0};
};

class RemoteUbRmaBuffer : public RemoteRmaBuffer {
public:
    explicit RemoteUbRmaBuffer(RdmaHandle rdmaHandle);

    RemoteUbRmaBuffer(
        uintptr_t addr, u64 size, u32 tokenId, u32 tokenValue, HcclMemType memType, const std::string& memInfo);

    RemoteUbRmaBuffer(RdmaHandle rdmaHandle1, const Serializable& rmtDto);

    ~RemoteUbRmaBuffer() override;

    RemoteUbRmaBuffer(const RemoteUbRmaBuffer& that) = delete;

    RemoteUbRmaBuffer& operator=(const RemoteUbRmaBuffer& that) = delete;

    std::string Describe() const final;

    uint32_t GetTokenId() const { return tokenId; }

    uint32_t GetTokenValue() const { return tokenValue; }

    uint64_t GetSegVa() const { return segVa; }

    uint32_t GetNotifyId() const { return notifyId; }

private:
    RdmaHandle rdmaHandle{nullptr};
    u8 key[HRT_UB_MEM_KEY_MAX_LEN]{0};
    u32 tokenId{0};
    u32 tokenValue{0};
    u32 keySize{0};
    u64 segVa{0};
    u32 notifyId{UINT32_MAX};
};

} // namespace Hccl
#endif
