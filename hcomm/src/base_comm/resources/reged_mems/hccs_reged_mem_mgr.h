/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCS_REGED_MEM_MGR_H
#define HCCS_REGED_MEM_MGR_H

#include <memory>
#include <vector>
#include <string>
#include "reged_mem_mgr.h"
#include "rma_buffer_mgr.h"
#include "buffer_key.h"
#include "hccl_mem_defs.h"
#include "inner/remote_ipc_rma_buffer.h"
#include "inner/local_ipc_rma_buffer.h"

namespace hcomm {
/**
 * @note 职责：用于通信设备EndPoint的注册内存信息管理，支持基于RmaBufferMgr类的重叠内存的检测报错等。
 */
class HccsRegedMemMgr : public RegedMemMgr {
public:
    using LocalIpcRmaBufferMgr
        = hcomm::RmaBufferMgr<hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<hccl::LocalIpcRmaBuffer>>;
    using RemoteIpcRmaBufferMgr
        = hcomm::RmaBufferMgr<hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<hccl::RemoteIpcRmaBuffer>>;

    HccsRegedMemMgr(HcclNetDevCtx netDevCtx);
    ~HccsRegedMemMgr() override;

    HcclResult RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle) override;
    HcclResult UnregisterMemory(void* memHandle) override;
    HcclResult
    MemoryExport(const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen) override;
    HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) override;
    HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) override;
    HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) override;

    HcclResult MemoryGrant(const HcommMemGrantInfo* remoteGrantInfo);
    HcclResult MemoryEnableP2P(const EndpointDesc& localEndpointDesc, const EndpointDesc& remoteEndpointDesc) const;
    HcclResult MemoryDisableP2P(const EndpointDesc& localEndpointDesc, const EndpointDesc& remoteEndpointDesc) const;
    HcclResult MemoryOpenRemoteIpc();
    HcclResult MemoryCloseRemoteIpc();
    HcclResult GetRemoteIpcRmaBuffer(std::vector<CommMem>& remoteIpcRmaBufferVec);
    HcclResult GetRemoteIpcRmaBufferEx(std::vector<HcclMemEx>& remoteIpcRmaBufferVecEx);
    HcclResult GetLocalIpcRmaBufferEx(std::vector<HcclMemEx>& localIpcRmaBufferVecEx) const;

private:
    HcclResult SerializeToMemDesc(
        const EndpointDesc& endpointDesc, hccl::LocalIpcRmaBuffer* localIpcRmaBuffer, void** memDesc,
        uint32_t* descLen) const;
    HcclResult MakeRemoteIpcRmaBuffer(
        std::string& ipcRmaBufferDesc, std::shared_ptr<hccl::RemoteIpcRmaBuffer>& remoteIpcRmaBuffer);
    HcclResult DeSerializeFromMemDesc(
        const void* memDesc, uint32_t descLen, EndpointDesc& endpointDesc,
        std::shared_ptr<hccl::RemoteIpcRmaBuffer>& remoteIpcRmaBuffer);

    HcclResult
    AddMem(hccl::BufferKey<uintptr_t, u64>& memKey, std::shared_ptr<hccl::RemoteIpcRmaBuffer>& remoteIpcRmaBuffer);
    HcclResult DeleteMem(hccl::BufferKey<uintptr_t, u64>& memKey);

    HcclNetDevCtx netDevCtx_{};
    std::vector<RegedBufferEntry<hccl::LocalIpcRmaBuffer>> allRegisteredBuffers_;
    std::vector<std::shared_ptr<hccl::LocalIpcRmaBuffer>> handlesRecords_;
    // for read/write with origin addr and len
    RemoteIpcRmaBufferMgr remoteIpcRmaBufferMgr_;
};
} // namespace hcomm

#endif // HCCS_REGED_MEM_MGR_H
