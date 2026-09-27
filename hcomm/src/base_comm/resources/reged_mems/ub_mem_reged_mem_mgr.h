/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UB_MEM_REGED_MEM_MGR_H
#define UB_MEM_REGED_MEM_MGR_H

#include <memory>
#include <vector>
#include <string>
#include "reged_mem_mgr.h"
#include "rma_buffer_mgr.h"
#include "buffer_key.h"
#include "local_ub_rma_buffer.h"
#include "remote_rma_buffer.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/buffer/local_ipc_rma_buffer_v2.h"

namespace hcomm {

class UbMemRegedMemMgr : public RegedMemMgr {
public:
    using LocalIpcRmaBufferMgr
        = hcomm::RmaBufferMgr<hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<Hccl::LocalIpcRmaBuffer>>;

    UbMemRegedMemMgr();
    ~UbMemRegedMemMgr() override = default;

    HcclResult RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle) override;
    HcclResult UnregisterMemory(void* memHandle) override;
    HcclResult
    MemoryExport(const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen) override;
    HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) override;
    HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) override;
    HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) override;

private:
    mutable std::mutex memMtx_;
    std::unique_ptr<LocalIpcRmaBufferMgr> localIpcRmaBufferMgr_{};
    std::vector<RegedBufferEntry<Hccl::LocalIpcRmaBuffer>> allRegisteredBuffers_;
};
} // namespace hcomm

#endif // UB_MEM_REGED_MEM_MGR_H
