/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_EXPERIMENTAL_URMA_MEM_H
#define HCOMM_EXPERIMENTAL_URMA_MEM_H

#include <memory>
#include <vector>
#include <string>
#include "reged_mem_mgr.h"
#include "rma_buffer_mgr.h"
#include "buffer_key.h"
#include "local_ub_rma_buffer.h"
#include "remote_rma_buffer.h"
#include "exchange_ub_buffer_dto.h"

namespace hcomm_experimental {
/**
 * @note 职责：用于通信设备EndPoint的注册内存信息管理，支持基于RmaBufferMgr类的重叠内存的检测报错等。
 */
class UbRegedMemMgr : public RegedMemMgr {
public:
    using LocalUbRmaBufferMgr
        = hcomm::RmaBufferMgr<Hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<Hccl::LocalUbRmaBuffer>>;
    using RemoteUbRmaBufferMgr
        = hcomm::RmaBufferMgr<Hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<Hccl::RemoteUbRmaBuffer>>;

    UbRegedMemMgr();
    ~UbRegedMemMgr() = default;

    HcclResult RegisterMemory(HcommMem mem, const char* memTag, void** memHandle) override;
    HcclResult UnregisterMemory(void* memHandle) override;
    HcclResult
    MemoryExport(const EndpointDesc endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen) override;
    HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) override;
    HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) override;
    HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) override;
    HcclResult GetMemDesc(const EndpointDesc endpointDesc, Hccl::LocalUbRmaBuffer* localUbRmaBuffer);
    HcclResult GetParamsFromMemDesc(
        const void* memDesc, uint32_t descLen, EndpointDesc& endpointDesc, Hccl::ExchangeUbBufferDto& dto);

private:
    std::unique_ptr<LocalUbRmaBufferMgr> localUbRmaBufferMgr_{};
    std::vector<std::shared_ptr<Hccl::LocalUbRmaBuffer>> allRegisteredBuffers_;
    std::unordered_map<EndpointDesc, std::unique_ptr<RemoteUbRmaBufferMgr>> remoteUbRmaBufferMgrs_;
};
} // namespace hcomm_experimental

#endif // HCOMM_EXPERIMENTAL_URMA_MEM_H
