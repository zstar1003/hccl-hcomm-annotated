/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_ROCE_REGED_MEM_MGR_H
#define AICPU_TS_ROCE_REGED_MEM_MGR_H

#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "reged_mem_mgr.h"
#include "hccl_mem_defs.h"
#include "hccl_network.h"
#include "transport_pub.h"
#include "local_rdma_rma_buffer.h"
#include "remote_rdma_rma_buffer.h"

namespace hcomm {
class AicpuTsRoceRegedMemMgr : public RegedMemMgr {
public:
    AicpuTsRoceRegedMemMgr(HcclNetDev netDev, RdmaHandle rdmaHandle);
    ~AicpuTsRoceRegedMemMgr() override = default;

    HcclResult RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle) override;
    HcclResult UnregisterMemory(void* memHandle) override;
    HcclResult
    MemoryExport(const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen) override;
    HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) override;
    HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) override;
    HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) override;

    HcclResult GetAllMemDetails(std::vector<RoceMemDetails>& localOut, std::vector<RoceMemDetails>& remoteOut) const;

private:
    using RemoteRdmaRmaBufferMgr
        = hcomm::RmaBufferMgr<hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<hccl::RemoteRdmaRmaBuffer>>;

    struct EndpointDescLess {
        bool operator()(const EndpointDesc& a, const EndpointDesc& b) const noexcept
        {
            return std::memcmp(&a, &b, sizeof(EndpointDesc)) < 0;
        }
    };

    HcclResult GetParamsFromMemDesc(
        const void* memDesc, uint32_t descLen, EndpointDesc& endpointDesc, std::string& rdmaBlob) const;
    void TrackRegisteredBuffer(const std::shared_ptr<hccl::LocalRdmaRmaBuffer>& localBuffer);
    // 注销流程的记录清理：exportDescByBuffer_ 擦除、allRegisteredBuffers_ 按 IsInTree 判定擦除或标记、
    // hcclBufRecords_ 移除该 handle 的条目。入参为本条注销涉及的 key 与 buffer 指针
    void CleanupBufferRecords(
        const void* memHandle, hccl::LocalRdmaRmaBuffer* buffer, const hccl::BufferKey<uintptr_t, u64>& ownKey);

    HcclResult GatherLocalMemDetails(std::vector<RoceMemDetails>& localOut) const;
    HcclResult AppendLocalNotifyMemDetails(std::vector<RoceMemDetails>& localOut) const;
    void GatherRemoteMemDetails(std::vector<RoceMemDetails>& remoteOut) const;

    HcclNetDev netDev_{nullptr};
    RdmaHandle rdmaHandle_{nullptr};
    std::shared_ptr<hccl::NetDevContext::LocalRdmaRmaBufferMgr> localRdmaRmaBufferMgr_{nullptr};
    std::vector<RegedBufferEntry<hccl::LocalRdmaRmaBuffer>> allRegisteredBuffers_;
    std::vector<HcclBuf> hcclBufRecords_;
    std::unordered_map<hccl::LocalRdmaRmaBuffer*, std::vector<char>> exportDescByBuffer_;
    std::map<EndpointDesc, std::unique_ptr<RemoteRdmaRmaBufferMgr>, EndpointDescLess> remoteRdmaRmaBufferMgrs_;
};

} // namespace hcomm
#endif // AICPU_TS_ROCE_REGED_MEM_MGR_H
