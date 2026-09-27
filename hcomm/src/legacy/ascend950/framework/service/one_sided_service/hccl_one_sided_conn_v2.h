/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ONE_SIDED_CONN_V2_H
#define HCCL_ONE_SIDED_CONN_V2_H

#include <hccl/hccl_types.h>
#include <hccl/base.h>
#include "socket_manager.h"
#include "hccl_one_sided_data.h"
#include "rma_buffer_mgr.h"
#include "exchange_ub_buffer_dto.h"
#include "local_ub_rma_buffer.h"
#include "remote_rma_buffer.h"
#include "kernel_param_lite.h"

namespace Hccl {
constexpr size_t TRANSPORT_EMD_ESC_SIZE = 512U - (sizeof(u32) * 2);

class TransportUrmaMem;

class HcclOneSidedConn {
public:
    HcclOneSidedConn(CommunicatorImpl* comm, LinkData linkData);
    ~HcclOneSidedConn();

    HcclResult Connect(const std::string& commId);
    void WaitOneSidedTransportReady();

    HcclResult ExchangeMemDesc(const HcclMemDescs& localMemDescs, HcclMemDescs& remoteMemDescs, u32& actualNumOfRemote);

    HcclResult EnableMemAccess(const HcclMemDesc& remoteMemDesc, HcclMem& remoteMem);
    HcclResult DisableMemAccess(const HcclMemDesc& remoteMemDesc);
    HcclResult BatchBufferSlice(
        const HcclOneSideOpDesc* oneSideDescs, u32 descNum,
        vector<HcclAicpuLocBufLite>& hostBatchPutGetLocalBufferSliceBufs,
        vector<HcclAicpuLocBufLite>& hostBatchPutGetRemoteBufferSliceBufs);

private:
    CommunicatorImpl* comm_{nullptr};
    LinkData linkData_;

    Socket* socket_{nullptr};
    std::shared_ptr<TransportUrmaMem> transportMemPtr_{};

    hcomm::RmaBufferMgr<BufferKey<uintptr_t, u64>, shared_ptr<HcclBuf>> remoteHcclBufMgr_{};
    std::unordered_map<std::string, shared_ptr<HcclBuf>> desc2HcclBufMapRemoteUb_{};
    std::unordered_map<std::string, HcclNetDev> desc2netDevMap_{};

private:
    HcclResult RmaMemDescCopyFromStr(RmaMemDesc& rmaMemDesc, const std::string& memDescStr) const
    {
        if (memcpy_s(rmaMemDesc.memDesc, TRANSPORT_EMD_ESC_SIZE, memDescStr.c_str(), memDescStr.size() + 1) != EOK) {
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }

    // 从 memDesc 转换为 string
    std::string RmaMemDescCopyToStr(const RmaMemDesc& rmaMemDesc) const
    {
        return std::string(rmaMemDesc.memDesc, TRANSPORT_EMD_ESC_SIZE);
    }

    HcclResult SendLocalMemDesc(const HcclMemDescs& localMemDescs);
    HcclResult ReceiveRemoteMemDesc(HcclMemDescs& remoteMemDescs, u32& actualNumOfRemote);
};
} // namespace Hccl
#endif // HCCL_ONE_SIDED_CONN_V2_H
