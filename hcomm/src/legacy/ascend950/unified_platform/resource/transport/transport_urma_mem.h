/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_URMA_MEM_H
#define TRANSPORT_URMA_MEM_H

#include "hccl_mem.h"
#include "net_device.h"
#include "local_ub_rma_buffer.h"
#include "remote_rma_buffer.h"
#include "ub_mem_transport.h"
#include "hccl_one_sided_data.h"
#include "rma_buffer_mgr.h"
#include "../mem/local_ub_rma_buffer_manager.h"

namespace Hccl {

struct RmaOpMem {
    void* addr;
    u64 size;
};

constexpr u64 MAX_DESC_NUM = 64; // 批量操作描述符个数上限

class TransportUrmaMem {
public:
    TransportUrmaMem(
        BaseMemTransport* transport,
        hcomm::RmaBufferMgr<BufferKey<uintptr_t, u64>, shared_ptr<HcclBuf>>& remoteHcclBufMgr);

    ~TransportUrmaMem();

    HcclResult BatchBufferSlice(
        const HcclOneSideOpDesc* oneSideDescs, u32 descNum, RmaBufferSlice* localRmaBufferSlice,
        RmtRmaBufferSlice* remoteRmaBufferSlice);

private:
    BaseMemTransport* transport_;

    hcomm::RmaBufferMgr<BufferKey<uintptr_t, u64>, shared_ptr<HcclBuf>>& remoteHcclBufMgr_;
    HcclNetDev netDev_;

private:
    HcclResult FillRmaBufferSlice(
        const RmaOpMem& localMem, const RmaOpMem& remoteMem, RmaBufferSlice& localRmaBufferSlice,
        RmtRmaBufferSlice& remoteRmaBufferSlice);
};

} // namespace Hccl
#endif // TRANSPORT_URMA_MEM_H
