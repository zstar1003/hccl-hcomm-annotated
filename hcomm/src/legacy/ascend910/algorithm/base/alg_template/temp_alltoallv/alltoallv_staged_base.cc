/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_staged_base.h"
#include "log.h"

namespace hccl {
using namespace std;

AlltoAllVStagedBase::AlltoAllVStagedBase(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AlltoAllVStagedBase::~AlltoAllVStagedBase() {}

HcclResult AlltoAllVStagedBase::Prepare(
    DeviceMem& sendMem, DeviceMem& recvMem, StageAlltoAllVAddrInfo& sendAddrInfo, StageAlltoAllVAddrInfo& recvAddrInfo,
    bool isAlltoAllZCopyMode, Stream& mainStream)
{
    sendMem_ = sendMem;
    recvMem_ = recvMem;
    sendAddrInfo_ = sendAddrInfo;
    recvAddrInfo_ = recvAddrInfo;
    isAlltoAllZCopyMode_ = isAlltoAllZCopyMode;
    mainStreamPtr_ = &mainStream;
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVStagedBase::LocalCopy(u32 rank)
{
    if (sendAddrInfo_.find(rank) == sendAddrInfo_.end()) {
        HCCL_ERROR("[AlltoAllVStaged][LocalCopy] Invalid Rank[%u]", rank);
        return HCCL_E_INTERNAL;
    }

    for (auto it = sendAddrInfo_[rank].begin(); it != sendAddrInfo_[rank].end(); it++) {
        u8* dstAddr = static_cast<u8*>(recvMem_.ptr()) + it->remoteOffset;
        u64 destMax = it->remoteLength;
        u8* srcAddr = static_cast<u8*>(sendMem_.ptr()) + it->localOffset;
        u64 size = it->localLength;
        DeviceMem dst = DeviceMem::create(dstAddr, destMax);
        DeviceMem src = DeviceMem::create(srcAddr, size);

        HCCL_DEBUG(
            "[AlltoAllVStagedPairwise][LocalCopy]: Rank[%u] destAddr[%p], destMax[%llu], srcAddr[%p], size[%llu]", rank,
            dstAddr, destMax, srcAddr, size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, *mainStreamPtr_));
    }

    return HCCL_SUCCESS;
}
} // namespace hccl
