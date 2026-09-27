/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rma_buffer_lite.h"
#include "log.h"
#include "string_util.h"
namespace Hccl {
RmaBufferLite::RmaBufferLite(u64 addr, u64 size) : type_(RmaType::IPC), addr_(addr), size_(size)
{
    HCCL_INFO("RmaBufferLite::RmaBufferLite:%s", Describe().c_str());
}

RmaBufferLite::RmaBufferLite(u64 addr, u64 size, u32 lkey) : type_(RmaType::RDMA), addr_(addr), size_(size), lkey_(lkey)
{
    HCCL_INFO("RmaBufferLite::RmaBufferLite:%s", Describe().c_str());
}

RmaBufferLite::RmaBufferLite(u64 addr, u64 size, u32 tokenId, u32 tokenValue)
    : type_(RmaType::UB),
      addr_(addr),
      size_(size),
      tokenId_(tokenId),
      tokenValue_(tokenValue)
{
    HCCL_INFO("RmaBufferLite::RmaBufferLite:%s", Describe().c_str());
}

RmaBufSliceLite RmaBufferLite::GetRmaBufSliceLite(u64 offset, u32 sliceSize) const
{
    return RmaBufSliceLite(addr_ + offset, sliceSize, lkey_, tokenId_);
}

std::string RmaBufferLite::Describe() const
{
    return StringFormat("RmaBufferLite[type=%s, addr=0x%llx, size=0x%llx", type_.Describe().c_str(), addr_, size_);
}
} // namespace Hccl
