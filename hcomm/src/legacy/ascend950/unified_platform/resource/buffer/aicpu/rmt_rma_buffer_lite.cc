/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rmt_rma_buffer_lite.h"
#include "string_util.h"
#include "log.h"
namespace Hccl {
RmtRmaBufferLite::RmtRmaBufferLite(u64 addr, u64 size) : type_(RmaType::IPC), addr_(addr), size_(size)
{
    HCCL_INFO("RmtRmaBufferLite::RmtRmaBufferLite:%s", Describe().c_str());
}

RmtRmaBufferLite::RmtRmaBufferLite(u64 addr, u64 size, u32 rkey)
    : type_(RmaType::RDMA),
      addr_(addr),
      size_(size),
      rkey_(rkey)
{
    HCCL_INFO("RmtRmaBufferLite::RmtRmaBufferLite:%s", Describe().c_str());
}

RmtRmaBufferLite::RmtRmaBufferLite(u64 addr, u64 size, u32 tokenId, u32 tokenValue)
    : type_(RmaType::UB),
      addr_(addr),
      size_(size),
      tokenId_(tokenId),
      tokenValue_(tokenValue)
{
    HCCL_INFO("RmtRmaBufferLite::RmtRmaBufferLite:%s", Describe().c_str());
}

RmtRmaBufSliceLite RmtRmaBufferLite::GetRmtRmaBufSliceLite(u64 offset, u64 sliceSize) const
{
    return RmtRmaBufSliceLite(addr_ + offset, sliceSize, rkey_, tokenId_, tokenValue_, UINT32_MAX);
}

std::string RmtRmaBufferLite::Describe() const
{
    return StringFormat(
        "RmtRmaBufferLite[type=%s, addr=0x%llx, size=0x%x, rkey=%u]", type_.Describe().c_str(), addr_, size_, rkey_);
}
} // namespace Hccl
