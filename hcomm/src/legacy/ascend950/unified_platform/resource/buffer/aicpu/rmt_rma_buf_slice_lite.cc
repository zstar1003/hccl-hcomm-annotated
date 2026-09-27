/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rmt_rma_buf_slice_lite.h"
#include "log.h"
#include "string_util.h"
namespace Hccl {
RmtRmaBufSliceLite::RmtRmaBufSliceLite(u64 addr, u64 size, u32 rkey, u32 tokenId, u32 tokenValue, u32 notifyId)
    : addr_(addr),
      size_(size),
      rkey_(rkey),
      tokenId_(tokenId),
      tokenValue_(tokenValue),
      notifyId_(notifyId)
{
    HCCL_INFO("RmtRmaBufSliceLite::RmtRmaBufSliceLite:%s", Describe().c_str());
}

std::string RmtRmaBufSliceLite::Describe() const
{
    return StringFormat(
        "RmtRmaBufSliceLite[addr=0x%llx, size=0x%llx, rkey=%u, notifyId=%u]", addr_, size_, rkey_, notifyId_);
}

} // namespace Hccl
