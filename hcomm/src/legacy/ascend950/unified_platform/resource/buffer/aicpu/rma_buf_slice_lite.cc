/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rma_buf_slice_lite.h"
#include "log.h"
#include "string_util.h"
namespace Hccl {
RmaBufSliceLite::RmaBufSliceLite(u64 addr, u64 size, u32 lkey, u32 tokenId)
    : addr_(addr),
      size_(size),
      lkey_(lkey),
      tokenId_(tokenId)
{
    HCCL_INFO("RmaBufSliceLite::RmaBufSliceLite:%s", Describe().c_str());
}

std::string RmaBufSliceLite::Describe() const
{
    return StringFormat("RmaBufSliceLite[addr=0x%llx, size=0x%llx]", addr_, size_);
}
} // namespace Hccl
