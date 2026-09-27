/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RMT_RMA_BUFFER_LITE_H
#define HCCLV2_RMT_RMA_BUFFER_LITE_H
#include <string>

#include "hccl/base.h"
#include "rmt_rma_buf_slice_lite.h"
#include "rma_type.h"
namespace Hccl {

class RmtRmaBufferLite {
public:
    RmtRmaBufferLite() = default;
    RmtRmaBufferLite(u64 addr, u64 size);
    RmtRmaBufferLite(u64 addr, u64 size, u32 rkey);
    RmtRmaBufferLite(u64 addr, u64 size, u32 tokenId, u32 tokenValue);

    RmtRmaBufSliceLite GetRmtRmaBufSliceLite(u64 offset, u64 sliceSize) const;

    inline u64 GetAddr() const { return addr_; }
    inline u64 GetSize() const { return size_; }
    inline u32 GetTokenId() const { return tokenId_; }
    inline u32 GetTokenValue() const { return tokenValue_; }
    inline u32 GetRkey() const { return rkey_; }

    std::string Describe() const;

private:
    RmaType type_;
    u64 addr_{0};
    u64 size_{0};
    u32 rkey_{0};
    u32 tokenId_{0};
    u32 tokenValue_{0};
};

} // namespace Hccl
#endif // HCCL_BUFFER_H
